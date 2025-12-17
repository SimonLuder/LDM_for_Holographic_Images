import os
import sys
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from pollen_datasets.poleno import PairwiseHolographyImageFolder

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from model.unet_v2 import UNet
from model.vqvae import VQVAE
from model.ddpm import Diffusion as DDPMDiffusion
from model.discriminator import DiffusionPatchGanDiscriminator
from model.conditioning import custom_conditions # required
from model.conditioning.transforms.registry import get_transforms
from model.conditioning.registry import build_encoder_from_registry
from utils.wandb import WandbManager
from utils.config import load_config
from utils.train_test_utils import get_image_encoder_names
from .ldm_validate_novel import validate


def train(config):

    # load configuration
    run_name                = config['name']
    config_path             = config['config_path']
    dataset_cfg             = config['dataset']
    transforms_cfg          = config['transforms']
    condition_cfg           = config['conditioning']
    ddpm_cfg                = config['ddpm']
    autoencoder_cfg         = config['autoencoder']
    train_cfg               = config['ldm_train']
    vae_model_cfg           = load_config(autoencoder_cfg["config_file"])["autoencoder"]

    ldm_ckpt_name = train_cfg['ldm_ckpt_name']
    discriminator_ckpt_name = train_cfg['ldm_discriminator_ckpt_name']

    train_with_discriminator = train_cfg['ldm_discriminator_weight'] > 0 and train_cfg['ldm_discriminator_start_step'] > 0
    train_with_perceptual_loss = train_cfg['ldm_perceptual_weight'] > 0

    # train on GPU if it is available else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup WandbManager
    wandb_manager = WandbManager(project="MSE_P9_LDM", run_name=run_name, config=config)
    # init run
    wandb_run = wandb_manager.get_run()

    # create checkpoints and sample paths
    Path(os.path.join(train_cfg['ckpt_folder'], run_name, ldm_ckpt_name)).mkdir(parents=True, exist_ok=True)  
    Path(os.path.join(train_cfg['ckpt_folder'], run_name, discriminator_ckpt_name)).mkdir(parents=True, exist_ok=True)   

    # copy config to checkpoint folder
    shutil.copyfile(config_path, os.path.join(train_cfg['ckpt_folder'], 
                                              run_name,
                                              os.path.basename(config_path)))

    ###################################### data ######################################

    # Transforms
    transform1 = get_transforms(transforms_cfg["transform1"], in_channels=dataset_cfg["img_channels"])
    transform2 = get_transforms(transforms_cfg["transform2"], in_channels=dataset_cfg["img_channels"])

    print(transform1)
    print(transform2)

    # Dataset
    dataset = PairwiseHolographyImageFolder(
        root=dataset_cfg["root"], 
        transform1=transform1,
        transform2=transform2, 
        dataset_cfg=dataset_cfg,
        cond_cfg=condition_cfg,
        labels=dataset_cfg.get("labels_train")
    )

    # Dataloader
    print("Initialize Dataloader")
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['ldm_batch_size'],
        num_workers=train_cfg.get('num_workers', 4),
        prefetch_factor=2,
        pin_memory=True,
        shuffle=True
    )
    
    ################################## autoencoder ###################################
    # Load Autoencoder if latents are not precalculated or are missing
    if os.path.exists(train_cfg["vqvae_latents_representations"]):
        latents_available = len(os.listdir(train_cfg["vqvae_latents_representations"])) > 0
    else:
        latents_available = False

    if not latents_available:
        print('Loading vqvae model as no latents found')
        vae = VQVAE(img_channels=dataset_cfg['img_channels'], config=vae_model_cfg).to(device)
        vae.eval()

        # Load VQVAE weights from checkpoint
        vae_ckpt_path = autoencoder_cfg["weights"]
        
        vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))
        print('Loaded autoencoder checkpoint')
        
        for param in vae.parameters():
            param.requires_grad = False

    ############################### context encoding ################################
    # Load context encoder
    if condition_cfg["enabled"] == "unconditional":
        print("Training unconditional model")
        context_encoder = None

    else:
        use_condition = condition_cfg["enabled"].split("+")
        used_image_encoders = get_image_encoder_names(condition_cfg)
        context_encoder = build_encoder_from_registry(
            wrapper_out_dim=ddpm_cfg['cond_emb_dim'],
            cond_cfg=condition_cfg,
            device=device
            )
        context_encoder.to(device)
        

    ##################################### u-net ######################################
    # UNet
    model = UNet(img_channels=vae_model_cfg['z_channels'], 
                 model_config=ddpm_cfg,
                 context_encoder=context_encoder).to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=train_cfg['ldm_lr'])

    criterion = torch.nn.MSELoss()

    # discriminator model
    if train_with_discriminator:

        discriminator = DiffusionPatchGanDiscriminator(img_channels=vae_model_cfg['z_channels']).to(device)
        discriminator.train()

        optimizer_d = Adam(discriminator.parameters(), lr=train_cfg['ldm_disc_lr'])
        # d_criterion = torch.nn.MSELoss()
        loss_functions = {"MSELoss" : torch.nn.MSELoss, 
                          "BCEWithLogits" : torch.nn.BCEWithLogitsLoss,}.get(
                              train_cfg["ldm_discriminator_loss"])
        d_criterion = loss_functions()

    discriminator_start_step = train_cfg["ldm_discriminator_start_step"] # starts discriminator after n batches
    
    # Load Diffusion Process
    if dataset_cfg.get('img_interpolation'):
        img_size = dataset_cfg['img_interpolation'] // 2 ** sum(vae_model_cfg['down_sample'])
    else:
        img_size = dataset_cfg['img_size'] // 2 ** sum(vae_model_cfg['down_sample'])
    
    diffusion = DDPMDiffusion(img_size=img_size, 
                              img_channels=vae_model_cfg['z_channels'],
                              noise_schedule="linear", 
                              beta_start=train_cfg["ldm_beta_start"], 
                              beta_end=train_cfg["ldm_beta_end"],
                              device=device,
                              )
    
    # lpips
    if train_with_perceptual_loss:
        lpips_model = LPIPS(net_type='alex').to(device)

    num_epochs = train_cfg['ldm_epochs']
    steps_per_optimization = train_cfg['ldm_steps_per_optimization']
    step_count = 0

    print(f"Start training with perceptual loss: {train_with_perceptual_loss}, discriminator: {train_with_perceptual_loss}")

    for epoch_idx in range(num_epochs):
        losses = []
        d_losses = []
        lpips_losses = []

        pbar = tqdm(dataloader)
        for (im1, im2), (cond1, cond2), _ in pbar:

            im1 = im1.float().to(device)
            im2 = im2.float().to(device)

            for cond in use_condition:

                # Use image as input condition
                if cond in used_image_encoders:
                    cond1[cond] = im2

                cond1[cond] = cond1[cond].to(device)
                cond2[cond] = cond2[cond].to(device)

            # randomly discard conditioning to train unconditionally if conditional training
            if condition_cfg["enabled"] == "unconditional" or np.random.random() < train_cfg["ldm_cfg_discard_prob"]:
                cond1 = None
                cond2 = None

            # train with discriminator?
            use_discriminator = (
                train_with_discriminator
                and step_count > discriminator_start_step
            )

            # autencode samples
            if not latents_available:
                with torch.no_grad():
                    im1, _ = vae.encode(im1)

            # sample timestep
            t = diffusion.sample_timesteps(im1.shape[0]).to(device)

            # noise image
            if use_discriminator:
                x_t, noise, x_t_neg_1 = diffusion.noise_images(im1, t, x_t_neg_1=True)
            else:
                x_t, noise = diffusion.noise_images(im1, t, x_t_neg_1=False)

            # predict noise
            noise_pred = model(x_t, t, cond1)

            # create image less noisy
            if use_discriminator:
                x_t_neg_1_pred = diffusion.denoising_step(x_t, t, noise_pred)

            # reconstruction loss
            loss = criterion(noise_pred, noise)

            logs = {"epoch" :               epoch_idx + 1,
                    "step" :                step_count + 1,
                    "loss" :                np.mean(loss.item()),
                    }

            # remove all predicted noise
            im_pred = x_t - noise_pred 

            # lpips loss
            if train_with_perceptual_loss:
                lpips_in = torch.clamp(im1, -1., 1.)
                lpips_in_pred = torch.clamp(im_pred, -1., 1.)

                if lpips_in.shape[1] == 1:
                    lpips_in = lpips_in.repeat(1,3,1,1)
                    lpips_in_pred = lpips_in_pred.repeat(1,3,1,1)

                if lpips_in.shape[1] != 3:
                    lpips_loss = 0
                    for i in range(lpips_in.shape[1]):
                        lpips_in_slice = lpips_in[:, i, :, :].unsqueeze(1).repeat(1,3,1,1)
                        lpips_in_pred_slice = lpips_in_pred[:, i, :, :].unsqueeze(1).repeat(1,3,1,1)
                        lpips_loss += (train_cfg['ldm_perceptual_weight'] * torch.mean(lpips_model(lpips_in_pred_slice, lpips_in_slice)))
                    lpips_loss = lpips_loss / lpips_in.shape[1]
                else:
                    lpips_loss = train_cfg['ldm_perceptual_weight'] * torch.mean(lpips_model(lpips_in_pred, lpips_in))

                loss += lpips_loss
                logs["lpips_loss"] = lpips_loss
                lpips_losses.append(train_cfg['ldm_perceptual_weight'] * lpips_loss.item())

            # discriminator loss (used only from "discriminator_start_step" onwards)
            if use_discriminator:
                disc_fake_pred = discriminator(x_t_neg_1_pred, t)
                disc_fake_loss = d_criterion(disc_fake_pred, torch.ones(disc_fake_pred.shape,
                                                                        device=disc_fake_pred.device))
                loss += train_cfg['ldm_discriminator_weight'] * disc_fake_loss

            logs["g_loss"] = loss
            losses.append(loss.item())

            loss = loss / steps_per_optimization

            loss.backward()

            # generator optimization step
            if (step_count % steps_per_optimization == 0) or (step_count == (len(dataloader) - 1)):
                optimizer.step()
                optimizer.zero_grad()

            ############################# discriminator #################################
            
            if use_discriminator:

                disc_fake_pred = discriminator(x_t_neg_1_pred.detach(), t)
                disc_real_pred = discriminator(x_t_neg_1, t)

                disc_fake_loss = d_criterion(disc_fake_pred, torch.zeros(disc_fake_pred.shape,
                                                                         device=disc_fake_pred.device))
                
                disc_real_loss = d_criterion(disc_real_pred, torch.ones(disc_real_pred.shape,
                                                                        device=disc_real_pred.device))
                
                d_loss = train_cfg['ldm_discriminator_weight'] * (disc_fake_loss + disc_real_loss) / 2

                d_losses.append(d_loss.item())
                logs["d_loss"] = np.mean(d_loss.item())

                # average per step
                d_loss = d_loss / steps_per_optimization

                # discriminator backpropagation
                d_loss.backward()

                # discriminator optimization step
                if (step_count % steps_per_optimization == 0) or (step_count == (len(dataloader) - 1)):
                    optimizer_d.step()
                    optimizer_d.zero_grad()

            ################################# validation ################################
            logs_val = None
            if (step_count % train_cfg["ldm_val_steps"] == 0) and (step_count >= train_cfg["ldm_val_start"]):
                logs_val = validate(config=config, model=model, vae=vae, diffusion=diffusion)

            pbar.set_postfix(Loss=loss.item())
            step_count += 1

            if logs_val is not None:
                logs.update(logs_val)

            # wandb logging
            wandb_run.log(data=logs)

            ################################ model saving ################################
            if step_count % train_cfg["ldm_ckpt_steps"] == 0:

                torch.save(model.state_dict(), os.path.join(train_cfg['ckpt_folder'], 
                                                            run_name, 
                                                            ldm_ckpt_name, 
                                                            "latest.pth")
                )

                torch.save(model.state_dict(), os.path.join(train_cfg['ckpt_folder'],
                                                            run_name,
                                                            ldm_ckpt_name,
                                                            f"{step_count}.pth")
                )
                
                if train_with_discriminator:
    
                    torch.save(discriminator.state_dict(), os.path.join(train_cfg['ckpt_folder'],
                                                                        run_name, 
                                                                        discriminator_ckpt_name, 
                                                                        "latest.pth")
                    )
                            
                    torch.save(discriminator.state_dict(), os.path.join(train_cfg['ckpt_folder'],
                                                                        run_name, 
                                                                        discriminator_ckpt_name,
                                                                        f"{step_count}.pth")
                    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Arguments for ldm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/base_ldm_config.yaml', type=str)
    args = parser.parse_args()

    config = load_config(args.config_path)

    train(config)