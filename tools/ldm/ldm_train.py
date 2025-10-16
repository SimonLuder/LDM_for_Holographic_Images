import os
import sys
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from model.unet_v2 import UNet
from model.vqvae import VQVAE
from model.embedding import ConditionEmbedding
from model.ddpm import Diffusion as DDPMDiffusion
from model.discriminator import DiffusionPatchGanDiscriminator
from utils.wandb import WandbManager
from utils.config import load_config
from .ldm_validate import validate
from pollen_datasets.poleno import HolographyImageFolder



def train(config_path):

    # load configuration
    config = load_config(config_path)
    dataset_config = config['dataset']
    ddpm_config = config['ddpm']
    autoencoder_model_config = config['autoencoder']
    train_config = config['ldm_train']

    train_with_discriminator = train_config['ldm_discriminator_weight'] > 0
    train_with_perceptual_loss = train_config['ldm_perceptual_weight'] > 0

    # train on GPU if it is available else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup WandbManager
    wandb_manager = WandbManager(project="MSE_P9_LDM", run_name=train_config['ldm_ckpt_name']  + "_ldm", config=config)
    # init run
    wandb_run = wandb_manager.get_run()

    # create checkpoints and sample paths
    Path(os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])).mkdir(parents=True, exist_ok=True)  
    Path(os.path.join(train_config['task_name'], train_config['ldm_discriminator_ckpt_name'])).mkdir(parents=True, exist_ok=True)   

    # copy config to checkpoint folder
    shutil.copyfile(config_path, os.path.join(train_config['task_name'], 
                                              train_config['ldm_ckpt_name'], 
                                              os.path.basename(config_path)))

    ###################################### data ######################################
    transforms_list = [torchvision.transforms.ToTensor()]

    if dataset_config.get("img_interpolation"):
        transforms_list.append(torchvision.transforms.Resize((dataset_config["img_interpolation"], 
                                                              dataset_config["img_interpolation"]),
                                                              interpolation = torchvision.transforms.InterpolationMode.BILINEAR))

    transforms_list.append(torchvision.transforms.Normalize((0.5) * dataset_config["img_channels"], 
                                                            (0.5) * dataset_config["img_channels"]))

    transforms = torchvision.transforms.Compose(transforms_list)

    # Dataset
    dataset = HolographyImageFolder(root=dataset_config["root"], 
                                    transform=transforms, 
                                    config=dataset_config,
                                    labels=dataset_config.get("labels_train"))

    # dataloader
    print("Initialize Dataloader")
    dataloader = DataLoader(dataset,
                            batch_size=train_config['ldm_batch_size'],
                            shuffle=True)
    
    ################################## autoencoder ###################################
    # Load Autoencoder if latents are not precalculated or are missing
    if os.path.exists(train_config["vqvae_latents_representations"]):
        latents_available = len(os.listdir(train_config["vqvae_latents_representations"])) > 0
    else:
        latents_available = False

    if not latents_available:

        print('Loading vqvae model as no latents found')
        vae = VQVAE(img_channels=dataset_config['img_channels'], config=autoencoder_model_config).to(device)
        vae.eval()

        # # Load VQVAE weights from checkpoint
        # vae_ckpt_path = os.path.join(train_config['task_name'],
        #                              train_config['vqvae_ckpt_dir'],
        #                              train_config['vqvae_ckpt_model']
        #                              )
        
        # vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))
        print('Loaded autoencoder checkpoint')
        
        for param in vae.parameters():
            param.requires_grad = False

    ############################### context encoding ################################
    # Load context encoder
    if train_config["conditioning"] == "unconditional":
        print("training unconditional model")
        context_encoder = None

    else:
        use_condition = train_config["conditioning"].split("+")
        print("training conditional model on:", use_condition)
        num_classes = int(max(dataset.class_labels)) + 1 if "class" in use_condition else None
        cls_emb_dim = ddpm_config['cls_emb_dim'] + 1 if "class" in use_condition else None
        tabular_in_dim = len(dataset_config["features"]) if "tabular" in use_condition else None
        tabular_out_dim = ddpm_config['tbl_emb_dim'] if "tabular" in use_condition else None
        img_out_dim = ddpm_config['img_emb_dim'] if "image" in use_condition else None


        context_encoder = ConditionEmbedding(out_dim=ddpm_config['cond_emb_dim'],
                                             num_classes=num_classes,
                                             cls_emb_dim=cls_emb_dim,
                                             tabular_in_dim=tabular_in_dim,
                                             tabular_out_dim=tabular_out_dim,
                                             img_in_channels=None, # TODO add image conditioning to training
                                             img_out_dim=img_out_dim
                                             )
        context_encoder.to(device)

    ##################################### u-net ######################################
    # UNet
    model = UNet(img_channels=autoencoder_model_config['z_channels'], 
                 model_config=ddpm_config,
                 context_encoder=context_encoder).to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])

    criterion = torch.nn.MSELoss()

    # discriminator model
    if train_with_discriminator:

        discriminator = DiffusionPatchGanDiscriminator(img_channels=autoencoder_model_config['z_channels']).to(device)
        discriminator.train()

        optimizer_d = Adam(discriminator.parameters(), lr=train_config['ldm_disc_lr'])
        # d_criterion = torch.nn.MSELoss()
        loss_functions = {"MSELoss" : torch.nn.MSELoss, 
                          "BCEWithLogits" : torch.nn.BCEWithLogitsLoss,}.get(
                              train_config["ldm_discriminator_loss"])
        d_criterion = loss_functions()

    discriminator_start_step = train_config["ldm_discriminator_start_step"] # starts discriminator after n batches
    
    # Load Diffusion Process
    if dataset_config.get('img_interpolation'):
        img_size = dataset_config['img_interpolation'] // 2 ** sum(autoencoder_model_config['down_sample'])
    else:
        img_size = dataset_config['img_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    diffusion = DDPMDiffusion(img_size=img_size, 
                              img_channels=autoencoder_model_config['z_channels'],
                              noise_schedule="linear", 
                              beta_start=train_config["ldm_beta_start"], 
                              beta_end=train_config["ldm_beta_end"],
                              device=device,
                              )
    
    # lpips
    if train_with_perceptual_loss:
        lpips_model = LPIPS(net_type='alex').to(device)

    num_epochs = train_config['ldm_epochs']
    steps_per_optimization = train_config['ldm_steps_per_optimization']
    step_count = 0

    print(f"Start training with \nperceptual loss: {train_with_perceptual_loss} \ndiscriminator: {train_with_perceptual_loss}")

    for epoch_idx in range(num_epochs):
        losses = []
        d_losses = []
        lpips_losses = []

        pbar = tqdm(dataloader)
        for (im, condition, _) in pbar:

            im = im.float().to(device)
            for cond in use_condition:
                condition[cond] = condition[cond].to(device)

            # randomly discard conditioning to train unconditionally if conditional training
            if train_config["conditioning"] == "unconditional" or np.random.random() < train_config["ldm_cfg_discard_prob"]:
                condition = None

            # autencode samples
            if not latents_available:
                with torch.no_grad():
                    im, _ = vae.encode(im)

            # sample timestep
            t = diffusion.sample_timesteps(im.shape[0]).to(device)

            # noise image
            x_t, noise, x_t_neg_1 = diffusion.noise_images(im, t, x_t_neg_1=True)

            # predict noise
            noise_pred = model(x_t, t, condition)

            # create image less noisy
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
                lpips_in = torch.clamp(im, -1., 1.)
                lpips_in_pred = torch.clamp(im_pred, -1., 1.)
                # lpips_in = torch.clamp(x_t_neg_1, -1., 1.)
                # lpips_in_pred = torch.clamp(x_t_neg_1_pred, -1., 1.)

                if lpips_in.shape[1] == 1:
                    lpips_in = lpips_in.repeat(1,3,1,1)
                    lpips_in_pred = lpips_in_pred.repeat(1,3,1,1)

                if lpips_in.shape[1] != 3:
                    lpips_loss = 0
                    for i in range(lpips_in.shape[1]):
                        lpips_in_slice = lpips_in[:, i, :, :].unsqueeze(1).repeat(1,3,1,1)
                        lpips_in_pred_slice = lpips_in_pred[:, i, :, :].unsqueeze(1).repeat(1,3,1,1)
                        lpips_loss += (train_config['ldm_perceptual_weight'] * torch.mean(lpips_model(lpips_in_pred_slice, lpips_in_slice)))
                    lpips_loss = lpips_loss / lpips_in.shape[1]
                else:
                    lpips_loss = train_config['ldm_perceptual_weight'] * torch.mean(lpips_model(lpips_in_pred, lpips_in))

                loss += lpips_loss
                logs["lpips_loss"] = lpips_loss
                lpips_losses.append(train_config['ldm_perceptual_weight'] * lpips_loss.item())

            # discriminator loss (used only from "discriminator_start_step" onwards)
            if (step_count > discriminator_start_step) and train_with_discriminator:
                disc_fake_pred = discriminator(x_t_neg_1_pred, t)
                disc_fake_loss = d_criterion(disc_fake_pred, torch.ones(disc_fake_pred.shape,
                                                                        device=disc_fake_pred.device))
                loss += train_config['ldm_discriminator_weight'] * disc_fake_loss

            logs["g_loss"] = loss
            losses.append(loss.item())

            loss = loss / steps_per_optimization

            loss.backward()

            # generator optimization step
            if (step_count % steps_per_optimization == 0) or (step_count == (len(dataloader) - 1)):
                optimizer.step()
                optimizer.zero_grad()

            ######################## discriminator ########################
            
            if (step_count >= discriminator_start_step) and train_with_discriminator:

                disc_fake_pred = discriminator(x_t_neg_1_pred.detach(), t)
                disc_real_pred = discriminator(x_t_neg_1, t)

                disc_fake_loss = d_criterion(disc_fake_pred, torch.zeros(disc_fake_pred.shape,
                                                                         device=disc_fake_pred.device))
                
                disc_real_loss = d_criterion(disc_real_pred, torch.ones(disc_real_pred.shape,
                                                                        device=disc_real_pred.device))
                
                d_loss = train_config['ldm_discriminator_weight'] * (disc_fake_loss + disc_real_loss) / 2

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

                ###############################################################

            ################################# validation #################################
            logs_val = None
            if (step_count % train_config["ldm_val_steps"] == 0) and (step_count >= train_config["ldm_val_start"]):
                logs_val = validate(config_path=config_path, model=model, vae=vae, diffusion=diffusion)
            #############################################################################

            pbar.set_postfix(Loss=loss.item())
            step_count += 1

            if logs_val is not None:
                logs.update(logs_val)

            # wandb logging
            wandb_run.log(data=logs)

            ################################ model saving ################################
            if step_count % train_config["ldm_ckpt_steps"] == 0:

                torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                            train_config['ldm_ckpt_name'],
                                                            "latest.pth"))
                        
                torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                            train_config['ldm_ckpt_name'],
                                                            f"{step_count}.pth"))
                
                if train_with_discriminator:
    
                    torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                                train_config['ldm_discriminator_ckpt_name'],
                                                                "latest.pth"))
                            
                    torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                                train_config['ldm_discriminator_ckpt_name'],
                                                                f"{step_count}.pth"))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Arguments for ldm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/base_ldm_config.yaml', type=str)
    args = parser.parse_args()

    train(args.config_path)