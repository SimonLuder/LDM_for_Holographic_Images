import os
import sys
import shutil
import random
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

# # add parent dir to path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir) 
# sys.path.append(parent_dir)

from model.vqvae import VQVAE
from utils.config import load_config
from utils.wandb import WandbManager
from utils.train_test_utils import get_transforms, save_sample_images, online_running_mean
from model.discriminator import PatchGanDiscriminator
from pollen_datasets.poleno import HolographyImageFolder
from .vqvae_validate import validate


def train(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_name = config["name"]
    config_path = config['config_path']
    model_cfg = config["autoencoder"]
    dataset_cfg = config["dataset"]
    train_cfg = config["vqvae_train"]
    
    ae_ckpt_name = f"autoencoder"
    d_ckpt_name = f"discriminator"

    if Path(os.path.join(train_cfg["ckpt_folder"], run_name,  ae_ckpt_name)).exists():
        raise FileExistsError(f"{run_name} already exists in {train_cfg['ckpt_folder']}. Stopping to avoid overwriting.")
    else:
        print(f"Creating new run: {run_name} in {train_cfg['ckpt_folder']}")

    train_with_discriminator = train_cfg['discriminator_weight'] > 0 and train_cfg['discriminator_start_step'] >= 0

    # create checkpoints and sample paths
    Path(os.path.join(train_cfg["ckpt_folder"], run_name, ae_ckpt_name)).mkdir(parents=True, exist_ok=True)   
    Path(os.path.join(train_cfg["ckpt_folder"], run_name, d_ckpt_name)).mkdir( parents=True, exist_ok=True)
    Path(os.path.join(train_cfg["ckpt_folder"], run_name, 'samples')).mkdir( parents=True, exist_ok=True)

    # copy config to checkpoint folder
    shutil.copyfile(config_path, os.path.join(train_cfg['ckpt_folder'], 
                                              run_name, 
                                              os.path.basename(config_path)))

    # setup WandbManager
    wandb_manager = WandbManager(project="MSE_P9_LDM", run_name=run_name, config=config)
    # init run
    wandb_run = wandb_manager.get_run()

    # set seeds
    seed = train_cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # transforms
    transforms = get_transforms(dataset_cfg)

    # dataset
    dataset = HolographyImageFolder(root=dataset_cfg["root"], 
                                    transform=transforms, 
                                    dataset_cfg=dataset_cfg,
                                    labels=dataset_cfg.get("labels_train"))

    # dataloader
    dataloader = DataLoader(dataset,
                            batch_size=train_cfg['autoencoder_batch_size'],
                            shuffle=True)
    
    # VAE model
    model = VQVAE(img_channels=dataset_cfg["img_channels"], 
                  config=model_cfg).to(device)
    
    optimizer_g = Adam(model.parameters(), lr=train_cfg['autoencoder_lr'], betas=(0.5, 0.999))
    
    # Discriminator model
    if train_with_discriminator:
        discriminator = PatchGanDiscriminator(img_channels=dataset_cfg['img_channels']).to(device)

        optimizer_d = Adam(discriminator.parameters(), lr=train_cfg['autoencoder_lr'], betas=(0.5, 0.999))

        loss_functions = {"MSELoss" : nn.MSELoss, 
                          "BCEWithLogits" : nn.BCEWithLogitsLoss,}.get(
                              train_cfg["discriminator_loss"])
        discriminator_loss = loss_functions()

        discriminator_start_step = train_cfg["discriminator_start_step"] # start discriminator after n batches

    # LPIPS perceptual loss
    lpips_model = LPIPS(net_type='alex').to(device)

    # L1/L2 loss for Reconstruction
    mse_loss = torch.nn.MSELoss()

    # optimize every n steps (for large badges)
    steps_per_optimization = train_cfg['autoencoder_steps_per_optimization']

    # save batch every n steps
    image_save_steps = train_cfg["autoencoder_img_save_steps"]

    num_epochs = train_cfg['autoencoder_epochs']

    step_count = 0

    for epoch_idx in range(num_epochs):
        ep_rec_loss = 0                     # reconstruction loss (l2)
        ep_codebook_loss = 0                # codebook loss between predicted and nearest codebook vector
        ep_lpips_loss = 0                   # preceptual loss (lpips)
        ep_d_loss = 0                       # discriminator loss
        ep_g_loss = 0                       # weighted sum of reconstruction, preceptual and discriminator scores

        pbar = tqdm(dataloader, disable=train_cfg.get("no_tqdm", False))
        for (im, _, _) in pbar:
    
            im = im.float().to(device)

            # autoencoder forward pass
            output, z, quantize_losses = model(im)

            # save images
            if step_count % image_save_steps == 0:
                save_as = os.path.join(train_cfg["ckpt_folder"], run_name, 'samples', f'step_{step_count}.png')
                save_sample_images(im, output, save_as)
            

            ########################## Autoencoder optimization ##########################
            # reconstruction loss
            rec_loss = mse_loss(output, im)
            ep_rec_loss = online_running_mean(ep_rec_loss, rec_loss.item(), step_count+1)

            # codebook & commitment loss loss
            g_loss = (rec_loss + 
                      train_cfg['codebook_weight'] * quantize_losses["codebook_loss"] + 
                      train_cfg['commitment_beta'] * quantize_losses["commitment_loss"]
                      )
            ep_codebook_loss = online_running_mean(
                ep_codebook_loss, 
                train_cfg['codebook_weight'] * quantize_losses['codebook_loss'].item(), 
                step_count+1
            )

            # lpips loss
            im_lpips = torch.clamp(im, -1., 1.)
            out_lpips = torch.clamp(output, -1., 1.)

            if im_lpips.shape[1] == 1:
                im_lpips = im_lpips.repeat(1,3,1,1)
                out_lpips = out_lpips.repeat(1,3,1,1)

            lpips_loss = train_cfg['perceptual_weight'] * torch.mean(lpips_model(out_lpips, im_lpips))
            ep_lpips_loss = online_running_mean(
                ep_lpips_loss, 
                train_cfg['perceptual_weight'] * lpips_loss.item(),
                step_count+1
            )
            g_loss += lpips_loss

            # discriminator loss (used only from "discriminator_start_step" onwards)
            if step_count > discriminator_start_step and train_with_discriminator:
                disc_fake_pred = discriminator(output)
                disc_fake_loss = discriminator_loss(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
                
                g_loss += train_cfg['discriminator_weight'] * disc_fake_loss

            ep_g_loss = online_running_mean(ep_g_loss, g_loss.item(), step_count+1)

            # average per step
            g_loss = g_loss / steps_per_optimization

            # VAE backpropagation
            g_loss.backward()

            # generator optimization step
            if (step_count % steps_per_optimization == 0) or (step_count == (len(dataloader) - 1)):
                optimizer_g.step()
                optimizer_g.zero_grad()
            ##############################################################################

            ######################### Discriminator optimization #########################
            d_loss = None
            if (step_count >= discriminator_start_step) and train_with_discriminator:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)

                disc_fake_loss = discriminator_loss(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                disc_real_loss = discriminator_loss(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,
                                                           device=disc_real_pred.device))
                
                d_loss = train_cfg['discriminator_weight'] * (disc_fake_loss + disc_real_loss) / 2
                ep_d_loss = online_running_mean(ep_d_loss, d_loss.item(), step_count+1)

                # average per step
                d_loss = d_loss / steps_per_optimization
                
                # discriminator backpropagation
                d_loss.backward()

                # discriminator optimization step
                if (step_count % steps_per_optimization == 0) or (step_count == (len(dataloader) - 1)):
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            ##############################################################################

            ################################ model saving ################################
            if step_count % train_cfg["autoencoder_ckpt_steps"] == 0:

                torch.save(model.state_dict(), 
                           os.path.join(train_cfg["ckpt_folder"],
                                        run_name,
                                        ae_ckpt_name,
                                        "latest.pth"))
                torch.save(model.state_dict(), 
                            os.path.join(train_cfg["ckpt_folder"],
                                         run_name,
                                         ae_ckpt_name,
                                         f"{step_count}.pth"))
                
                if train_with_discriminator:

                    torch.save(discriminator.state_dict(), 
                           os.path.join(train_cfg["ckpt_folder"],
                                        run_name,
                                        d_ckpt_name,
                                        "latest.pth"))
                    
                    torch.save(discriminator.state_dict(), 
                            os.path.join(train_cfg["ckpt_folder"],
                                         run_name,
                                         d_ckpt_name,
                                         f"{step_count}.pth"))
                    
            ##############################################################################

            ################################# validation #################################
            logs_val = None
            if (step_count % train_cfg["autoencoder_val_steps"] == 0) and (step_count >= train_cfg["autoencoder_val_start"]):
                logs_val = validate(config=config, model=model, step_count=step_count)
            #############################################################################

            ################################### logging ##################################
            logs = {"epoch" :               epoch_idx + 1,
                    "step" :                step_count + 1,
                    "reconstruction_loss" : rec_loss,
                    "lpips_loss" :          lpips_loss,
                    "codebook_loss" :       quantize_losses["codebook_loss"],
                    "g_loss" :              g_loss,
                }
            
            if d_loss is not None:
                logs["d_loss"] = d_loss

            if logs_val is not None:
                logs.update(logs_val)

            # wandb logging
            wandb_run.log(data=logs)
            ##############################################################################

            step_count += 1
            pbar.set_postfix(g_loss=g_loss.item())

        ################################### logging ##################################
        logs = {"epoch" :                     epoch_idx + 1,
                "epoch_reconstruction_loss" : ep_rec_loss,
                "epoch_lpips_loss" :          ep_lpips_loss,
                "epoch_codebook_loss" :       ep_codebook_loss,
                "epoch_g_loss" :              ep_g_loss,
                }
        
        if ep_d_loss > 0:
            logs["epoch_d_loss"] = ep_d_loss
        
        # wandb logging
        wandb_run.log(data=logs)

        ##############################################################################

            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for ldm training')
    parser.add_argument('--config', dest='config_path', default='config/base_vqvae_config.yaml', type=str)
    args = parser.parse_args()

    config = load_config(args.config_path)

    train(config)