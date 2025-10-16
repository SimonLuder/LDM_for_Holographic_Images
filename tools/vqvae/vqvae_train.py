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
from torchvision.utils import make_grid
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

# # add parent dir to path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir) 
# sys.path.append(parent_dir)

from model.vqvae import VQVAE
from utils.config import load_config
from utils.wandb import WandbManager
from model.discriminator import PatchGanDiscriminator
from pollen_datasets.poleno import HolographyImageFolder
from .vqvae_validate import validate


def train(config_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config(config_path)
    model_config = config["autoencoder"]
    dataset_config = config["dataset"]
    train_config = config["vqvae_train"]

    train_with_discriminator = train_config['discriminator_weight'] > 0 and train_config['discriminator_start_step'] >= 0

    # create checkpoints and sample paths
    Path(os.path.join(train_config["task_name"], train_config['vqvae_autoencoder_ckpt_name'])).mkdir(parents=True, exist_ok=True)   
    Path(os.path.join(train_config["task_name"], train_config['vqvae_discriminator_ckpt_name'])).mkdir( parents=True, exist_ok=True)
    Path(os.path.join(train_config["task_name"], train_config["vqvae_autoencoder_ckpt_name"], 'samples')).mkdir( parents=True, exist_ok=True)

    # copy config to checkpoint folder
    shutil.copyfile(config_path, os.path.join(train_config['task_name'], 
                                              train_config['vqvae_autoencoder_ckpt_name'], 
                                              os.path.basename(config_path)))

    # setup WandbManager
    wandb_manager = WandbManager(project="MSE_P9_LDM", run_name=train_config['vqvae_autoencoder_ckpt_name'] + "_vqvae", config=config)
    # init run
    wandb_run = wandb_manager.get_run()

    # set seeds
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # transforms
    transforms_list = []

    transforms_list.append(torchvision.transforms.ToTensor())

    if dataset_config.get("img_interpolation"):
        transforms_list.append(torchvision.transforms.Resize((dataset_config["img_interpolation"], 
                                                              dataset_config["img_interpolation"]),
                                                              interpolation = torchvision.transforms.InterpolationMode.BILINEAR))

    transforms_list.append(torchvision.transforms.Normalize((0.5) * dataset_config["img_channels"], 
                                                            (0.5) * dataset_config["img_channels"]))

    transforms = torchvision.transforms.Compose(transforms_list)

    #dataset
    dataset = HolographyImageFolder(root=dataset_config["root"], 
                                    transform=transforms, 
                                    config=dataset_config,
                                    labels=dataset_config.get("labels_train"))

    # dataloader
    dataloader = DataLoader(dataset,
                            batch_size=train_config['autoencoder_batch_size'],
                            shuffle=True)
    
    # VAE model
    model = VQVAE(img_channels=dataset_config["img_channels"], 
                  config=model_config).to(device)
    
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    
    # Discriminator model
    if train_with_discriminator:
        discriminator = PatchGanDiscriminator(img_channels=dataset_config['img_channels']).to(device)

        optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

        loss_functions = {"MSELoss" : nn.MSELoss, 
                          "BCEWithLogits" : nn.BCEWithLogitsLoss,}.get(
                              train_config["discriminator_loss"])
        discriminator_loss = loss_functions()

        discriminator_start_step = train_config["discriminator_start_step"] # start discriminator after n batches

    # LPIPS perceptual loss
    lpips_model = LPIPS(net_type='alex').to(device)

    # L1/L2 loss for Reconstruction
    mse_loss = torch.nn.MSELoss()

    # optimize every n steps (for large badges)
    steps_per_optimization = train_config['autoencoder_steps_per_optimization']

    # save batch every n steps
    image_save_steps = train_config["autoencoder_img_save_steps"]

    num_epochs = train_config['autoencoder_epochs']

    img_save_count = 0
    step_count = 0

    for epoch_idx in range(num_epochs):
        reconstruction_losses = []          # reconstruction loss (l2)
        codebook_losses = []                # codebook loss between predicted and nearest codebook vector
        lpips_losses = []                   # preceptual loss (lpips)
        d_losses = []                       # discriminator loss
        g_losses = []                       # weighted sum of reconstruction, preceptual and discriminator scores

        pbar = tqdm(dataloader)
        for (im, _, _) in pbar:
    
            im = im.float().to(device)

            # autoencoder forward pass
            output, z, quantize_losses = model(im)

            # save images
            if step_count % image_save_steps == 0:
                sample_size = min(8, im.shape[0])
                save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                save_output = ((save_output + 1) / 2)
                save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                
                grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                img = torchvision.transforms.ToPILImage()(grid)

                img.save(os.path.join(train_config["task_name"], 
                                      train_config["vqvae_autoencoder_ckpt_name"], 
                                      'samples',
                                      f'sample_{epoch_idx+1}_{step_count}.png'))
                img_save_count += 1
                img.close()

            ########################## Autoencoder optimization ##########################
            # reconstruction loss
            rec_loss = mse_loss(output, im)
            reconstruction_losses.append(rec_loss.item())

            # codebook & commitment loss loss
            g_loss = (rec_loss + 
                      train_config['codebook_weight'] * quantize_losses["codebook_loss"] + 
                      train_config['commitment_beta'] * quantize_losses["commitment_loss"]
                      )
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())

            # lpips loss
            im_lpips = torch.clamp(im, -1., 1.)
            out_lpips = torch.clamp(output, -1., 1.)

            if im_lpips.shape[1] == 1:
                im_lpips = im_lpips.repeat(1,3,1,1)
                out_lpips = out_lpips.repeat(1,3,1,1)

            lpips_loss = train_config['perceptual_weight'] * torch.mean(lpips_model(out_lpips, im_lpips))
            lpips_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += lpips_loss

            # discriminator loss (used only from "discriminator_start_step" onwards)
            if step_count > discriminator_start_step and train_with_discriminator:
                disc_fake_pred = discriminator(output)
                disc_fake_loss = discriminator_loss(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
                
                g_loss += train_config['discriminator_weight'] * disc_fake_loss

            g_losses.append(g_loss.item())

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
                
                d_loss = train_config['discriminator_weight'] * (disc_fake_loss + disc_real_loss) / 2

                d_losses.append(d_loss.item())

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
            if step_count % train_config["autoencoder_ckpt_steps"] == 0:

                torch.save(model.state_dict(), 
                           os.path.join(train_config["task_name"],
                                        train_config['vqvae_autoencoder_ckpt_name'],
                                        "latest.pth"))
                torch.save(model.state_dict(), 
                            os.path.join(train_config["task_name"],
                                         train_config['vqvae_autoencoder_ckpt_name'],
                                         f"{step_count}.pth"))
                
                if train_with_discriminator:

                    torch.save(discriminator.state_dict(), 
                           os.path.join(train_config["task_name"],
                                        train_config['vqvae_discriminator_ckpt_name'],
                                        "latest.pth"))
                    
                    torch.save(discriminator.state_dict(), 
                            os.path.join(train_config["task_name"],
                                         train_config['vqvae_discriminator_ckpt_name'],
                                         f"{step_count}.pth"))
                    
            ##############################################################################

            ################################# validation #################################
            logs_val = None
            if (step_count % train_config["autoencoder_val_steps"] == 0) and (step_count >= train_config["autoencoder_val_start"]):
                logs_val = validate(config_path=config_path, model=model)
            #############################################################################

            ################################### logging ##################################
            logs = {"epoch" :               epoch_idx + 1,
                    "step" :                step_count + 1,
                    "reconstruction_loss" : rec_loss,
                    "lpips_loss" :          lpips_loss,
                    "codebook_loss" :       quantize_losses["codebook_loss"],
                    "g_loss" :              g_loss,
                }
            
            if len(d_losses) > 0:
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
                "epoch_reconstruction_loss" : np.mean(reconstruction_losses),
                "epoch_lpips_loss" :          np.mean(lpips_losses),
                "epoch_codebook_loss" :       np.mean(codebook_losses),
                "epoch_g_loss" :              np.mean(g_losses),
                }
        
        if len(d_losses) > 0:
            logs["epoch_d_loss"] = np.mean(d_losses)
        
        # wandb logging
        wandb_run.log(data=logs)

        ##############################################################################

            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for ldm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/base_vqvae_config.yaml', type=str)
    args = parser.parse_args()

    train(args.config_path)