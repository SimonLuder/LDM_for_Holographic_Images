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
from model.ldm_trainer import NVLDMTrainer


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
    Path(os.path.join(train_cfg['ckpt_folder'], run_name, "checkpoints")).mkdir(parents=True, exist_ok=True)  

    # copy config to checkpoint folder
    shutil.copyfile(config_path, os.path.join(train_cfg['ckpt_folder'], 
                                              run_name,
                                              os.path.basename(config_path)))

    ###################################### data ######################################

    # Transforms
    transform1 = get_transforms(transforms_cfg["transform1"], in_channels=dataset_cfg["img_channels"])
    transform2 = get_transforms(transforms_cfg["transform2"], in_channels=dataset_cfg["img_channels"])

    print("transform1", transform1)
    print("transform2", transform2)

    # Dataset
    dataset = PairwiseHolographyImageFolder(
        root=dataset_cfg["root"], 
        transform1=transform1,
        transform2=transform2, 
        dataset_cfg=dataset_cfg,
        cond_cfg=condition_cfg,
        labels=dataset_cfg.get("labels_train")
    )

    # Dataset
    dataset_val = PairwiseHolographyImageFolder(
        root=dataset_cfg["root"], 
        transform1=transform1,
        transform2=transform2,
        dataset_cfg=dataset_cfg,
        cond_cfg=condition_cfg,
        labels=dataset_cfg.get("labels_val")
    )

    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['ldm_batch_size'],
        num_workers=train_cfg.get('num_workers', 4),
        prefetch_factor=2,
        pin_memory=True,
        shuffle=True
    )

    # Dataloader
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=train_cfg['ldm_batch_size'],
        num_workers=train_cfg.get('num_workers', 4),
        prefetch_factor=2,
        pin_memory=True,
        shuffle=False
    )
        
    print("Initialized Dataloader")
    
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
        print("Training model with conditions:", condition_cfg["enabled"])
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

    # Discriminator model
    if train_with_discriminator:

        discriminator = DiffusionPatchGanDiscriminator(img_channels=vae_model_cfg['z_channels']).to(device)
        discriminator.train()

        optimizer_d = Adam(discriminator.parameters(), lr=train_cfg['ldm_disc_lr'])
        # d_criterion = torch.nn.MSELoss()
        loss_functions = {"MSELoss" : torch.nn.MSELoss, 
                          "BCEWithLogits" : torch.nn.BCEWithLogitsLoss,}.get(
                              train_cfg["ldm_discriminator_loss"])
        d_criterion = loss_functions()
    
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


    print(f"Start training with perceptual loss: {train_with_perceptual_loss}, discriminator: {train_with_discriminator}")

    trainer = NVLDMTrainer(
        config=config,
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        vae=vae if not latents_available else None,
        criterion=criterion,
        lpips_model=lpips_model if train_with_perceptual_loss else None,
        discriminator=discriminator if train_with_discriminator else None,
        optimizer_d=optimizer_d if train_with_discriminator else None,
        d_criterion=d_criterion if train_with_discriminator else None,
        wandb_run=wandb_run,
        dataloader_val=dataloader_val,
    )

    trainer.train()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Arguments for ldm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/base_ldm_config.yaml', type=str)
    args = parser.parse_args()

    config = load_config(args.config_path)

    train(config)