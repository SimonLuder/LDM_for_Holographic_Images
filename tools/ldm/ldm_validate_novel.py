import os
import sys
import argparse
import numpy as np

import torch
import torch.nn
from torch.utils.data import DataLoader
import torchvision

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from utils.config import load_config
from utils.train_test_utils import get_image_encoder_names
from model.conditioning.transforms.registry import get_transforms
from pollen_datasets.poleno import PairwiseHolographyImageFolder


def validate(config, model=None, vae=None, diffusion=None, model_ckpt=None):

    global dataloader_val

    run_name                = config['name']
    dataset_cfg             = config['dataset']
    transforms_cfg          = config['transforms']
    condition_cfg           = config['conditioning']
    train_cfg               = config['ldm_train']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ###################################### data ######################################
    if not "dataloader_val" in globals():

        # Transforms
        transform1 = get_transforms(transforms_cfg["transform1"], in_channels=dataset_cfg["img_channels"])
        transform2 = get_transforms(transforms_cfg["transform2"], in_channels=dataset_cfg["img_channels"])

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
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=train_cfg['ldm_batch_size'],
            num_workers=train_cfg.get('num_workers', 4),
            prefetch_factor=2,
            pin_memory=True,
            shuffle=False
        )
        
    # Conditioning
    if condition_cfg["enabled"] == "unconditional":
        print("Training unconditional model")
    else:
        use_condition = condition_cfg["enabled"].split("+")
        used_image_encoders = get_image_encoder_names(condition_cfg)
    
    # define criterions for evaluation
    criterion = torch.nn.MSELoss()

    reconstruction_losses = []
 
    for (im1, im2), (cond1, cond2), _ in dataloader_val:

        with torch.no_grad():
            
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

            # autencode samples
            im1, _ = vae.encode(im1)

            # sample timestep
            t = diffusion.sample_timesteps(im1.shape[0]).to(device)

            # noise image
            x_t, noise, x_t_neg_1 = diffusion.noise_images(im1, t, x_t_neg_1=True)

            # predict noise
            noise_pred = model(x_t, t, cond1)

            # reconstruction loss
            rec_loss = criterion(noise_pred, noise)
            reconstruction_losses.append(rec_loss.item())  
 
    logs = {"val_epoch_reconstructon_loss" : np.mean(reconstruction_losses)}

    model.train()
    vae.train()

    return logs


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Arguments for ldm validation')
    parser.add_argument('--config', dest='config_path',
                        default='config/base_ldm_config.yaml', type=str)
    args = parser.parse_args()

    config = load_config(args.config_path)

    validate(config)