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
from utils.train_test_utils import get_transforms
from pollen_datasets.poleno import HolographyImageFolder


def validate(config, model=None, vae=None, diffusion=None, model_ckpt=None):

    global dataloader_val

    run_name                = config['name']
    dataset_cfg             = config['dataset']
    condition_cfg           = config['conditioning']
    ddpm_model_cfg          = config['ddpm']
    train_cfg               = config['ldm_train']
    inference_cfg           = config["ddpm_inference"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ###################################### data ######################################
    if not "dataloader_val" in globals():

        # Transforms
        transforms = get_transforms(dataset_cfg)

        # Dataset
        dataset_val = HolographyImageFolder(root=dataset_cfg["root"], 
                                    transform=transforms, 
                                    dataset_cfg=dataset_cfg,
                                    cond_cfg=condition_cfg,
                                    labels=dataset_cfg.get("labels_val"))

        # Dataloader
        dataloader_val = DataLoader(dataset_val,
                                batch_size=train_cfg['ldm_batch_size'],
                                shuffle=False)
    
    # define criterions for evaluation
    criterion = torch.nn.MSELoss()

    reconstruction_losses = []
 
    for (im, condition, _) in dataloader_val:

        with torch.no_grad():
            
            im = im.float().to(device)
            for key in condition.keys():
                condition[key] = condition[key].to(device)

            # randomly discard conditioning to train unconditionally if conditional training
            if condition_cfg["enabled"] == "unconditional" or np.random.random() < train_cfg["ldm_cfg_discard_prob"]:
                condition = None

            # autencode samples
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