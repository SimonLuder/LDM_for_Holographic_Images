import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from model.vqvae import VQVAE
from pollen_datasets.poleno import HolographyImageFolder
from utils.config import load_config
from utils.train_test_utils import save_json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(config_path, model=None, model_ckpt=None):

    print("entered val loop")

    # global model
    global lpips_model
    global dataloader_val

    config = load_config(config_path)

    dataset_config = config['dataset']
    autoencoder_config = config['autoencoder']
    train_config = config['vqvae_train']
    inference_config = config["vqvae_inference"]

    # create checkpoint paths
    Path(os.path.join(train_config['task_name'], 
                      train_config['vqvae_autoencoder_ckpt_name']
                      )).mkdir(parents=True, exist_ok=True)

    if not "dataloader_val" in globals(): # singleton design pattern
        # transforms
        transforms_list = [torchvision.transforms.ToTensor()]

        if dataset_config.get("img_interpolation"):
            transforms_list.append(torchvision.transforms.Resize((dataset_config["img_interpolation"], 
                                                                dataset_config["img_interpolation"]),
                                                                interpolation = torchvision.transforms.InterpolationMode.BILINEAR))

        transforms_list.append(torchvision.transforms.Normalize((0.5) * dataset_config["img_channels"], 
                                                                (0.5) * dataset_config["img_channels"]))

        transforms = torchvision.transforms.Compose(transforms_list)

        #dataset
        dataset_val = HolographyImageFolder(root=dataset_config["root"], 
                                        transform=transforms, 
                                        config=dataset_config,
                                        labels=dataset_config.get("labels_val"))

        # dataloader
        dataloader_val = DataLoader(dataset_val,
                                batch_size=train_config['autoencoder_batch_size'],
                                shuffle=False)
        
        print("Instanciate validation dataloader")

    # load pretrained vqvae
    if model is None: 
        print("Instanciate pretrained model for validation")
        model = VQVAE(img_channels=dataset_config['img_channels'], config=autoencoder_config).to(device)
    
        model.load_state_dict(
            torch.load(os.path.join(train_config['task_name'], 
                                    train_config['vqvae_autoencoder_ckpt_name'], 
                                    model_ckpt), 
                                    map_location=device))
    model.eval()

    # mse reconstruction criterion
    mse_loss = torch.nn.MSELoss()

    # LPIPS perceptual criterion
    if not "lpips_model" in globals(): # singleton design pattern
        print("Instanciate lpips for validation")
        lpips_model = LPIPS(net_type='alex').to(device)

    folders = []
    sample_filenames = []
    reconstruction_losses = []          # reconstruction loss (l2)
    codebook_losses = []                # codebook loss between predicted and nearest codebook vector
    lpips_losses = []                   # preceptual loss (lpips)
    
    with torch.no_grad():

        pbar = tqdm(dataloader_val)
        for (im, folder, filenames) in pbar:

            im = im.float().to(device)

            folders.append(folder)
            sample_filenames.append(filenames)

            # autoencoder forward pass
            output, z, quantize_losses = model(im)

            # reconstruction loss
            rec_loss = mse_loss(output, im)
            reconstruction_losses.append(rec_loss.item())

            # codebook & commitment loss
            codebook_losses.append(quantize_losses["codebook_loss"].item())

            # lpips loss
            im_lpips = torch.clamp(im, -1., 1.)
            out_lpips = torch.clamp(output, -1., 1.)

            if im_lpips.shape[1] == 1:
                im_lpips = im_lpips.repeat(1,3,1,1)
                out_lpips = out_lpips.repeat(1,3,1,1)

            lpips_loss = train_config['perceptual_weight'] * torch.mean(lpips_model(out_lpips, im_lpips))
            lpips_losses.append(train_config['perceptual_weight'] * lpips_loss.item())


    logs = {"val_epoch_reconstructon_loss"    : np.mean(reconstruction_losses),
            "val_epoch_codebook_loss"         : np.mean(codebook_losses),
            "val_epoch_lpips_loss"            : np.mean(lpips_losses)
            }
            
    model.train()
    return logs

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for vqvae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/base_vqvae_config.yaml', type=str)
    
    args = parser.parse_args()

    config = load_config(args.config_path)
    validate_ckpts = config["vqvae_validation"]["model_ckpts"]
    log_filepath = os.path.join(config["train"]["task_name"], config["train"]["vqvae_autoencoder_ckpt_name"], "val_logs.json")

    logs = []
    
    for model_ckpt in validate_ckpts:
        log = validate(args.config_path, model_ckpt=model_ckpt)
        log["step"] = model_ckpt
        logs.append(log)
    save_json(logs, log_filepath)
    




