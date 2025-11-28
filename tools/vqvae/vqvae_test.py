import os
import sys
import argparse
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
from utils.train_test_utils import save_images_batch, save_json, get_transforms

def test(config):

    run_name = config["name"]
    dataset_cfg = config['dataset']
    autoencoder_config = config['autoencoder']
    train_cfg = config['vqvae_train']
    test_config = config["vqvae_test"]

    ae_ckpt_name = f"autoencoder"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images_save_dir = os.path.join(train_cfg['ckpt_folder'], run_name, "test", "images")
    log_save_name = os.path.join(train_cfg['ckpt_folder'], run_name, "test", "test_logs.json")

    # create checkpoint paths
    Path(images_save_dir).mkdir(parents=True, exist_ok=True)

    # transforms
    transforms = get_transforms(dataset_cfg)

    #dataset
    dataset_test = HolographyImageFolder(root=dataset_cfg["root"], 
                                         transform=transforms, 
                                         dataset_cfg=dataset_cfg,
                                         labels=dataset_cfg.get("labels_test"))

    # dataloader
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=1,
                                 shuffle=False)
    

    # load pretrained vqvae
    model = VQVAE(img_channels=dataset_cfg['img_channels'], config=autoencoder_config).to(device)

    model.load_state_dict(
        torch.load(os.path.join(train_cfg['ckpt_folder'], 
                                run_name,
                                ae_ckpt_name, 
                                test_config["model_ckpt"]), 
                                map_location=device))
    model.eval()

    # mse reconstruction criterion
    mse_loss = torch.nn.MSELoss()

    # lpips perceptual criterion
    lpips_model = LPIPS(net_type='alex').to(device)

    folders = []
    sample_filenames = []
    reconstruction_losses = []          # reconstruction loss (l2)
    codebook_losses = []                # codebook loss between predicted and nearest codebook vector
    lpips_losses = []                   # preceptual loss (lpips)
    
    with torch.no_grad():

        pbar = tqdm(dataloader_test)
        for (im, folder, filenames) in pbar:

            im = im.float().to(device)

            # autoencoder forward pass
            output, z, quantize_losses = model(im)

            folders.extend(folder)
            sample_filenames.extend(filenames)

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

            lpips_loss = torch.mean(lpips_model(out_lpips, im_lpips))
            lpips_losses.append(lpips_loss.item())

            # save images
            if test_config["save_images"]:
                output = torch.clamp(output, -1., 1.)
                output = (output + 1) / 2
                save_images_batch(output, filenames, save_dir=images_save_dir)

            break


        logs = {"test_reconstructon_loss"    : reconstruction_losses,
                "test_codebook_loss"         : codebook_losses,
                "test_lpips_loss"            : lpips_losses,
                "filenames"                  : sample_filenames,
                }
        
        save_json(logs, log_save_name)

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for vqvae inference')
    parser.add_argument('--config', dest='config_path', default='config/base_vqvae_config.yaml', type=str)
    args = parser.parse_args()

    config = load_config(args.config_path)

    test(config)

    




