import os
import sys
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.utils import make_grid

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from model.vqvae import VQVAE
from pollen_datasets.poleno import HolographyImageFolder
from utils.config import load_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(config_file):

    config = load_config(config_file)

    dataset_config = config['dataset']
    autoencoder_config = config['autoencoder']
    train_config = config['vqvae_train']
    inference_config = config["vqvae_inference"]

    Path(os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'], 'inference')).mkdir(parents=True, exist_ok=True)

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
    dataset = HolographyImageFolder(root=dataset_config["root"], 
                                         transform=transforms, 
                                         config=dataset_config,
                                         labels=dataset_config.get("labels_test"))

    # dataloader
    dataloader = DataLoader(dataset,
                            batch_size=train_config['autoencoder_batch_size'],
                            shuffle=False)
    
    # vqvae
    model = VQVAE(img_channels=dataset_config['img_channels'],
                  config=autoencoder_config).to(device)
    
    model.load_state_dict(
        torch.load(os.path.join(train_config['task_name'], 
                                train_config['vqvae_autoencoder_ckpt_name'], 
                                inference_config["model_ckpt"]), 
                                map_location=device))
    
    model.eval()

    idxs = torch.randint(0, len(dataset) - 1, (inference_config['num_samples'],)).numpy()
    ims = torch.cat([dataset[idx][0][None, :] for idx in idxs]).float()
    ims = ims.to(device)

    with torch.no_grad():
        
        encoded_output, _ = model.encode(ims)
        decoded_output = model.decode(encoded_output)
        encoded_output = torch.clamp(encoded_output, -1., 1.)
        encoded_output = (encoded_output + 1) / 2
        decoded_output = torch.clamp(decoded_output, -1., 1.)
        decoded_output = (decoded_output + 1) / 2
        ims = (ims + 1) / 2

        # only select first 3 dimensions of the codebook
        if encoded_output.shape[1] > 3:
            encoded_output = encoded_output[:, :3, :, :]

        # interpolate encoded output to same size as input
        if inference_config['upscale_latent_dim']:
            encoded_output = F.interpolate(encoded_output, size=(ims.shape[-2], ims.shape[-1]), mode="nearest")

        encoder_grid = make_grid(encoded_output.cpu(), nrow=inference_config['num_grid_rows'])
        decoder_grid = make_grid(decoded_output.cpu(), nrow=inference_config['num_grid_rows'])
        input_grid = make_grid(ims.cpu(), nrow=inference_config['num_grid_rows'])

        encoder_grid = torchvision.transforms.ToPILImage()(encoder_grid)
        decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
        input_grid = torchvision.transforms.ToPILImage()(input_grid)
        
        input_grid.save(os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'], 
                                     'inference', 'input_samples.png'))
        encoder_grid.save(os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'], 
                                       'inference', 'encoded_samples.png'))
        decoder_grid.save(os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'], 
                                       'inference', 'reconstructed_samples.png'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for vqvae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/base_vqvae_config.yaml', type=str)
    args = parser.parse_args()

    inference(args.config_path)

# python tools/vqvae_inference.py --config holographic_pollen/vqvae_autoencoder_ckpt_512_3/ldm3_config.yaml