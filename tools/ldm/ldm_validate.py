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

from model.unet_v2 import UNet
from model.vqvae import VQVAE
from model.ddpm import Diffusion as DDPMDiffusion
from utils.config import load_config
from pollen_datasets.poleno import HolographyImageFolder


def validate(config_path, model=None, vae=None, diffusion=None, model_ckpt=None):

    global dataloader_val

    config = load_config(config_path)
    dataset_config = config['dataset']
    ddpm_model_config = config['ddpm']
    autoencoder_model_config = config['autoencoder']
    train_config = config['ldm_train']
    inference_config = config["ddpm_inference"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ###################################### data ######################################
    if not "dataloader_val" in globals():
        transforms_list = []

        transforms_list.append(torchvision.transforms.ToTensor())

        if dataset_config.get("img_interpolation"):
            transforms_list.append(torchvision.transforms.Resize((dataset_config["img_interpolation"], 
                                                                dataset_config["img_interpolation"]),
                                                                interpolation = torchvision.transforms.InterpolationMode.BILINEAR))

        transforms_list.append(torchvision.transforms.Normalize((0.5) * dataset_config["img_channels"], 
                                                                (0.5) * dataset_config["img_channels"]))

        transforms = torchvision.transforms.Compose(transforms_list)

        # dataset
        dataset_val = HolographyImageFolder(root=dataset_config["root"], 
                                    transform=transforms, 
                                    config=dataset_config,
                                    labels=dataset_config.get("labels_val"))

        # dataloader
        dataloader_val = DataLoader(dataset_val,
                                batch_size=train_config['ldm_batch_size'],
                                shuffle=False)
    
    ################################## autoencoder ###################################
    # # Load Autoencoder
    # if vae is None: 
    #     vae = VQVAE(img_channels=dataset_config['img_channels'], config=autoencoder_model_config).to(device)

    #     vae_ckpt_path = os.path.join(train_config['task_name'],
    #                                  train_config['vqvae_ckpt_dir'],
    #                                  train_config['vqvae_ckpt_model']
    #                                  )

    #     vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))
    #     print(f'Loaded autoencoder checkpoint from {vae_ckpt_path}')

    #     for param in vae.parameters():
    #         param.requires_grad = False
    
    # vae.eval()

    ############################### context encoding ################################

    # if train_config["conditioning"] == "unconditional":
    #     context_encoder = None
    # else:
    #     use_condition = train_config["conditioning"].split("+")
    #     num_classes = int(max(dataset_val.class_labels)) + 1 if "class" in use_condition else None
    #     cls_emb_dim = ddpm_model_config['cond_emb_dim'] + 1 if "class" in use_condition else None
    #     tabular_in_dim = len(dataset_config["features"]) if "tabular" in use_condition else None
    #     tabular_out_dim = ddpm_model_config['cond_emb_dim'] if "tabular" in use_condition else None

    #     context_encoder = ConditionEmbedding(out_dim=ddpm_model_config['time_emb_dim'],
    #                                          num_classes=num_classes,
    #                                          cls_emb_dim=cls_emb_dim,
    #                                          tabular_in_dim=tabular_in_dim,
    #                                          tabular_out_dim=tabular_out_dim,
    #                                          img_in_channels=None, # TODO add image conditioning to training
    #                                          img_out_dim=None
    #                                          )
    #     context_encoder.to(device)

    ##################################### u-net ######################################
    # # Load UNet
    # if model is None: 
    #     model = UNet(img_channels=autoencoder_model_config['z_channels'], 
    #                  model_config=ddpm_model_config,
    #                  context_encoder=context_encoder).to(device)

        

    #     unet_ckpt_path = os.path.join(train_config['task_name'], 
    #                                 train_config['ldm_ckpt_name'], 
    #                                 inference_config["ddpm_model_ckpt"]
    #                                 )
        
    #     model.load_state_dict(torch.load(unet_ckpt_path, map_location=device))
    #     print(f'Loaded unet checkpoint from {unet_ckpt_path}')

    # model.eval()

    ################################### diffusion ####################################
    # if diffusion is None:
    #     # init diffusion class
    #     if dataset_config.get('img_interpolation'):
    #         img_size = dataset_config['img_interpolation'] // 2 ** sum(autoencoder_model_config['down_sample'])
    #     else:
    #         img_size = dataset_config['img_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    #     diffusion = DDPMDiffusion(img_size=img_size, 
    #                             img_channels=autoencoder_model_config['z_channels'],
    #                             noise_schedule="linear", 
    #                             beta_start=train_config["ldm_beta_start"], 
    #                             beta_end=train_config["ldm_beta_end"],
    #                             device=device,
    #                             )

    # define criterions for evaluation
    criterion = torch.nn.MSELoss()

    reconstruction_losses = []
 
    for (im, condition, _) in dataloader_val:

        with torch.no_grad():
            
            im = im.float().to(device)
            for key in condition.keys():
                condition[key] = condition[key].to(device)

            # randomly discard conditioning to train unconditionally if conditional training
            if train_config["conditioning"] == "unconditional" or np.random.random() < train_config["ldm_cfg_discard_prob"]:
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

    validate(args.config_path)