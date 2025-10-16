import os
import sys
import argparse
from pathlib import Path
from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader
import torchvision

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from model.unet_v2 import UNet
from model.vqvae import VQVAE
from model.embedding import ConditionEmbedding
from model.ddpm import Diffusion as DDPMDiffusion
from utils.config import load_config
from pollen_datasets.poleno import HolographyImageFolder
from utils.train_test_utils import save_images_batch, save_tensors_batch


def test(config_path):

    config = load_config(config_path)
    dataset_config = config['dataset']
    ddpm_model_config = config['ddpm']
    autoencoder_model_config = config['autoencoder']
    train_config = config['ldm_train']
    inference_config = config['ddpm_inference']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images_save_dir = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'], "test", "images")
    latents_save_dir = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'], "test", "latents")

    Path(images_save_dir).mkdir(parents=True, exist_ok=True)
    Path(latents_save_dir).mkdir(parents=True, exist_ok=True)

    ################################### transforms ###################################
    transforms_list = []

    transforms_list.append(torchvision.transforms.ToTensor())

    if dataset_config.get("img_interpolation"):
        transforms_list.append(torchvision.transforms.Resize((dataset_config["img_interpolation"], 
                                                              dataset_config["img_interpolation"]),
                                                              interpolation = torchvision.transforms.InterpolationMode.BILINEAR))

    transforms_list.append(torchvision.transforms.Normalize((0.5) * dataset_config["img_channels"], 
                                                            (0.5) * dataset_config["img_channels"]))

    transforms = torchvision.transforms.Compose(transforms_list)

    ###################################### data ######################################

    #dataset
    dataset = HolographyImageFolder(root=dataset_config["root"], 
                                    transform=transforms, 
                                    config=dataset_config,
                                    labels=dataset_config.get("labels_test"))

    # dataloader
    dataloader = DataLoader(dataset,
                            batch_size=train_config['ldm_batch_size'],
                            shuffle=True)
    
    
    ################################## autoencoder ###################################
    # Load Autoencoder
    vae = VQVAE(img_channels=dataset_config['img_channels'], config=autoencoder_model_config).to(device)
    vae.eval()

    vae_ckpt_path = os.path.join(train_config['task_name'],
                                 train_config['vqvae_ckpt_dir'],
                                 train_config['vqvae_ckpt_model']
                                 )

    vae.load_state_dict(torch.load(vae_ckpt_path, map_location=device))
    print(f'Loaded autoencoder checkpoint from {vae_ckpt_path}')

    for param in vae.parameters():
        param.requires_grad = False


    ############################### context encoding ################################
    # Load context encoder
    if train_config["conditioning"] == "unconditional":
        print("training unconditional model")
        context_encoder = None
    else:
        use_condition = train_config["conditioning"].split("+")
        print("sampling conditional model with:", use_condition)
        # num_classes = int(max(dataset.class_labels)) + 1 if "class" in use_condition else None

        unet_ckpt_path = os.path.join(train_config['task_name'], 
                                  train_config['ldm_ckpt_name'], 
                                  inference_config["ddpm_model_ckpt"])
        
        state_dict = torch.load(unet_ckpt_path, map_location=device)
        num_classes = state_dict["context_encoder.class_emb.weight"].shape[0] if "class" in use_condition else None
        print("nr of classes:", num_classes)

        cls_emb_dim = ddpm_model_config['cls_emb_dim'] + 1 if "class" in use_condition else None
        tabular_in_dim = len(dataset_config["features"]) if "tabular" in use_condition else None
        tabular_out_dim = ddpm_model_config['tbl_emb_dim'] if "tabular" in use_condition else None
        img_out_dim = ddpm_model_config['img_emb_dim'] if "image" in use_condition else None

        
        context_encoder = ConditionEmbedding(out_dim=ddpm_model_config['time_emb_dim'],
                                             num_classes=num_classes,
                                             cls_emb_dim=cls_emb_dim,
                                             tabular_in_dim=tabular_in_dim,
                                             tabular_out_dim=tabular_out_dim,
                                             img_in_channels=None, # TODO add image conditioning to training
                                             img_out_dim=img_out_dim
                                             )
        context_encoder.to(device)

    ##################################### u-net ######################################
    # Load UNet
    model = UNet(img_channels=autoencoder_model_config['z_channels'], 
                 model_config=ddpm_model_config, 
                 context_encoder=context_encoder).to(device)
    model.eval()

    unet_ckpt_path = os.path.join(train_config['task_name'], 
                                  train_config['ldm_ckpt_name'], 
                                  inference_config["ddpm_model_ckpt"]
                                  )
       
    model.load_state_dict(torch.load(unet_ckpt_path, map_location=device))
    print(f'Loaded unet checkpoint from {unet_ckpt_path}')

    ################################### diffusion ####################################
    # init diffusion class
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

    with torch.no_grad():

        for batch_idx, (img_gt, condition, filenames) in enumerate(dataloader):

            print("batch", batch_idx)

            img_gt = img_gt.float().to(device)
            for cond in use_condition:
                condition[cond] = condition[cond].to(device)

            img_latent = diffusion.sample(model, 
                                          condition=condition, 
                                          n=train_config['ldm_batch_size'], 
                                          cfg_scale=3,
                                          to_uint8=False)

            # upsample with vqvae
            img = vae.decode(img_latent)

            # save latents
            save_tensors_batch(img_latent.cpu(), filenames, save_dir=latents_save_dir)

            # pca image reduction
            if img_latent.shape[1] > 3:
                img_latent = pca_channel_reduction(img_latent, out_channels=3)

            #save the generated latent representation
            img_latent = torch.clamp(img_latent, -1., 1.)
            img_latent = (img_latent + 1) / 2
            save_images_batch(img_latent.cpu(), filenames, save_dir=latents_save_dir)

            #save the generated image
            img = torch.clamp(img, -1., 1.)
            img = (img + 1) / 2
            save_images_batch(img.cpu(), filenames, save_dir=images_save_dir)


def pca_channel_reduction(batch, out_channels = 3):

    # Initialize PCA
    pca = PCA(n_components=out_channels)

    # List to store PCA transformed images
    x_pca_batch = []

    # Loop over each image in the batch
    for im in batch:
        # Get the shape of the image
        C, H, W = im.shape

        # Flatten the image
        x_2d = im.view(C, -1).cpu().numpy()

        # Fit PCA on the image
        pca.fit(x_2d)

        # Transform the image using PCA and reshape it back to original shape
        x_pca = pca.components_.reshape(out_channels, H, W)
        x_pca = ((x_pca - x_pca.min()) / (x_pca.max() - x_pca.min()))

        # Append the PCA transformed image to the list
        x_pca_batch.append(x_pca)

    # Convert the list to a tensor
    x_pca_batch = [torch.from_numpy(arr) for arr in x_pca_batch]
    x_pca_batch = torch.stack(x_pca_batch)

    return x_pca_batch


# Run with command: python -m tools.ldm.ldm_test --config config/base_ldm_config.yaml
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Arguments for ldm inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/base_ldm_config.yaml', type=str)
    args = parser.parse_args()

    test(args.config_path)