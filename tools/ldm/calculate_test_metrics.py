import os
from PIL import Image
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

from pollen_datasets.poleno import HolographyImageFolder


import torch
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from model.conditioning.transforms.registry import get_transforms
from utils.config import load_config
from utils.train_test_utils import deep_update
from evaluation.particle import rotation_invariant_particle_metrics
from evaluation.regionprops import calculate_regionprops

def load_image(full_path):

    with Image.open(full_path) as img:
        mode = img.mode
        img = np.array(img).astype(np.float32)

    if mode == 'I;16' or mode == 'I':
        img /= 65535.0
    elif mode == 'L':
        img /= 255.0
    elif mode == 'RGB':
        img /= 255.0
    else:
        raise ValueError(f"Unsupported image mode: {mode}")

    return img


def calc_test_metrics(config):

    dataset_cfg = config["dataset"]
    test_cfg = config["test"]

    
    transform_cfg = {"name": "test_image_transform", "img_interpolation": 256}
    transforms = get_transforms(transform_cfg, in_channels=dataset_cfg["img_channels"])

    gt_dataset = HolographyImageFolder(
        dataset_cfg=dataset_cfg, 
        transform=transforms,
        labels=dataset_cfg.get("labels_test"),
        root=dataset_cfg["root"]
        )

    gt_dataloader = DataLoader(
        gt_dataset,
        batch_size=1,
    )

    # LPIPS (with normalize=True expects values between [0, 1])
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)

    df = pd.DataFrame()

    for i, (gt_img_tensor, _, filename) in enumerate(tqdm(gt_dataloader)):

        filename = filename[0]

        full_path = os.path.join(test_cfg["gen_images_root"], filename)

        if not os.path.exists(full_path):
            continue

        if not ".0.0." in full_path:
            continue

        df.loc[i, "rec_path"] = filename

        # load predicted image
        pred_img_np = load_image(full_path)

        pred_img_tensor = transforms(pred_img_np)
        pred_img_tensor = pred_img_tensor.unsqueeze(1)

        gt_img_np = gt_img_tensor.squeeze().cpu().numpy()
        pred_img_np = pred_img_np.squeeze()

        # Image MSE
        mse = torch.mean((pred_img_tensor - gt_img_tensor) ** 2)
        df.loc[i, "mse"] = mse.item()

        # Image LPIPS
        lpips = lpips_metric(
            pred_img_tensor.repeat(1, 3, 1, 1), 
            gt_img_tensor.repeat(1, 3, 1, 1)
        )
        df.loc[i, "lpips"] = lpips.item()
        df.loc[i, "rec_path"] = filename
        
        # Particle Regionprops
        calculate_regionprops(df, i, full_path)

        # Particle DICE & MSE
        p_dice, p_iou, p_mse = rotation_invariant_particle_metrics(
            gt_img_np, pred_img_np, angle_step=5, combined_mask="intersection",
            )
        
        df.loc[i, "particle_dice"] = p_dice
        df.loc[i, "particle_iou"] = p_iou
        df.loc[i, "particle_mse"] = p_mse

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for postprocessing')

    parser.add_argument(
        '--ckpt_dir',  
        type=str,
        nargs='+', 
        required=True,
        help="Dir paths to the checkpoint",
    )

    parser.add_argument(
        '--gt_img_root',  
        type=str,
        default="Z:/marvel/marvel-fhnw/data/",
        help="Ground truth image data root",
    )

    parser.add_argument(
        '--outfile',  
        type=str,
        default="scores.csv",
        help="Output filename",
    )

    args = parser.parse_args()     

    for ckpt_dir in args.ckpt_dir:

        test_samples = os.listdir(os.path.join(ckpt_dir, "test"))

        for test_sample in test_samples:

            print("Testing:", test_sample)

            output_dir              = os.path.join(ckpt_dir, "test", test_sample)
            config_file             = os.path.join(output_dir, "test_config.yaml")
            gen_img_root            = os.path.join(output_dir, "images")
            outfile                 = os.path.join(output_dir, args.outfile)
            
            data_to_update = {
                "dataset": {
                    "root": args.gt_img_root,
                },
                "test": {
                    "gen_images_root": gen_img_root,
                }
            }

            config = load_config(config_file)
            config = deep_update(config, data_to_update)

            df_metrics = calc_test_metrics(config)

            df_metrics.to_csv(outfile, index=False)
            print("Saved at:", outfile)
    

# Example use
#  python -m tools.ldm.calculate_test_metrics --ckpt_dir "Z:/simon_luder/LDM/LDM_for_Holographic_Images/checkpoints/ldm_nv_vicreg_rp_avgpool_knn_L"  