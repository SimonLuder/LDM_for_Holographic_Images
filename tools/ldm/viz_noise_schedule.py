"""
Visualizes the forward diffusion (noising) process on holography images.

This script loads images from a dataset, applies a DDPM diffusion noise
schedule, and saves:
- the original (clean) image,
- the fully noised image,
- a grid showing noise progression over timesteps.

Run:
python -m tools.ldm.viz_noise_schedule
"""


import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import make_grid

from model.ddpm import Diffusion
from pollen_datasets.poleno import HolographyImageFolder
from model.conditioning.transforms.registry import get_transforms
from utils.config import load_config


def save_unnoised_and_fully_noised(
    im,
    diffusion,
    noise_steps,
    path_clean,
    path_noisy,
    ):
    """
    im: (1, C, H, W) clean image in [-1, 1]
    diffusion: Diffusion object
    noise_steps: total number of diffusion steps
    """

    device = im.device

    def _save_single(x, path):
        x = (x.clamp(-1, 1) + 1) / 2 * 255
        x = x.byte()
        # x = add_border(x, border=border, value=0)
        x = x.squeeze(0)  # (C, H, W)
        img = transforms.ToPILImage()(x)
        img.save(path)

    # t = 0 → unnoised
    _save_single(im, path_clean)

    # t = T-1 → fully noised
    t_full = torch.tensor([noise_steps - 1], device=device)
    x_T, _ = diffusion.noise_images(im, t_full)

    _save_single(x_T, path_noisy)


def load_single_image_from_dataset(config, index=0, device="cpu"):
    dataset_cfg = config["dataset"]
    transforms_cfg = config["transforms"]

    transforms = get_transforms(transforms_cfg["transform1"], in_channels=dataset_cfg["img_channels"])

    dataset = HolographyImageFolder(
        root=dataset_cfg["root"],
        transform=transforms,
        dataset_cfg=dataset_cfg,
        labels=dataset_cfg.get("labels_test")
    )

    im, folder, filename = dataset[index]
    im = im.unsqueeze(0).float().to(device)  # add batch dim

    return im, folder, filename


def add_border(x, border=2, value=0):
    """
    x: (1, C, H, W)
    border: pixels
    value: 0=black, 255=white (after scaling)
    """
    return F.pad(
        x,
        pad=(border, border, border, border),  # left, right, top, bottom
        mode="constant",
        value=value
    )


def save_grid(images, path, nrow=11, border=1):
    bordered = []

    for x in images:
        x = (x.clamp(-1, 1) + 1) / 2 * 255
        x = x.byte()
        x = add_border(x, border=border, value=0)
        bordered.append(x)

    x = torch.cat(bordered, dim=0)  # (N, C, H, W)
    grid = make_grid(x, nrow=nrow, padding=0)

    img = transforms.ToPILImage()(grid)
    img.save(path)


def main(indices, config, output_dir="images/noised_steps", noise_schedule="linear",
         noise_steps=1000, save_every=100, img_size=200,
         device="cuda" if torch.cuda.is_available() else "cpu"):

    os.makedirs(output_dir, exist_ok=True)

    diffusion = Diffusion(
        img_size=img_size,
        img_channels=1,
        noise_schedule=noise_schedule,
        noise_steps=noise_steps,
        device=device,
    )

    for index in indices:
        im, folder, filename = load_single_image_from_dataset(config, index=index, device=device)

        print(folder)

        collected = []

        # original
        collected.append(im)

        for t in range(0, noise_steps):
            t_tensor = torch.tensor([t], device=device)
            x_t, _ = diffusion.noise_images(im, t_tensor)

            if t % save_every == 0 or t == noise_steps:
                collected.append(x_t)

        save_grid(
            collected,
            os.path.join(output_dir, f"noise_progression_{noise_schedule}_{index}_{filename}.png")
        )

        save_unnoised_and_fully_noised(
            im=im,
            diffusion=diffusion,
            noise_steps=noise_steps,
            path_clean=os.path.join(
                output_dir, f"clean_{noise_schedule}_{index}_{filename}.png"
            ),
            path_noisy=os.path.join(
                output_dir, f"fully_noised_{noise_schedule}_{index}_{filename}.png"
            ),
        )

    print("Saved noise progression")

if __name__ == "__main__":

    config_path = "config/base_ldm_config.yaml"

    main(
        indices=[0, 500, 1000, 2000, 3000], # image indices in dataset
        config=load_config(config_path),
        noise_schedule="cosine",  # "cosinie" or "linear"
        save_every=100,
    )