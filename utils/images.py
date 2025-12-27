import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.utils import make_grid



def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def save_images_batch(images, filenames, save_dir):
    """
    Save images from a PyTorch batch as individual image files.

    Args:
    images (torch.Tensor): a 4D mini-batch Tensor of shape (B x C x H x W) 
                    or a list of images all of the same size.
    filenames (str): filenames from the batch images.
    save_dir (str): path where to save the batch.
    """
    for image, filename in zip(images, filenames):
        image = torchvision.transforms.ToPILImage()(image)
        filename = os.path.join(save_dir, os.path.basename(filename))
        image.save(filename)


# save images
def save_sample_images(input_img, output_img, filepath):
    """Saves a grid of input and output images for comparison."""
    sample_size = min(8, input_img.shape[0])
    save_output = torch.clamp(output_img[:sample_size], -1., 1.).detach().cpu()
    save_output = ((save_output + 1) / 2)
    save_input = ((input_img[:sample_size] + 1) / 2).detach().cpu()
    
    grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
    img = torchvision.transforms.ToPILImage()(grid)

    img.save(filepath)
    img.close()