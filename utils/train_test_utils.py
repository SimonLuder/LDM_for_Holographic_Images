import os
import json
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path

import torch
import torchvision


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


def save_json(data_dict, filepath):
    """
    Save a dictionary as a JSON file.

    Args:
        data_dict (dict): Dictionary to save.
        filepath (str): Path to the save the JSON file.
    """
    path, _ = os.path.split(filepath)
    Path(path).mkdir( parents=True, exist_ok=True )
    with open(filepath, "w") as file:
        json.dump(data_dict, file)


def load_json(filepath):
    """
    Load a dictionary from a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Loaded dictionary. If the file does not exist, returns an empty list.
    """
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            data_dict = json.load(file)
        return data_dict
    else:
        return []
  

def save_tensors_batch(tensors, filenames, save_dir):
    for filename, tensor in zip(filenames, tensors):
        base_name = os.path.splitext(os.path.basename(filename))[0]  # Remove current extension
        new_filename = os.path.join(save_dir, f'{base_name}.pt')  # Add .pt extension
        torch.save(tensor, new_filename)
     

def load_tensors_batch(save_dir, filenames):
    loaded_tensors = []
    for filename in filenames:
        base_name = os.path.splitext(os.path.basename(filename))[0]  # Remove current extension
        tensor_path = os.path.join(save_dir, f'{base_name}.pt')  # Add .pt extension
        tensor = torch.load(tensor_path)
        loaded_tensors.append(tensor)
    return loaded_tensors
