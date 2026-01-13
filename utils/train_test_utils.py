import os
import json
from pathlib import Path
import torch


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


def online_running_mean(mu, x, step):
    mu += (x - mu) / step
    return mu


def get_image_encoder_names(condition_config):
    image_encoders = []
    for name, cfg in condition_config["encoders"].items():
        if cfg.get("type") == "image" and name in condition_config["enabled"]:
            image_encoders.append(name)
    return image_encoders


def deep_update(d, u):
    """Update a nested dictionary d with another nested dictionary u"""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d