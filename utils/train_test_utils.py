import os
import yaml
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path
import json

import torch
import torchvision
from torch.utils.data import DataLoader

import clip

# from dataset import ImageClassDataset, ImageSentenceDataset, ImageTabularDataset, ImageImageDataset


def resume_from_checkpoint():
    raise NotImplementedError(f"'resume_from_checkpoint' is no implemented!")


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

    Parameters:
    images (Tensor): a 4D mini-batch Tensor of shape (B x C x H x W) 
                    or a list of images all of the same size.
    filenames (str): filename from the batch.
    """
    for image, filename in zip(images, filenames):
        image = torchvision.transforms.ToPILImage()(image)
        filename = os.path.join(save_dir, os.path.basename(filename))
        image.save(filename)


def get_dataloader(args):

    if args.image_channels == 3:
        transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    elif args.image_channels == 1:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5), (0.5))
            ])
    

    if args.cfg_encoding == None or args.cfg_encoding == "classes":
        print("Loading ImageFolder")

        if vars(args).get("train_images"):
            print(args.train_images)
            train_dataset = ImageClassDataset(root=args.train_images, transform=transforms)
        if vars(args).get("val_images"):
            val_dataset   = ImageClassDataset(root=args.val_images, transform=transforms)
        if vars(args).get("test_images"):
            test_dataset  = ImageClassDataset(root=args.test_images, transform=transforms)

    if args.cfg_encoding == "tabular":
        
        print("Loading ImageTabularDataset")

        if vars(args).get("train_images"):
            train_dataset = ImageTabularDataset(labels_path=args.train_labels, transform=transforms)
        if vars(args).get("val_images"):
            val_dataset   = ImageTabularDataset(labels_path=args.val_labels, transform=transforms)
        if vars(args).get("test_images"):
            test_dataset  = ImageTabularDataset(labels_path=args.test_labels, transform=transforms)

    if args.cfg_encoding == "clip_text":

        print("Loading ImageSentenceDataset")

        preprocess = lambda x: clip.tokenize(x)[0]

        if vars(args).get("train_images"):
            train_dataset = ImageSentenceDataset(labels_path=args.train_labels, transform=transforms, preprocess=preprocess)
        if vars(args).get("val_images"):
            val_dataset   = ImageSentenceDataset(labels_path=args.val_labels, transform=transforms, preprocess=preprocess)
        if vars(args).get("test_images"):
            test_dataset  = ImageSentenceDataset(labels_path=args.test_labels, transform=transforms, preprocess=preprocess)

    if args.cfg_encoding == "clip_image":

        print("Loading ImageImageDataset")

        _, preprocessor = clip.load(args.encoder_model, device=args.device)
        preprocess = lambda img: preprocessor(img)

        if vars(args).get("train_images"):
            train_dataset = ImageImageDataset(labels_path=args.train_labels, transform=transforms, preprocess=preprocess)
        if vars(args).get("val_images"):
            val_dataset   = ImageImageDataset(labels_path=args.val_labels, transform=transforms, preprocess=preprocess)
        if vars(args).get("test_images"):
            test_dataset  = ImageImageDataset(labels_path=args.test_labels, transform=transforms, preprocess=preprocess)

    if args.cfg_encoding == "cnn_image":

        print("Loading ImageImageDataset")

        if vars(args).get("train_images"):
            train_dataset = ImageImageDataset(labels_path=args.train_labels, transform=transforms, preprocess=transforms)
        if vars(args).get("val_images"):
            val_dataset   = ImageImageDataset(labels_path=args.val_labels, transform=transforms, preprocess=transforms)
        if vars(args).get("test_images"):
            test_dataset  = ImageImageDataset(labels_path=args.test_labels, transform=transforms, preprocess=transforms)

    dataloader = dict()
    if vars(args).get("train_images"):
        dataloader["train"] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if vars(args).get("val_images"):
        dataloader["val"]   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    if vars(args).get("test_images"):
        dataloader["test"]  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return dataloader
    

def initialize_model_weights(model, weight_path, device):
    model_weights_dict = torch.load(f=weight_path, map_location=device)
    model.load_state_dict(model_weights_dict)

def yaml_load_config(config_path):
    if config_path:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
def save_json(data_dict, filepath):
    path, _ = os.path.split(filepath)
    Path(path).mkdir( parents=True, exist_ok=True )
    with open(filepath, "w") as file:
        json.dump(data_dict, file)


def load_json(filepath):
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
