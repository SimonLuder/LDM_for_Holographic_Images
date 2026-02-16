import torch
import numpy as np
from pollen_datasets.poleno.registry import register_condition_fn


@register_condition_fn("relative_viewpoint_rotation")
def relative_viewpoint_rotation(val1, val2, meta):
    """
    Rotation-aware Zero-1-to-3 viewpoint conditioning.
    
    Baseline geometry (no rotation, no swap):
        img0 is located on +x axis
        img1 is located on +z axis
        => relative azimuth φ = +90° = +pi/2
        => Zero-1-to-3 encoding = [0, sin(φ), cos(φ), 0]

    If the image pair is rotated by N degrees, the relative viewpoint
    rotates by the same amount.

    If swapped, img0 <-> img1, the direction reverses sign.
    """

    # dataset augmentations
    rotation = meta.get("rotation", 0)       # 0, 90, 180, 270 degrees
    swapped  = meta.get("swapped", False)

    # baseline azimuth difference:
    #   img0 on +x, img1 on +z  →  +90° = +pi/2
    base_phi = np.pi / 2

    # convert applied rotation to radians
    rot_rad = np.deg2rad(rotation)

    # swapping reverses direction
    # meaning: φ -> -φ
    if swapped:
        rot_rad = -rot_rad
        base_phi = -base_phi

    # total relative azimuth
    phi = base_phi + rot_rad

    # Zero-1-to-3 conditioning vector:
    # [ Δθ,  sin(φ), cos(φ), Δr ]
    cond = torch.tensor([
        0.0,                # Δθ = 0 (no polar difference in this setup)
        np.sin(phi),
        np.cos(phi),
        0.0                 # Δr = 0 (no radius change)
    ], dtype=torch.float32)

    # Return conditioning for both images
    # (symmetrical: model takes cond with each img)
    return cond, cond


@register_condition_fn("dual_image_indices")
def dual_image_condition_index(val1, val2, meta):
    """Returns the indices for the images"""
    return 0, 1


@register_condition_fn("fourier_orientation")
def pair_fourier_orientation(val1, val2, meta):
    
    def fourier_orientation(theta, num_freqs=4):
        theta = torch.tensor(theta, dtype=torch.float32)

        freqs = 2 ** torch.arange(num_freqs)
        angles = theta * freqs

        features = torch.cat([
            torch.sin(angles),
            torch.cos(angles)
        ])
        return features  # (num_freqs*2,)

    return (
        fourier_orientation(val1),
        fourier_orientation(val2),
    )
    
