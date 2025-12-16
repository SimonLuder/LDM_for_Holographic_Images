from typing import Callable, Dict, Any
import torch
import torchvision.transforms as T


def get_transforms(transform_cfg: dict, in_channels: int):
    """
    Build a transform from config.
    """
    if transform_cfg is None:
        return None

    name = transform_cfg["name"]

    params = {
        "img_channels": in_channels,
        "img_interpolation": transform_cfg.get("img_interpolation"),
    }

    builder = get_cond_transform_builder(name)
    return builder(params)


# ------------------------------------------------------------------ #
# Condition Registry

CondTransformBuilder = Callable[[Dict[str, Any]], Callable]

COND_TRANSFORM_REGISTRY: Dict[str, CondTransformBuilder] = {}

def register_cond_transform(name: str):
    name = name.lower()
    def decorator(fn: CondTransformBuilder):
        COND_TRANSFORM_REGISTRY[name] = fn
        return fn
    return decorator


def get_cond_transform_builder(name: str) -> CondTransformBuilder:
    name = name.lower()
    if name not in COND_TRANSFORM_REGISTRY:
        raise KeyError(
            f"Unknown condition transform '{name}'. "
            f"Registered: {list(COND_TRANSFORM_REGISTRY.keys())}"
        )
    return COND_TRANSFORM_REGISTRY[name]


# -----------------------------------------------------------------
# Custom transformations

@register_cond_transform("base_transform")
def build_base_transform(params: Dict[str, Any]):
    """
    params:
      img_channels: int
      img_interpolation: int | None
    """

    img_channels      = params["img_channels"]
    img_interpolation = params.get("img_interpolation", None)

    transforms_list = []

    # To tensor
    transforms_list.append(T.ToTensor())

    # Resize if specified
    if img_interpolation is not None:
        transforms_list.append(
            T.Resize(
                (img_interpolation, img_interpolation),
                interpolation=T.InterpolationMode.BILINEAR
            )
        )

    # Normalize to [-1, 1]
    transforms_list.append(
        T.Normalize(
            mean=[0.5] * img_channels,
            std=[0.5] * img_channels
        )
    )

    return T.Compose(transforms_list)


@register_cond_transform("clip_image_transform")
def build_clip_transform(params: Dict[str, Any]):
    """
    params:
      img_channels: int        # input channels (1 or 3)
      img_interpolation: int   # output spatial size (usually 224)
    """

    img_channels      = params["img_channels"]
    img_interpolation = params["img_interpolation"]

    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std  = (0.26862954, 0.26130258, 0.27577711)

    def ensure_three_channels(x: torch.Tensor) -> torch.Tensor:
        # x: (C,H,W) or (H,W)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1 and img_channels == 1:
            x = x.repeat(3, 1, 1)
        return x

    transforms_list = [
        T.ToTensor(),
        T.Lambda(ensure_three_channels),
        T.Resize(
            (img_interpolation, img_interpolation),
            interpolation=T.InterpolationMode.BICUBIC
        ),
        T.Normalize(mean=clip_mean, std=clip_std),
    ]

    return T.Compose(transforms_list)