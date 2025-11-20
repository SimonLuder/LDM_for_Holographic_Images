from __future__ import annotations
from typing import Callable, Dict, Any
import torch.nn as nn
from .embeddings import MLP
from .wrapper import ConditionEmbeddingWrapper

# ------------------------------------------------------------------ #
# builder function

def build_encoder_from_registry(wrapper_out_dim, cond_cfg, device="cpu"):
    """
    Build context encoder purely from config:
    - Each encoder must define params.out_dim explicitly in config.
    """

    enabled = cond_cfg["enabled"]
    enabled_names = [c for c in enabled.replace(" ", "").split("+") if c]

    encoders = {}
    encoders_out_dims = {}

    for name in enabled_names:
        cfg = cond_cfg["encoders"][name]

        encoder_type = cfg["encoder"] # e.g., "MLP", "Embedding"
        params       = cfg.get("params", {}).copy()

        # Build encoder from registry
        builder = get_encoder_builder(encoder_type)
        encoder = builder(params).to(device)
        encoders[name] = encoder

        # Use config out_dim directly
        try:
            encoders_out_dims[name] = params["out_dim"]
        except KeyError:
            raise KeyError(f"Config missing required parameter: 'params.out_dim' in {name}")
        
    # Final combined embedding wrapper
    context_encoder = ConditionEmbeddingWrapper(
        encoders=encoders,
        encoders_out_dims=encoders_out_dims,
        out_dim=wrapper_out_dim
    ).to(device)

    return context_encoder

# ------------------------------------------------------------------ #
# Registry

EncoderBuilder = Callable[[Dict[str, Any]], nn.Module]

ENCODER_REGISTRY: Dict[str, EncoderBuilder] = {}

def register_encoder(name: str):
    name = name.lower()
    def decorator(fn: EncoderBuilder):
        ENCODER_REGISTRY[name] = fn
        return fn
    return decorator

def get_encoder_builder(name: str) -> EncoderBuilder:
    name = name.lower()
    if name not in ENCODER_REGISTRY:
        raise KeyError(f"Unknown encoder_type '{name}'. Registered: {list(ENCODER_REGISTRY.keys())}")
    return ENCODER_REGISTRY[name]

# ------------------------------------------------------------------ #
# Built-in encoders add more below

@register_encoder("embedding")
def build_embedding(params: Dict[str, Any]) -> nn.Module:
    num_classes = params["num_classes"]
    out_dim = params["out_dim"]
    return nn.Embedding(num_classes, out_dim)

@register_encoder("mlp")
def build_mlp(params: Dict[str, Any]) -> nn.Module:
    in_dim = params["in_dim"]
    hidden_dim = params["hidden_dim"]
    out_dim = params["out_dim"]
    return MLP(in_dim, hidden_dim, out_dim)

@register_encoder("identity")
def build_identity(params: Dict[str, Any]) -> nn.Module:
    return nn.Identity() # expects input already in desired out_dim

# @register_encoder("clip_image")
# def build_clip_image(params: Dict[str, Any]) -> nn.Module:
#     out_dim = params["out_dim"]
#     pretrained = params.get("pretrained")
#     return CLIPImageEncoder(out_dim=out_dim, pretrained=pretrained)
