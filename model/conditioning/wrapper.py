import torch
from torch import nn
from typing import Optional


class ConditionEmbeddingWrapper(nn.Module):
    """
    A modular conditional embedding block that supports arbitrary encoders.

    Each encoder should be a callable (e.g., nn.Module) that takes an input tensor
    and returns an embedding tensor. For example:
        encoders = {
            "class": nn.Embedding(num_classes, 128),
            "tabular": nn.Linear(tab_in_dim, 64),
            "image": CLIPImageEmbedding(out_dim=256, device="cuda"),
            "angle": SomeAngleEncoder(...)
        }
    """

    def __init__(self, encoders: dict[str, nn.Module], out_dim: int, encoders_out_dims: Optional[dict[str, int]] = None):
        """
        Embedding wrapper class to combine multiple embeddings. For ddp save training give the encoders_out_dims 
        argument or manualy run self.build_fc(...) before starting the distributed training !

        Args:
            encoders (dict[str, nn.Module]): Dict of encoders
            out_dim (int): Final output dimension of this encoder wrapper
            encoders_out_dims (dict[str, int] | None, optional): Output dimensions of the individual encoders. 
                Require same keys as the encoders argument. Defaults to None. e.g. {"class": 64, "angle": 4}

        Example:
            encoders = {
                "class": SimpleEncoder(in_dim=10, out_dim=64),
                "angle": nn.Identity()
            }
            encoders_out_dims = {
                "class": 64,
                "angle": 8,
            }
            embedding = ConditionEmbeddingWrapper(encoders=encoders, out_dim=512, encoders_out_dims=encoders_out_dims)
        """
        super().__init__()
        self.encoders = nn.ModuleDict(encoders) 
        self.out_dim = out_dim
        if encoders_out_dims is None:
            self.fully_connected = None  # initialize at first forward pass
        else:
            self.build_fc(encoders_out_dims)


    def build_fc(self, encoders_out_dims: dict):
        """Builds the final projection layer once encoder output dims are known."""
        in_dim = sum(encoders_out_dims.values())
        device = next(self.parameters()).device

        self.fully_connected = nn.Sequential(
            nn.Linear(in_dim, self.out_dim),
        ).to(device)


    def forward(self, inputs: dict[str, torch.Tensor]):
        """
        inputs: a dict matching keys of encoders, e.g.:
            {
                "class": class_idx_tensor,
                "angle": angle_tensor,
                ...
            }
        """
        embeddings = {}
        for key, encoder in self.encoders.items():
            if key not in inputs or inputs[key] is None:
                continue
            x = inputs[key]
            emb = encoder(x)
            embeddings[key] = emb

        # dynamically build FC layer if needed
        if self.fully_connected is None:
            with torch.no_grad():
                dims = {k: v.shape[-1] for k, v in embeddings.items()}
            self.build_fc(dims)

        # concatenate embeddings
        x = torch.cat(list(embeddings.values()), dim=-1)
        return self.fully_connected(x)