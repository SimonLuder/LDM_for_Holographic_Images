import clip
import torch
import torch.nn as nn


class MLP(nn.Module):
    
    def __init__(self, input_dim:int, hidden_dim: int, out_dim:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.net(x)
        return emb
    

class CLIPImageEmbedding(nn.Module):
    """
    CLIP based image embedding. The network consists of a pre-trained CLIP model and a fully connected layer block. 
    The output of the CLIP model is passed through the fully connected layers to generate the final embedding.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Defines the computation performed at every call.
        get_embedding_dim() -> int: Returns the embedding dimension of the CLIP model.
    """
    
    def __init__(self, out_dim:int, model_name:str="ViT-B/32", device:str="cpu"):
        super().__init__()
        
        self.device = device
        self.clip_encoder, _ = clip.load(name=model_name, device=device)
        self.embedding_dim = self.get_embedding_dim()
        self.fully_connected = nn.Sequential(
            nn.Linear(self.embedding_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # we only want to train the fully connected layers without clip
        with torch.no_grad():
            x = self.clip_encoder.encode_image(x).type(torch.float32)
        emb = self.fully_connected(x)
        return emb
    
    def get_embedding_dim(self):
        """
        Returns the embedding dimension of the CLIP model.

        Returns:
        int: The embedding dimension.
        """
        # Create a dummy input tensor
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Pass the dummy input through the clip_encoder to get the embedding size
        with torch.no_grad():
            embedding_dim = self.clip_encoder.encode_image(dummy_input).size(1)
            return embedding_dim