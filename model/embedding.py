import torch
from torch import nn
import clip


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


class ConditionEmbedding(nn.Module):

    def __init__(self, 
                 out_dim:int, 
                 num_classes:int=None, 
                 cls_emb_dim:int=None, 
                 tabular_in_dim:int=None,
                 tabular_out_dim:int=None,
                 img_in_channels:int=None,
                 img_out_dim:int=None):
        
        super(ConditionEmbedding, self).__init__()

        self.out_dim = out_dim
        self.use_cls_cond = False
        self.use_tbl_cond = False
        self.use_img_cond = False

        if num_classes:
            self.use_cls_cond= True
            self.class_emb = nn.Embedding(num_classes, cls_emb_dim)  # class embeddings
        
        if tabular_in_dim:
            self.use_tbl_cond = True
            self.tabular_emb = nn.Linear(in_features=tabular_in_dim, out_features=tabular_out_dim) # tabular embedding

        if img_in_channels:
            self.use_img_cond = True
            # TODO
            
            self.image_emb = CLIPImageEmbedding(out_dim=img_out_dim, device="cuda")

        in_dim = sum(dim for dim in (cls_emb_dim, tabular_out_dim, img_out_dim) if dim is not None)
        self.fully_connected = nn.Sequential(nn.Linear(in_dim, out_dim), 
                                             nn.SiLU(), 
                                             nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):

        x_cls = x.get("class", None)
        x_tbl = x.get("tabular", None)
        x_img = x.get("image", None)
                
        if x_cls is not None and self.use_cls_cond:
            x_cls_emb = self.class_emb(x_cls.long())
        else:
            x_cls_emb = None

        if x_tbl is not None and self.use_tbl_cond:
            x_tbl_emb = self.tabular_emb(x_tbl.float())
        else:
            x_tbl_emb = None

        if x_img is not None and self.use_img_cond:
            x_img_emb = self.image_emb(x_img.float().repeat(1,3,1,1)) # TODO
        else:
            x_img_emb = None

        x = torch.cat([emb for emb in (x_cls_emb, x_tbl_emb, x_img_emb) if emb is not None], dim=-1)
        x = self.fully_connected(x)

        return x

