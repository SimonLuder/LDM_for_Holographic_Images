import torch
import torch.nn as nn
from model.blocks import DownBlock, MidBlock, UpBlock

class UNet(nn.Module):
    
    def __init__(self, img_channels, model_config, context_encoder=None):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample'] 
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
        self.cond_emb_dim = model_config['cond_emb_dim']
        self.cond_insert_type = model_config['cond_insert_type']

        self.cross_attn = self.cond_insert_type == "crossattention"

        # null context vector for cross-attention
        if self.cross_attn: 
            self.null_context = nn.Parameter(torch.randn(1, 1, self.cond_emb_dim) * 0.02)

        self.context_encoder = context_encoder

        # Validating Unet Model configurations
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_sample)

        # input layer
        self.conv_in = nn.Conv2d(img_channels, self.down_channels[0], kernel_size=3, padding=1)

        # time embedding projection layers
        self.t_proj = nn.Sequential(nn.Linear(self.t_emb_dim, self.t_emb_dim),
                                    nn.SiLU(),
                                    nn.Linear(self.t_emb_dim, self.t_emb_dim))

        if self.cond_insert_type == "additive":
            pass

        elif self.cond_insert_type == "concatenation":
            self.proj_concat_layer = nn.Linear(self.t_emb_dim + self.cond_emb_dim, self.t_emb_dim)

        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], 
                                        self.down_channels[i + 1], 
                                        self.t_emb_dim,
                                        down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        attn=self.attns[i], 
                                        norm_channels=self.norm_channels,
                                        cross_attn=self.cross_attn,
                                        context_dim=self.t_emb_dim))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], 
                                      self.mid_channels[i + 1], 
                                      self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels,
                                      cross_attn=self.cross_attn,
                                      context_dim=self.t_emb_dim))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlock(self.down_channels[i + 1],
                                    self.down_channels[i],
                                    self.t_emb_dim, 
                                    up_sample=self.down_sample[i],
                                    num_heads=self.num_heads,
                                    num_layers=self.num_up_layers,
                                    attn=self.attns[i],
                                    norm_channels=self.norm_channels,
                                    skip_connection=True,
                                    cross_attn=self.cross_attn,
                                    context_dim=self.t_emb_dim))

        self.conv_ups_out = nn.Conv2d(self.down_channels[0], self.conv_out_channels, kernel_size=3, padding=1)
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, img_channels, kernel_size=3, padding=1)


    def forward(self, x, t, cond=None):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # [B x C x H x W]
        x = self.conv_in(x)
        # [B x C1 x H x W]

        # time embedding
        t_emb = get_position_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb) 

        if cond is not None:
            # context embedding
            cond_emb = self.context_encoder(cond)

            if self.cond_insert_type == "additive":
                t_emb += cond_emb

            if self.cond_insert_type == "concatenation":
                t_emb = torch.cat([t_emb, cond_emb], dim=1)
                t_emb = self.proj_concat_layer(t_emb)

            if self.cond_insert_type == "crossattention":
                if cond_emb.dim() == 2:
                    cond_emb = cond_emb.unsqueeze(1)

        else:
            # pass learned unconditional vector
            if self.cross_attn:
                cond_emb = self.null_context.expand(x.size(0), -1, -1)
            else:
                # if not using cross-attention, no need for cond_emb
                cond_emb = None

        down_outs = []
        for down in self.downs:
            down_outs.append(x)
            x = down(x, t_emb, cond_emb)
        # down_outs  [[B x C1 x H x W], [B x C2 x H/2 x W/2], [B x C3 x H/4 x W/4]]
        # out [B x C4 x H/4 x W/4]

        for mid in self.mids:
            x = mid(x, t_emb, cond_emb)
        # out [B x C3 x H/4 x W/4]

        for up in self.ups:
            down_out = down_outs.pop()
            x = up(x, down_out, t_emb, cond_emb)
            # out [[B x C2 x H/4 x W/4], [B x C1 x H/2 x W/2], [B x 16 x H x W]]

        x = self.conv_ups_out(x)
        x = self.norm_out(x)
        x = nn.SiLU()(x)
        x = self.conv_out(x)

        # out [B x C x H x W]
        return x
    
    
def get_position_embedding(time_steps, temb_dim):
    """
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "positional embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2)))
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat(tensors=[torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb