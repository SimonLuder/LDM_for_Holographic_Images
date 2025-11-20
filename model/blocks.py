import torch
import torch.nn as nn
# modified from https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main

class DownBlock(nn.Module):
    """Down convolutional block with attention of sequence:
        1. Resnet block with time embedding
        2. Attention block
        3. Down sampling

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, 
                 num_heads, num_layers, attn, norm_channels, cross_attn=None, context_dim=None):
        
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.t_emb_dim = t_emb_dim
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        
        # first resnet layer
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(num_groups=norm_channels, num_channels=in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, 
                              out_channels, kernel_size=3, stride=1, padding=1)
                ) 
                for i in range(num_layers)
            ]
        )

        # time embedding
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
                ]
            )

        # second resnet layer
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        # multihead self attention
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            
            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
        
        # residual connection
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        # cross attention
        if self.cross_attn:
            assert self.context_dim is not None, "Context dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        # downsampling layer
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 
                                          kernel_size=4, stride=2, padding=1) if self.down_sample else nn.Identity()
        

    def forward(self, x, t_emb=None, context=None):

        for i in range(self.num_layers):

            # residual block
            resnet_input = x
            x = self.resnet_conv_first[i](x)
            if self.t_emb_dim is not None:
                x = x + self.t_emb_layers[i](t_emb)[:, :, None, None]
            x = self.resnet_conv_second[i](x)
            x = x + self.residual_input_conv[i](resnet_input)

            # self attention
            if self.attn:
                B, C, H, W = x.shape
                in_attn = x.reshape(B, C, H * W)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2)
                out_attn = out_attn.reshape(B, C, H, W)
                x = x + out_attn

            # cross-attention
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                B, C, H, W = x.shape
                in_attn = x.reshape(B, C, H * W)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                print("in attn shape", in_attn.shape)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                print("context", context.shape)
                context_proj = self.context_proj[i](context)
                print("context_proj", context_proj.shape)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(B, C, H, W)
                x = x + out_attn  

        # Downsample
        x = self.down_sample_conv(x)
        return x
        

class MidBlock(nn.Module):

    def __init__(self, in_channels, out_channels, t_emb_dim, 
                 num_heads, num_layers, norm_channels, cross_attn=None, context_dim=None):
        
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.cross_attn = cross_attn
        self.context_dim = context_dim

        # first resnet layer
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )

        # time embedding
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers + 1)
            ])

        # second resnet layer
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )

        # residual connection
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

        # multihead self attention
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)]
        )

        # cross attention
        if self.cross_attn:
            assert self.context_dim is not None, "Context dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )


    def forward(self, x, t_emb=None, context=None):
        
        # first residual block
        resnet_input = x
        x = self.resnet_conv_first[0](x)
        if self.t_emb_dim is not None:
            x = x + self.t_emb_layers[0](t_emb)[:, :, None, None]
        x = self.resnet_conv_second[0](x)
        x = x + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):

            # self attention
            B, C, H, W = x.shape
            in_attn = x.reshape(B, C, H * W)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2)
            out_attn = out_attn.reshape(B, C, H, W)
            x = x + out_attn

            # residual bock
            resnet_input = x
            x = self.resnet_conv_first[i + 1](x)
            if self.t_emb_dim is not None:
                x = x + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            x = self.resnet_conv_second[i + 1](x)
            x = x + self.residual_input_conv[i + 1](resnet_input)

            # cross-attention
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                B, C, H, W = x.shape
                in_attn = x.reshape(B, C, H * W)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(B, C, H, W)
                x = x + out_attn
        
        return x
    

class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, 
                 num_heads, num_layers, attn, norm_channels, skip_connection=False, cross_attn=None, context_dim=None):
        
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.cross_attn = cross_attn
        self.context_dim = context_dim
   
        # first resnet layer
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, 
                                 (in_channels + out_channels) if i == 0 and skip_connection else (in_channels if i == 0 else out_channels)),
                    nn.SiLU(),
                    nn.Conv2d((in_channels + out_channels) if i == 0 and skip_connection else (in_channels if i == 0 else out_channels), 
                              out_channels, kernel_size=3, stride=1, padding=1),
                )
                for i in range(num_layers)
            ]
        )

        
        # time embedding
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        
        # second resnet layer
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        # residual connection
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d((in_channels + out_channels) if i == 0 and skip_connection else (in_channels if i == 0 else out_channels), 
                          out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        # multihead self attention
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )

        # cross attention
        if self.cross_attn:
            assert self.context_dim is not None, "Context dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        # upsample convolution
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, 
                                                 kernel_size=4, stride=2, padding=1) \
            if self.up_sample else nn.Identity()
        

    def forward(self, x, out_down=None, t_emb=None, context=None):

        # upsample
        x = self.up_sample_conv(x)

        # concat with DownBlock output
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)

        for i in range(self.num_layers):

            # residual block
            resnet_input = x
            x = self.resnet_conv_first[i](x)
            if self.t_emb_dim is not None:
                x = x + self.t_emb_layers[i](t_emb)[:, :, None, None]
            x = self.resnet_conv_second[i](x)
            x = x + self.residual_input_conv[i](resnet_input)

            # self attention
            if self.attn:
                B, C, H, W = x.shape
                in_attn = x.reshape(B, C, H * W)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2)
                out_attn = out_attn.reshape(B, C, H, W)
                x = x + out_attn

            # cross-attention
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                B, C, H, W = x.shape
                in_attn = x.reshape(B, C, H * W)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                print("in attn shape", in_attn.shape)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                print("context", context.shape)
                context_proj = self.context_proj[i](context)
                print("context_proj", context_proj.shape)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(B, C, H, W)
                x = x + out_attn            

        return x