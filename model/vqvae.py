import torch
import torch.nn as nn
from model.blocks import DownBlock, MidBlock, UpBlock

# https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main

class VQVAE(nn.Module):

    def __init__(self, img_channels, config):
        super().__init__()
        self.img_channels = img_channels
        self.down_channels = config["down_channels"]
        self.mid_channels = config["mid_channels"]
        self.down_sample = config["down_sample"]
        self.num_down_layers = config["num_down_layers"]
        self.num_mid_layers = config["num_mid_layers"]
        self.num_up_layers = config["num_up_layers"]
        self.norm_channels = config["norm_channels"]

        # disable attention in DownBlock of encoder and UpBlock of Decoder
        self.attns = config["attns"]
        self.num_heads = config["num_heads"]

        # latent dimension
        self.z_channels = config["z_channels"]
        self.codebook_size = config["codebook_size"]

        # mirror the downsampling steps from the encoder when upsampling in the decoder
        self.up_sample = list(reversed(self.down_sample))

        ####################### Encoder #######################

        # input convolution
        self.encoder_conv_in = nn.Conv2d(self.img_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))

        # DownBlock
        self.encoder_downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_downs.append(
                DownBlock(in_channels=self.down_channels[i], 
                          out_channels=self.down_channels[i + 1], 
                          t_emb_dim=None,
                          down_sample=self.down_sample[i], 
                          num_heads=self.num_heads, 
                          num_layers=self.num_down_layers, 
                          attn=self.attns[i], 
                          norm_channels=self.norm_channels
                          )
            )

        # MidBock
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(
                MidBlock(in_channels=self.mid_channels[i], 
                         out_channels=self.mid_channels[i + 1],
                         t_emb_dim=None,
                         num_heads=self.num_heads,
                         num_layers=self.num_mid_layers,
                         norm_channels=self.norm_channels
                         )
            )

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)

        # pre quantization convolution
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)

        # codebook
        self.embedding = nn.Embedding(num_embeddings=self.codebook_size, embedding_dim=self.z_channels)
        #######################################################

        ####################### Decoder #######################

        # post quantization convolution
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))

        # MidBlock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(
                MidBlock(in_channels=self.mid_channels[i], 
                         out_channels=self.mid_channels[i - 1],
                         t_emb_dim=None,
                         num_heads=self.num_heads,
                         num_layers=self.num_mid_layers,
                         norm_channels=self.norm_channels
                         )
            )

        # UpBlock
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(
                 UpBlock(in_channels=self.down_channels[i], 
                         out_channels=self.down_channels[i - 1],
                         t_emb_dim=None, 
                         up_sample=self.down_sample[i - 1],
                         num_heads=self.num_heads,
                         num_layers=self.num_up_layers,
                         attn=self.attns[i-1],
                         norm_channels=self.norm_channels
                         )
            )
           
        # groupnorm and output convolution
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], self.img_channels, kernel_size=3, padding=1)
        #######################################################

    
    def quantize(self, x):
        B, C, H, W = x.shape
        
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        
        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))
        
        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        
        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2) 
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }
        # Straight through estimation (required for backpropagation)
        quant_out = x + (quant_out - x).detach()
        
        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices


    def encode(self, x):
        x = self.encoder_conv_in(x)
        for down in self.encoder_downs:
            x = down(x)
        for mid in self.encoder_mids:
            x = mid(x)
        x = self.encoder_norm_out(x)
        x = nn.SiLU()(x)
        x = self.encoder_conv_out(x)
        x = self.pre_quant_conv(x)
        z, quant_losses, _ = self.quantize(x)
        return z, quant_losses


    def decode(self, z):
        x = self.post_quant_conv(z)
        x = self.decoder_conv_in(x)
        for mid in self.decoder_mids:
            x = mid(x)
        for up in self.decoder_layers:
            x = up(x)
        x = self.decoder_norm_out(x)
        x = nn.SiLU()(x)
        x = self.decoder_conv_out(x)
        return x
    

    def forward(self, x):
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        return out, z, quant_losses
