import torch
import torch.nn as nn
from model.unet_v2 import get_position_embedding

class PatchGanDiscriminator(nn.Module):

    """PatchGan discriminator
    """

    def __init__(self, img_channels, conv_channels=[64, 128, 256], 
                 kernels=[4,4,4,4], strides=[2,2,2,2], paddings=[1,1,1,1]
                 ):
        super().__init__()
        self.im_channels = img_channels
        activation = nn.LeakyReLU(0.2)
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i !=0 else True),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DiffusionPatchGanDiscriminator(nn.Module):

    """PatchGan discriminator with sinusoidal timestep embedding
    """

    def __init__(self, img_channels, conv_channels=[64, 128, 256], t_emb_dim=512,
                 kernels=[4,4,4,4], strides=[2,2,2,2], paddings=[1,1,1,1],
                 
                 ):
        super().__init__()
        self.im_channels = img_channels
        self.t_emb_dim = t_emb_dim
        activation = nn.LeakyReLU(0.2)
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=False if i !=0 else True),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])

        # time embedding
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, layers_dim[i])
                )
                for i in range(len(layers_dim) - 1)
                ]
            )

    def forward(self, x, t=None):

        for i, layer in enumerate(self.layers):
            
            # time embedding
            if self.t_emb_dim is not None and t is not None:
                
                t_emb = get_position_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
                x = x + self.t_emb_layers[i](t_emb)[:, :, None, None]

            # layer block
            x = layer(x)

        return x

if __name__ == '__main__':
    x = torch.randn((2,3, 256, 256))
    prob = PatchGanDiscriminator(im_channels=3)(x)
    print(prob.shape)