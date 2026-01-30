import torch
import numpy as np
from tqdm import tqdm


class Diffusion:
    """
    Class for Diffusion process: noise schedule, forward process, and reverse process.
    """

    def __init__(self, img_size=256, img_channels=3, noise_schedule="linear", noise_steps=1000, 
                 beta_start=1e-4, beta_end=0.02, s=0.008, device="cpu"):
        
        self.img_size = img_size
        self.img_channels = img_channels
        self.noise_schedule = noise_schedule
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.s = s
        self.device = device

        self.prepare_noise_schedule()
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)


    def prepare_noise_schedule(self):
        
        if self.noise_schedule == "linear":
            self.linear_noise_schedule()

        elif self.noise_schedule == "cosine":
            self.cosine_noise_schedule()


    def linear_noise_schedule(self):
        # create linear noise schedule
        self.beta =  torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        # Formula: α = 1 - β
        self.alpha = 1. - self.beta
        # The cumulative product of α.
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)


    def cosine_noise_schedule(self):
        # create cosine noise schedule
        steps = self.noise_steps + 1
        s = self.s
        t = torch.linspace(0, 1, steps)
        alpha_hat = torch.cos(((t + s) / (1 + s)) * (np.pi / 2)).pow(2)
        self.alpha_hat = alpha_hat / alpha_hat[0]
        # caluculate β
        self.beta = 1 - alpha_hat[1:] / alpha_hat[:-1]
        # Formula: α = 1 - β
        self.alpha = 1. - self.beta


    def noise_images(self, x, t, x_t_neg_1=False):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        
        if x_t_neg_1 is True:
            sqrt_alpha_hat_neg_1 = torch.sqrt(self.alpha_hat[t - 1])[:, None, None, None]
            sqrt_one_minus_alpha_hat_neg_1 = torch.sqrt(1 - self.alpha_hat[t - 1])[:, None, None, None]
            return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ, \
                sqrt_alpha_hat_neg_1 * x + sqrt_one_minus_alpha_hat_neg_1 * Ɛ
        
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ


    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    

    def denoising_step(self, x, t, predicted_noise):
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]

        t = t.view(-1, 1, 1, 1)
        
        if (t > 1).any():  # if any t > 1
            noise = torch.where(t > 1, torch.randn_like(x), torch.zeros_like(x))
        else:
            noise = torch.zeros_like(x)
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x


    def sample(self, model, condition, n=None, x_init=None, cfg_scale=3, to_uint8=True):
        if n is None:
            n = len(condition)
        model.eval()
        with torch.no_grad():

            if x_init is None:
                x = torch.randn((n, self.img_channels, self.img_size, self.img_size), device=self.device)
            else:
                x = x_init.clone()

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, condition)

                if cfg_scale > 0:
                    predicted_noise_unconditional = model(x, t)
                    predicted_noise = (1 + cfg_scale) * predicted_noise - cfg_scale * predicted_noise_unconditional # https://arxiv.org/pdf/2207.12598.pdf eq. 6
 
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        model.train()

        if to_uint8:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)

        return x

