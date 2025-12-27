import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.train_test_utils import get_image_encoder_names


class NVLDMTrainer:
    def __init__(
        self,
        config,
        model,
        diffusion,
        dataloader,
        optimizer,
        device,
        vae=None,
        discriminator=None,
        optimizer_d=None,
        lpips_model=None,
        criterion=None,
        d_criterion=None,
        wandb_run=None,
        dataloader_val=None,
        ):
        
        self.model = model
        self.diffusion = diffusion
        self.vae = vae
        self.discriminator = discriminator

        self.optimizer = optimizer
        self.optimizer_d = optimizer_d

        self.optimizer.zero_grad()
        if self.optimizer_d is not None:
            self.optimizer_d.zero_grad() 

        self.lpips_model = lpips_model
        self.criterion = criterion or torch.nn.MSELoss()
        self.d_criterion = d_criterion

        self.device = device
        self.dataloader = dataloader
        self.dataloader_val = dataloader_val
        self.wandb_run = wandb_run

        self.step_count = 0
        self.start_epoch = 0

        self.init_from_config(config)

    def init_from_config(self, config):
        """
        Populate all config-dependent trainer attributes.
        Must be called once before training or resume.
        """

        self.config = config
        self.run_name = config["name"]
        self.train_cfg = config["ldm_train"]
        self.condition_cfg = config["conditioning"]

        if self.condition_cfg["enabled"] == "unconditional":
            self.use_condition = []
            self.used_image_encoders = []
        else:
            self.use_condition = self.condition_cfg["enabled"].split("+")
            self.used_image_encoders = get_image_encoder_names(self.condition_cfg)

        self.train_with_perceptual_loss = (self.train_cfg.get("ldm_perceptual_weight", 0) > 0)

        self.train_with_discriminator = (
            self.train_cfg.get("ldm_discriminator_weight", 0) > 0
            and self.train_cfg.get("ldm_discriminator_start_step", 0) > 0
        )

        self.discriminator_start_step = self.train_cfg.get(
            "ldm_discriminator_start_step", 0
        )

        lat_dir = self.train_cfg.get("vqvae_latents_representations", None)
        if lat_dir and os.path.exists(lat_dir):
            self.latents_available = len(os.listdir(lat_dir)) > 0
        else:
            self.latents_available = False

        self.validate_fn = getattr(self, "validate_fn", None)

        # Sanity checks
        if self.train_with_discriminator:
            assert self.optimizer_d is not None, "optimizer_d must be provided"
            assert self.d_criterion is not None, "d_criterion must be provided"

        if self.train_with_perceptual_loss:
            assert self.lpips_model is not None, "lpips_model must be provided"

        # Checkpoints
        self.resume_from_ckpt = self.train_cfg.get("resume_from_ckpt", None)
        self.ckpt_exists = os.path.exists(self.resume_from_ckpt)
        if self.resume_from_ckpt is not None and self.ckpt_exists:
            self.load_checkpoint(self.resume_from_ckpt)

        print("Trainer setup from config complete")

    # Checkpointing ---------------------------------------------------

    def save_checkpoint(self, path, epoch):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": self.step_count,
        }

        if self.discriminator:
            ckpt["discriminator"] = self.discriminator.state_dict()
            ckpt["optimizer_d"] = self.optimizer_d.state_dict()

        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)  
        torch.save(ckpt, path)


    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        self.step_count = ckpt["step"]
        self.start_epoch = ckpt["epoch"] + 1

        if self.discriminator and "discriminator" in ckpt:
            self.discriminator.load_state_dict(ckpt["discriminator"])
            self.optimizer_d.load_state_dict(ckpt["optimizer_d"])

        print(f"Resumed from step {self.step_count}, epoch {self.start_epoch}")

    # Training steps ---------------------------------------------------
 
    def prepare_batch(self, batch):
        (im1, im2), (cond1, cond2), _ = batch

        im1 = im1.float().to(self.device)
        im2 = im2.float().to(self.device)

        if self.condition_cfg["enabled"] != "unconditional":

            for cond in self.use_condition:

                if cond in self.used_image_encoders:
                    cond1[cond] = im1
                    cond2[cond] = im2

                cond1[cond] = cond1[cond].to(self.device)
                cond2[cond] = cond2[cond].to(self.device)

            if np.random.random() < self.train_cfg["ldm_cfg_discard_prob"]:
                cond1 = None
                cond2 = None
        else:
            cond1 = None
            cond2 = None

        return (im1, im2), (cond1, cond2)


    def diffusion_forward(self, im1, use_discriminator=False):

        # autencode samples
        if not self.latents_available:
            with torch.no_grad():
                im1, _ = self.vae.encode(im1)

        # sample timestep
        t = self.diffusion.sample_timesteps(im1.shape[0]).to(self.device)

        # noise image
        if use_discriminator:
            return self.diffusion.noise_images(im1, t, x_t_neg_1=True) + (t,)
        else:
            x_t, noise = self.diffusion.noise_images(im1, t, x_t_neg_1=False)
            return x_t, noise, None, t
        
    # Losses ---------------------------------------------------

    def compute_rec_loss(self, noise, noise_pred):
        # reconstruction loss
        loss = self.criterion(noise_pred, noise)
        return loss
    

    def compute_lpips_loss(self, im, im_pred):

        lp_in = torch.clamp(im, -1., 1.)
        lp_in_pred = torch.clamp(im_pred, -1., 1.)

        if lp_in.shape[1] == 1:
            lp_in = lp_in.repeat(1,3,1,1)
            lp_in_pred = lp_in_pred.repeat(1,3,1,1)

        if lp_in.shape[1] != 3:
            lp_loss = 0
            for i in range(lp_in.shape[1]):
                lp_slice = lp_in[:, i, :, :].unsqueeze(1).repeat(1,3,1,1)
                lp_pred_slice = lp_in_pred[:, i, :, :].unsqueeze(1).repeat(1,3,1,1)
                lp_loss += torch.mean(self.lpips_model(lp_pred_slice, lp_slice))
            lp_loss = lp_loss / lp_in.shape[1]
        else:
            lp_loss = torch.mean(self.lpips_model(lp_in_pred, lp_in))

        return lp_loss


    def compute_discriminator_loss(self, x_t, x_t_neg_1, noise_pred, t):
        x_t_neg_1_pred = self.diffusion.denoising_step(x_t, t, noise_pred)

        fake_pred_g = self.discriminator(x_t_neg_1_pred, t)
        fake_pred_d = self.discriminator(x_t_neg_1_pred.detach(), t)
        real_pred_d = self.discriminator(x_t_neg_1, t)

        g_fake_loss = self.d_criterion(fake_pred_g, torch.ones_like(fake_pred_g))
        d_fake_loss = self.d_criterion(fake_pred_d, torch.zeros_like(fake_pred_d))
        d_real_loss = self.d_criterion(real_pred_d, torch.ones_like(real_pred_d))

        g_loss = g_fake_loss
        d_loss =  (d_fake_loss + d_real_loss) / 2

        return g_loss, d_loss
    
    # Forward pass ---------------------------------------------------

    def forward_step(self, batch):
        
        (im1, im2), (cond1, cond2) = self.prepare_batch(batch)

        use_disc = (
            self.train_with_discriminator
            and self.step_count >= self.discriminator_start_step
        )

        x_t, noise, x_t_neg_1, t = self.diffusion_forward(im1, use_disc)

        noise_pred = self.model(x_t, t, cond2)

        # reconstruction loss
        rec_loss = self.compute_rec_loss(noise, noise_pred)
        loss = rec_loss

        # lpips loss
        lpips_loss = None
        if self.train_with_perceptual_loss:
            # remove all predicted noise
            im_pred = x_t - noise_pred
            lpips_loss = self.compute_lpips_loss(im1, im_pred)
            loss += self.train_cfg["ldm_perceptual_weight"] * lpips_loss
            
        # discriminator loss
        g_disc_loss, d_disc_loss = None, None
        if use_disc:
            g_disc_loss, d_disc_loss = self.compute_discriminator_loss(
                x_t, x_t_neg_1, noise_pred, t
            )
            loss += self.train_cfg["ldm_discriminator_weight"] * g_disc_loss

        return loss, {
            "g_loss": loss,
            "rec_loss": rec_loss,
            "lpips_loss": lpips_loss,
            "g_disc_loss": g_disc_loss,
            "d_disc_loss": d_disc_loss,
        }

    # Training loop ---------------------------------------------------

    def train(self):
        steps_per_opt = self.train_cfg["ldm_steps_per_optimization"]
        num_epochs = self.train_cfg["ldm_epochs"]

        for epoch in range(self.start_epoch, num_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(pbar):

                self.step_count += 1

                # Forward (generator)
                loss, aux = self.forward_step(batch)
                loss_scaled = loss / steps_per_opt
                loss_scaled.backward()

                # Generator optimizer step
                if (
                        self.step_count % steps_per_opt == 0
                        or batch_idx == len(self.dataloader) - 1
                    ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Discriminator optimizer step
                if aux.get("d_disc_loss") is not None:
                    d_loss = aux["d_disc_loss"] / steps_per_opt
                    d_loss.backward()

                    if (
                            self.step_count % steps_per_opt == 0
                            or batch_idx == len(self.dataloader) - 1
                        ):
                        self.optimizer_d.step()
                        self.optimizer_d.zero_grad()

                # Logging
                if self.wandb_run:
                    log_dict = {
                        "g_loss": aux["g_loss"].item(),
                        "rec_loss": aux["rec_loss"].item(),
                        "step": self.step_count,
                        "epoch": epoch,
                    }

                    if aux.get("lpips_loss") is not None:
                        log_dict["lpips_loss"] = aux["lpips_loss"].item()

                    if aux.get("g_disc_loss") is not None:
                        log_dict["g_disc_loss"] = aux["g_disc_loss"].item()
                        log_dict["d_disc_loss"] = aux["d_disc_loss"].item()

                    self.wandb_run.log(log_dict, step=self.step_count)

                # Validation
                if (
                    (self.step_count % self.train_cfg["ldm_val_steps"] == 0)
                    and (self.step_count >= self.train_cfg["ldm_val_start"])
                    ):

                    logs_val = self.validate()
                    if self.wandb_run and logs_val is not None:
                        self.wandb_run.log(logs_val, step=self.step_count)


                # Checkpointing
                if self.step_count % self.train_cfg["ldm_ckpt_steps"] == 0:
                    ckpt_path = os.path.join(self.train_cfg['ckpt_folder'], self.run_name, "ckpts")
                    self.save_checkpoint(path= os.path.join(ckpt_path, "latest.ckpt"), epoch=epoch)
                    self.save_checkpoint(path= os.path.join(ckpt_path, f"{self.step_count}.ckpt"), epoch=epoch)

                pbar.set_postfix(Loss=loss.item())
                
    # Validation loop ---------------------------------------------------

    def validate(self):

        if self.dataloader_val is None:
            return None

        self.model.eval()
        if self.vae is not None:
            self.vae.eval()

        reconstruction_losses = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader_val):
                (im1, _), (_, cond2) = self.prepare_batch(batch)


                x_t, noise, x_t_neg_1, t = self.diffusion_forward(im1)

                # predict noise
                noise_pred = self.model(x_t, t, cond2)

                # reconstruction loss
                rec_loss = self.criterion(noise_pred, noise)
                reconstruction_losses.append(rec_loss.item())

        self.model.train()
        if self.vae is not None:
            self.vae.train()

        return {
            "val_epoch_reconstructon_loss": float(np.mean(reconstruction_losses))
        }
    
    def sample_batch(self, batch, cfg_scale=3, to_uint8=False):

        self.model.eval()
        if self.vae is not None:
            self.vae.eval()

        with torch.no_grad():
            (gt_im1, _), (_, cond2) = self.prepare_batch(batch)

            sampled_latents = self.diffusion.sample(
                self.model, 
                condition=cond2, 
                n=batch.size()[0], 
                cfg_scale=cfg_scale,
                to_uint8=to_uint8
            )

            # upsample with vqvae
            sampled_imgs = self.vae.decode(sampled_latents)

        return sampled_imgs, gt_im1