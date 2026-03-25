import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.train_test_utils import get_image_encoder_names
from utils.wandb import wandb_make_batch_grid
from utils import regionprops


class NVFlowTrainer:
    """
    Trainer for Rectified Flow (RF), tuple-based.

    Tuple contract everywhere:
        ((im1, im2), (cond1, cond2))
    """

    def __init__(
        self,
        config,
        model,
        flow,                    # Rectified Flow object
        dataloader,
        optimizer,
        device,
        vae=None,
        lpips_model=None,
        criterion=None,
        wandb_run=None,
        dataloader_val=None,
    ):
        self.model = model
        self.flow = flow
        self.vae = vae

        self.optimizer = optimizer
        self.optimizer.zero_grad()

        self.lpips_model = lpips_model
        self.criterion = criterion or torch.nn.MSELoss()

        self.device = device
        self.dataloader = dataloader
        self.dataloader_val = dataloader_val
        self.wandb_run = wandb_run

        self.step_count = 0
        self.start_epoch = 0

        self.init_from_config(config)

    # ------------------------------------------------------------------
    # Config & setup
    # ------------------------------------------------------------------

    def init_from_config(self, config):
        self.config = config
        self.run_name = config["name"]
        self.train_cfg = config["ldm_train"]
        self.condition_cfg = config["conditioning"]

        self.ckpt_root = self.train_cfg["ckpt_folder"]
        self.run_dir = os.path.join(self.ckpt_root, self.run_name)
        self.ckpt_dir = os.path.join(self.run_dir, "ckpts")
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)

        # Conditioning
        if self.condition_cfg["enabled"] == "unconditional":
            self.use_condition = []
            self.used_image_encoders = []
        else:
            self.use_condition = self.condition_cfg["enabled"].split("+")
            self.used_image_encoders = get_image_encoder_names(self.condition_cfg)

        # LPIPS only used in validation
        self.train_with_perceptual_loss = False
        assert self.lpips_model is not None, "lpips_model must be provided for validation"

        # Latents
        lat_dir = self.train_cfg.get("vqvae_latents_representations", None)
        self.latents_available = (
            lat_dir and os.path.exists(lat_dir) and len(os.listdir(lat_dir)) > 0
        )

        # Resume
        self.resume_training = self.train_cfg.get("resume_training", False)
        self.resume_ckpt = self.train_cfg.get("resume_ckpt", None)

        if self.resume_training:
            if self.resume_ckpt is None or not os.path.exists(self.resume_ckpt):
                raise FileNotFoundError(f"Resume checkpoint not found: {self.resume_ckpt}")
            self.load_checkpoint(self.resume_ckpt)
        else:
            print("Starting training from scratch")

        # Fixed validation samples
        self.num_val_lpips = self.train_cfg.get("ldm_val_lpips_samples", 8)
        if self.dataloader_val is not None:
            val_len = len(self.dataloader_val.dataset)
            self.val_lpips_indices = np.linspace(
                0, val_len - 1, self.num_val_lpips, dtype=int
            ).tolist()
        else:
            self.val_lpips_indices = []

        self.val_lpips_noise = torch.randn(
            self.num_val_lpips,
            self.flow.img_channels,
            self.flow.img_size,
            self.flow.img_size,
            device=self.device,
        )

        print("Rectified Flow trainer setup complete")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path, epoch):
        tmp_path = path + ".tmp"
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": self.step_count,
        }
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt["step"]
        self.start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from step {self.step_count}, epoch {self.start_epoch}")

    # ------------------------------------------------------------------
    # Batch preparation (tuple-preserving)
    # ------------------------------------------------------------------

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
        else:
            cond1 = None
            cond2 = None

        return (im1, im2), (cond1, cond2)

    # ------------------------------------------------------------------
    # Rectified Flow forward
    # ------------------------------------------------------------------

    def flow_forward(self, im1):
        if not self.latents_available:
            with torch.no_grad():
                im1, _ = self.vae.encode(im1)

        t = self.flow.sample_times(im1.shape[0])   # t in [0,1]
        x_t, v_target, x0 = self.flow.make_xt_and_target(im1, t)
        return x_t, v_target, t, x0

    # ------------------------------------------------------------------
    # Forward step
    # ------------------------------------------------------------------

    def forward_step(self, batch):
        (im1, _), (_, cond2) = self.prepare_batch(batch)

        x_t, v_target, t, _ = self.flow_forward(im1)
        v_pred = self.model(x_t, t, cond2)

        loss = self.criterion(v_pred, v_target)

        return loss, {
            "g_loss": loss,
            "rec_loss": loss,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        steps_per_opt = self.train_cfg["ldm_steps_per_optimization"]
        num_epochs = self.train_cfg["ldm_epochs"]

        for epoch in range(self.start_epoch, num_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(pbar):
                self.step_count += 1

                loss, aux = self.forward_step(batch)
                (loss / steps_per_opt).backward()

                if (
                    self.step_count % steps_per_opt == 0
                    or batch_idx == len(self.dataloader) - 1
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.wandb_run:
                    self.wandb_run.log(
                        {
                            "loss": aux["g_loss"].item(),
                            "step": self.step_count,
                            "epoch": epoch,
                        },
                        step=self.step_count,
                    )

                if (
                    self.dataloader_val is not None
                    and self.step_count % self.train_cfg["ldm_val_steps"] == 0
                    and self.step_count >= self.train_cfg["ldm_val_start"]
                ):
                    logs = self.validate_samples()
                    if self.wandb_run and logs is not None:
                        self.wandb_run.log(logs, step=self.step_count)

                if self.step_count % self.train_cfg["ldm_ckpt_steps"] == 0:
                    self.save_checkpoint(
                        os.path.join(self.ckpt_dir, "latest.ckpt"),
                        epoch,
                    )

                pbar.set_postfix(loss=loss.item())

    # ------------------------------------------------------------------
    # Validation helpers (tuple-preserving)
    # ------------------------------------------------------------------

    def get_fixed_val_samples(self, indices):
        samples = [self.dataloader_val.dataset[i] for i in indices]
        collate_fn = self.dataloader_val.collate_fn
        batch = collate_fn(samples)
        return self.prepare_batch(batch)

    def validate_samples(self):
        if self.dataloader_val is None or len(self.val_lpips_indices) == 0:
            return None

        self.model.eval()
        self.vae.eval()
        self.lpips_model.eval()

        with torch.no_grad():
            (gt_im1, _), (_, cond2) = self.get_fixed_val_samples(self.val_lpips_indices)

            sampled_latents = self.flow.sample(
                model=self.model,
                condition=cond2,
                n=gt_im1.size(0),
                x_init=self.val_lpips_noise,
                steps=self.train_cfg.get("ldm_sampling_steps", 50),
                to_uint8=False,
            )

            sampled_imgs = self.vae.decode(sampled_latents)

            # Metrics
            gt_rps = regionprops.regionprops_from_torch_batch(gt_im1)
            gen_rps = regionprops.regionprops_from_torch_batch(sampled_imgs)
            rp_error = regionprops.abs_prop_rps_error(gt_rps, gen_rps)

            lpips_loss = self.compute_lpips_loss(gt_im1, sampled_imgs)
            mse_loss = ((sampled_imgs - gt_im1) ** 2).mean()

            if self.wandb_run:
                grid_image = wandb_make_batch_grid(gt_im1, sampled_imgs, self.step_count)
                logs = rp_error.copy()
                logs.update(
                    {
                        "val_mse": mse_loss.item(),
                        "val_lpips": lpips_loss.item(),
                        "val_lpips_batch": grid_image,
                    }
                )
                self.wandb_run.log(logs, step=self.step_count)

        self.model.train()
        self.vae.train()

        return {
            "val_lpips": lpips_loss.item(),
            "val_mse": mse_loss.item(),
        }

    # ------------------------------------------------------------------
    # LPIPS
    # ------------------------------------------------------------------

    def compute_lpips_loss(self, im_gt, im_pred):
        im_gt = torch.clamp(im_gt, -1, 1)
        im_pred = torch.clamp(im_pred, -1, 1)

        if im_gt.shape[1] == 1:
            im_gt = im_gt.repeat(1, 3, 1, 1)
            im_pred = im_pred.repeat(1, 3, 1, 1)

        if im_gt.shape[1] != 3:
            lp = 0.0
            for i in range(im_gt.shape[1]):
                gt_i = im_gt[:, i:i+1].repeat(1, 3, 1, 1)
                pr_i = im_pred[:, i:i+1].repeat(1, 3, 1, 1)
                lp += torch.mean(self.lpips_model(pr_i, gt_i))
            return lp / im_gt.shape[1]

        return torch.mean(self.lpips_model(im_pred, im_gt))
