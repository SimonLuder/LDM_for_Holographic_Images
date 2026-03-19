# Latent Diffusion Model for Holographic Images

## Overview

This repository contains code for training a **VQ-VAE** and a **latent diffusion model (LDM)** for holographic image generation. The project is organized around:

- **VQ-VAE** training and inference
- **DDPM / latent diffusion** training, validation, and testing
- **Conditional generation** using multiple conditioning sources
- **Evaluation utilities** for particle- and region-property-based analysis

## Repository Structure

```text
LDM_for_Holographic_Images/
├── config/              # Base and slurm configs for VQ-VAE and LDM
├── evaluation/          # Evaluation helpers
├── model/               # Core model implementations
│   ├── conditioning/
│   ├── blocks.py
│   ├── ddpm.py
│   ├── discriminator.py
│   ├── ldm_trainer.py
│   ├── unet_v2.py
│   └── vqvae.py
├── notebooks/           # Experiment notebooks
├── tools/
│   ├── ldm/             # LDM train/test/validate scripts
│   └── vqvae/           # VQ-VAE scripts
├── utils/               # Config, image, training, and logging utilities
├── requirements.txt
└── README.md



## Setup

### Option 1: Local (pip)

```bash
pip install -r requirements.txt
```

### Option 2: SLURM + Singularity

```bash
cd /path/to/repo
srun -p performance -t 60 --mem=32G --pty bash
singularity build --fakeroot ../ldm_training.sif dockerfile/singularity_container.def
```

For full instructions, see [SLURM Setup](slurm/README.md)