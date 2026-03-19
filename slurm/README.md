# SLURM + Singularity Setup

This guide explains how to build and run the Singularity container on a SLURM-based HPC cluster. 

Note: This repository must already be set up in an environment with access to SLURM.

---

## 1. Navigate to Project Directory

```bash
cd /mnt/.../LDM_for_Holographic_Images
```

---

## 2. Start an Interactive SLURM Session

```bash
srun -p performance -t 60 --mem=32G --pty bash
```

* `-p performance` → partition
* `-t 60` → time (minutes)
* `--mem=32G` → memory allocation

---

## 3. Build the Singularity Container

```bash
cd repo
singularity build --fakeroot ../ldm_training.sif dockerfile/singularity_container.def
```

This creates a `ldm_training.sif` container outside the repository which is used by the setup files in the subfolders.

---
