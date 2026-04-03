# Grayscale Image Colorization using Generative Adversarial Networks

Bachelor's Thesis — Faculty of Automation, Computers and Electronics, University of Craiova, 2026

---

## Overview

This project implements an automatic grayscale image colorization system based on a Generative Adversarial Network (GAN). The generator uses a ResNet-18 encoder with U-Net skip connections, and the discriminator is a PatchGAN. The model operates in the LAB color space, taking the L (lightness) channel as input and predicting the a and b (color) channels.

The training dataset is a 13,000-image subset of Microsoft COCO. The system introduces two novel loss functions — **Spatial Color Affinity Loss** and **No Grey Loss** — alongside standard GAN, L1, Total Variation, and Contrast losses.

---

## Repository Structure

```
grayscale-to-color-unet-gan/
├── notebooks/          # Colab notebooks (experiments, training runs)
├── scripts/
│   └── prepare_dataset.py      # Dataset cleaning and subset creation
├── src/
│   ├── data/
│   │   └── dataset.py          # ColorizationDataset and make_dataloaders
│   ├── models/                 # Generator and discriminator definitions
│   ├── losses/                 # Loss function implementations
│   ├── training/               # Training loop and utilities
│   └── utils/                  # Visualization and metrics
├── results/
│   └── sample_outputs/         # Colorization output samples
└── paper/
    ├── thesis.tex              # LaTeX source
    └── references.bib          # Bibliography
```

---

## Setup

All experiments run on **Google Colab** with Google Drive as storage. There is no local installation required.

**1. Mount Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**2. Clone the repository into Colab**

```python
!git clone https://github.com/your-username/grayscale-to-color-unet-gan.git
%cd grayscale-to-color-unet-gan
```

**3. Install dependencies**

```python
!pip install torch torchvision scikit-image tqdm Pillow numpy
```

---

## Dataset Preparation

The dataset preparation is done once and consists of three steps. Edit the paths at the top of the script before running.

```python
# scripts/prepare_dataset.py — configure these paths first
DRIVE_DATASET_PATH      = "/content/drive/MyDrive/datasets/coco"
SUBSET_DATASET_PATH     = "/content/drive/MyDrive/datasets/coco_subset_16000"
NUM_IMAGES_TO_EXTRACT   = 16000
```

Run the script from Colab:

```python
!python scripts/prepare_dataset.py
```

Or import individual steps if you have already completed some:

```python
from scripts.prepare_dataset import remove_grayscale_images, create_subset, count_images

remove_grayscale_images(DRIVE_DATASET_PATH)   # Step 1 — remove grayscale images
create_subset(...)                             # Step 2 — create a 16k subset
count_images(SUBSET_DATASET_PATH)             # Step 3 — verify final count
```

After running, manually inspect a sample of images in the subset to catch any near-grayscale images the automated filter may have missed.

---

## Training

Open the training notebook in `notebooks/` and run the cells in order. The notebook documents each experiment with observations and intermediate results.

The main configurable parameters:

| Parameter     | Default | Notes                              |
|---------------|---------|-------------------------------------|
| `batch_size`  | 16      | Reduce to 8 if OOM on T4           |
| `n_epochs`    | 20      | Pretraining: 20, full GAN: +20     |
| `lr`          | 2e-4    | Adam, β₁=0.5, β₂=0.999            |
| `SIZE`        | 256     | Input resolution                    |

---

## Results

Sample colorization outputs are saved in `results/sample_outputs/`. Quantitative results (PSNR, SSIM, FID) and the ablation study are reported in `paper/thesis.pdf`.
