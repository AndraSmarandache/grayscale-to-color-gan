import numpy as np
from PIL import Image
from skimage.color import rgb2lab

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class ColorizationDataset(Dataset):
    """
    Loads color images and converts them to LAB color space for GAN training.

    The model receives the L (lightness) channel as input and predicts the
    a and b (color) channels. Working in LAB keeps the luminance and color
    information cleanly separated, so the generator only needs to learn color.

    Normalization:
        L  : divided by 50, shifted to [-1, 1]  (original range: [0, 100])
        ab : divided by 110, range ~= [-1, 1]    (original range: [-128, 127])

    Parameters
    ----------
    paths : list[str]
        File paths to the color images.
    split : str
        'train' applies random horizontal flip augmentation; 'val' does not.
    size : int
        Images are resized to (size x size) with BICUBIC interpolation.
    """

    def __init__(self, paths, split='train', size=256):
        self.size = size
        self.paths = paths
        self.split = split

        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((size, size), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),  # simple augmentation to increase effective dataset size
            ])
        elif split == 'val':
            # no augmentation for validation - we want a clean, repeatable evaluation
            self.transforms = transforms.Resize((size, size), Image.BICUBIC)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)

        # Convert RGB to LAB
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)

        # Separate and normalize channels
        L = img_lab[[0], ...] / 50. - 1.   # lightness -> [-1, 1]
        ab = img_lab[[1, 2], ...] / 110.    # color    -> [-1, 1] approx

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)


def make_dataloaders(paths, split, batch_size=16, n_workers=2, pin_memory=True):
    """
    Create a DataLoader for the given image paths.

    Parameters
    ----------
    paths : list[str]
        File paths passed directly to ColorizationDataset.
    split : str
        'train' or 'val' — controls augmentation and shuffling.
    batch_size : int
        Number of images per batch. Larger = faster training but more GPU memory.
        Recommended: 16 for T4, 32 if memory allows.
    n_workers : int
        Parallel data loading workers. Set to 0 if Colab raises
        "DataLoader worker exited unexpectedly".
    pin_memory : bool
        Speeds up CPU to GPU transfer. Automatically disabled when no GPU is available.
    """
    dataset = ColorizationDataset(paths=paths, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory if torch.cuda.is_available() else False,  # pin_memory only helps on GPU
        shuffle=(split == 'train'),                                      # shuffle only during training
        persistent_workers=(n_workers > 0),                             # keep workers alive between epochs
    )
    return dataloader
