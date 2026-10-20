"""Pair RGB images with single-channel class-ID PNG masks."""
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from satellite_segmentation_config import CLASS_NAMES, IGNORE_INDEX, IMAGE_SIZE

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def list_pairs(image_dir, mask_dir):
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected image and mask folders: {image_dir} and {mask_dir}")
    masks = {p.stem:p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower()==".png"}
    pairs = [(p,masks[p.stem]) for p in sorted(image_dir.iterdir()) if p.suffix.lower() in EXTENSIONS and p.stem in masks]
    if not pairs:
        raise ValueError(f"No pairs found. Image/mask stems must match: {image_dir} and {mask_dir}")
    return pairs

class SatelliteDataset(Dataset):
    def __init__(self, pairs, train=False):
        self.pairs = pairs
        aug = [A.Resize(IMAGE_SIZE, IMAGE_SIZE)]
        if train:
            aug += [A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5), A.RandomRotate90(p=.5), A.ColorJitter(p=.3)]
        aug += [A.Normalize(mean=(.485,.456,.406), std=(.229,.224,.225)), ToTensorV2()]
        self.transform = A.Compose(aug)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        image = np.array(Image.open(ip).convert("RGB"))
        mask = np.array(Image.open(mp))
        if mask.ndim != 2: raise ValueError(f"Mask must be single-channel class IDs: {mp}")
        valid = np.isin(mask, list(range(len(CLASS_NAMES))) + [IGNORE_INDEX])
        if not valid.all(): raise ValueError(f"Unexpected label IDs in {mp}: {np.unique(mask[~valid])[:20]}")
        data = self.transform(image=image, mask=mask.astype(np.int64))
        return {"pixel_values":data["image"], "labels":data["mask"].long()}
