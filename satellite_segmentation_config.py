"""Shared configuration for satellite/drone semantic segmentation."""
from pathlib import Path

DATA_ROOT = Path("data")
TRAIN_IMAGE_DIR = DATA_ROOT / "train/images"
TRAIN_MASK_DIR = DATA_ROOT / "train/masks"
VAL_IMAGE_DIR = DATA_ROOT / "val/images"
VAL_MASK_DIR = DATA_ROOT / "val/masks"
OUTPUT_DIR = Path("outputs")
MODEL_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"
IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 6e-5
IGNORE_INDEX = 255
NUM_WORKERS = 2
CLASS_NAMES = [
    "background", "building", "road", "water", "barren", "forest",
    "agricultural", "fire", "smoke", "other", "occluded/unknown",
    "industrial/factory", "flood",
]
NUM_CLASSES = len(CLASS_NAMES)
# RGB colors used for visualization; indexed by class ID.
CLASS_COLORS = [
    (245,245,245), (255,82,82), (255,190,0), (33,150,243), (161,136,127),
    (76,175,80), (139,195,74), (255,30,0), (180,180,180), (156,39,176),
    (55,55,55), (96,125,139), (121,85,52),
]
