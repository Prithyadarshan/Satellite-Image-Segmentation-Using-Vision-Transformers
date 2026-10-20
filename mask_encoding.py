"""Mask ID validation and RGB-to-class-ID conversion helpers."""
import numpy as np
from satellite_segmentation_config import CLASS_NAMES, CLASS_COLORS, IGNORE_INDEX

def validate_class_id_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2: raise ValueError("Expected a 2-D single-channel class-ID mask")
    allowed = np.r_[np.arange(len(CLASS_NAMES)), IGNORE_INDEX]
    invalid = np.setdiff1d(np.unique(mask), allowed)
    if invalid.size: raise ValueError(f"Unknown class IDs: {invalid.tolist()}")
    return mask.astype(np.uint8)

def rgb_mask_to_ids(rgb_mask, color_to_id=None):
    """Convert exact RGB legend colors to IDs; unmatched pixels become IGNORE_INDEX."""
    rgb = np.asarray(rgb_mask)
    if rgb.ndim != 3 or rgb.shape[2] < 3: raise ValueError("Expected an RGB mask")
    palette = color_to_id or {tuple(color):i for i,color in enumerate(CLASS_COLORS)}
    out = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
    for color, class_id in palette.items(): out[np.all(rgb[:,:,:3] == color, axis=2)] = class_id
    return out
