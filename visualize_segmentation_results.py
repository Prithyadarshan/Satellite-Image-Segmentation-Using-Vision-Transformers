"""Colorize class-ID masks and create prediction overlays."""
import numpy as np
from PIL import Image
from satellite_segmentation_config import CLASS_COLORS, CLASS_NAMES

def colorize(mask):
    mask=np.asarray(mask); colors=np.asarray(CLASS_COLORS,dtype=np.uint8)
    rgb=np.zeros((*mask.shape,3),dtype=np.uint8)
    valid=(mask>=0)&(mask<len(colors)); rgb[valid]=colors[mask[valid].astype(int)]
    return rgb

def save_visuals(image, mask, prefix):
    image=np.asarray(image.convert("RGB") if isinstance(image,Image.Image) else image)
    color=colorize(mask)
    overlay=(.55*image.astype(np.float32)+.45*color.astype(np.float32)).clip(0,255).astype(np.uint8)
    Image.fromarray(color).save(f"{prefix}_segmentation.png")
    Image.fromarray(overlay).save(f"{prefix}_overlay.png")
    return color,overlay
