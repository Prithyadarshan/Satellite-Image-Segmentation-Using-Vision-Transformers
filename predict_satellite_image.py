"""Load a trained checkpoint and predict a full-size image tile by tile."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
from satellite_segmentation_config import *
from tile_large_satellite_image import iter_tiles

TRANSFORM = A.Compose([A.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225)),ToTensorV2()])
@torch.inference_mode()
def predict_array(image, checkpoint=OUTPUT_DIR/"best_segformer", device=None):
    image=np.asarray(image.convert("RGB") if isinstance(image,Image.Image) else image)
    device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SegformerForSemanticSegmentation.from_pretrained(checkpoint).to(device).eval()
    h,w=image.shape[:2]; best=np.full((h,w),-1.,np.float32); pred=np.zeros((h,w),np.uint8)
    for top,left,th,tw,tile in iter_tiles(image,IMAGE_SIZE,min(64,IMAGE_SIZE//4)):
        x=TRANSFORM(image=tile,mask=np.zeros((IMAGE_SIZE,IMAGE_SIZE),np.uint8))["image"].unsqueeze(0).to(device)
        logits=F.interpolate(model(pixel_values=x).logits,size=(IMAGE_SIZE,IMAGE_SIZE),mode="bilinear",align_corners=False)
        confidence,labels=torch.softmax(logits,1)[0,:,:th,:tw].max(0)
        confidence=confidence.cpu().numpy(); labels=labels.cpu().numpy().astype(np.uint8)
        region=best[top:top+th,left:left+tw]; update=confidence>region
        region[update]=confidence[update]; pred[top:top+th,left:left+tw][update]=labels[update]
    return pred

def predict_file(path, output_path=None):
    path=Path(path); image=Image.open(path).convert("RGB"); mask=predict_array(image)
    output_path=Path(output_path or f"{path.stem}_class_ids.png")
    Image.fromarray(mask).save(output_path)
    return mask, output_path
