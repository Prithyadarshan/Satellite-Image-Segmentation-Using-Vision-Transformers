"""Save or reload SegFormer checkpoints with the project class mapping."""
import json
from pathlib import Path
from transformers import SegformerForSemanticSegmentation
from satellite_segmentation_config import CLASS_NAMES, OUTPUT_DIR

def save_model(model, directory=OUTPUT_DIR/"best_segformer"):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(directory)
    (directory/"class_names.json").write_text(json.dumps(CLASS_NAMES,indent=2))
    return directory

def load_model(directory=OUTPUT_DIR/"best_segformer", device="cpu"):
    return SegformerForSemanticSegmentation.from_pretrained(directory).to(device)
