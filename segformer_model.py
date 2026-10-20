"""SegFormer model construction."""
from transformers import SegformerForSemanticSegmentation
from satellite_segmentation_config import CLASS_NAMES, MODEL_CHECKPOINT, NUM_CLASSES

def create_model(checkpoint=MODEL_CHECKPOINT, pretrained=True):
    if pretrained:
        return SegformerForSemanticSegmentation.from_pretrained(
            checkpoint, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True,
            id2label=dict(enumerate(CLASS_NAMES)), label2id={n:i for i,n in enumerate(CLASS_NAMES)})
    from transformers import SegformerConfig
    config = SegformerConfig(num_labels=NUM_CLASSES, id2label=dict(enumerate(CLASS_NAMES)),
                             label2id={n:i for i,n in enumerate(CLASS_NAMES)})
    return SegformerForSemanticSegmentation(config)
