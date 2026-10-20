"""Evaluate a trained segmentation model on a data loader."""
import torch
import torch.nn.functional as F
from satellite_segmentation_config import NUM_CLASSES
from segmentation_metrics import update_confusion, scores

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    for batch in loader:
        pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
        logits = model(pixel_values=pixels).logits
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        update_confusion(confusion, labels, logits.argmax(1))
    iou, accuracy, miou = scores(confusion)
    return {"per_class_iou":iou.cpu().numpy(), "pixel_accuracy":accuracy, "mean_iou":miou, "confusion_matrix":confusion}
