"""Confusion matrix, per-class IoU, mean IoU, and pixel accuracy."""
import torch
from satellite_segmentation_config import NUM_CLASSES, IGNORE_INDEX

def update_confusion(confusion, labels, predictions):
    valid = labels != IGNORE_INDEX
    ids = labels[valid].long() * NUM_CLASSES + predictions[valid].long()
    confusion += torch.bincount(ids.cpu(), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    return confusion

def scores(confusion):
    intersection = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
    iou = intersection / union.clamp_min(1)
    accuracy = intersection.sum() / confusion.sum().clamp_min(1)
    return iou, float(accuracy), float(iou.mean())
