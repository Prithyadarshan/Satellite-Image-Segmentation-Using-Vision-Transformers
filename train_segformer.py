"""Fine-tune SegFormer on paired satellite/drone images and class-ID masks."""
import torch
from torch.utils.data import DataLoader
from satellite_segmentation_config import *
from dataset_loader import list_pairs, SatelliteDataset
from segformer_model import create_model
from evaluate_segmentation import evaluate

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = SatelliteDataset(list_pairs(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR), train=True)
    val = SatelliteDataset(list_pairs(VAL_IMAGE_DIR, VAL_MASK_DIR), train=False)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=.01)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best = -1
    for epoch in range(EPOCHS):
        model.train(); total = 0
        for batch in train_loader:
            pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(pixel_values=pixels, labels=labels).loss
            loss.backward(); optimizer.step(); total += loss.item()
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} loss={total/max(1,len(train_loader)):.4f} mIoU={metrics['mean_iou']:.4f} acc={metrics['pixel_accuracy']:.4f}")
        if metrics["mean_iou"] > best:
            best = metrics["mean_iou"]
            model.save_pretrained(OUTPUT_DIR / "best_segformer")
            (OUTPUT_DIR / "class_names.json").write_text(__import__('json').dumps(CLASS_NAMES, indent=2))
    print("Best checkpoint:", OUTPUT_DIR / "best_segformer")

if __name__ == "__main__": main()
