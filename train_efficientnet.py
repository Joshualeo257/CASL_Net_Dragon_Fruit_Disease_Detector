"""
CASL-Net Step 6a: EfficientNet Backbone Training
==================================================
This script:
  1. Crops lesion regions from training images using YOLO labels
  2. Trains EfficientNet-B3 to classify cropped lesions into 3 disease classes
  3. Saves the best model weights

Why EfficientNet on top of YOLOv8?
  - YOLOv8 detects lesion locations (bounding boxes)
  - EfficientNet classifies what disease each cropped lesion is
  - The 3 diseases are visually similar — EfficientNet's deeper feature
    extraction reduces misclassification between them
  - This is the "Confusion-Aware" core of CASL-Net
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_IMG_DIR = r"C:\Data\MSRIT\8th SEM project\CASL-Net\dataset\images\train"
TRAIN_LBL_DIR = r"C:\Data\MSRIT\8th SEM project\CASL-Net\dataset\labels\train"
VAL_IMG_DIR   = r"C:\Data\MSRIT\8th SEM project\CASL-Net\dataset\images\val"
VAL_LBL_DIR   = r"C:\Data\MSRIT\8th SEM project\CASL-Net\dataset\labels\val"
OUTPUT_DIR    = r"C:\Data\MSRIT\8th SEM project\CASL-Net\models\efficientnet"

CROP_SIZE     = 128       # EfficientNet input size for cropped lesions
BATCH_SIZE    = 32
EPOCHS        = 30
LR            = 1e-4
NUM_CLASSES   = 3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED   = 42
PADDING       = 10        # extra pixels around bbox crop

CLASS_NAMES   = {0: "Anthracnose", 1: "StemRot", 2: "Canker"}

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# HELPER: Crop lesion from image using YOLO bbox
# ─────────────────────────────────────────────
def crop_lesion(image, cx, cy, bw, bh, padding=PADDING):
    """
    Converts normalized YOLO bbox to pixel coords and crops the lesion.
    Adds padding around the crop for context.
    """
    h, w  = image.shape[:2]
    x1    = int((cx - bw / 2) * w) - padding
    y1    = int((cy - bh / 2) * h) - padding
    x2    = int((cx + bw / 2) * w) + padding
    y2    = int((cy + bh / 2) * h) + padding
    x1    = max(0, x1); y1 = max(0, y1)
    x2    = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return image[y1:y2, x1:x2]


# ─────────────────────────────────────────────
# DATASET: LesionCropDataset
# Reads images + YOLO labels, crops each lesion,
# returns (crop_tensor, class_id)
# ─────────────────────────────────────────────
class LesionCropDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.samples   = []
        self.transform = transform

        img_dir = Path(img_dir)
        lbl_dir = Path(lbl_dir)

        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if not lbl_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with open(lbl_path) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(float(parts[0]))
                    cx, cy, bw, bh = [float(x) for x in parts[1:]]

                    crop = crop_lesion(image, cx, cy, bw, bh)
                    if crop is None:
                        continue

                    # Resize to CROP_SIZE x CROP_SIZE
                    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                    self.samples.append((crop, class_id))

        print(f"  📦 Loaded {len(self.samples)} lesion crops from {img_dir.name}")
        class_counts = {0: 0, 1: 0, 2: 0}
        for _, c in self.samples:
            class_counts[c] = class_counts.get(c, 0) + 1
        for cid, cnt in class_counts.items():
            print(f"     [{cid}] {CLASS_NAMES[cid]}: {cnt} crops")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop, class_id = self.samples[idx]
        if self.transform:
            crop = self.transform(crop)
        return crop, class_id


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# MODEL: EfficientNet-B3 via timm
# ─────────────────────────────────────────────
def build_model(num_classes=NUM_CLASSES):
    """
    Loads pretrained EfficientNet-B3 and replaces the
    classifier head with a 3-class output layer.
    """
    model = timm.create_model(
        'efficientnet_b3',
        pretrained   = True,
        num_classes  = num_classes,
    )
    return model


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for crops, labels in loader:
        crops  = crops.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(crops)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * crops.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += crops.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# VALIDATION LOOP
# ─────────────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for crops, labels in loader:
            crops   = crops.to(device)
            labels  = labels.to(device)
            outputs = model(crops)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * crops.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += crops.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─────────────────────────────────────────────
# SAVE TRAINING CURVES
# ─────────────────────────────────────────────
def save_training_curves(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'],   label='Val Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'],   label='Val Acc')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(path)
    plt.close()
    print(f"  📈 Training curves saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n🖥️  Device: {DEVICE}")

    # ── Datasets ──
    print("\n📂 Loading datasets...")
    train_dataset = LesionCropDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, train_transform)
    val_dataset   = LesionCropDataset(VAL_IMG_DIR,   VAL_LBL_DIR,   val_transform)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)

    # ── Model ──
    print("\n📦 Building EfficientNet-B3...")
    model     = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    print("✅ EfficientNet-B3 ready")

    # ── Training ──
    print(f"\n🚀 Training for {EPOCHS} epochs...\n")
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [],
               'train_acc':  [], 'val_acc':  []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, DEVICE)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path    = os.path.join(OUTPUT_DIR, "efficientnet_best.pth")
            torch.save(model.state_dict(), best_path)
            saved = "  ✅ saved"

        print(f"  Epoch [{epoch:02d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{saved}")

    # ── Final Report ──
    print(f"\n{'='*57}")
    print(f"🎉 TRAINING COMPLETE")
    print(f"   Best Val Accuracy : {best_val_acc:.4f}")
    print(f"   Weights saved     : {best_path}")

    print(f"\n📋 Classification Report (best epoch):")
    print(classification_report(
        val_labels, val_preds,
        target_names=list(CLASS_NAMES.values())
    ))

    save_training_curves(history, OUTPUT_DIR)
    print(f"{'='*57}")
    print("\n✅ Next: Run inference pipeline (Step 7)")


if __name__ == "__main__":
    main()