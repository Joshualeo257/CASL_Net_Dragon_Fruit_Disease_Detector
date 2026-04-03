"""
Test script to verify efficientnet_best.pth works correctly.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import timm
from pathlib import Path
from torchvision import transforms

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH   = r"C:\Data\MSRIT\8th SEM project\CASL-Net\models\efficientnet\efficientnet_best.pth"
VAL_IMG_DIR  = r"C:\Data\MSRIT\8th SEM project\CASL-Net\dataset\images\val"
VAL_LBL_DIR  = r"C:\Data\MSRIT\8th SEM project\CASL-Net\dataset\labels\val"
NUM_IMAGES   = 10
CROP_SIZE    = 128
PADDING      = 10
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES  = {0: "Anthracnose", 1: "StemRot", 2: "Canker"}

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("\n📦 Loading EfficientNet-B3...")
model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"✅ Model loaded from: {MODEL_PATH}")
print(f"   Device: {DEVICE}")

# ─────────────────────────────────────────────
# HELPER: Crop lesion from bbox
# ─────────────────────────────────────────────
def crop_lesion(image, cx, cy, bw, bh):
    h, w = image.shape[:2]
    x1   = int((cx - bw / 2) * w) - PADDING
    y1   = int((cy - bh / 2) * h) - PADDING
    x2   = int((cx + bw / 2) * w) + PADDING
    y2   = int((cy + bh / 2) * h) + PADDING
    x1   = max(0, x1); y1 = max(0, y1)
    x2   = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return image[y1:y2, x1:x2]

# ─────────────────────────────────────────────
# HELPER: Predict class for a single crop
# ─────────────────────────────────────────────
def predict_crop(crop_bgr):
    crop_rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (CROP_SIZE, CROP_SIZE))
    tensor    = transform(crop_resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)[0]
        pred_class  = probs.argmax().item()
        confidence  = probs[pred_class].item()
    return pred_class, confidence, probs.cpu().numpy()

# ─────────────────────────────────────────────
# MAIN TEST
# ─────────────────────────────────────────────
print(f"\n🔍 Testing on {NUM_IMAGES} validation images...\n")

val_images = list(Path(VAL_IMG_DIR).glob("*.jpg"))[:NUM_IMAGES]

correct = 0
total   = 0

for img_path in val_images:
    lbl_path = Path(VAL_LBL_DIR) / img_path.with_suffix(".txt").name
    if not lbl_path.exists():
        continue

    image = cv2.imread(str(img_path))
    if image is None:
        continue

    print(f"📸 {img_path.name}")

    with open(lbl_path) as f:
        lines = f.readlines()

    if not lines:
        print("   ⚠️  No labels\n")
        continue

    for line in lines:
        parts    = line.strip().split()
        if len(parts) != 5:
            continue
        true_class = int(float(parts[0]))
        cx, cy, bw, bh = [float(x) for x in parts[1:]]

        crop = crop_lesion(image, cx, cy, bw, bh)
        if crop is None:
            continue

        pred_class, confidence, probs = predict_crop(crop)

        match   = "✅" if pred_class == true_class else "❌"
        correct += 1 if pred_class == true_class else 0
        total   += 1

        print(f"   {match} True: {CLASS_NAMES[true_class]:12s} | "
              f"Pred: {CLASS_NAMES[pred_class]:12s} | "
              f"Conf: {confidence:.4f}")
        print(f"      Probs → "
              f"Anthracnose: {probs[0]:.3f} | "
              f"StemRot: {probs[1]:.3f} | "
              f"Canker: {probs[2]:.3f}")

    print()

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
accuracy = correct / total if total > 0 else 0
print("=" * 55)
print(f"✅ TEST COMPLETE")
print(f"   Total lesions tested : {total}")
print(f"   Correct predictions  : {correct}")
print(f"   Accuracy             : {accuracy:.4f} ({accuracy*100:.1f}%)")
print("=" * 55)