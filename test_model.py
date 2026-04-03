"""
Quick test to verify best.pt works correctly.
Run this before moving to SAM segmentation.
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH   = r"C:\Data\MSRIT\8th SEM project\CASL-Net\models\yolov8\best.pt"
TEST_IMG_DIR = r"C:\Data\MSRIT\8th SEM project\CASL-Net\dataset\images\val"
OUTPUT_DIR   = r"C:\Data\MSRIT\8th SEM project\CASL-Net\outputs\test_results"
NUM_IMAGES   = 5  # how many val images to test on

CLASS_NAMES  = {0: "Anthracnose", 1: "StemRot", 2: "Canker"}

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("\n📦 Loading model...")
model = YOLO(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

# ─────────────────────────────────────────────
# PICK A FEW VALIDATION IMAGES
# ─────────────────────────────────────────────
val_images = list(Path(TEST_IMG_DIR).glob("*.jpg"))[:NUM_IMAGES]
print(f"\n🖼️  Testing on {len(val_images)} validation images...\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────────
for img_path in val_images:
    results = model.predict(
        source  = str(img_path),
        conf    = 0.25,
        iou     = 0.5,
        device  = 'cpu',
        verbose = False,
    )

    result = results[0]
    boxes  = result.boxes

    print(f"📸 {img_path.name}")

    if boxes is None or len(boxes) == 0:
        print("   ⚠️  No detections\n")
        continue

    for box in boxes:
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        class_name = CLASS_NAMES.get(class_id, "Unknown")
        print(f"   ✅ {class_name:15s} | Confidence: {confidence:.2f} | Box: ({x1},{y1}) → ({x2},{y2})")

    # Save annotated image
    annotated = result.plot()
    out_path  = os.path.join(OUTPUT_DIR, f"detected_{img_path.name}")
    cv2.imwrite(out_path, annotated)
    print(f"   💾 Saved to: {out_path}\n")

print("="*50)
print("✅ TEST COMPLETE")
print(f"   Annotated images saved to: {OUTPUT_DIR}")
print("="*50)