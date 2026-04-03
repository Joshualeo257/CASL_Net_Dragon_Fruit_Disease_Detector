"""
CASL-Net Step 4: SAM Segmentation
===================================
This script:
  1. Loads a dragon fruit stem image
  2. Runs YOLOv8 to detect disease lesions (bounding boxes)
  3. Feeds each bounding box into SAM as a prompt
  4. SAM generates a precise pixel mask for each lesion
  5. Calculates:
       - Individual lesion area (pixels)
       - Total infection area per disease class
       - Stem area (non-background pixels)
       - Infection percentage = infection area / stem area * 100
  6. Saves visualizations with masks overlaid

This output feeds directly into Step 5 (Severity Estimation).
"""

import cv2
import torch
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

# SAM imports
from segment_anything import sam_model_registry, SamPredictor

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH   = r"C:\Data\MSRIT\8th SEM project\CASL-Net\models\yolov8\best.pt"
SAM_CKPT     = r"C:\Data\MSRIT\8th SEM project\CASL-Net\sam\sam_vit_h_4b8939.pth"
SAM_TYPE     = "vit_h"
TEST_IMG_DIR = r"C:\Data\MSRIT\8th SEM project\CASL-Net\dataset\images\val"
OUTPUT_DIR   = r"C:\Data\MSRIT\8th SEM project\CASL-Net\outputs\sam_results"
NUM_IMAGES   = 5        # how many images to test on
CONF_THRESH  = 0.35     # raised from 0.25 to reduce over-detection
IOU_THRESH   = 0.5

CLASS_NAMES  = {0: "Anthracnose", 1: "StemRot", 2: "Canker"}

# Overlay colors per class (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 165, 255),    # Orange  — Anthracnose
    1: (0, 0, 255),      # Red     — StemRot
    2: (255, 0, 0),      # Blue    — Canker
}

# ─────────────────────────────────────────────
# HELPER: Estimate stem area
# We approximate stem area as the largest
# contiguous green/dark region in the image
# using simple HSV thresholding.
# ─────────────────────────────────────────────
def estimate_stem_area(image_bgr):
    """
    Estimates the pixel area of the dragon fruit stem
    by masking out the background (sky/white areas).
    Returns stem area in pixels.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Mask for non-background (not white/light sky)
    # Background is typically bright white or light blue sky
    lower_bg = np.array([0,   0,   200])   # bright white
    upper_bg = np.array([180, 30,  255])
    bg_mask  = cv2.inRange(hsv, lower_bg, upper_bg)

    # Also mask out very bright light blue sky
    lower_sky = np.array([90,  10, 180])
    upper_sky = np.array([130, 80, 255])
    sky_mask  = cv2.inRange(hsv, lower_sky, upper_sky)

    # Combine background masks
    combined_bg   = cv2.bitwise_or(bg_mask, sky_mask)
    stem_mask     = cv2.bitwise_not(combined_bg)

    # Clean up with morphology
    kernel    = np.ones((5, 5), np.uint8)
    stem_mask = cv2.morphologyEx(stem_mask, cv2.MORPH_CLOSE, kernel)
    stem_mask = cv2.morphologyEx(stem_mask, cv2.MORPH_OPEN,  kernel)

    stem_area = int(np.sum(stem_mask > 0))

    # Fallback: if stem detection fails, use 60% of total image area
    total_area = image_bgr.shape[0] * image_bgr.shape[1]
    if stem_area < total_area * 0.1:
        stem_area = int(total_area * 0.60)

    return stem_area, stem_mask


# ─────────────────────────────────────────────
# HELPER: Draw colored mask overlay on image
# ─────────────────────────────────────────────
def overlay_mask(image, mask, color, alpha=0.45):
    """Overlay a binary mask on image with given color and transparency."""
    overlay       = image.copy()
    colored_mask  = np.zeros_like(image)
    colored_mask[mask > 0] = color
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    # Draw mask contour
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


# ─────────────────────────────────────────────
# MAIN SEGMENTATION FUNCTION
# ─────────────────────────────────────────────
def segment_image(image_path, yolo_model, sam_predictor):
    """
    Full pipeline for one image:
      1. YOLOv8 detection
      2. SAM segmentation per detection
      3. Area calculations
    Returns result dict with all metrics.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"  ❌ Cannot read: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w      = image_bgr.shape[:2]

    # ── Step 1: YOLOv8 Detection ──
    results  = yolo_model.predict(
        source  = str(image_path),
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        device  = 'cpu',
        verbose = False,
    )
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print(f"  ⚠️  No detections in {image_path.name}")
        return None

    # ── Step 2: Estimate Stem Area ──
    stem_area, stem_mask = estimate_stem_area(image_bgr)

    # ── Step 3: SAM Setup ──
    sam_predictor.set_image(image_rgb)

    # Track results per class
    class_areas = {0: 0, 1: 0, 2: 0}   # total pixel area per disease
    detections  = []
    vis_image   = image_bgr.copy()

    # ── Step 4: SAM Segmentation per bounding box ──
    for box in boxes:
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

        # Clamp to image bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        # Skip degenerate boxes
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        # Feed bounding box as prompt to SAM
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = sam_predictor.predict(
            point_coords = None,
            point_labels = None,
            box          = input_box[None, :],  # SAM expects (1, 4)
            multimask_output = False,
        )

        # Best mask (highest SAM confidence)
        best_mask  = masks[0].astype(np.uint8)   # shape: (H, W)
        mask_area  = int(np.sum(best_mask > 0))

        class_areas[class_id] += mask_area

        detections.append({
            'class_id'  : class_id,
            'class_name': CLASS_NAMES[class_id],
            'confidence': confidence,
            'bbox'      : (x1, y1, x2, y2),
            'mask_area' : mask_area,
        })

        # Visualize: overlay mask + draw bbox + label
        color     = CLASS_COLORS[class_id]
        vis_image = overlay_mask(vis_image, best_mask, color)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
        cv2.putText(vis_image, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ── Step 5: Infection Percentage per Disease ──
    infection_pct = {}
    for cid, area in class_areas.items():
        pct = (area / stem_area * 100) if stem_area > 0 else 0.0
        infection_pct[CLASS_NAMES[cid]] = round(pct, 4)

    total_infection_area = sum(class_areas.values())
    total_infection_pct  = round((total_infection_area / stem_area * 100)
                                  if stem_area > 0 else 0.0, 4)

    return {
        'image_name'         : image_path.name,
        'image_size'         : (w, h),
        'stem_area_px'       : stem_area,
        'detections'         : detections,
        'class_areas_px'     : {CLASS_NAMES[k]: v for k, v in class_areas.items()},
        'infection_pct'      : infection_pct,
        'total_infection_pct': total_infection_pct,
        'vis_image'          : vis_image,
    }


# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────
def print_results(result):
    print(f"\n{'='*55}")
    print(f"📸 Image   : {result['image_name']}")
    print(f"📐 Size    : {result['image_size'][0]}x{result['image_size'][1]} px")
    print(f"🌿 Stem area: {result['stem_area_px']:,} px")
    print(f"\n  Detections ({len(result['detections'])} total):")
    for d in result['detections']:
        print(f"    [{d['class_name']:12s}] conf={d['confidence']:.2f} "
              f"| mask area={d['mask_area']:,} px "
              f"| bbox={d['bbox']}")
    print(f"\n  Infection Area per Disease:")
    for cls_name, area_px in result['class_areas_px'].items():
        pct = result['infection_pct'][cls_name]
        print(f"    {cls_name:12s}: {area_px:,} px  →  {pct:.2f}% of stem")
    print(f"\n  Total Infection : {result['total_infection_pct']:.2f}% of stem")
    print(f"{'='*55}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load YOLOv8
    print("\n📦 Loading YOLOv8...")
    yolo_model = YOLO(MODEL_PATH)
    print("✅ YOLOv8 loaded")

    # Load SAM
    print("\n📦 Loading SAM (this may take a moment)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    print("✅ SAM loaded")

    # Get test images
    val_images = list(Path(TEST_IMG_DIR).glob("*.jpg"))[:NUM_IMAGES]
    print(f"\n🔍 Running segmentation on {len(val_images)} images...\n")

    all_results = []
    for img_path in val_images:
        result = segment_image(img_path, yolo_model, sam_predictor)
        if result is None:
            continue

        print_results(result)

        # Save visualization
        out_path = os.path.join(OUTPUT_DIR, f"sam_{result['image_name']}")
        cv2.imwrite(out_path, result['vis_image'])
        print(f"  💾 Saved: {out_path}")

        all_results.append(result)

    # Final summary
    print(f"\n{'='*55}")
    print(f"🎉 SAM SEGMENTATION COMPLETE")
    print(f"   Processed : {len(all_results)} images")
    print(f"   Output dir: {OUTPUT_DIR}")
    print(f"{'='*55}")
    print("\n✅ Next step: Severity Estimation (Step 5)")

    return all_results


if __name__ == "__main__":
    main()