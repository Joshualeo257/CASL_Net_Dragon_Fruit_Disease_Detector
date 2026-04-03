"""
CASL-Net Step 7: Full Inference Pipeline
==========================================
Complete end-to-end pipeline:
  1. Load image
  2. YOLOv8  -> detect lesion bounding boxes + confidence scores
  3. EfficientNet -> reclassify each cropped lesion (confusion-aware)
  4. SAM -> segment each lesion, compute pixel mask area
  5. Severity Estimation -> infection % + severity index + label
  6. Save annotated output image + CSV report
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import timm
import os
import csv
from pathlib import Path
from torchvision import transforms
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
YOLO_MODEL_PATH = r"C:\Data\MSRIT\8th SEM project\CASL-Net\models\yolov8\best.pt"
EFFICIENTNET_PATH = r"C:\Data\MSRIT\8th SEM project\CASL-Net\models\efficientnet\efficientnet_best.pth"
SAM_CKPT        = r"C:\Data\MSRIT\8th SEM project\CASL-Net\sam\sam_vit_h_4b8939.pth"
SAM_TYPE        = "vit_h"

# Input: folder of images to run inference on
INPUT_DIR       = r"C:\Data\MSRIT\8th SEM project\CASL-Net\dataset\images\val"
OUTPUT_DIR      = r"C:\Data\MSRIT\8th SEM project\CASL-Net\outputs\inference_results"
NUM_IMAGES      = 10

CONF_THRESH     = 0.35
IOU_THRESH      = 0.5
MIN_MASK_AREA   = 300
CROP_SIZE       = 128
PADDING         = 10
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES     = {0: "Anthracnose", 1: "StemRot", 2: "Canker"}

# Disease weights for severity index
WEIGHTS = {"Anthracnose": 1.0, "StemRot": 1.5, "Canker": 1.2}

# Colors per class (BGR)
CLASS_COLORS = {
    0: (0, 165, 255),   # Orange — Anthracnose
    1: (0, 0, 255),     # Red    — StemRot
    2: (255, 0, 0),     # Blue   — Canker
}

SEVERITY_COLORS = {
    "Healthy"           : (0, 255, 0),
    "Moderate"          : (0, 255, 255),
    "Slightly Infected" : (0, 165, 255),
    "Highly Infected"   : (0, 0, 255),
}

# ─────────────────────────────────────────────
# SEVERITY LABEL
# ─────────────────────────────────────────────
def get_severity_label(index):
    if index < 5:   return "Healthy"
    elif index < 10: return "Moderate"
    elif index < 15: return "Slightly Infected"
    else:            return "Highly Infected"

# ─────────────────────────────────────────────
# EFFICIENTNET TRANSFORM
# ─────────────────────────────────────────────
eff_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# HELPER: Crop lesion from bbox
# ─────────────────────────────────────────────
def crop_lesion(image, x1, y1, x2, y2, padding=PADDING):
    h, w  = image.shape[:2]
    x1    = max(0, x1 - padding)
    y1    = max(0, y1 - padding)
    x2    = min(w, x2 + padding)
    y2    = min(h, y2 + padding)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return image[y1:y2, x1:x2]

# ─────────────────────────────────────────────
# HELPER: EfficientNet prediction on crop
# ─────────────────────────────────────────────
def classify_crop(crop_bgr, eff_model):
    crop_rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (CROP_SIZE, CROP_SIZE))
    tensor       = eff_transform(crop_resized).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs    = eff_model(tensor)
        probs      = torch.softmax(outputs, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()
    return pred_class, confidence

# ─────────────────────────────────────────────
# HELPER: Estimate stem area
# ─────────────────────────────────────────────
def estimate_stem_area(image_bgr):
    hsv         = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    bg_mask     = cv2.inRange(hsv, np.array([0,0,200]),   np.array([180,30,255]))
    sky_mask    = cv2.inRange(hsv, np.array([90,10,180]), np.array([130,80,255]))
    combined_bg = cv2.bitwise_or(bg_mask, sky_mask)
    stem_mask   = cv2.bitwise_not(combined_bg)
    kernel      = np.ones((5,5), np.uint8)
    stem_mask   = cv2.morphologyEx(stem_mask, cv2.MORPH_CLOSE, kernel)
    stem_mask   = cv2.morphologyEx(stem_mask, cv2.MORPH_OPEN,  kernel)
    stem_area   = int(np.sum(stem_mask > 0))
    total_area  = image_bgr.shape[0] * image_bgr.shape[1]
    if stem_area < total_area * 0.1:
        stem_area = int(total_area * 0.60)
    return stem_area

# ─────────────────────────────────────────────
# HELPER: Overlay SAM mask on image
# ─────────────────────────────────────────────
def overlay_mask(image, mask, color, alpha=0.45):
    overlay      = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    contours, _  = cv2.findContours(mask.astype(np.uint8),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay

# ─────────────────────────────────────────────
# HELPER: Draw info panel on image
# ─────────────────────────────────────────────
def draw_panel(image, result):
    panel_h = 150
    h, w    = image.shape[:2]
    panel   = image.copy()
    cv2.rectangle(panel, (0, h - panel_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.75, image, 0.25, 0, panel)

    label       = result['severity_label']
    sev_index   = result['severity_index']
    label_color = SEVERITY_COLORS.get(label, (255,255,255))

    y = h - panel_h + 22
    cv2.putText(panel, f"Severity: {label}  (Index: {sev_index:.2f})",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
    y += 24
    cv2.putText(panel, f"Stem area: {result['stem_area_px']:,} px  |  "
                       f"Total infection: {result['total_infection_pct']:.2f}%",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    y += 22
    for cls_name, pct in result['infection_pct'].items():
        cid   = [k for k,v in CLASS_NAMES.items() if v == cls_name][0]
        color = CLASS_COLORS.get(cid, (255,255,255))
        cv2.putText(panel, f"{cls_name}: {pct:.2f}%",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        y += 18
    y += 4
    cv2.putText(panel, f"Detections: {result['num_detections']} lesions",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
    return panel

# ─────────────────────────────────────────────
# CORE: Run full pipeline on one image
# ─────────────────────────────────────────────
def run_inference(image_path, yolo_model, eff_model, sam_predictor):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w      = image_bgr.shape[:2]

    # ── 1. YOLOv8 Detection ──
    yolo_results = yolo_model.predict(
        source  = str(image_path),
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        device  = 'cpu',
        verbose = False,
    )
    boxes = yolo_results[0].boxes

    # ── 2. Stem Area ──
    stem_area = estimate_stem_area(image_bgr)

    class_areas = {0: 0, 1: 0, 2: 0}
    detections  = []
    vis_image   = image_bgr.copy()

    if boxes is not None and len(boxes) > 0:
        sam_predictor.set_image(image_rgb)

        for box in boxes:
            yolo_class_id = int(box.cls[0])
            yolo_conf     = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            x1 = max(0,x1); y1 = max(0,y1)
            x2 = min(w,x2); y2 = min(h,y2)

            if x2-x1 < 5 or y2-y1 < 5:
                continue

            # ── 3. EfficientNet Reclassification ──
            crop = crop_lesion(image_bgr, x1, y1, x2, y2)
            if crop is not None:
                eff_class_id, eff_conf = classify_crop(crop, eff_model)
            else:
                eff_class_id = yolo_class_id
                eff_conf     = yolo_conf

            # Final class = EfficientNet (more accurate for visually similar diseases)
            final_class_id = eff_class_id

            # ── 4. SAM Segmentation ──
            input_box        = np.array([x1, y1, x2, y2])
            masks, scores, _ = sam_predictor.predict(
                point_coords     = None,
                point_labels     = None,
                box              = input_box[None, :],
                multimask_output = False,
            )
            best_mask = masks[0].astype(np.uint8)
            mask_area = int(np.sum(best_mask > 0))

            if mask_area < MIN_MASK_AREA:
                continue

            class_areas[final_class_id] += mask_area

            detections.append({
                'yolo_class'  : CLASS_NAMES[yolo_class_id],
                'eff_class'   : CLASS_NAMES[final_class_id],
                'yolo_conf'   : yolo_conf,
                'eff_conf'    : eff_conf,
                'bbox'        : (x1, y1, x2, y2),
                'mask_area'   : mask_area,
            })

            # Visualize
            color     = CLASS_COLORS[final_class_id]
            vis_image = overlay_mask(vis_image, best_mask, color)
            cv2.rectangle(vis_image, (x1,y1), (x2,y2), color, 2)
            label = (f"{CLASS_NAMES[final_class_id]} "
                     f"Y:{yolo_conf:.2f} E:{eff_conf:.2f}")
            cv2.putText(vis_image, label, (x1, max(y1-8,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    # ── 5. Severity Estimation ──
    infection_pct = {}
    for cid, area in class_areas.items():
        pct = round((area / stem_area * 100) if stem_area > 0 else 0.0, 4)
        infection_pct[CLASS_NAMES[cid]] = pct

    total_infection_pct = round(sum(infection_pct.values()), 4)
    severity_index      = round(
        WEIGHTS["Anthracnose"] * infection_pct["Anthracnose"] +
        WEIGHTS["StemRot"]     * infection_pct["StemRot"]     +
        WEIGHTS["Canker"]      * infection_pct["Canker"],
        4
    )
    severity_label = get_severity_label(severity_index)

    result = {
        'image_name'         : image_path.name,
        'stem_area_px'       : stem_area,
        'num_detections'     : len(detections),
        'detections'         : detections,
        'class_areas_px'     : {CLASS_NAMES[k]: v for k,v in class_areas.items()},
        'infection_pct'      : infection_pct,
        'total_infection_pct': total_infection_pct,
        'severity_index'     : severity_index,
        'severity_label'     : severity_label,
        'vis_image'          : vis_image,
    }

    result['vis_image'] = draw_panel(vis_image, result)
    return result

# ─────────────────────────────────────────────
# PRINT RESULT
# ─────────────────────────────────────────────
def print_result(result):
    colors = {
        "Healthy": "\033[92m", "Moderate": "\033[93m",
        "Slightly Infected": "\033[33m", "Highly Infected": "\033[91m"
    }
    reset = "\033[0m"
    c     = colors.get(result['severity_label'], "")

    print(f"\n{'='*60}")
    print(f"📸  {result['image_name']}")
    print(f"🌿  Stem area     : {result['stem_area_px']:,} px")
    print(f"🔍  Detections    : {result['num_detections']} lesions")
    if result['detections']:
        print(f"\n  Lesion Details:")
        for d in result['detections']:
            match = "✅" if d['yolo_class'] == d['eff_class'] else "🔄"
            print(f"    {match} YOLO: {d['yolo_class']:12s}({d['yolo_conf']:.2f}) → "
                  f"EfficientNet: {d['eff_class']:12s}({d['eff_conf']:.2f}) | "
                  f"mask={d['mask_area']:,}px")
    print(f"\n  Infection % per Disease:")
    for cls_name, pct in result['infection_pct'].items():
        print(f"    {cls_name:12s}: {pct:.4f}%")
    print(f"  Total Infection  : {result['total_infection_pct']:.4f}%")
    print(f"\n  Severity Index   : {result['severity_index']:.4f}")
    print(f"  Severity Label   : {c}{result['severity_label']}{reset}")
    print(f"{'='*60}")

# ─────────────────────────────────────────────
# SAVE CSV REPORT
# ─────────────────────────────────────────────
def save_csv(all_results, output_dir):
    csv_path   = os.path.join(output_dir, "inference_report.csv")
    fieldnames = [
        'image_name', 'stem_area_px', 'num_detections',
        'anthracnose_area_px', 'stemrot_area_px', 'canker_area_px',
        'anthracnose_pct', 'stemrot_pct', 'canker_pct',
        'total_infection_pct', 'severity_index', 'severity_label',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                'image_name'         : r['image_name'],
                'stem_area_px'       : r['stem_area_px'],
                'num_detections'     : r['num_detections'],
                'anthracnose_area_px': r['class_areas_px']['Anthracnose'],
                'stemrot_area_px'    : r['class_areas_px']['StemRot'],
                'canker_area_px'     : r['class_areas_px']['Canker'],
                'anthracnose_pct'    : r['infection_pct']['Anthracnose'],
                'stemrot_pct'        : r['infection_pct']['StemRot'],
                'canker_pct'         : r['infection_pct']['Canker'],
                'total_infection_pct': r['total_infection_pct'],
                'severity_index'     : r['severity_index'],
                'severity_label'     : r['severity_label'],
            })
    print(f"\n📊 CSV report saved: {csv_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load YOLOv8 ──
    print("\n📦 Loading YOLOv8...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLOv8 loaded")

    # ── Load EfficientNet ──
    print("\n📦 Loading EfficientNet-B3...")
    eff_model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=3)
    eff_model.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=DEVICE))
    eff_model.to(DEVICE)
    eff_model.eval()
    print("✅ EfficientNet-B3 loaded")

    # ── Load SAM ──
    print("\n📦 Loading SAM...")
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    print("✅ SAM loaded")

    # ── Run Inference ──
    images = list(Path(INPUT_DIR).glob("*.jpg"))[:NUM_IMAGES]
    print(f"\n🚀 Running full pipeline on {len(images)} images...\n")

    all_results  = []
    label_counts = {"Healthy": 0, "Moderate": 0,
                    "Slightly Infected": 0, "Highly Infected": 0}

    for img_path in images:
        result = run_inference(img_path, yolo_model, eff_model, sam_predictor)
        if result is None:
            continue

        print_result(result)

        # Save annotated image
        out_path = os.path.join(OUTPUT_DIR, f"result_{result['image_name']}")
        cv2.imwrite(out_path, result['vis_image'])

        label_counts[result['severity_label']] += 1
        all_results.append(result)

    save_csv(all_results, OUTPUT_DIR)

    # ── Final Summary ──
    print(f"\n{'='*60}")
    print(f"🎉  CASL-NET INFERENCE COMPLETE")
    print(f"    Images processed : {len(all_results)}")
    print(f"\n    Severity Distribution:")
    for label, count in label_counts.items():
        print(f"      {label:20s}: {count} images")
    print(f"\n    Output saved to  : {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()