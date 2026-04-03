"""
CASL-Net Step 2: Preprocessing Script (FIXED)
===============================================
Fixes applied:
  - GaussNoise: var_limit -> std_range (albumentations 2.x compatibility)
  - Degenerate bbox filter: removes zero-width/height annotations before
    processing and augmentation so they don't silently corrupt labels

Class Mapping:
  Anthracnose    -> 0 (Anthracnose)
  Brown_Stem_Spot-> 0 (Anthracnose)  [merged, similar lesion type]
  Soft_Rot       -> 1 (StemRot)
  Gray_Blight    -> 2 (Canker)
  Stem_Canker    -> 2 (Canker)

Output structure:
  dataset/
    images/train/
    images/val/
    labels/train/
    labels/val/
    data.yaml
"""

import os
import cv2
import random
import numpy as np
import albumentations as A
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION — paths already set
# ─────────────────────────────────────────────
ANNOTATED_DIR = r"C:\Data\MSRIT\8th SEM project\CASL-Net\Dragon Fruit (Pitahaya)\Annotated Files"
OUTPUT_DIR    = r"C:\Data\MSRIT\8th SEM project\CASL-Net\dataset"

TARGET_SIZE   = 640
TRAIN_SPLIT   = 0.8
TARGET_TOTAL  = 2000
RANDOM_SEED   = 42
MIN_BBOX_SIZE = 0.002   # Minimum normalized width AND height for a valid bbox

# ─────────────────────────────────────────────
# CLASS MAPPING
# ─────────────────────────────────────────────
CLASS_MAP = {
    "Anthracnose":     0,
    "Brown_Stem_Spot": 0,
    "Soft_Rot":        1,
    "Gray_Blight":     2,
    "Stem_Canker":     2,
}

CLASS_NAMES = {
    0: "Anthracnose",
    1: "StemRot",
    2: "Canker",
}

# ─────────────────────────────────────────────
# AUGMENTATION PIPELINE
# FIX 1: var_limit -> std_range for albumentations 2.x
# ─────────────────────────────────────────────
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.4),
    A.Rotate(limit=20, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(std_range=(0.02, 0.1), p=0.2),   # FIXED: was var_limit=(10, 50)
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.RandomShadow(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


# ─────────────────────────────────────────────
# FIX 2: Filter degenerate bounding boxes
# Removes any bbox where width or height is effectively zero
# These are annotation errors in the original dataset
# ─────────────────────────────────────────────
def filter_degenerate_bboxes(labels, min_size=MIN_BBOX_SIZE):
    valid   = []
    removed = 0
    for label in labels:
        _, cx, cy, bw, bh = label
        if bw < min_size or bh < min_size:
            removed += 1
        else:
            valid.append(label)
    return valid, removed


# ─────────────────────────────────────────────
# HELPER: Determine class from filename prefix
# ─────────────────────────────────────────────
def get_class_from_filename(filename):
    for prefix, class_id in CLASS_MAP.items():
        if filename.startswith(prefix):
            return class_id
    return None


# ─────────────────────────────────────────────
# HELPER: Letterbox resize to TARGET_SIZE x TARGET_SIZE
# ─────────────────────────────────────────────
def letterbox_image(image, size=TARGET_SIZE):
    h, w    = image.shape[:2]
    scale   = size / max(h, w)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas  = np.zeros((size, size, 3), dtype=np.uint8)
    pad_top  = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, pad_top, pad_left


# ─────────────────────────────────────────────
# HELPER: Transform YOLO bbox for letterbox
# ─────────────────────────────────────────────
def transform_bbox_letterbox(cx, cy, bw, bh, orig_w, orig_h, scale, pad_top, pad_left, size=TARGET_SIZE):
    px_cx  = cx * orig_w
    px_cy  = cy * orig_h
    px_bw  = bw * orig_w
    px_bh  = bh * orig_h
    new_cx = (px_cx * scale + pad_left) / size
    new_cy = (px_cy * scale + pad_top)  / size
    new_bw = (px_bw * scale) / size
    new_bh = (px_bh * scale) / size
    new_cx = max(0.0, min(1.0, new_cx))
    new_cy = max(0.0, min(1.0, new_cy))
    new_bw = max(0.0, min(1.0, new_bw))
    new_bh = max(0.0, min(1.0, new_bh))
    return new_cx, new_cy, new_bw, new_bh


# ─────────────────────────────────────────────
# HELPER: Read YOLO label file
# ─────────────────────────────────────────────
def read_label(txt_path):
    labels = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return labels


# ─────────────────────────────────────────────
# HELPER: Write YOLO label file
# ─────────────────────────────────────────────
def write_label(txt_path, labels):
    with open(txt_path, 'w') as f:
        for label in labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


# ─────────────────────────────────────────────
# STEP 1: Load all valid image-label pairs
# ─────────────────────────────────────────────
def load_dataset(annotated_dir):
    data          = []
    annotated_path= Path(annotated_dir)
    jpg_files     = sorted(annotated_path.glob("*.jpg"))

    print(f"\n📂 Scanning: {annotated_dir}")
    class_counts  = {0: 0, 1: 0, 2: 0}
    skipped       = 0
    total_removed = 0

    for jpg_path in jpg_files:
        txt_path = jpg_path.with_suffix('.txt')
        if not txt_path.exists():
            skipped += 1
            continue

        class_id = get_class_from_filename(jpg_path.name)
        if class_id is None:
            print(f"  ⚠️  Unrecognized prefix: {jpg_path.name} — skipping")
            skipped += 1
            continue

        # Read raw labels, remap class ID, then filter degenerate bboxes
        raw_labels    = read_label(str(txt_path))
        mapped_labels = [[class_id] + l[1:] for l in raw_labels]
        valid_labels, removed = filter_degenerate_bboxes(mapped_labels)
        total_removed += removed

        # Skip image entirely if no valid labels remain after filtering
        if not valid_labels:
            skipped += 1
            continue

        class_counts[class_id] += 1
        data.append({
            'img_path':     jpg_path,
            'clean_labels': valid_labels,
            'new_class_id': class_id,
        })

    print(f"\n✅ Loaded {len(data)} valid image-label pairs ({skipped} skipped)")
    if total_removed > 0:
        print(f"   🧹 Removed {total_removed} degenerate (zero-size) bboxes from annotations")
    print(f"   Class breakdown (before augmentation):")
    for cid, name in CLASS_NAMES.items():
        print(f"     [{cid}] {name}: {class_counts[cid]} images")

    return data


# ─────────────────────────────────────────────
# STEP 2: Letterbox + save a single image
# ─────────────────────────────────────────────
def process_single(item, out_img_path, out_lbl_path):
    img = cv2.imread(str(item['img_path']))
    if img is None:
        print(f"  ❌ Cannot read: {item['img_path']}")
        return None, None

    orig_h, orig_w = img.shape[:2]
    lb_img, scale, pad_top, pad_left = letterbox_image(img)

    new_labels = []
    for label in item['clean_labels']:
        _, cx, cy, bw, bh = label
        new_cx, new_cy, new_bw, new_bh = transform_bbox_letterbox(
            cx, cy, bw, bh, orig_w, orig_h, scale, pad_top, pad_left
        )
        # Final degenerate check after letterbox transform
        if new_bw >= MIN_BBOX_SIZE and new_bh >= MIN_BBOX_SIZE:
            new_labels.append([item['new_class_id'], new_cx, new_cy, new_bw, new_bh])

    if not new_labels:
        return None, None

    cv2.imwrite(str(out_img_path), lb_img)
    write_label(out_lbl_path, new_labels)
    return lb_img, new_labels


# ─────────────────────────────────────────────
# STEP 3: Augment and save
# ─────────────────────────────────────────────
def augment_and_save(img, labels, out_img_path, out_lbl_path):
    if not labels:
        return False
    bboxes       = [[l[1], l[2], l[3], l[4]] for l in labels]
    class_labels = [l[0] for l in labels]
    try:
        result     = augment(image=img, bboxes=bboxes, class_labels=class_labels)
        aug_img    = result['image']
        aug_bboxes = result['bboxes']
        aug_cls    = result['class_labels']
        if not aug_bboxes:
            return False
        aug_labels = [[aug_cls[i]] + list(aug_bboxes[i]) for i in range(len(aug_bboxes))]
        cv2.imwrite(str(out_img_path), aug_img)
        write_label(out_lbl_path, aug_labels)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# STEP 4: Generate data.yaml
# ─────────────────────────────────────────────
def generate_yaml(output_dir):
    yaml_path = Path(output_dir) / "data.yaml"
    content   = f"""# CASL-Net Dataset Config
path: {output_dir}
train: images/train
val: images/val

nc: 3
names:
  0: Anthracnose
  1: StemRot
  2: Canker
"""
    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f"\n📄 data.yaml saved to: {yaml_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    out = Path(OUTPUT_DIR)
    for split in ['train', 'val']:
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    data = load_dataset(ANNOTATED_DIR)
    if not data:
        print("❌ No data found. Check ANNOTATED_DIR.")
        return

    random.shuffle(data)
    split_idx  = int(len(data) * TRAIN_SPLIT)
    train_data = data[:split_idx]
    val_data   = data[split_idx:]
    print(f"\n📊 Split: {len(train_data)} train / {len(val_data)} val")

    # ── Validation (no augmentation) ──
    print("\n🔄 Processing validation set...")
    val_done = 0
    for i, item in enumerate(val_data):
        img_out = out / 'images' / 'val' / f"val_{i:04d}.jpg"
        lbl_out = out / 'labels' / 'val' / f"val_{i:04d}.txt"
        img, labels = process_single(item, img_out, lbl_out)
        if img is not None:
            val_done += 1
    print(f"✅ Validation set done: {val_done} images")

    # ── Training + augmentation ──
    print("\n🔄 Processing training set + augmentation...")
    processed_train = []
    for i, item in enumerate(train_data):
        img_out = out / 'images' / 'train' / f"train_{i:04d}.jpg"
        lbl_out = out / 'labels' / 'train' / f"train_{i:04d}.txt"
        img, labels = process_single(item, img_out, lbl_out)
        if img is not None:
            processed_train.append((img, labels, i))

    current_count = len(processed_train)
    needed        = TARGET_TOTAL - val_done - current_count
    print(f"   Original training images : {current_count}")
    print(f"   Augmented images needed  : {needed}")

    aug_index    = 0
    aug_count    = 0
    attempts     = 0
    max_attempts = needed * 5

    while aug_count < needed and attempts < max_attempts:
        img, labels, _ = processed_train[aug_index % len(processed_train)]
        aug_index += 1
        attempts  += 1
        img_out = out / 'images' / 'train' / f"train_aug_{aug_count:05d}.jpg"
        lbl_out = out / 'labels' / 'train' / f"train_aug_{aug_count:05d}.txt"
        if augment_and_save(img, labels, img_out, lbl_out):
            aug_count += 1
            if aug_count % 100 == 0:
                print(f"   ... {aug_count}/{needed} augmented images done")

    total_train = current_count + aug_count
    print(f"✅ Training set done: {total_train} images ({current_count} original + {aug_count} augmented)")

    generate_yaml(OUTPUT_DIR)

    print("\n" + "="*55)
    print("🎉 PREPROCESSING COMPLETE")
    print("="*55)
    print(f"  Train images : {total_train}")
    print(f"  Val images   : {val_done}")
    print(f"  Total        : {total_train + val_done}")
    print(f"  Image size   : {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"  Classes      : 0=Anthracnose, 1=StemRot, 2=Canker")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print("\n✅ Next step: Train YOLOv8 using data.yaml")


if __name__ == "__main__":
    main()