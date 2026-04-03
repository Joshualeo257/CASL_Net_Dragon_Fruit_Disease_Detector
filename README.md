# CASL-Net: Confusion-Aware Similarity Learning Network
### Visually Similar Dragon Fruit Stem Disease Classification

A deep learning pipeline for detecting, classifying, and estimating the severity of dragon fruit stem diseases using YOLOv8, EfficientNet-B3, and the Segment Anything Model (SAM).

**Diseases Detected:** Anthracnose · Stem Rot · Canker  
**Dataset:** 724 images → augmented to 2,000 (640×640)  
**Models:** YOLOv8m + EfficientNet-B3 + SAM ViT-H

---

## Project Structure

```
CASL-Net/
├── dataset/
│   ├── images/
│   │   ├── train/          # 1891 training images
│   │   └── val/            # 109 validation images
│   ├── labels/
│   │   ├── train/          # YOLO format .txt labels
│   │   └── val/
│   └── data.yaml           # YOLOv8 dataset config
├── models/
│   ├── yolov8/
│   │   └── best.pt         # Trained YOLOv8 weights
│   └── efficientnet/
│       └── efficientnet_best.pth   # Trained EfficientNet weights
├── sam/
│   └── sam_vit_h_4b8939.pth        # SAM checkpoint (~2.4GB)
├── outputs/
│   ├── test_results/       # YOLOv8 test output images
│   ├── sam_results/        # SAM segmentation output images
│   ├── severity_results/   # Severity estimation outputs + CSV
│   └── inference_results/  # Final pipeline outputs + CSV
├── preprocess.py           # Step 2: Dataset preprocessing
├── train_yolo.py           # Step 3: YOLOv8 training (local)
├── train_efficientnet.py   # Step 6: EfficientNet training (Colab)
├── test_model.py           # YOLOv8 model verification
├── test_efficientnet.py    # EfficientNet model verification
├── segment_sam.py          # Step 4: SAM segmentation test
├── severity.py             # Step 5: Severity estimation test
├── inference.py            # Step 7: Full inference pipeline
└── requirements.txt
```

---

## Quick Start

### Step 1 — Create the Virtual Environment

```bash
# In your project root (CASL-Net/)
python -m venv caslnet-env

# Activate (Windows)
caslnet-env\Scripts\activate

# Activate (Mac/Linux)
source caslnet-env/bin/activate
```

### Step 2 — Install Dependencies

```bash
pip install ultralytics
pip install torch torchvision
pip install timm
pip install segment-anything
pip install opencv-python
pip install numpy pandas
pip install matplotlib seaborn
pip install scikit-learn
pip install PyYAML
pip install albumentations
pip install tqdm
pip install Pillow
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 3 — Set Up the SAM Folder

Create the `sam/` folder in your project root and download the SAM checkpoint:

```bash
mkdir sam
```

Then download the checkpoint (~2.4GB) by pasting this URL directly into your browser:
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Save the downloaded file to:
```
CASL-Net/sam/sam_vit_h_4b8939.pth
```

### Step 4 — Verify You Have the Trained Model Weights

Make sure these two files exist before running inference:
```
CASL-Net/models/yolov8/best.pt
CASL-Net/models/efficientnet/efficientnet_best.pth
```

---

## Final Execution — Full Inference Pipeline

Once the environment is set up and all model weights are in place, run the full pipeline:

```bash
python inference.py
```

This runs the complete CASL-Net pipeline on validation images and saves:
- Annotated output images with disease masks and severity panel
- `inference_report.csv` with all metrics per image

### What the Pipeline Does

```
Input Image
    │
    ▼
YOLOv8m ──────────────► Detects lesion bounding boxes + confidence scores
    │
    ▼
EfficientNet-B3 ───────► Reclassifies each cropped lesion (confusion-aware)
    │                     Corrects visually similar disease misclassifications
    ▼
SAM (ViT-H) ───────────► Segments each lesion, computes pixel mask area
    │
    ▼
Severity Estimator ────► Calculates infection % and severity index
    │
    ▼
Output Image + CSV Report
```

### Output Example

```
📸  val_0003.jpg
🌿  Stem area     : 310,992 px
🔍  Detections    : 1 lesions

  Lesion Details:
    ✅ YOLO: Anthracnose(0.65) → EfficientNet: Anthracnose(1.00) | mask=15,034px

  Infection % per Disease:
    Anthracnose : 4.8342%
    StemRot     : 0.0000%
    Canker      : 0.0000%
  Total Infection  : 4.8342%

  Severity Index   : 4.8342
  Severity Label   : Healthy
```

---

## Additional Steps & Scripts

> The following scripts are the individual steps used to build up to the final pipeline. Run these if you need to retrain models, reprocess data, or test individual components.

---

### Preprocessing (Step 2)

**Script:** `preprocess.py`

Preprocesses the raw annotated dataset into a YOLO-ready structure.

**What it does:**
- Remaps 5 original dataset classes to 3 project classes
- Letterbox-resizes all images to 640×640 with black padding
- Augments training data to reach 2,000 total images using Albumentations
- Splits into 80% train / 20% val
- Generates `data.yaml`

**Class Mapping:**
| Original Class | → Project Class |
|---|---|
| Anthracnose | → Anthracnose (0) |
| Brown_Stem_Spot | → Anthracnose (0) |
| Soft_Rot | → StemRot (1) |
| Gray_Blight | → Canker (2) |
| Stem_Canker | → Canker (2) |

**Run:**
```bash
python preprocess.py
```

**Output:** `dataset/` folder with train/val split, labels, and `data.yaml`

---

### YOLOv8 Training (Step 3)

**Script:** `train_yolo.py`

Trains YOLOv8m on the preprocessed dataset. Run on Google Colab with a T4 GPU for best performance.

**Configuration:**
- Model: YOLOv8m (medium)
- Epochs: 75 (early stopping patience: 20)
- Image size: 640×640
- Batch size: 16 (Colab GPU)

**Training Results (mAP50):**
| Class | mAP50 | Recall |
|---|---|---|
| Anthracnose | 0.838 | 0.826 |
| StemRot | 0.432 | 0.030 |
| Canker | 0.574 | 0.455 |
| Overall | 0.614 | 0.437 |

> Note: StemRot has lower recall due to fewer original training samples (111 vs ~220 for other classes). EfficientNet compensates for this in the full pipeline.

**Run (locally on CPU — slow):**
```bash
python train_yolo.py
```

**Run on Colab (recommended):**
Upload `train_yolo.py` and `dataset/` to Google Drive, then run with GPU enabled.

**Output:** `models/yolov8/best.pt`

---

### YOLOv8 Model Test (Step 3 Verification)

**Script:** `test_model.py`

Verifies that `best.pt` works correctly by running inference on 5 validation images.

**Run:**
```bash
python test_model.py
```

**Output:** Annotated images saved to `outputs/test_results/`

---

### SAM Segmentation (Step 4)

**Script:** `segment_sam.py`

Tests the SAM segmentation pipeline independently. Takes YOLOv8 bounding boxes and feeds them as prompts into SAM to generate precise pixel masks for each lesion.

**What it computes:**
- Pixel mask area per detected lesion
- Stem area (estimated by background removal)
- Infection percentage = `infection_area / stem_area × 100`

**Run:**
```bash
python segment_sam.py
```

**Output:** Mask-overlaid images saved to `outputs/sam_results/`

> Note: SAM (ViT-H) takes 30–60 seconds to load on first run. This is normal.

---

### Severity Estimation (Step 5)

**Script:** `severity.py`

Applies the severity index formula to compute a weighted disease severity score and assigns a label.

**Infection % Formula:**
```
infection_pct = infection_area / stem_area × 100
```

**Severity Index Formula:**
```
severity_index = w1 × anthracnose_pct + w2 × stemrot_pct + w3 × canker_pct

Weights:
  w1 = 1.0  (Anthracnose — surface lesions, slowest spread)
  w2 = 1.5  (StemRot — internal damage, fastest spread, most destructive)
  w3 = 1.2  (Canker — disrupts nutrient flow, moderately destructive)
```

**Severity Labels:**
| Severity Index | Label |
|---|---|
| 0 – 4.99 | Healthy |
| 5 – 9.99 | Moderate |
| 10 – 14.99 | Slightly Infected |
| 15+ | Highly Infected |

**Run:**
```bash
python severity.py
```

**Output:** Annotated images + `severity_report.csv` in `outputs/severity_results/`

---

### EfficientNet Training (Step 6)

**Script:** `train_efficientnet.py`

Trains EfficientNet-B3 as a secondary classifier on cropped lesion regions. Run on Google Colab with a T4 GPU.

**Why EfficientNet on top of YOLOv8?**
The three diseases are visually similar, especially under varying lighting conditions. YOLOv8 detects where lesions are, but can misclassify between similar-looking diseases. EfficientNet-B3 looks at each cropped lesion individually and performs a more accurate classification — this is the "Confusion-Aware" component of CASL-Net.

**Why EfficientNet-B3 and not B4?**
- Input crops are 128×128 — B4's higher resolution advantage is not useful at this size
- CPU/limited GPU training — B4 is ~30% slower with minimal accuracy gain
- 26,971 training crops is well-suited to B3's capacity without overfitting

**Training Results:**
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Anthracnose | 0.97 | 0.98 | 0.97 |
| StemRot | 0.97 | 0.98 | 0.98 |
| Canker | 0.97 | 0.95 | 0.96 |
| **Overall** | **0.97** | **0.97** | **0.97** |

**Colab Setup (recommended):**

Cell 1 — Mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Cell 2 — Install timm:
```python
!pip install timm -q
```

Cell 3 — Copy dataset locally for fast I/O:
```python
import shutil, os
shutil.copytree("/content/drive/MyDrive/CASL-Net/dataset", "/content/dataset")
shutil.copy("/content/drive/MyDrive/CASL-Net/train_efficientnet.py", "/content/train_efficientnet.py")
os.makedirs("/content/models/efficientnet", exist_ok=True)
```

Cell 4 — Fix paths and train:
```python
with open("/content/train_efficientnet.py", "r") as f:
    script = f.read()
# Replace Windows paths with Colab paths before exec
exec(script)
```

Cell 5 — Save weights to Drive:
```python
import shutil, os
os.makedirs("/content/drive/MyDrive/CASL-Net/models/efficientnet", exist_ok=True)
shutil.copy("/content/models/efficientnet/efficientnet_best.pth",
            "/content/drive/MyDrive/CASL-Net/models/efficientnet/efficientnet_best.pth")
```

**Output:** `models/efficientnet/efficientnet_best.pth`

---

### EfficientNet Model Test (Step 6 Verification)

**Script:** `test_efficientnet.py`

Verifies that `efficientnet_best.pth` works correctly by classifying cropped lesions from validation images and comparing against ground truth labels.

**Run:**
```bash
python test_efficientnet.py
```

**Expected output:** 97–100% accuracy on validation lesion crops.

---

## Model Summary

| Model | Role | Accuracy |
|---|---|---|
| YOLOv8m | Lesion detection + localization | mAP50: 0.614 overall |
| EfficientNet-B3 | Lesion classification (confusion-aware) | 97% accuracy |
| SAM ViT-H | Lesion segmentation + area measurement | Pixel-level masks |

---

## Dataset

**Source:** Dragon Fruit Stem Disease: An Annotated High-Resolution Image Dataset  
**DOI:** 10.17632/v3brsrm2f7.1  
**Contributor:** Sushmoy Md Abu Rayhan Sushmoy  
**License:** CC BY 4.0

Original: 724 images, 6 classes, 640×480px  
After preprocessing: 2,000 images, 3 classes, 640×640px

---

## Notes

- Severity index thresholds are research-defined based on biological disease weights. In a production system these would be validated by a plant pathologist with field data.
- StemRot has fewer original samples (111 vs ~220 for other classes). EfficientNet compensates for YOLOv8's lower StemRot recall by reclassifying detected lesions with 98% recall.
- SAM ViT-H is the largest and most accurate SAM variant. Use `vit_b` for faster inference at the cost of some segmentation accuracy.
