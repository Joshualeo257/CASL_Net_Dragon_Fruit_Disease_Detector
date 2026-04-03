from ultralytics import YOLO

# Load YOLOv8 pretrained model (medium size — good balance of speed/accuracy)
model = YOLO("yolov8m.pt")

model.train(
    data    = r"C:\Data\MSRIT\8th SEM project\CASL-Net\dataset\data.yaml",
    epochs  = 75,          # midpoint of your 50-100 range
    imgsz   = 640,
    batch   = 8,           # lower if you get out-of-memory errors (try 4)
    name    = "caslnet_yolov8",
    project = r"C:\Data\MSRIT\8th SEM project\CASL-Net\outputs",
    patience= 20,          # early stopping if no improvement
    device  = 0,           # GPU (use 'cpu' if no GPU)
    workers = 4,
    conf    = 0.25,        # confidence threshold
    iou     = 0.5,
    plots   = True,        # saves training curves and confusion matrix
    save    = True,
    verbose = True,
)