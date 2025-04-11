import os
from ultralytics import YOLO

import preparation_ds

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO("yolov8n.pt")

model.train(
    data="uavdt.yaml",
    epochs=5,
    imgsz=640,
    batch=32,
    show=False,
    name="yolov8-uavdt",
    project="runs/train"
)
