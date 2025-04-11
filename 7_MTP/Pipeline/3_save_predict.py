import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import shutil
import os

root_dir = "runs/train"

best_weights = []

for subdir in os.listdir(root_dir):
    full_path = os.path.join(root_dir, subdir, "weights", "best.pt")
    if os.path.isfile(full_path):
        best_weights.append(full_path)

# Print all found paths
for path in best_weights:
    print(path)


# Paths
image_dir = "dataset/UAVDT-processed/val/images"
output_img_dir = "dataset/UAVDT-new/val/images"
output_lbl_dir = "dataset/UAVDT-new/val/labels"

# Create directories if they don't exist
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# Load model
model = YOLO(path)

# Process each image
for img_file in tqdm(sorted(os.listdir(image_dir))):
    if not img_file.endswith(".jpg"):
        continue

    img_path = os.path.join(image_dir, img_file)
    results = model(img_path, conf=0.25)

    # Get image dimensions
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Save prediction as YOLO .txt
    label_path = os.path.join(output_lbl_dir, img_file.replace(".jpg", ".txt"))
    with open(label_path, "w") as f:
        for box in results[0].boxes.data.tolist():
            cls, x1, y1, x2, y2, conf = int(box[5]), *box[:4], box[4]
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # Copy image to new images folder
    shutil.copy(img_path, os.path.join(output_img_dir, img_file))

print("✅ New dataset created at: dataset/UAVDT-new/val/")
