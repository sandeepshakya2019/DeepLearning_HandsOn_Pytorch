import os
import cv2
import matplotlib.pyplot as plt
import random

# Paths
image_dir = "dataset/UAVDT-new/val/images"
label_dir = "dataset/UAVDT-new/val/labels"

# Class names and colors (modify if your class list differs)
class_names = ["car", "truck", "bus", "vehicle"]
class_colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
}

# List all image files
image_files = sorted([
    f for f in os.listdir(image_dir) if f.endswith(".jpg")
])

# Show N random images
N = 5
sample_images = random.sample(image_files, min(N, len(image_files)))

for img_file in sample_images:
    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Draw boxes
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                cls, x, y, bw, bh = map(float, line.strip().split())
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                color = class_colors.get(int(cls), (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, class_names[int(cls)], (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {img_file}")
    plt.show()
