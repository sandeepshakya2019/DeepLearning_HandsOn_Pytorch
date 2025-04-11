import os
from ultralytics import YOLO

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


# model = YOLO(path)

# # Test on a single image (put your own image path)
# results = model("dataset/UAVDT-processed/val/images/img000003.jpg", save=True)

# # Show predictions inline
# results[0].show()

import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the trained YOLO model
model = YOLO(path)

# Directory with validation images
image_dir = "dataset/UAVDT-processed/val/images"

# Optional: show only first N images
max_to_show = 1
count = 0

# Iterate over image files
for img_file in os.listdir(image_dir):
    if not img_file.endswith(".jpg"):
        continue

    img_path = os.path.join(image_dir, img_file)
    print(f"Running prediction on: {img_file}")

    # Run prediction
    results = model(img_path, conf=0.25)

    # Get rendered image with boxes
    rendered_img = results[0].plot()  # This returns a BGR image

    # Convert to RGB for matplotlib
    rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(10, 8))
    plt.imshow(rendered_img)
    plt.title(f"Predictions for {img_file}")
    plt.axis('off')
    plt.show()

    count += 1
    if count >= max_to_show:
        break

