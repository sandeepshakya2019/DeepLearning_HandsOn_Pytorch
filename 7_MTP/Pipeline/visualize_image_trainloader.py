import os
import cv2
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Class names and colors
class_names = ["car", "truck", "bus", "vehicle"]
class_colors = {
    0: (255, 0, 0),     # Blue for car
    1: (0, 255, 0),     # Green for truck
    2: (0, 0, 255),     # Red for bus
    3: (255, 255, 0),   # Cyan for vehicle
}

class UAVDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith(".jpg")
        ])
        self.label_dir = label_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = os.path.join(
            self.label_dir,
            os.path.basename(img_path).replace(".jpg", ".txt")
        )

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, w, h = map(float, parts)

                    x_center, y_center = x * width, y * height
                    w_pix, h_pix = w * width, h * height
                    x1 = int(x_center - w_pix / 2)
                    y1 = int(y_center - h_pix / 2)
                    x2 = int(x_center + w_pix / 2)
                    y2 = int(y_center + h_pix / 2)

                    boxes.append((int(cls), x1, y1, x2, y2))

        return img, boxes


def collate_fn(batch):
    images, boxes = zip(*batch)
    return list(images), list(boxes)


def visualize_random_from_loader(dataloader):
    for batch_imgs, batch_boxes in dataloader:
        idx = random.randint(0, len(batch_imgs) - 1)
        img = batch_imgs[idx].copy()
        boxes = batch_boxes[idx]

        for cls, x1, y1, x2, y2 in boxes:
            color = class_colors.get(cls, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_names[cls], (x1, max(10, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Random Sample from Batch (Index {idx})")
        plt.show()
        break  # Only show one batch


if __name__ == "__main__":
    dataset = UAVDataset(
        image_dir="dataset/UAVDT-processed/train/images",
        label_dir="dataset/UAVDT-processed/train/labels"
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    print(len(dataset), len(dataloader))
    visualize_random_from_loader(dataloader)
