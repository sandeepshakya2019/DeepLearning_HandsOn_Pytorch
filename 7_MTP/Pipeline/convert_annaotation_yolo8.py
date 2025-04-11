import os
from PIL import Image
from tqdm.auto import tqdm

def convert_annotation(input_path, output_path, image_path, stats):
    try:
        image = Image.open(image_path)
        image_width, image_height = image.size
    except Exception as e:
        print(f"⚠️  Could not open image: {image_path} — {e}")
        stats["missing_image"] += 1
        return

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            parts = line.strip().split(",")
            if len(parts) < 6:
                stats["malformed"] += 1
                continue

            x, y, w, h = map(int, parts[:4])
            class_id_raw = int(parts[5])
            stats["total"] += 1

            if class_id_raw not in [0, 1, 2, 3]:
                stats["skipped"][class_id_raw] += 1
                continue

            stats["converted"] += 1

            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            w_norm = w / image_width
            h_norm = h / image_height

            outfile.write(f"{class_id_raw} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


# if __name__ == "__main__":
    # root_dir = "./dataset/UAVDT-2024"
    # annotation_paths = glob(os.path.join(root_dir, "M*/annotations/*.txt"))
    # total_files = len(annotation_paths)

    # stats = {
    #     "total": 0,
    #     "converted": 0,
    #     "malformed": 0,
    #     "missing_image": 0,
    #     "skipped": defaultdict(int)
    # }

    # print(f"🔄 Converting {total_files} annotation files to YOLO format with dynamic image size...")

    # for anno_path in tqdm(annotation_paths, desc="Converting", unit="file"):
    #     sequence_dir = os.path.dirname(os.path.dirname(anno_path))  # Mxxxx
    #     file_name = os.path.basename(anno_path)

    #     label_dir = os.path.join(sequence_dir, "labels")
    #     os.makedirs(label_dir, exist_ok=True)

    #     label_path = os.path.join(label_dir, file_name)

    #     # 🔍 Construct image path
    #     image_name = file_name.replace(".txt", ".jpg")
    #     image_path = os.path.join(sequence_dir, "images", image_name)

    #     convert_annotation(anno_path, label_path, image_path, stats)

    # print("\n✅ Conversion complete.")
    # print(f"📊 Total boxes:     {stats['total']}")
    # print(f"✅ Converted boxes: {stats['converted']}")
    # print(f"❌ Skipped boxes:   {sum(stats['skipped'].values())}")
    # for cls, count in sorted(stats["skipped"].items()):
    #     print(f"   - Skipped class {cls}: {count}")
    # print(f"⚠️ Malformed lines: {stats['malformed']}")
    # print(f"🖼️  Missing images: {stats['missing_image']}")
