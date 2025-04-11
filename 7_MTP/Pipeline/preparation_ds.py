import download_package
import dataset_download
from convert_annaotation_yolo8 import convert_annotation
from newdataset_for_yolo import copy_split_sequences
from visualize_single_image import visualize_from_path

from glob import glob
import os
from collections import defaultdict
from tqdm.auto import tqdm

root_dir = "./dataset/UAVDT-2024"
annotation_paths = glob(os.path.join(root_dir, "M*/annotations/*.txt"))
total_files = len(annotation_paths)

stats = {
    "total": 0,
    "converted": 0,
    "malformed": 0,
    "missing_image": 0,
    "skipped": defaultdict(int)
}

print(f"🔄 Converting {total_files} annotation files to YOLO format with dynamic image size...")

for anno_path in tqdm(annotation_paths, desc="Converting", unit="file"):
    sequence_dir = os.path.dirname(os.path.dirname(anno_path))  # Mxxxx
    file_name = os.path.basename(anno_path)

    label_dir = os.path.join(sequence_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    label_path = os.path.join(label_dir, file_name)

        # 🔍 Construct image path
    image_name = file_name.replace(".txt", ".jpg")
    image_path = os.path.join(sequence_dir, "images", image_name)

    convert_annotation(anno_path, label_path, image_path, stats)

print("\n✅ Conversion complete.")
print(f"📊 Total boxes:     {stats['total']}")
print(f"✅ Converted boxes: {stats['converted']}")
print(f"❌ Skipped boxes:   {sum(stats['skipped'].values())}")
for cls, count in sorted(stats["skipped"].items()):
    print(f"   - Skipped class {cls}: {count}")
print(f"⚠️ Malformed lines: {stats['malformed']}")
print(f"🖼️  Missing images: {stats['missing_image']}")


src_root = "./dataset/UAVDT-2024"
dst_root = "./dataset/UAVDT-processed"
copy_split_sequences(src_root, dst_root)


visualize_from_path(
        image_dir="dataset/UAVDT-processed/val/images",
        label_dir="dataset/UAVDT-processed/val/labels"
    )
