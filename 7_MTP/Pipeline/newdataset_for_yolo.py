# save this as split_uavdt_train_val.py

import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

def copy_split_sequences(src_root, dst_root, train_ratio=0.8):
    all_sequences = sorted(glob(os.path.join(src_root, "M*")))
    train_seqs, val_seqs = train_test_split(all_sequences, train_size=train_ratio, random_state=42)

    for split_name, split_list in zip(['train', 'val'], [train_seqs, val_seqs]):
        for seq_path in tqdm(split_list):
            images_src = os.path.join(seq_path, "images")
            labels_src = os.path.join(seq_path, "labels")
          
            images_dst = os.path.join(dst_root, split_name, "images")
            labels_dst = os.path.join(dst_root, split_name, "labels")

            os.makedirs(images_dst, exist_ok=True)
            os.makedirs(labels_dst, exist_ok=True)

            for img_file in glob(os.path.join(images_src, "*.jpg")):
                shutil.copy(img_file, os.path.join(images_dst, os.path.basename(img_file)))
                
            for label_file in glob(os.path.join(labels_src, "*.txt")):
                shutil.copy(label_file, os.path.join(labels_dst, os.path.basename(label_file)))
            

    print("✅ Split complete into train/ and val/")
