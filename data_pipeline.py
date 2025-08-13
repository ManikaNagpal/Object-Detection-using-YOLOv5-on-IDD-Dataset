import os
import numpy as np
from PIL import Image
import cv2
import shutil

# -----------------------------
# Mapping for semantic → YOLO
# -----------------------------
id_mapping = {
    0: "road",
    1: "parking",
    2: "sidewalk",
    3: "non_drivable",
    4: "person/animal",
    5: "rider",
    6: "vehicles",
}

class_names = sorted(set(id_mapping.values()))
name_to_id = {name: idx for idx, name in enumerate(class_names)}

def convert_semantic_to_yolo(gtfine_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    for root, _, files in os.walk(gtfine_dir):
        sem_files = [f for f in files if f.endswith("_label.png") and "_inst_label" not in f]
        for sem_file in sem_files:
            sem_path = os.path.join(root, sem_file)
            sem_mask = np.array(Image.open(sem_path))
            h, w = sem_mask.shape
            label_lines = []

            for sem_class in np.unique(sem_mask):
                if sem_class == 255 or sem_class not in id_mapping:
                    continue
                mask = (sem_mask == sem_class).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(mask)

                for comp_id in range(num_labels):
                    ys, xs = np.where(labels == comp_id)
                    if ys.size == 0:
                        continue
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()

                    x_center = (x_min + x_max) / 2 / w
                    y_center = (y_min + y_max) / 2 / h
                    box_width = (x_max - x_min) / w
                    box_height = (y_max - y_min) / h

                    yolo_class_id = name_to_id[id_mapping[sem_class]]
                    label_lines.append(
                        f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                    )

            base_name = os.path.splitext(sem_file)[0].replace("_label", "")
            txt_path = os.path.join(output_label_dir, base_name + ".txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(label_lines))

    print("Conversion complete! YOLO label files saved to:", output_label_dir)


# -----------------------------
# Dataset mirroring for YOLO
# -----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def mirror_dataset(images_root, labels_root, dataset_root, splits=["train","val"]):
    if os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)
    for split in splits:
        os.makedirs(os.path.join(dataset_root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, split, "labels"), exist_ok=True)

    def mirror_split(split):
        img_split_root = os.path.join(images_root, split)
        lbl_split_root = os.path.join(labels_root, split)
        out_img_root = os.path.join(dataset_root, split, "images")
        out_lbl_root = os.path.join(dataset_root, split, "labels")

        for root, _, files in os.walk(img_split_root):
            img_files = [f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
            if not img_files:
                continue
            rel = os.path.relpath(root, img_split_root)
            out_img_dir = os.path.join(out_img_root, rel)
            out_lbl_dir = os.path.join(out_lbl_root, rel)
            ensure_dir(out_img_dir)
            ensure_dir(out_lbl_dir)

            for f in img_files:
                img_src = os.path.join(root, f)
                img_dst = os.path.join(out_img_dir, f)
                if not os.path.exists(img_dst):
                    try:
                        os.symlink(img_src, img_dst)
                    except Exception:
                        shutil.copy2(img_src, img_dst)

                base = os.path.splitext(f)[0]
                if base.endswith("_image"):
                    base = base[:-6]
                lbl_src = os.path.join(lbl_split_root, rel, base + ".txt")
                lbl_dst = os.path.join(out_lbl_dir, base + "_image.txt")

                if os.path.exists(lbl_src):
                    if not os.path.exists(lbl_dst):
                        try:
                            os.symlink(lbl_src, lbl_dst)
                        except Exception:
                            shutil.copy2(lbl_src, lbl_dst)
                else:
                    if not os.path.exists(lbl_dst):
                        with open(lbl_dst, "w") as _:
                            pass

    for split in splits:
        mirror_split(split)
    print("Dataset mirroring complete!")
