import os
from data_pipeline import convert_semantic_to_yolo, mirror_dataset
from training import install_yolov5, create_yaml, train_yolov5, run_inference

# ----------------------------
# Specify Paths 
# ----------------------------
gtfine_dir = ".....idd20k_lite/gtFine/"
yolo_label_dir = "...../yolo_labels"
images_root = "..../idd20k_lite/leftImg8bit"
dataset_root = "...../idd_yolo"
classes = ["road","parking","sidewalk","non_drivable","person/animal","rider","vehicles"]
test_images = "...../idd20k_lite/leftImg8bit/test/**/*.jpg"

# ----------------------------
# 1) Convert semantic labels to YOLO
# ----------------------------
convert_semantic_to_yolo(gtfine_dir, yolo_label_dir)

# ----------------------------
# 2) Mirror dataset for YOLO training
# ----------------------------
mirror_dataset(images_root, yolo_label_dir, dataset_root)

# ----------------------------
# 3) Install YOLOv5
# ----------------------------
install_yolov5()

# ----------------------------
# 4) Create dataset YAML
# ----------------------------
yaml_path = create_yaml(dataset_root, classes)

# ----------------------------
# 5) Train YOLOv5
# ----------------------------
exp_dir = train_yolov5(yaml_path, img_size=320, batch_size=16, epochs=1, weights="yolov5s.pt", experiment_name="idd_yolo_s_quick")

# ----------------------------
# 6) Run inference on test images
# ----------------------------
weights_path = os.path.join(exp_dir, "weights/best.pt")
run_inference(weights_path, test_images)
