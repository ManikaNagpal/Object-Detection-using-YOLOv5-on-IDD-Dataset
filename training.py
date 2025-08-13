import os
import shutil
import glob
import subprocess

def install_yolov5():
    if not os.path.exists("yolov5"):
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"], check=True)
    subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"], check=True)
    subprocess.run(["pip", "install", "-U", "albumentations==1.0.3"], check=True)

def create_yaml(dataset_root, classes, yaml_path="data/idd.yaml"):
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    yaml_text = f"""# IDD YOLO dataset
path: {dataset_root}
train: train/images
val: val/images

nc: {len(classes)}
names: {classes}
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_text)
    print("YAML dataset config created:\n", yaml_text)
    return yaml_path

def train_yolov5(yaml_path, img_size=320, batch_size=16, epochs=1, weights="yolov5s.pt", experiment_name="idd_yolo"):
    cwd = os.getcwd()
    os.chdir("yolov5")
    subprocess.run([
        "python", "train.py",
        "--img", str(img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", yaml_path,
        "--weights", weights,
        "--name", experiment_name,
        "--cache"
    ], check=True)
    os.chdir(cwd)
    exp_dir = sorted(glob.glob(os.path.join("yolov5", "runs", "train", experiment_name + "*")))[-1]
    print("Training completed! Experiment dir:", exp_dir)
    return exp_dir

def run_inference(weights_path, source_images, img_size=640, conf=0.25):
    cwd = os.getcwd()
    os.chdir("yolov5")
    subprocess.run([
        "python", "detect.py",
        "--weights", weights_path,
        "--source", source_images,
        "--img", str(img_size),
        "--conf", str(conf)
    ], check=True)
    os.chdir(cwd)
