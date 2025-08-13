# Object-Detection-using-YOLOv5-on-IDD-Dataset
This repository provides a complete pipeline for training YOLOv5 on the [**Indian Driving Dataset (IDD) by IIT Bombay**](<https://idd.insaan.iiit.ac.in/>). It includes modular Python scripts for preprocessing, dataset preparation, and model training, along with an optional Google Colab notebook for quick experimentation.

## Repository Structure

The repository is organized as follows:
"
.
|- main.py                # Orchestrates the entire workflow: preprocessing, dataset mirroring, training, inference
|- data_pipeline.py       # Handles semantic label processing and YOLO-compatible dataset creation
|- training.py            # Installs YOLOv5, sets up dataset YAML, trains the model, and runs inference
|- idd_yolo_colab.ipynb   # Google Colab notebook version (messy/unorganized but usable)
\- README.md
"

Each file serves a specific purpose:

- **main.py**: Integrates all steps from data preprocessing to YOLOv5 training and inference.  
- **data_pipeline.py**: Contains functions to convert semantic masks to YOLO labels and mirror the dataset structure.  
- **training.py**: Handles YOLOv5 installation, dataset YAML creation, model training, and running inference.  
- **idd_yolo_colab.ipynb**: Interactive Colab notebook for step-by-step experimentation, though less organized.

## Dataset

The pipeline uses the **IDD Lite Dataset**.  

- **Download link:** [IDD20k Lite](https://idd.insaan.iiit.ac.in/accounts/login/?next=/dataset/download/)  
- The dataset should be placed in your drive or local directory. By default, the scripts expect the following structure:

"
idd20k_lite/
|- gtFine/
   |- train/
   |- val/
|- leftImg8bit/
   |- train/
   |- val/
"

## Features

1. **Semantic → YOLO Conversion**  
   Converts semantic masks to YOLO-compatible bounding boxes (`<class_id> <x_center> <y_center> <width> <height>`).  

2. **Dataset Mirroring**  
   Mirrors the dataset structure (`images/` + `labels/`) with symlinks or copies for efficient training.  

3. **YOLOv5 Training & Inference**  
   Installs YOLOv5 automatically, creates dataset YAML, trains the model, and allows running inference on test images.

4. **Optional Colab Notebook**  
   Quick experimentation in Google Colab (`idd_yolo_colab.ipynb`), though less organized than the modular scripts.

## Usage

### 1. Clone the repository and mount dataset
"""bash
git clone <repo-url>
"""

### 2. Run the main pipeline
"""bash
python main.py
"""

This will:

- Convert semantic labels to YOLO format  
- Mirror dataset for YOLO training  
- Install YOLOv5 and dependencies  
- Create the dataset YAML file  
- Train YOLOv5  
- Run inference on test images

### 3. Optional: Use the Colab notebook
Open `idd_yolo_colab.ipynb` in Google Colab for an interactive, step-by-step workflow.

## Customization

- **Classes:** Modify the `classes` list in `main.py` if you want to adjust the mapped classes.  
- **Training parameters:** Adjust `img_size`, `batch_size`, `epochs`, and `weights` in `main.py` for different experiments.  
- **Dataset paths:** Update the paths in `main.py` if your dataset is stored elsewhere.

## Notes

- The modular Python scripts are recommended for structured experiments and reproducibility.  
- The Colab notebook is more of a quick-start guide and may be messy.  
- YOLOv5 installation and training are handled programmatically via `subprocess`, so no notebook-specific commands are needed.

## References

- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)  
- [Indian Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/accounts/login/?next=/dataset/download/)

## Author

This pipeline was developed to provide a clean, modular workflow for training YOLOv5 on the IDD dataset, along with an optional Colab notebook for experimentation."

