```markdown
# Vehicle License Plate Detection and Tracking

This repository contains a Google Colab notebook that demonstrates a complete pipeline for **vehicle license plate detection and tracking**. The project leverages state-of-the-art deep learning techniques to accurately locate license plates in images and track them across video sequences.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Setup and Usage](#setup-and-usage)
    - [Running in Google Colab](#running-in-google-colab)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
- [Model Training](#model-training)
- [Evaluation Results](#evaluation-results)
- [Video Tracking (DeepSORT)](#video-tracking-deepsort)
- [Exported Models](#exported-models)
- [Future Improvements](#future-improvements)
- [About the Author](#about-the-author)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

The goal of this project is to build an end-to-end system for detecting and tracking vehicle license plates. This involves:

1.  Acquiring a suitable dataset.
2.  Training a robust object detection model (YOLOv11).
3.  Integrating a tracking algorithm (DeepSORT) to follow detected license plates in videos.
4.  Evaluating the model's performance.
5.  Providing options for model deployment (export).

The entire process is documented and executed within a single Google Colab notebook for ease of use and reproducibility, especially for those with access to Colab's free GPU resources.

## Technologies Used

* **Python**
* **Ultralytics YOLO** (Specifically YOLOv11n for detection)
* **DeepSORT** (for tracking)
* **PyTorch** (Deep Learning Framework)
* **Roboflow** (for dataset management and download)
* **OpenCV** (for image/video processing and visualization)
* **Matplotlib & Seaborn** (for data visualization)
* **PyYAML** (for handling configuration files)
* **gdown** (for downloading data from Google Drive)
* **ffmpeg** (for video processing and compression)

## Dataset

The project uses the **License Plate Recognition** dataset from Roboflow Universe. This dataset contains **7,035 images with 2,195 license plate instances annotated**. It is split into training, validation, and test sets.

## Setup and Usage

The easiest way to run this project is directly in Google Colab.

### Running in Google Colab

1.  Open the notebook file (`your_notebook_name.ipynb`) in Google Colab.
2.  Ensure you are using a GPU runtime (`Runtime -> Change runtime type -> GPU`).
3.  Execute the cells sequentially from top to bottom.

### Prerequisites

If you wish to run parts of this code locally (requires a suitable environment with potentially a GPU), ensure you have:

* Python 3.7+
* The dependencies listed in the [Technologies Used](#technologies-used) section installed (e.g., via `pip install -r requirements.txt`).
* FFmpeg installed on your system.

### Steps

The Colab notebook guides you through the following steps:

1.  Installing dependencies.
2.  Importing libraries and configuration.
3.  Downloading the dataset from Roboflow (requires a Roboflow API key, which is included in the notebook for the specific dataset used).
4.  Exploring and analyzing the dataset.
5.  Visualizing dataset samples with annotations.
6.  Preparing the YAML configuration for YOLO.
7.  Initializing the YOLOv11 model.
8.  Training the model.
9.  Analyzing training results and metrics (mAP, Precision, Recall).
10. Visualizing training curves.
11. Testing the trained model on sample images.
12. Saving and exporting the model (ONNX, PT).
13. Downloading a sample video for tracking.
14. Performing object tracking on the video using the trained model and DeepSORT (Note: The final video compression step encountered an issue during the provided notebook execution).

## Model Training

The model was trained for **75 epochs** using the YOLOv11n architecture. Key training parameters are detailed in Table 1 of your report.

## Evaluation Results

After training, the model was evaluated on the test dataset. The results demonstrate high performance in license plate detection:

* **mAP50:** **97.1%**
* **mAP50-95:** **70.5%**
* **Precision:** **98.0%**
* **Recall:** **95.0%**

These metrics indicate that the model is highly effective at identifying and locating license plates with great accuracy and a low error rate.

## Video Tracking (DeepSORT)

The notebook integrates DeepSORT to track the detected license plates across video frames. This allows for assigning unique IDs to each detected plate and following its movement. The output of the tracking process is intended to be saved as a video.

## Exported Models

The trained `best.pt` model is saved and exported into different formats for potential deployment:

* `license_plate_model.pt` (PyTorch format)
* `license_plate_model.onnx` (ONNX format)
* An attempt was made to export to `license_plate_model.engine` (TensorRT format), but it failed on the current Colab setup.

These files are stored in the `exported_models` directory within the repository (or generated upon running the notebook).

## Future Improvements

* Implement the character recognition (OCR) step to read the license plate numbers.
* Improve the video processing and compression pipeline.
* Experiment with different YOLO models (e.g., larger ones) or other detection architectures.
* Refine DeepSORT parameters or explore alternative tracking algorithms.
* Build a user interface or API for the system.
* Integrate with automated billing and account management systems.

## About the Author

This project was developed by **ALOUACH Abdennour** as part of the Master's degree program in **MIATE - Artificial Intelligence and Emerging Technologies** at Mohammed Premier University, Multidisciplinary Faculty of Nador.

## License

This project is licensed under the **MIT License**.

## Acknowledgements

* [Ultralytics](https://github.com/ultralytics/ultralytics) for the powerful YOLO framework.
* [Roboflow](https://roboflow.com/) for the dataset and platform.
* The authors of DeepSORT and the underlying tracking algorithms.
* Google Colab for providing the computing environment.
```
