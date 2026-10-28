# 🚗 Smart Parking Lot Occupancy Detection using Lightweight CNN

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.0%25-brightgreen.svg)]()
[![Inference Latency](https://img.shields.io/badge/Latency-8.2ms%2Fslot-blue.svg)]()

> **An automated, end-to-end, privacy-preserving, and license-free parking slot occupancy detector using a lightweight Convolutional Neural Network (CNN). Deployed with an asynchronous FastAPI inference service and an interactive Streamlit driver guidance dashboard.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [CNN Model Architecture & Benchmarks](#-cnn-model-architecture--benchmarks)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the System](#-running-the-system)
  - [1. Start FastAPI Backend](#1-start-the-fastapi-backend)
  - [2. Start Streamlit Dashboard](#2-start-the-streamlit-web-dashboard)
  - [3. Environment Verification](#3-verify-environment)
- [REST API Reference](#-rest-api-reference)
- [Dataset & Evaluation](#-dataset--evaluation)
- [Model Training & Evaluation Scripts](#-model-training--evaluation-scripts)
- [Academic Details](#-academic-details)
- [License](#-license)

---

## 📖 Overview

Searching for available parking spaces (*cruising for parking*) accounts for up to **30% of urban traffic congestion**, wasting millions of barrels of fuel and aggravating city emissions. Traditional smart parking deployments require invasive, in-ground physical IoT sensors (ultrasonic, geomagnetic) that incur prohibitive road drilling, trenching, and battery maintenance costs. Conversely, vision-based systems relying on **Automatic License Plate Recognition (ALPR)** introduce severe privacy violations under GDPR statutes and degrade under steep camera perspectives and visual occlusions.

**CNN Parking Lot** overcomes these challenges by utilizing existing overhead surveillance cameras to monitor parking bays with zero physical sensors:
1. It maps pre-configured quadrilateral slot regions (ROIs) from JSON configurations or auto-detects candidate parking bays.
2. Crops and bilinearly scales each slot patch to $96 \times 96 \times 3$.
3. Classifies occupancy using a custom **63K-parameter CNN** with batch normalization, global average pooling, and calibrated thresholding ($p \ge 0.30$).
4. Applies a **3-frame temporal rolling filter** to eliminate flicker and transient false alarms.
5. Surfaces real-time parking metrics through an interactive **Streamlit dashboard** and **FastAPI REST endpoints**.

---

## ✨ Key Features

- **🔒 Privacy-Preserving & License-Free:** Operates strictly on slot occupancy patches without logging vehicle registration plates, facial identities, or personal data (100% GDPR-compliant).
- **⚡ Ultra-Lightweight Edge Model:** Only **63,137 parameters** (~0.75 MB `.keras` weight footprint). Delivers sub-10ms latency per slot on standard commodity CPUs without requiring enterprise GPUs.
- **🌦️ All-Weather Resilience:** Trained and rigorously validated on 144,958+ crops from the **CNR-EXT** dataset across sunny, cloudy, rainy, overcast, and shadowed environments.
- **🎯 Dual-Mode Inference:**
  - **Calibrated Polygon Mode (`/detect`):** Precision 4-point quadrilateral coordinate mapping tailored to arbitrary camera angles and irregular lot layouts.
  - **Automated Detection Mode (`/auto-detect`):** Combines object detection candidate generation with CNN occupancy verification for rapid zero-calibration deployments.
- **⏱️ Temporal Multi-Frame Smoothing:** 3-frame FIFO probability buffer suppresses transient noise, passing pedestrians, and moving tree shadows.
- **🌐 Full-Stack Deployment:** Production-ready asynchronous FastAPI microservice with Swagger docs (`/docs`) and an intuitive Streamlit frontend.

---

## 🏗️ System Architecture
                 +-----------------------------------+
                   |      SURVEILLANCE VIDEO FEED      |
                   | (RTSP IP Camera / Video / Images) |
                   +-----------------+-----------------+
                                     |
                                     v
                   +-----------------------------------+
                   |      ROI EXTRACTION & RESIZING    |
                   |  - JSON Polygon Coordinate Map    |
                   |  - Bilinear Resize to 96x96x3     |
                   |  - Min-Max Normalization [0, 1]   |
                   +-----------------+-----------------+
                                     |
                                     v
                   +-----------------------------------+
                   |       CUSTOM FAST CNN ENGINE      |
                   |  - Conv2D (32, 64, 128) + BN +    |
                   |    ReLU + MaxPool                 |
                   |  - GlobalAvgPool + Dropout(0.4)   |
                   |  - Dense(1, Sigmoid) --> p >= 0.30|
                   +-----------------+-----------------+
                                     |
                                     v
                   +-----------------------------------+
                   |     TEMPORAL SMOOTHING BUFFER     |
                   |  - 3-Frame Moving Window Filter   |
                   |  - State Debouncing Logic         |
                   +-----------------+-----------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
   +---------------------------+           +---------------------------+
   |   FASTAPI REST SERVICE    |           |    STREAMLIT DASHBOARD    |
   |  - POST /predict (Single) |           |  - Live Annotated Feed    |
   |  - POST /detect (Full)    |           |  - Available Bay Counters |
   |  - POST /auto-detect      |           |  - Occupancy KPI Cards    |
   |  - GET  /health           |           |  - Per-Slot Confidence    |
   +---------------------------+           +---------------------------+
   
---

## 🧠 CNN Model Architecture & Benchmarks

The model is intentionally engineered to avoid the parameter bloat of heavy backbones (like ResNet or VGG) while outperforming shallow hand-crafted classifiers:

### Layer Breakdown
| Stage | Layer Type | Output Shape | Parameters | Activation / Regularization |
| :---: | :--- | :---: | :---: | :--- |
| **Input** | `InputLayer` | $(96, 96, 3)$ | 0 | Min-Max $[0.0, 1.0]$ |
| **Block 1** | `Conv2D (3x3)` + `BatchNorm` + `ReLU` + `MaxPool2D (2x2)` | $(48, 48, 32)$ | 992 | Spatial downsampling |
| **Block 2** | `Conv2D (3x3)` + `BatchNorm` + `ReLU` + `MaxPool2D (2x2)` | $(24, 24, 64)$ | 18,688 | Intermediate feature extraction |
| **Block 3** | `Conv2D (3x3)` + `BatchNorm` + `ReLU` + `MaxPool2D (2x2)` | $(12, 12, 128)$ | 43,520 | High-level semantic features |
| **Head** | `GlobalAveragePooling2D` + `Dropout(0.4)` | $(128)$ | 0 | Prevents overfitting |
| **Output** | `Dense (1)` | $(1)$ | 129 | `sigmoid` ($p \ge 0.30 = \text{Occupied}$) |
| **Total** | | | **63,137** | **Memory: ~0.75 MB** |

### Benchmark Comparison (CNR-EXT Held-Out Test Set)

| Method / Model | Accuracy | Precision | Recall | F1-Score | Parameters | CPU Latency / Slot |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HOG + SVM** *(Almeida et al.)* | 89.42% | 88.60% | 89.15% | 88.87% | ~12K | 3.1 ms |
| **mAlexNet** *(Amato et al.)* | 93.80% | 93.25% | 93.90% | 93.57% | ~2.4M | 18.4 ms |
| **ResNet-50 Transfer Learning** | 95.21% | 95.10% | 94.85% | 94.97% | ~25.6M | 42.6 ms |
| **Proposed Fast CNN (Ours)** | **95.998%** | **96.04%** | **96.10%** | **96.00%** | **63,137** | **8.2 ms** |

---

## 📁 Project Structure

```text
cnn-parking-lot/
├── api/                                # FastAPI backend microservice
│   └── main.py                         # REST endpoints (/predict, /detect, /auto-detect)
├── configs/                            # Configuration & slot coordinate files
│   ├── config.yaml                     # Global model and training parameters
│   ├── parking_slots.json              # Slot polygon coordinates (default lot)
│   ├── parking_slots_new_camera.json   # Multi-bay polygon definitions (2048x1529)
│   └── parking_slots_demo.json         # Camera 7 demo polygon mappings
├── dataset/                            # Dataset manifests and metadata
│   ├── cnrpark_ext/                    # CNR-EXT camera frame indexes & splits
│   └── splits/                         # Manifest splits (train, val, test)
├── models/                             # Trained deep learning models
│   ├── cnr_ext_cnn_fast.keras          # Core production lightweight CNN model weights
│   └── cnr_ext_fast_checkpoints/       # Periodic epoch checkpoints
├── outputs/                            # Evaluation artifacts and media samples
│   ├── cnr_ext_fast_demo/              # Demo video outputs, confusion matrices, CSVs
│   └── parking_video_sample.jpg        # Sample test frame
├── parking_drawer_frontend/            # Interactive web-based polygon coordinate drawer
│   └── build/index.html                # Compiled canvas drawing utility
├── scripts/                            # Operational, training & evaluation scripts
│   ├── train_cnr_ext_fast.py           # Model training script with date-grouped splits
│   ├── test_cnr_ext_fast_full_frame.py # Full-frame surveillance evaluation
│   ├── validate_cnr_ext_full_frames.py # Multi-camera validation across weather conditions
│   ├── create_cnr_demo_video.py        # Generates annotated demo video clips
│   ├── compare_full_frame_thresholds.py# Optimal threshold analysis (F1 vs threshold)
│   └── generate_report_docx.py         # Academic project report generator
├── src/                                # Core modular Python library
│   ├── data/                           # Ingestion, crop extraction, and normalization
│   ├── detection/                      # Occupancy detector, polygon annotator, video processor
│   ├── training/                       # CNN model builder, trainer, error analyzer
│   └── utils/                          # YAML and JSON config loaders
├── CNN_Parking_Lot_Mini_Project_Report.docx # Complete academic project report
├── main.py                             # Environment diagnostic & hardware check
├── requirements.txt                    # Frontend & data processing dependencies
├── requirements-api.txt                # FastAPI and deep learning backend dependencies
├── streamlit_app.py                    # Interactive Streamlit driver guidance web application
└── yolov8n.pt                          # Lightweight vehicle candidate localization weights
```
⚙️ Installation & Setup
Prerequisites

Before installing the project, ensure you have the following:

Python: 3.10 or 3.11
pip: Python package manager
Optional: NVIDIA GPU with CUDA support for accelerated training. Inference runs comfortably on a CPU.
Step 1: Clone the Repository
bash
git clone https://github.com/your-username/cnn-parking-lot.git
cd cnn-parking-lot

Step 2: Create and Activate a Virtual Environment
Windows (PowerShell)
powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

Linux / macOS
bash
python3 -m venv .venv
source .venv/bin/activate

Step 3: Install Dependencies

Install the core frontend and utility dependencies:

bash
pip install -r requirements.txt


Install the API and deep learning inference dependencies:

bash
pip install -r requirements-api.txt

🚀 Running the System
1. Verify the Environment

Run the diagnostic script to ensure OpenCV, TensorFlow, and Streamlit are correctly configured.

bash
python main.py

2. Start the FastAPI Backend

Start the asynchronous API server using Uvicorn:

bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload


Once the server is running, access the API documentation:

Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
3. Start the Streamlit Web Dashboard

Open a separate terminal, activate the virtual environment, and run:

bash
streamlit run streamlit_app.py


Open the dashboard in your browser:

http://localhost:8501

The dashboard provides an interface for interacting with the real-time parking occupancy detection and driver guidance system.

🔌 REST API Reference

The FastAPI backend exposes endpoints for health monitoring, parking slot classification, configured-bay detection, and automatic parking bay detection.

Method	Endpoint	Description	Request Payload	Response
GET	/	API health and status check	None	Project information and running status
GET	/health	Service diagnostics and model availability	None	Health status and configured slot count
POST	/predict	Classify a single parking slot crop	Image file (multipart form data)	Prediction and occupied probability
POST	/detect	Detect parking bays using configured JSON polygon coordinates	Surveillance image	Annotated Base64 JPEG, free/occupied totals, and slot details
POST	/auto-detect	Automatically detect parking bays using YOLOv8 and classify occupancy using the CNN	Image file and threshold	Annotated Base64 JPEG, bounding boxes, and occupancy statuses
Example: Python API Request

The following example sends an image to the /auto-detect endpoint and prints the parking occupancy statistics.

python
import requests

url = "http://localhost:8000/auto-detect"

with open("outputs/parking_video_sample.jpg", "rb") as f:
    response = requests.post(
        url,
        files={"file": f}
    )

response.raise_for_status()
data = response.json()

print(f"Total Bays Detected: {data['total_spaces']}")
print(
    f"Vacant: {data['free_spaces']} | "
    f"Occupied: {data['occupied_spaces']}"
)

📊 Dataset & Evaluation

The network was trained and evaluated using the CNR-EXT (Visual Parking Lot Dataset), collected by the National Research Council (CNR) in Pisa, Italy.

Dataset Overview
Metric	Details
Total Crops Monitored	144,958 images
Camera Perspectives	9 distinct perspectives
Training Set	98,542 samples
Validation Set	27,801 samples
Held-Out Test Set	18,615 samples
Calibrated Threshold	0.30
Dataset Split Strategy

A strict date-grouped partitioning strategy was used to minimize data leakage between training, validation, and test sets. Images from identical calendar dates were kept within their respective partitions.

Training set: Clear, sunny conditions with strong and harsh shadows.
Validation set: Overcast conditions with diffuse lighting.
Held-out test set: Heavy rain, asphalt puddles, and camera glare.
Threshold Calibration

An occupancy classification threshold of 0.30 was selected to maximize the macro F1-score across challenging visual conditions, including non-uniform sunlight, shadows, and reflections.

🛠️ Model Training & Evaluation Scripts

The repository includes scripts for training the CNN, evaluating full-frame parking detection, generating demonstration videos, and producing the academic project report.

1. Train the CNN Model

Retrain the lightweight CNN from scratch:

bash
python scripts/train_cnr_ext_fast.py

2. Evaluate Full-Frame Accuracy

Evaluate multi-camera parking detection performance across different weather and lighting scenarios:

bash
python scripts/validate_cnr_ext_full_frames.py

3. Generate a Demonstration Video

Create an annotated demonstration video with temporal smoothing:

bash
python scripts/create_cnr_demo_video.py

4. Generate the Academic Project Report

Regenerate the official academic project report in Microsoft Word (.docx) format:

bash
python scripts/generate_report_docx.py

