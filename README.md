
# **Pothole Detection and Quantification System**

An end-to-end pothole detection, quantification, and analysis system integrating Deep Learning, Machine Learning, and 3D Reconstruction approaches.


## 🚀 Project Overview

Potholes are a major contributor to road accidents, vehicle damage, and high road maintenance costs. This project proposes a multi-modal pothole detection system combining real-time detection with YOLOv8, 3D point cloud quantification, and monocular depth estimation for volume calculation, providing actionable data for road maintenance authorities.
## 📂 Project Features

- Real-time pothole detection using YOLOv8 object detection models

- Machine learning based vibration-analysis pothole detection

- 3D point cloud-based pothole volume estimation using Alpha Shapes algorithm

- Depth estimation using MiDaS v3.1 for monocular depth-based size approximation

- Automatic repair cost estimation

- Edge-friendly lightweight deployment options for real-world integration

## 🔧 Technologies Used

- YOLOv8 (Ultralytics) - for real-time object detection

- PyTorch - core deep learning framework

- MiDaS v3.1 - monocular depth estimation

- Streamlit - UI for data upload, visualization and processing

- Matplotlib, Pandas - result visualization & reporting

- Roboflow - data augmentation and preprocessing
## 📊 Dataset

Custom YOLO Dataset (2000+ images):

- Roboflow pothole dataset

- YouTube video extracted frames (manually annotated)

- Additional open-source datasets
## 🧠 Model Architecture Summary

1️⃣ YOLOv8 Detection Pipeline

    Backbone: C2f & ELAN modules

    Neck: PANet + BiFPN for multi-scale feature fusion

    Head: Dense anchor-free object detection

    Trained on 720x720 resolution images for diverse real-world road conditions.

2️⃣ MiDaS v3.1 Depth Estimation Pipeline

    Encoder: BEiT-512 or SwinV2 for high resolution depth maps

    Inference resolution: 512x512

    Relative depth scaling for real-world size estimation (requires camera metadata)

3️⃣ Cost Estimation

    Repair cost = Volume (cm3) × INR cost factor

    Supports multiple camera models for accurate real-world unit conversion


## ⚙️ Installation Guide

```bash
# Clone the repo
git clone https://github.com/Mehwish4610/automatic-pothole-detection-system.git
cd pothole-detection-system

# Create virtual environment
python -m venv pothole-env
source pothole-env/bin/activate  # Linux
pothole-env\Scripts\activate.bat  # Windows

```
## 💻 Running The Application

``` bash
# Start Streamlit interface
streamlit run app.py
```

- Upload image / video

- Perform YOLOv8 detection

- Chose if you want to perform real-world scaling

- Generate repair cost estimate
## 📈 Performance Metrics

| Model   | mAP@0.5 | Inference Time |
|---------|---------|----------------|
YOLOv8n   |  0.88   |  12 ms/image
YOLOv8s   |  0.91   | 8.8 ms/image
YOLOv8m   |  0.911  | 8.8 ms/image
MiDaS v3.1|28% higher relative accuracy over v3.0|         |

## 🔮 Future Work

- Integration with autonomous repair robots

- Full real-time edge deployment

- Improved stereo camera pipelines for better 3D reconstruction

- Adaptive hyperparameter tuning via AutoML

- Large-scale city-wide monitoring dashboard
## Author

- [@Mehwish4610](https://github.com/Mehwish4610)


## Screenshots

Login Page
![App Screenshot](https://github.com/Mehwish4610/automatic-pothole-detection-system/blob/main/Screenshot%202025-06-12%20140524.png)


Main Dashboard
![App Screenshot](https://github.com/Mehwish4610/automatic-pothole-detection-system/blob/main/Screenshot%202025-06-12%20140611.png)



Detection Result
![App Screenshot](https://github.com/Mehwish4610/automatic-pothole-detection-system/blob/main/Screenshot%202025-05-07%20234541.png)


Detection Summary
![App Screenshot](https://github.com/Mehwish4610/automatic-pothole-detection-system/blob/main/Screenshot%202025-05-07%20234554.png)

Detection History Page
![App Screenshot](https://github.com/Mehwish4610/automatic-pothole-detection-system/blob/main/Screenshot%202025-06-12%20140630.png)

