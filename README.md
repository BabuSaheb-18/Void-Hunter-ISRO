# 🚀 Void Hunter

> AI-powered Infrared Image Enhancement and Colorization System for ISRO Smart India Hackathon 2026

---

## 📌 Problem Statement

**Problem Statement 10**

Infrared Image Colorization and Enhancement for Improved Object Interpretation.

The objective is to enhance infrared satellite imagery by improving image quality and generating realistic RGB representations while preserving semantic information.

---

## 🎯 Objectives

- Enhance infrared image quality
- Perform realistic IR-to-RGB colorization
- Preserve semantic information
- Improve object interpretation
- Generate downloadable analysis reports

---

## ✨ Features

- Modern ISRO-inspired web interface
- Infrared image upload
- AI-based image enhancement
- IR to RGB colorization
- Image comparison
- Automatic PDF report generation
- Deep Learning inference
- Responsive UI

---

## 🧠 AI Model

Model Architecture

- U-Net
- Custom Loss Function
- LPIPS Perceptual Loss
- SSIM Loss

Evaluation Metrics

- PSNR
- SSIM
- MSE
- LPIPS
- Training Loss
- Validation Loss

---

## 📂 Project Structure

```
Void-Hunter-ISRO
│
├── backend/
├── dataset/
├── docs/
├── logs/
├── models/
├── output/
├── scripts/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Void-Hunter-ISRO.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python backend/app.py
```

---

## 📊 Dataset

Dataset Used

- Landsat 8 / Landsat 9
- USGS Earth Explorer

Dataset contains

- Infrared Images
- RGB Images
- Thermal Bands

---

## 🤖 Training

Training is performed using PyTorch.

```bash
python models/train.py
```

Best model is saved as

```
models/checkpoints/best_model.pth
```

---

## 🔍 Inference

```bash
python models/predict.py
```

or use the Flask web interface.

---

## 🛠 Technology Stack

Frontend

- HTML5
- CSS3
- JavaScript

Backend

- Flask

Deep Learning

- PyTorch
- TorchVision
- OpenCV
- NumPy
- Pillow

Utilities

- ReportLab
- Matplotlib
- Scikit-Image

---

## 📈 Evaluation Metrics

- PSNR
- SSIM
- MSE
- LPIPS

---

## 🚀 Future Improvements

- Real-time inference
- Transformer-based architecture
- Multi-spectral support
- Larger satellite datasets
- Cloud deployment

---

## 👥 Team

**Team Name**

Void Hunter

Smart India Hackathon 2026

---

## 📄 License

This project is developed for educational and research purposes under the Smart India Hackathon 2026.