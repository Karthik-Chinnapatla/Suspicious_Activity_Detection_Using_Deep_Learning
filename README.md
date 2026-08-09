# Suspicious Activity Detection System Using CNN & OpenCV

A modern, production-grade Django web application for automated surveillance video analysis, frame extraction, and real-time activity recognition using Convolutional Neural Networks (CNN) and OpenCV.

[![Live Demo]https://suspicious-activity-detection-using-deep.onrender.com

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11-brightgreen)
![Django](https://img.shields.io/badge/Django-5.2-success)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-blue)

---

## 🌟 Key Features

1. **Secure Admin Authentication**: Integrated with Django's built-in authentication system with protected dashboard, detection, and history pages.
2. **Interactive CCTV Control Center Dashboard**: Provides real-time stats (Total Videos Analyzed, Suspicious Threats, Normal Activity, Latest Detection Status).
3. **OpenCV Video Stream Processing**:
   - Frame extraction from surveillance footage (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
   - Configurable frame sampling strategy for high-performance processing vs accuracy tradeoff.
   - Real-time HUD overlay rendering (bounding box headers, color-coded status badges, frame counter, time code stamp).
4. **Real CNN Model Inference Engine**:
   - Model singleton loader inspecting `.h5` model architecture dynamically.
   - RGB frame normalization `(224, 224, 3)`.
   - Vectorized batch prediction for multi-class classification (**Fighting**, **Fire**, **Burglary**, **Shooting**, **Normal**).
5. **Detailed Result Breakdown**: Side-by-side player, confidence score gauge bar, frame rate, resolution, duration metrics.
6. **SQLite Detection History**: Full searchable and filterable activity logs with deletion cleanup of media assets.
7. **Synthetic Demo Video Generator**: Allows one-click generation of sample surveillance videos for immediate testing.

---

## 📁 Project Structure

```
Suspicious_Activity_Detection_Using_Deep_Learning/
│
├── manage.py                       # Django management utility
├── requirements.txt                 # Python dependencies
├── README.md                       # Complete documentation
├── create_admin.py                 # Initial superuser bootstrap script
├── test_detection_pipeline.py      # End-to-end verification script
│
├── config/                         # Main Django configuration
│   ├── __init__.py
│   ├── settings.py                 # App settings, media/static configuration
│   ├── urls.py                     # Main URL router
│   ├── wsgi.py                     # WSGI server entrypoint
│   └── asgi.py                     # ASGI server entrypoint
│
├── detection/                      # Core Django application
│   ├── migrations/                 # Database migrations
│   ├── templates/
│   │   └── detection/
│   │       ├── base.html           # Master layout template (Cyber CCTV theme)
│   │       ├── dashboard.html      # Surveillance control center dashboard
│   │       ├── detect.html         # Drag-and-drop video uploader
│   │       ├── result.html         # Detailed analysis result page
│   │       └── history.html        # Searchable detection history table
│   ├── static/
│   │   └── detection/
│   │       ├── css/
│   │       │   └── style.css       # Dark mode CCTV design system
│   │       └── js/
│   │           └── main.js         # Upload drag-and-drop & UI scripts
│   ├── models.py                   # SQLite Detection model schema
│   ├── views.py                    # Dashboard, upload, results & history views
│   ├── urls.py                     # App routing definitions
│   ├── forms.py                    # VideoUploadForm & format validation
│   ├── admin.py                    # Django Admin registration
│   ├── cnn_model.py                # Model loader & frame preprocessor
│   ├── video_processor.py         # OpenCV frame extractor & HUD renderer
│   └── utils.py                    # Stats aggregators & demo video generator
│
├── templates/
│   └── registration/
│       └── login.html              # Admin login portal card
│
├── media/                          # Uploaded and processed media storage
│   ├── uploads/                    # Raw uploaded surveillance footage
│   └── processed/                  # Output videos with AI visual overlays
│
└── models/                         # Trained CNN model directory
    ├── build_and_train_model.py    # Keras CNN model architecture builder
    └── suspicious_activity_model.h5# Trained CNN model file
```

---

## ⚙️ Technology Stack

- **Backend Framework**: Django 5.2 / Python 3.10
- **AI / Deep Learning**: TensorFlow 2.21 / Keras H5 Model
- **Computer Vision**: OpenCV (`cv2`) 5.0
- **Database**: SQLite3
- **Frontend**: Django Templates, HTML5, Vanilla JavaScript, Bootstrap 5, FontAwesome, Custom CSS Glassmorphism

---

## 🚀 Quick Setup & Running Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed on your machine.

### 2. Virtual Environment Setup & Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 3. Generate Trained CNN Model File
If `models/suspicious_activity_model.h5` does not exist, build it by running:
```bash
python models/build_and_train_model.py
```

### 4. Run Database Migrations
```bash
python manage.py makemigrations detection
python manage.py migrate
```

### 5. Create Admin Superuser
```bash
python create_admin.py
```
*Default Admin Credentials:*
- **Username**: `admin`
- **Password**: `admin123`

### 6. Launch Django Development Server
```bash
python manage.py runserver 8000
```
Open your web browser and navigate to: **`http://127.0.0.1:8000/`**

---

## 🧠 CNN Model Architecture & Workflow

The model uses a 4-Block Convolutional Neural Network architecture:
- **Input Tensor**: `(224, 224, 3)` RGB frame array
- **Conv2D + Batch Normalization + MaxPooling2D** blocks for feature extraction
- **Global Average Pooling** to flatten feature maps
- **Dense Head + Dropout (0.4)** to prevent overfitting
- **Softmax Activation Layer** outputting probability distribution across 5 activity categories:
  1. `Fighting` (Suspicious)
  2. `Fire` (Suspicious)
  3. `Burglary` (Suspicious)
  4. `Shooting` (Suspicious)
  5. `Normal` (Normal)

---

## 🛡️ OpenCV Video Processing Pipeline

1. **Video Ingestion**: `cv2.VideoCapture` loads the uploaded file and inspects resolution, framerate, and frame count.
2. **Frame Extraction & Sampling**: Extract frames at intervals defined by `sample_rate` (e.g. sample every 5th frame).
3. **Preprocessing**: Converts OpenCV BGR frames to RGB, resizes to `(224, 224)`, and normalizes pixel values to `[0.0, 1.0]`.
4. **Prediction Aggregation**: Runs CNN prediction across sampled frames and calculates maximum confidence scores.
5. **HUD Annotation**: OpenCV draws visual security overlays onto output frames.
6. **Video Encoding**: `cv2.VideoWriter` encodes the annotated frames into web-compatible MP4 format.

---

## 🧪 Automated Testing

Run the end-to-end pipeline test script:
```bash
python test_detection_pipeline.py
```

---

## 📜 License & Academic Usage
Created for Final-Year Academic Project: *"Suspicious Activity Detection Using Deep Learning"*.
