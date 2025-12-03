# CSC 8830: Computer Vision Application

**Author:** Shruthi Ledalla

A production-grade, web-based platform featuring real-time tracking, image processing, panoramic stitching, and geometric calibration. This application is engineered for cloud performance, utilizing advanced optimization techniques to run heavy AI models on serverless infrastructure.

---

## 🔗 Quick Links

| Resource | Link |
| :--- | :--- |
| **🚀 Live Website** | [View Website](https://computer-vision-production-4bab.up.railway.app) |
| **💻 GitHub Repo** | [View Source Code](https://github.com/shruthi0103/Computer-Vision) |
| **🎥 Demo Video** | [Watch Walkthrough](https://drive.google.com/drive/folders/1GzJfdRxJs3Ik8vpEBkWoy7ECRUFMo8Cr?usp=sharing)|

---

## ✨ Modules & Functionality

### Module 1 – 📏 Camera Calibration & Dimension Estimation
- Computes camera intrinsic parameters (`fx`, `fy`, `cx`, `cy`).
- **Perspective projection–based** real-world object measurement.
- Click-based point selection for dimensional estimation.
- Outputs all measurements in **centimeters (cm)**.
- Forms the geometric foundation for stereo and measurement modules.

### Module 2 – 🧩 Template Matching & Image Reconstruction
- **Multi-scale** template matching.
- Robust detection under scale and rotation variations.
- **Image reconstruction & restoration**.
- Handles partial occlusions using reconstruction logic.
- 📊 Evaluation metrics – https://github.com/shruthi0103/Computer-Vision/blob/main/Evaluation_mod2.txt

This module evaluates the performance of multi-object template matching using Center Localization Error and Scale Deviation as primary metrics. The evaluation measures how accurately the detected bounding boxes align with the ground-truth object locations and sizes across multiple real-world objects. The results show high accuracy for rigid, well-textured objects, while performance degrades under clutter, occlusion, and large scale variations.

### Module 3 – 🧠 Feature Analysis & Boundary Detection
Includes:
- Gradient Analysis (Sobel X & Y), Magnitude & Orientation.
- Laplacian of Gaussian (LoG).
- Edge Detection, Corner Detection, and ArUco Marker Detection.
- Boundary Detection using Convex Hulls and SAM2 Mask-Based Segmentation.

This module provides the full **low-level vision + geometric boundary extraction pipeline**.

### Module 4 – 🖼️ Panorama Stitching & SIFT Matching
- Automatic **panoramic image stitching**.
- SIFT feature extraction and correspondence matching.
- Homography estimation using RANSAC.
- Seamless warped panorama generation with border cropping.

### Module 5 – 🎥 Motion Tracking (Without Tracker)
- Frame-to-frame motion detection.
- Hand/object exclusion masking.
- Bounding box estimation without pre-trained trackers.
- Motion stabilization via adaptive thresholding.

### Module 6 – 🏃 Motion Tracking with Trackers & SAM2
- CSRT tracker–based object tracking.
- SAM2 segmentation-guided tracking.
- Robust tracking under occlusion.
- Mask-refined bounding boxes with real-time performance.

### Module 7 – 🧍 Pose & Hand Tracking + 📐 Calibrated Stereo Estimation
**Pose & Hand Tracking**
- **MediaPipe Holistic** pipeline tracking 33 pose landmarks and 21 hand landmarks (left/right).
- Live webcam streaming with automatic **CSV logging** and browser-based download.

**Calibrated Stereo Estimation**
- **Manual stereo triangulation**.
- User inputs: `fx`, `fy`, `cx`, `cy`, `baseline` (cm).
- **Click-based 4-point correspondence** to compute 3D (X, Y, Z) coordinates and object dimensions.

---

## ⚙️ Technical Architecture & Cloud Optimizations

This project is deployed on **Railway (Free Tier)**, which imposes a strict **512MB RAM** limit. To run heavy AI models like MediaPipe and OpenCV in this environment, several architectural optimizations were implemented:

### 1. Lazy Loading & Memory Management
Instead of loading all libraries at startup (which causes a SIGKILL/Crash), libraries like `cv2`, `numpy`, and `mediapipe` are imported **lazily** inside functions.
* **Result:** Server boot time reduced from 20s to <1s.
* **Stability:** Memory usage stays low until a specific module is requested.

### 2. HTTP POST Streaming (Module 7)
Standard WebSockets (`socket.io`) proved unstable on restrictive firewalls.
* **Solution:** Implemented a robust "Fetch Loop" architecture.
* **Flow:** Client captures frame → Encodes to Base64 (0.5 quality) → HTTP POST → Server Processes (MediaPipe) → Response.
* **Benefit:** Zero connection drops and better error handling.

### 3. Explicit Garbage Collection
Python's automatic garbage collection is sometimes too slow for video processing.
* **Optimization:** Added manual `gc.collect()` calls after every frame processing to prevent RAM spikes.

---

## 🌐 Deployment Status

**Platform:** Railway  
**Status:** ✅ Live & Production Ready


### Auto-Deployment

Every push to `main` branch automatically deploys to Railway:

```bash
git add .
git commit -m "Your changes"
git push origin main
```

Railway will rebuild and deploy in ~3-5 minutes.

---

## 💻 Local Development

### Prerequisites

- Python 3.9+
- Git

### Setup

1. **Clone repository:**
```bash
git clone [https://github.com/shruthi0103/Computer-Vision.git](https://github.com/shruthi0103/Computer-Vision.git)
cd Computer-Vision
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run application:**
```bash
python app.py
```

5. **Access locally:**
```
http://localhost:5000
```

---


## 📁 Project Structure



```text
CV-Application/
├── app.py                  # Main Entry Point: Contains all Flask routes and logic
├── requirements.txt        # Optimized dependency list (No heavy dlib/GUI libs)
├── Dockerfile              # Lightweight Python 3.9 Bookworm container config
├── scene.py                # Helper module for Template Matching logic
├── templates/              # HTML Frontend
│   ├── home.html           # Main Dashboard
│   ├── face_unlock.html    # Landing/Login Page
│   ├── index.html          # Module 1 
│   ├── module2.html        # Module 2 
│   ├── module3.html        # Module 3 
│   ├── module4.html        # Module 4
│   ├── module6.html        # Module 6 
│   └── module7.html        # Module 7 
└── static/                 # Static Assets
    ├── css/                # Stylesheets
    ├── marker/             # Marker images
    ├── mod2/               # Template datasets
    └── js/                 # Client-side scripts

---

## 🛠️ Technical Stack

**Backend:**
- Flask 3.0.3
- Python 3.9
- Gunicorn 21.2.0

**Frontend:**
- HTML5, CSS3, JavaScript(Fetch API)

**Deployment:**
- Docker
- Railway (auto-deployment)

---
2025 Shruthi Ledalla | Educational Project for Computer Vision | **Deployed on Railway** 
