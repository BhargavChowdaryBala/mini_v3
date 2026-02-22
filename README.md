# Bus Monitoring & ANPR System

A professional, real-time Bus Monitoring and Automatic Number Plate Recognition (ANPR) system featuring a high-performance dashboard, mobile-camera integration, and intelligent logging.

## ✨ Features

- **Real-time ANPR**: Detects and tracks vehicle license plates using YOLOv8 and ByteTrack.
- **OCR Integration**: Extracts registration text using PaddleOCR.
- **Interactive Dashboard**: Premium dark-mode UI with glassmorphic effects.
- **Mobile Vision**: Seamlessly switch between fixed CCTV feeds and mobile-phone back camera processing.
- **Intelligent Logging**: Automatic entry/exit tracking with MongoDB integration.
- **Guidelines**: Built-in perspective guidelines for precision alignment.

## 🚀 Tech Stack

- **Backend**: Python, Flask, OpenCV, Ultralytics (YOLOv8), PaddleOCR.
- **Frontend**: React, Vite, Framer Motion, Lucide React, Axios.
- **Database**: MongoDB.

## 📦 Installation

### Backend
1. `cd backend`
2. Create virtual environment: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows)
4. Install: `pip install -r requirements.txt`
5. Run: `python main.py`

### Frontend
1. `cd frontend`
2. Install: `npm install`
3. Run: `npm run dev`

## 🌍 Deployment

- **Backend**: Ready for [Render](https://render.com/).
- **Frontend**: Ready for [Vercel](https://vercel.com/).

---
© 2026 AI Transportation Solutions - IEEE Research Demo
