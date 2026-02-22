# 🚀 Waresight – AI-Powered Smart Surveillance & Analytics

Waresight is a real-time AI surveillance and crowd analytics system built using **YOLOv8, FastAPI, and React**. It detects and tracks people in video streams, generates movement heatmaps, monitors crowd density, and visualizes insights through a live dashboard.

---

## 🧠 Problem

Traditional CCTV systems only record footage. They do not provide intelligent, real-time insights. Manual monitoring is inefficient and not scalable.

Waresight converts raw video into actionable analytics.

---

## ✨ Core Features

- Real-time person detection (YOLOv8)
- Frame-by-frame tracking logic
- Crowd count monitoring
- Movement heatmap generation
- Entry/Exit analysis
- Live WebSocket streaming
- Interactive React dashboard

---

## 🏗️ Architecture


Video Input
↓
YOLOv8 Detection
↓
Tracking + Analytics Engine
↓
FastAPI Backend
↓
WebSocket
↓
React Dashboard


---

## 🛠️ Tech Stack

**Frontend:** React, Vite, TailwindCSS  
**Backend:** FastAPI, Uvicorn, OpenCV, NumPy  
**Model:** YOLOv8 (Ultralytics)

---

# ⚙️ Setup Instructions

## 🔹 Prerequisites
- Python 3.9+
- Node.js 18+
- Git

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/nishaanshu303/waresight.git
cd waresight
