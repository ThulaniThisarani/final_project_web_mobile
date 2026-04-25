<div align="center">

<img src="https://img.shields.io/badge/🌿-CinnaGuard-2d7a4f?style=for-the-badge&logoColor=white" alt="CinnaGuard" width="300"/>

# 🌿 CinnaGuard — Cinnamon Leaf Disease Detector

*AI-powered cinnamon leaf disease detection using deep learning*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react&logoColor=black)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6?style=flat-square&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=flat-square&logo=mysql&logoColor=white)](https://mysql.com)

---

Upload a photo of a cinnamon leaf → Get an instant AI-powered disease diagnosis 🔬

</div>

---

## 📖 Table of Contents

- [✨ Features](#-features)
- [🦠 Detectable Diseases](#-detectable-diseases)
- [🏗️ Architecture](#️-architecture)
- [🗂️ Project Structure](#️-project-structure)
- [⚙️ Backend Setup](#️-backend-setup)
- [🌐 Frontend Setup](#-frontend-setup)
- [🔌 API Reference](#-api-reference)
- [🤖 Model Details](#-model-details)
- [🛠️ Tech Stack](#️-tech-stack)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 *AI Disease Detection* | EfficientNet-based model classifies 4 cinnamon leaf diseases with confidence scores |
| 📊 *Severity Rating* | Automatically grades disease severity (Low / Medium / High) |
| 🕐 *Prediction History* | Full log of past scans, filterable per user |
| 📈 *Statistics Dashboard* | Visual analytics on disease distribution and scan counts |
| 👤 *User Auth* | Register & login with hashed password authentication |
| ⚡ *Fast API* | Sub-second predictions via FastAPI + Uvicorn |
| 🌓 *Responsive UI* | Mobile-friendly React frontend with dark/light mode |

---

## 🦠 Detectable Diseases

CinnaGuard can identify the following cinnamon leaf conditions:

<div align="center">

| 🟤 Black Sooty Mold | 🟠 Blight Disease | 🟡 Leaf Gall Disease | 🟡 Yellow Leaf Spots |
|:---:|:---:|:---:|:---:|
| Fungal coating on leaf surface caused by honeydew-secreting insects | Rapid browning & wilting of leaves and shoots | Abnormal growths or galls on the leaf tissue | Chlorotic yellowing patches, often nutrient-related |

</div>

---

## 🏗️ Architecture


┌──────────────────────┐         ┌──────────────────────────────────┐
│                      │         │                                  │
│   React + TypeScript │ ──────► │   FastAPI (Python)               │
│   (Vite + Tailwind)  │  HTTP   │                                  │
│   shadcn/ui          │ ◄────── │   ├── EfficientNet Model (.keras) │
│                      │  JSON   │   ├── OpenCV Image Preprocessing  │
└──────────────────────┘         │   └── MySQL Database             │
                                 │                                  │
                                 └──────────────────────────────────┘


---

## 🗂️ Project Structure


📦 CinnaGuard/
├── 🐍 python-backend/
│   ├── main.py                  # FastAPI app — all routes & logic
│   ├── app.py                   # Single-image CLI prediction script
│   ├── modeldiagnostic.py       # Model evaluation & diagnostic tools
│   ├── test_api.py              # API endpoint tests
│   ├── database_setup.sql       # MySQL schema
│   ├── requirements.txt         # Python dependencies
│   ├── .env                     # Environment variables (not committed)
│   ├── best_cinnamon_model.keras # Trained Keras model
│   └── model.weights.h5         # Model weights
│
└── 🌐 web-site/
    ├── src/
    │   ├── App.tsx              # Root router
    │   ├── pages/
    │   │   ├── Index.tsx        # Main dashboard
    │   │   └── Auth.tsx         # Login / Register
    │   └── components/
    │       ├── ImageUploader.tsx
    │       ├── PredictionResult.tsx
    │       ├── DiseaseInfoCard.tsx
    │       ├── HistoryCard.tsx
    │       ├── StatsCard.tsx
    │       └── Header.tsx
    ├── package.json
    ├── vite.config.ts
    └── tailwind.config.ts


---

## ⚙️ Backend Setup

### Prerequisites

- Python 3.10+
- MySQL 8.0+
- pip

### 1. Clone the repository

bash
git clone https://github.com/your-username/cinnaguard.git
cd cinnaguard/python-backend


### 2. Create & activate virtual environment

bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate


### 3. Install dependencies

bash
pip install -r requirements.txt


### 4. Configure the database

bash
mysql -u root -p < database_setup.sql


### 5. Set environment variables

Create a .env file in python-backend/:

env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=cinnamon_db
MODEL_PATH=./best_cinnamon_model.keras


### 6. Run the API server

bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000


> 🚀 API is now live at http://localhost:8000  
> 📖 Interactive docs at http://localhost:8000/docs

---

## 🌐 Frontend Setup

### Prerequisites

- Node.js 18+
- npm or bun

### 1. Navigate to the web-site directory

bash
cd cinnaguard/web-site


### 2. Install dependencies

bash
npm install
# or
bun install


### 3. Start the development server

bash
npm run dev


> 🌐 Frontend is now live at http://localhost:5173

### 4. Build for production

bash
npm run build


---

## 🔌 API Reference

Base URL: http://localhost:8000

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|:---:|
| GET | / | Health check & model status | ❌ |
| POST | /register | Create a new user account | ❌ |
| POST | /login | Authenticate & receive session | ❌ |
| POST | /predict | Upload leaf image → get diagnosis | ✅ |
| GET | /history | Retrieve past prediction history | ✅ |
| GET | /stats | Get disease statistics | ✅ |
| DELETE | /history/{id} | Delete a prediction record | ✅ |
| POST | /reload-model | Hot-reload the ML model | ✅ |

### Example: Predict Disease

bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@leaf_photo.jpg" \
  -F "user_id=1"


*Response:*
json
{
  "disease": "Blight_Disease",
  "confidence": 0.9423,
  "severity": "High",
  "all_probabilities": {
    "Black_Sooty_Mold": 0.0211,
    "Blight_Disease": 0.9423,
    "Leaf_Gall_Disease": 0.0189,
    "Yellow_leaf_spots": 0.0177
  },
  "timestamp": "2026-01-04T11:12:00"
}


---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| *Architecture* | EfficientNet (transfer learning) |
| *Input Size* | 224 × 224 px |
| *Output Classes* | 4 disease categories |
| *Framework* | TensorFlow / Keras |
| *Preprocessing* | EfficientNet standard normalization |
| *Format* | .keras (SavedModel compatible) |

The model uses EfficientNet as a backbone with a custom classification head trained on cinnamon leaf disease imagery. A custom CompatibleRandomFlip layer ensures cross-platform compatibility with different TensorFlow versions.

---

## 🛠️ Tech Stack

<div align="center">

### 🐍 Backend

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-222222?style=for-the-badge&logo=gunicorn&logoColor=white)

### 🌐 Frontend

![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)
![shadcn/ui](https://img.shields.io/badge/shadcn%2Fui-000000?style=for-the-badge&logo=shadcnui&logoColor=white)
![React Query](https://img.shields.io/badge/React_Query-FF4154?style=for-the-badge&logo=reactquery&logoColor=white)

</div>

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. *Fork* the repository
2. *Create* a feature branch: git checkout -b feature/my-feature
3. *Commit* your changes: git commit -m 'Add my feature'
4. *Push* to the branch: git push origin feature/my-feature
5. *Open* a Pull Request

Please make sure to update tests as appropriate.

---

## 📄 License

This project is licensed under the *MIT License* — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with 💚 for Sri Lankan cinnamon farmers

Helping protect one of Sri Lanka's most valuable crops through AI

⭐ *Star this repo if you found it useful!* ⭐

</div>
