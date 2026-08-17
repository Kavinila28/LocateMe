# LocateMe — AI-Assisted Missing Person Screening

LocateMe is an AI-assisted missing-person screening platform designed to help authorized operators compare faces from photographs and surveillance footage against a registered reference gallery.

The system combines **MTCNN face detection**, **InceptionResnetV1 facial embeddings**, **cosine similarity**, a **FastAPI backend**, a **Streamlit operator dashboard**, and **Supabase-backed storage** to provide an end-to-end prototype for controlled missing-person search assistance.

> **Important:** LocateMe is a controlled prototype and human-in-the-loop screening system. Similarity scores indicate algorithmic feature proximity and do **not** constitute positive identification.

---

## 🚀 Live Demo

### Dashboard

**https://locateme-dashboard.onrender.com**

### Backend API

**https://locateme-x7h2.onrender.com**

### API Documentation

**https://locateme-x7h2.onrender.com/docs**

---

## ✨ Key Features

* 🗂️ **Missing Persons Gallery**

  * Register reference photographs
  * Maintain a searchable gallery of registered individuals
  * Precompute facial embeddings for faster screening

* 📸 **Photo & Snapshot Screening**

  * Upload a surveillance photograph
  * Detect one or multiple faces
  * Generate 512-dimensional facial embeddings
  * Compare detected faces against the registered gallery
  * Display similarity scores and confidence tiers

* 📹 **CCTV Video Analysis**

  * Upload recorded surveillance footage
  * Sample video frames for processing
  * Detect faces throughout the video
  * Compare detected faces against registered candidates
  * Generate an annotated surveillance video
  * Produce a downloadable JSON screening report
  * Display a detection timeline

* ⚙️ **Configurable Screening**

  * Adjustable cosine similarity threshold
  * Configurable video frame sampling rate
  * Device and gallery status monitoring

* ☁️ **Cloud-Ready Architecture**

  * FastAPI REST backend
  * Streamlit dashboard
  * Supabase integration
  * Render deployment

* 🔐 **Ethical Boundaries**

  * Human-in-the-loop workflow
  * Designed for authorized missing-person operations
  * No claim of definitive biometric identification

---

## 🧠 AI / ML Pipeline

LocateMe uses the following processing pipeline:

```text
Input Image / Video
        │
        ▼
   Face Detection
      MTCNN
        │
        ▼
 Face Alignment & Crop
        │
        ▼
InceptionResnetV1
   VGGFace2 Weights
        │
        ▼
512-D Face Embedding
        │
        ▼
Cosine Similarity
        │
        ▼
Registered Gallery
        │
        ▼
Candidate Screening Result
```

### Face Detection

**MTCNN (Multi-task Cascaded Convolutional Networks)** is used to locate and align faces before feature extraction.

### Face Embeddings

**InceptionResnetV1**, pretrained on **VGGFace2**, generates normalized **512-dimensional facial embeddings**.

### Similarity Measurement

Candidate comparison uses cosine similarity:

```text
Cosine Similarity(u, v)
        = (u · v) / (||u|| ||v||)
```

The gallery embeddings are stored in a matrix and compared against the query embedding using vectorized operations.

---

## 📊 Screening Thresholds

The dashboard provides configurable similarity thresholds.

| Similarity    | Interpretation                                |
| ------------- | --------------------------------------------- |
| `0.85 – 1.00` | High-confidence candidate                     |
| `0.68 – 0.84` | Moderate candidate — human review recommended |
| `0.55 – 0.67` | Borderline candidate                          |
| `< 0.55`      | Non-matching individual                       |

These thresholds are intended for controlled evaluation and should not be interpreted as proof of identity.

---

## 🏗️ System Architecture

```text
                         ┌─────────────────────────┐
                         │   LocateMe Dashboard    │
                         │       Streamlit         │
                         │                         │
                         │ • Gallery               │
                         │ • Photo Screening       │
                         │ • CCTV Analysis         │
                         │ • Technical/Ethics      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      FastAPI API        │
                         │                         │
                         │ • REST Endpoints        │
                         │ • Gallery Operations    │
                         │ • Health Monitoring     │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌────────────────────┐             ┌────────────────────┐
          │     ML Pipeline    │             │      Supabase      │
          │                    │             │                    │
          │ MTCNN              │             │ Database           │
          │ InceptionResnetV1  │             │ Storage            │
          │ Embeddings         │             │ pgvector           │
          │ Matching           │             │                    │
          └────────────────────┘             └────────────────────┘
```

---

## 📁 Project Structure

```text
LocateMe/
│
├── api/
│   └── main.py
│
├── app/
│   └── dashboard.py
│
├── ml/
│   ├── embedding.py
│   ├── face_detector.py
│   ├── gallery.py
│   ├── matcher.py
│   └── video_processor.py
│
├── data/
│   ├── registered/
│   ├── test_images/
│   └── test_videos/
│
├── supabase/
│
├── tests/
│
├── process_video.py
├── run_demo.py
├── requirements.txt
├── runtime.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🛠️ Technology Stack

### Frontend / Dashboard

* Python
* Streamlit

### Backend

* FastAPI
* Uvicorn

### Machine Learning

* PyTorch
* torchvision
* facenet-pytorch
* MTCNN
* InceptionResnetV1
* OpenCV
* NumPy
* Pillow

### Data / Cloud

* Supabase
* PostgreSQL
* pgvector

### Deployment

* Render

### Testing

* Pytest

---

## ⚙️ Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/Kavinila28/LocateMe.git
cd LocateMe
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

Do **not** commit `.env` or any secret credentials to GitHub.

---

## ▶️ Running the Application

### Start the FastAPI backend

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Start the Streamlit dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard will normally be available at:

```text
http://localhost:8501
```

---

## 🔌 API

The FastAPI backend exposes endpoints for application health and gallery operations.

Example health endpoint:

```text
GET /health
```

Example gallery endpoint:

```text
GET /api/v1/gallery
```

Interactive API documentation is available through FastAPI Swagger UI:

```text
https://locateme-x7h2.onrender.com/docs
```

---

## 📹 Video Processing

LocateMe supports recorded surveillance footage in formats including:

* `.mp4`
* `.avi`
* `.mov`

The video processor can:

1. Read the input video.
2. Sample frames according to the configured frame step.
3. Detect faces.
4. Generate embeddings.
5. Compare embeddings against the gallery.
6. Record candidate sighting events.
7. Generate an annotated output video.
8. Export a JSON screening report.

Example report information includes:

```text
Timestamp
Frame Index
Candidate Name
Similarity Score
Face Detection Confidence
Confidence Tier
```

---

## 🧪 Testing

Run the complete test suite with:

```bash
pytest
```

Or use the provided Windows batch script:

```powershell
.\run_all_tests.bat
```

---

## 🔒 Privacy & Ethical Considerations

LocateMe is designed around a controlled, authorized workflow.

The system should be used only when appropriate authorization exists for the photographs, surveillance footage, and registered-person data being processed.

The matching result is an **algorithmic similarity measurement**, not a verified identity.

Potential sources of error include:

* Lighting conditions
* Camera angle
* Image resolution
* Occlusion
* Facial expression
* Age-related appearance changes
* Detection errors
* False positives
* False negatives

Therefore, any potential candidate should undergo appropriate human verification and operational review.

---

## 🎯 Intended Use

LocateMe is intended as a:

* Hackathon prototype
* Research and experimentation platform
* Demonstration of computer vision pipelines
* Human-assisted missing-person screening system
* Educational AI/ML application

It is **not intended to replace trained investigators, law-enforcement procedures, or formal identity verification systems.**

---

## 📌 Current Deployment

LocateMe is deployed using Render:

```text
Streamlit Dashboard
        │
        ▼
https://locateme-dashboard.onrender.com

FastAPI Backend
        │
        ▼
https://locateme-x7h2.onrender.com

API Documentation
        │
        ▼
https://locateme-x7h2.onrender.com/docs
```

---

## 👩‍💻 Author

**Kavinila Prabhakaran**

Computer Science & Engineering — Artificial Intelligence & Machine Learning

GitHub:
https://github.com/Kavinila28

LinkedIn:
https://linkedin.com/in/kavinila-prabhakaran-079319332

---

## 📜 License

This project is intended for educational, research, and controlled demonstration purposes.
