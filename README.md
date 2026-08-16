# LocateMe — Real-Time Missing Person Detection in Public Feeds

> **Controlled Prototype Notice**: LocateMe is an AI-assisted research and hackathon prototype designed strictly for evaluation on authorized, consenting benchmark datasets and sample video feeds. It is **not** an unrestricted public surveillance system.

---

## 1. Project Objective

LocateMe accelerates missing-person search and rescue operations by automatically screening video frames, CCTV feeds, and test images against registered missing-person reference galleries. When potential matches are detected, the system produces actionable candidate alerts with similarity confidence metrics and detection metadata for human operator verification.

---

## 2. Complete Architecture & Tech Stack

```
┌────────────────────────────────────────────────────────┐
│     Registered Missing Person Gallery (data/registered)│
└───────────────────────────┬────────────────────────────┘
                            │ (Offline Precomputation)
                            ▼
┌────────────────────────────────────────────────────────┐
│     512-D Embedding Matrix Cache (gallery_cache.npz)   │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────┐
│                  VIDEO / CCTV INGESTION                │
│    (MP4, AVI, RTSP Stream, or Live Webcam Feed)        │
└───────────────────────────┬────────────────────────────┘
                            │ (Configurable Frame Sampling: e.g., 5 frames)
                            ▼
┌────────────────────────────────────────────────────────┐
│           MTCNN Multi-Face Detection & Alignment       │
│                (Crop & Scale to 160x160)               │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│        InceptionResnetV1 Feature Extractor             │
│            (Pretrained on VGGFace2)                    │
│            Output: Unit-Norm 512-D Vector              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│        Fast Vectorized 1-to-N Gallery Screening        │
│          Cosine Similarity: S = G · q / ||q||          │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌───────────────────────────┐   ┌────────────────────────┐
│ Similarity >= Threshold   │   │ Similarity < Threshold │
│ (Default: 0.68)           │   │                        │
│ [POTENTIAL MATCH]         │   │ [UNRECOGNIZED FACE]    │
│ • Green Bounding Box      │   │ • Gray Bounding Box    │
│ • Log Match Event + Crop  │   │ • Background Tracking  │
└─────────────┬─────────────┘   └────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────┐
│   OPERATOR INTERFACES & SERVING LAYERS                 │
│   • Streamlit Web Dashboard  (app/dashboard.py)        │
│   • FastAPI REST Backend API (api/main.py)             │
│   • Video Screening CLI      (process_video.py)        │
│   • Pairwise Image CLI       (run_demo.py)             │
└────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
LocateMe/
├── api/
│   ├── __init__.py          # FastAPI package init
│   ├── main.py              # REST API endpoints & static file mounts
│   └── schemas.py           # Pydantic request/response schemas
├── app/
│   └── dashboard.py         # Streamlit Operator & Hackathon Dashboard
├── ml/
│   ├── __init__.py          # Core package exports
│   ├── face_detector.py     # MTCNN face detection & multi-face alignment
│   ├── embedding.py         # InceptionResnetV1 512-D embedding extraction
│   ├── matcher.py           # Cosine similarity & threshold screening
│   ├── gallery.py           # Gallery caching & vectorized 1-to-N search
│   └── video_processor.py   # Video frame decoding, annotation & event logging
├── data/
│   ├── registered/          # Reference missing person photos
│   │   ├── person_a_ref.jpg
│   │   └── person_b_ref.jpg
│   ├── test_images/         # Query test photos & static CCTV frames
│   ├── test_videos/         # Benchmark video clips & screening outputs
│   │   ├── sample_cctv_feed.mp4
│   │   ├── cctv_annotated_feed.mp4
│   │   ├── cctv_match_report.json
│   │   └── detected_crops/  # Cropped candidate sightings
│   └── gallery_cache.npz    # Precomputed gallery embeddings cache
├── tests/
│   ├── test_pipeline.py       # Phase 1 unit & pipeline tests
│   ├── test_video_pipeline.py # Phase 2 video & gallery test suite
│   └── test_api.py            # Phase 3 FastAPI endpoint test suite
├── run_demo.py              # CLI tool for pairwise image comparison
├── process_video.py         # CLI tool for CCTV/video feed screening
├── requirements.txt         # Pinned reproducible python dependencies
├── .env.example             # Environment variable configuration template
├── .gitignore               # Ignored cache, secrets, and environments
└── README.md                # Project documentation
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Windows, Linux, or macOS

### Step 1: Navigate to Project
```bash
cd LocateMe
```

### Step 2: Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Running Automated Tests

Run the entire automated test suite with `pytest`:
```bash
pytest tests/ -v
```

Expected output:
```
tests/test_api.py::test_health_endpoint PASSED                           [  8%]
tests/test_api.py::test_list_gallery_endpoint PASSED                     [ 16%]
tests/test_api.py::test_screen_image_endpoint PASSED                     [ 25%]
tests/test_api.py::test_screen_image_blank PASSED                        [ 33%]
tests/test_pipeline.py::test_face_detector_invalid_input PASSED          [ 41%]
tests/test_pipeline.py::test_face_detector_no_face PASSED                [ 50%]
tests/test_pipeline.py::test_embedding_dimensions_and_norm PASSED        [ 58%]
tests/test_pipeline.py::test_matcher_cosine_similarity PASSED            [ 66%]
tests/test_pipeline.py::test_matcher_threshold_evaluation PASSED         [ 75%]
tests/test_video_pipeline.py::test_gallery_manager_loading_and_search PASSED [ 83%]
tests/test_video_pipeline.py::test_gallery_cache_saving_and_loading PASSED [ 91%]
tests/test_video_pipeline.py::test_video_processor_screening PASSED      [100%]

======================= 12 passed in 19.67s =======================
```

---

## 6. Running Applications & Tools

### 1. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```
Opens the web dashboard in your browser (`http://localhost:8501`) featuring:
- **Missing Persons Gallery**: Browse active entries and register new persons with drag-and-drop portraits.
- **Photo Screening**: Upload test photos and view side-by-side reference comparisons with confidence gauges.
- **CCTV Video Analysis**: Upload surveillance video clips, view live detection progress, watch annotated playback with HUD overlays, inspect candidate sighting timelines, and download JSON reports.
- **Technical Specs**: Threshold calibration guidelines and ethical notices.

### 2. Launch FastAPI REST Backend
```bash
uvicorn api.main:app --reload --port 8000
```
Interactive Swagger API documentation is available at `http://127.0.0.1:8000/docs`.

### 3. CLI CCTV Video Screening Tool
```bash
python process_video.py --video data/test_videos/sample_cctv_feed.mp4 --threshold 0.68 --frame-step 5
```

### 4. CLI Pairwise Image Comparison Tool
```bash
python run_demo.py --reference data/registered/person_a_ref.jpg --test data/test_images/person_a_cctv.jpg
```

---

## 7. Experimental Threshold Calibration

- **`0.85 - 1.00`**: High confidence match. Extremely strong facial similarity.
- **`0.68 - 0.84`**: Moderate candidate match. Recommended screening range for human review.
- **`0.55 - 0.67`**: Borderline candidate. High false-positive rate; requires cautious review.
- **`< 0.55`**: Non-matching individuals.
