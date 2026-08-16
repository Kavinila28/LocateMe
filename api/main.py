"""
LocateMe — FastAPI REST Backend Service (Phase 4 Dual-Mode)
Provides missing person gallery management, photo screening, and CCTV video processing.
Supports both Supabase PostgreSQL + pgvector cloud persistence and offline local fallback.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Ensure project root is in sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    CandidateMatch,
    DetectedFaceScreening,
    GalleryListResponse,
    HealthResponse,
    ImageScreeningResponse,
    MatchEventSchema,
    PersonRegistrationResponse,
    PersonSummary,
    VideoScreeningResponse,
)
from ml.embedding import FaceEmbeddingGenerator, get_default_generator
from ml.face_detector import FaceDetector, get_default_detector
from ml.gallery import GalleryManager, RegisteredPerson
from ml.matcher import DEFAULT_SIMILARITY_THRESHOLD, DISCLAIMER_TEXT
from ml.supabase_client import get_supabase_client
from ml.video_processor import VideoProcessor, VideoProcessingSummary

# App initialization
app = FastAPI(
    title="LocateMe — Missing Person Detection API",
    description="RESTful AI-assisted missing person screening service with Supabase pgvector & local storage.",
    version="1.0.0",
)

# Enable CORS for local dashboards and frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directories exist
DATA_DIR = PROJECT_ROOT / "data"
REG_DIR = DATA_DIR / "registered"
TEST_IMG_DIR = DATA_DIR / "test_images"
TEST_VID_DIR = DATA_DIR / "test_videos"

for d in [DATA_DIR, REG_DIR, TEST_IMG_DIR, TEST_VID_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Mount static media directories for local mode
app.mount("/media/registered", StaticFiles(directory=str(REG_DIR)), name="registered_media")
app.mount("/media/test_videos", StaticFiles(directory=str(TEST_VID_DIR)), name="test_videos_media")

# Pipeline singletons
_detector: Optional[FaceDetector] = None
_generator: Optional[FaceEmbeddingGenerator] = None
_gallery: Optional[GalleryManager] = None


def get_pipeline():
    """Lazily initialize and return the ML pipeline components."""
    global _detector, _generator, _gallery
    if _detector is None:
        _detector = get_default_detector()
    if _generator is None:
        _generator = get_default_generator()
    if _gallery is None:
        _gallery = GalleryManager(
            gallery_dir=REG_DIR,
            detector=_detector,
            generator=_generator,
            supabase_client=get_supabase_client(),
        )
    return _detector, _generator, _gallery


def _resolve_image_url(person: RegisteredPerson) -> str:
    """Helper to return remote CDN URL or local static URL."""
    if person.image_url and (person.image_url.startswith("http://") or person.image_url.startswith("https://")):
        return person.image_url
    filename = Path(person.image_path).name
    return f"/media/registered/{filename}"


# ---------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """Returns server status, registered gallery count, storage mode, and compute device."""
    detector, generator, gallery = get_pipeline()
    return HealthResponse(
        status="ok",
        registered_persons_count=gallery.count,
        compute_device=str(detector.device),
        storage_mode=gallery.storage_mode,
    )


# ---------------------------------------------------------
# Gallery Management Endpoints
# ---------------------------------------------------------

@app.get("/api/v1/gallery", response_model=GalleryListResponse, tags=["Gallery"])
def list_gallery():
    """List all currently registered missing persons."""
    _, _, gallery = get_pipeline()
    persons_summary: List[PersonSummary] = []

    for p in gallery.persons:
        persons_summary.append(
            PersonSummary(
                person_id=p.person_id,
                name=p.name,
                image_url=_resolve_image_url(p),
                registered_at=p.registered_at,
                embedding_dimension=len(p.embedding) if p.embedding is not None else 512,
            )
        )

    return GalleryListResponse(
        total_count=len(persons_summary),
        persons=persons_summary,
    )


@app.post("/api/v1/gallery/register", response_model=PersonRegistrationResponse, tags=["Gallery"])
async def register_person(
    name: str = Form(..., description="Full Name of the missing person"),
    file: UploadFile = File(..., description="Frontal reference portrait photo"),
    person_id: Optional[str] = Form(None, description="Optional custom unique ID"),
):
    """
    Register a new missing person photo into the live gallery.
    Precomputes 512-D face embeddings and persists to Supabase pgvector or local cache.
    """
    detector, generator, gallery = get_pipeline()

    # Validate file extension
    suffix = Path(file.filename or "photo.jpg").suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image format. Please upload JPG, PNG, or WEBP.",
        )

    pid = person_id or Path(file.filename).stem
    save_filename = f"{pid}{suffix}"
    dest_path = REG_DIR / save_filename

    # Save uploaded file locally
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save image file: {str(e)}",
        )

    # Register with GalleryManager (handles Supabase upload + local caching)
    reg = gallery.register_person(name=name, image_path=dest_path, person_id=pid)
    if reg is None:
        if dest_path.exists():
            dest_path.unlink()  # Clean up failed image
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No clear face detected in the uploaded photo. Please provide a clear frontal portrait.",
        )

    gallery.save_cache()

    summary = PersonSummary(
        person_id=reg.person_id,
        name=reg.name,
        image_url=_resolve_image_url(reg),
        registered_at=reg.registered_at,
        embedding_dimension=len(reg.embedding),
    )

    return PersonRegistrationResponse(
        success=True,
        message=f"Successfully registered '{name}' in LocateMe gallery ({gallery.storage_mode} mode).",
        person=summary,
    )


@app.delete("/api/v1/gallery/{person_id}", tags=["Gallery"])
def delete_person(person_id: str):
    """Remove a registered person from the gallery (Supabase database and local cache)."""
    _, _, gallery = get_pipeline()
    deleted = gallery.delete_person(person_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person with ID '{person_id}' not found in gallery.",
        )

    # Remove local image file if exists
    for f in REG_DIR.glob(f"{person_id}.*"):
        try:
            f.unlink()
        except Exception:
            pass

    return {"success": True, "message": f"Person ID '{person_id}' removed from gallery ({gallery.storage_mode} mode)."}


# ---------------------------------------------------------
# Image Screening Endpoint
# ---------------------------------------------------------

@app.post("/api/v1/screen/image", response_model=ImageScreeningResponse, tags=["Screening"])
async def screen_image(
    file: UploadFile = File(..., description="Query photo or CCTV frame snapshot"),
    threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD, description="Similarity threshold (0.0 to 1.0)"),
):
    """
    Screen a static photo or surveillance snapshot against all registered missing persons.
    """
    detector, generator, gallery = get_pipeline()

    if gallery.count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gallery is empty. Please register at least one reference person first.",
        )

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}",
        )

    face_results = detector.detect_all_faces(pil_img)
    detections: List[DetectedFaceScreening] = []
    potential_matches_count = 0

    for idx, det in enumerate(face_results):
        if not det.is_detected or det.face_tensor is None or det.box is None:
            continue

        emb = generator.extract_from_tensor(det.face_tensor)
        if emb is None:
            continue

        all_matches = gallery.search(emb, threshold=threshold)
        candidates_schema: List[CandidateMatch] = []
        best_candidate: Optional[CandidateMatch] = None

        for person, match_res in all_matches:
            cand = CandidateMatch(
                person_id=person.person_id,
                person_name=person.name,
                similarity_score=match_res.similarity_score,
                is_match=match_res.is_match,
                confidence_tier=match_res.confidence_tier,
                match_status=match_res.match_status,
                reference_image_url=_resolve_image_url(person),
            )
            candidates_schema.append(cand)

        if candidates_schema:
            best_candidate = candidates_schema[0]
            if best_candidate.is_match:
                potential_matches_count += 1

        detections.append(
            DetectedFaceScreening(
                face_index=idx,
                bounding_box=[int(v) for v in det.box],
                detection_confidence=round(det.probability or 0.0, 4),
                best_match=best_candidate if (best_candidate and best_candidate.is_match) else None,
                all_candidates=candidates_schema[:5],  # top 5
            )
        )

    return ImageScreeningResponse(
        total_faces_detected=len(detections),
        potential_matches_found=potential_matches_count,
        detections=detections,
        threshold_used=threshold,
        disclaimer=DISCLAIMER_TEXT,
    )


# ---------------------------------------------------------
# Video Screening Endpoint
# ---------------------------------------------------------

@app.post("/api/v1/screen/video", response_model=VideoScreeningResponse, tags=["Screening"])
async def screen_video(
    file: UploadFile = File(..., description="Surveillance / CCTV video clip (.mp4, .avi, etc.)"),
    threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD, description="Similarity threshold (0.0 to 1.0)"),
    frame_step: int = Form(5, description="Frame sampling step (e.g. 5 = every 5th frame)"),
):
    """
    Screen a video feed against the registered missing person gallery.
    Produces annotated video playback and a chronological detection event log.
    """
    detector, generator, gallery = get_pipeline()

    if gallery.count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gallery is empty. Please register at least one reference person first.",
        )

    # Save uploaded video
    temp_suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    temp_in = TEST_VID_DIR / f"upload_{file.filename}"
    temp_out = TEST_VID_DIR / f"annotated_{file.filename}"
    temp_report = TEST_VID_DIR / f"report_{Path(file.filename).stem}.json"
    crops_dir = TEST_VID_DIR / "detected_crops"

    try:
        with open(temp_in, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save video: {str(e)}",
        )

    processor = VideoProcessor(
        gallery=gallery,
        detector=detector,
        generator=generator,
        threshold=threshold,
        frame_step=frame_step,
    )

    summary: VideoProcessingSummary = processor.process_video(
        video_source=str(temp_in),
        output_video_path=temp_out,
        export_crops_dir=crops_dir,
    )
    summary.save_json(temp_report)

    # Convert match events to schema
    match_schemas: List[MatchEventSchema] = []
    for m in summary.matches:
        crop_url = f"/media/test_videos/detected_crops/{m.crop_filename}" if m.crop_filename else None
        match_schemas.append(
            MatchEventSchema(
                frame_index=m.frame_index,
                timestamp_seconds=m.timestamp_seconds,
                timestamp_formatted=m.timestamp_formatted,
                person_id=m.person_id,
                person_name=m.person_name,
                similarity_score=m.similarity_score,
                detection_confidence=m.detection_confidence,
                box=m.box,
                confidence_tier=m.confidence_tier,
                crop_url=crop_url,
            )
        )

    return VideoScreeningResponse(
        total_frames=summary.total_frames,
        processed_frames=summary.processed_frames,
        duration_seconds=summary.duration_seconds,
        processing_time_seconds=summary.elapsed_time_seconds,
        processing_fps=summary.processing_fps,
        total_matches_detected=summary.total_matches_detected,
        unique_matched_candidates=summary.unique_candidates_matched,
        annotated_video_url=f"/media/test_videos/{temp_out.name}",
        report_url=f"/media/test_videos/{temp_report.name}",
        matches=match_schemas,
        disclaimer=DISCLAIMER_TEXT,
    )
