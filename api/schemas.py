"""
LocateMe — Pydantic Schemas for FastAPI REST endpoints
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "LocateMe Missing Person Screening API"
    version: str = "1.0.0"
    registered_persons_count: int
    compute_device: str
    model_architecture: str = "MTCNN + InceptionResnetV1 (VGGFace2)"


class PersonSummary(BaseModel):
    person_id: str
    name: str
    image_url: str
    registered_at: str
    embedding_dimension: int = 512


class GalleryListResponse(BaseModel):
    total_count: int
    persons: List[PersonSummary]


class PersonRegistrationResponse(BaseModel):
    success: bool
    message: str
    person: Optional[PersonSummary] = None


class CandidateMatch(BaseModel):
    person_id: str
    person_name: str
    similarity_score: float
    is_match: bool
    confidence_tier: str
    match_status: str
    reference_image_url: Optional[str] = None


class DetectedFaceScreening(BaseModel):
    face_index: int
    bounding_box: List[int]  # [x1, y1, x2, y2]
    detection_confidence: float
    best_match: Optional[CandidateMatch] = None
    all_candidates: List[CandidateMatch] = Field(default_factory=list)


class ImageScreeningResponse(BaseModel):
    total_faces_detected: int
    potential_matches_found: int
    detections: List[DetectedFaceScreening]
    threshold_used: float
    disclaimer: str


class MatchEventSchema(BaseModel):
    frame_index: int
    timestamp_seconds: float
    timestamp_formatted: str
    person_id: str
    person_name: str
    similarity_score: float
    detection_confidence: float
    box: List[int]
    confidence_tier: str
    crop_url: Optional[str] = None


class VideoScreeningResponse(BaseModel):
    total_frames: int
    processed_frames: int
    duration_seconds: float
    processing_time_seconds: float
    processing_fps: float
    total_matches_detected: int
    unique_matched_candidates: List[str]
    annotated_video_url: Optional[str] = None
    report_url: Optional[str] = None
    matches: List[MatchEventSchema] = Field(default_factory=list)
    disclaimer: str
