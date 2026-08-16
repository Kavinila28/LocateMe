"""
LocateMe Machine Learning Pipeline Package
Phase 1: Local Face Detection, Feature Embedding, and Similarity Matching
Phase 2: Video Stream Processing & Vectorized Gallery Screening
"""

from ml.face_detector import (
    FaceDetector,
    FaceDetectionResult,
    detect_face,
    get_default_detector,
)
from ml.embedding import (
    FaceEmbeddingGenerator,
    generate_embedding,
    get_default_generator,
    EXPECTED_EMBEDDING_DIM,
)
from ml.matcher import (
    FaceMatcher,
    MatchResult,
    compute_cosine_similarity,
    match_faces,
    DEFAULT_SIMILARITY_THRESHOLD,
    DISCLAIMER_TEXT,
)
from ml.gallery import (
    GalleryManager,
    RegisteredPerson,
)
from ml.video_processor import (
    VideoProcessor,
    MatchEvent,
    VideoProcessingSummary,
)

__all__ = [
    "FaceDetector",
    "FaceDetectionResult",
    "detect_face",
    "get_default_detector",
    "FaceEmbeddingGenerator",
    "generate_embedding",
    "get_default_generator",
    "EXPECTED_EMBEDDING_DIM",
    "FaceMatcher",
    "MatchResult",
    "compute_cosine_similarity",
    "match_faces",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DISCLAIMER_TEXT",
    "GalleryManager",
    "RegisteredPerson",
    "VideoProcessor",
    "MatchEvent",
    "VideoProcessingSummary",
]
