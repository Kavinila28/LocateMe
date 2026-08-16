"""
LocateMe — Phase 2 Video Pipeline Test Suite
Verifies GalleryManager vectorized search/caching and VideoProcessor CCTV feed screening.
"""

import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.gallery import GalleryManager, RegisteredPerson
from ml.video_processor import VideoProcessor, VideoProcessingSummary, MatchEvent
from ml.face_detector import FaceDetector
from ml.embedding import FaceEmbeddingGenerator


def create_sample_video_clip_if_needed() -> Path:
    """
    Synthesizes a realistic 3-second (90 frames, 30 FPS) benchmark video clip
    containing Person A and Person B sequences from registered photos.
    """
    video_dir = PROJECT_ROOT / "data" / "test_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "sample_cctv_feed.mp4"

    if video_path.exists() and video_path.stat().st_size > 1000:
        return video_path

    reg_dir = PROJECT_ROOT / "data" / "registered"
    img_a_path = reg_dir / "person_a_ref.jpg"
    img_b_path = reg_dir / "person_b_ref.jpg"

    w, h = 640, 480
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

    # Helper to create frame with centered face on simulated street background
    def generate_frame(face_img_path: Path, bg_color: tuple, pan_x: int) -> np.ndarray:
        frame = np.full((h, w, 3), bg_color, dtype=np.uint8)
        if face_img_path.exists():
            face_img = cv2.imread(str(face_img_path))
            if face_img is not None:
                # Resize face to 240x240
                face_resized = cv2.resize(face_img, (240, 240))
                # Insert at pan position
                start_x = max(0, min(w - 240, 200 + pan_x))
                start_y = 100
                frame[start_y:start_y+240, start_x:start_x+240] = face_resized
        return frame

    # Sequence 1: Person A (frames 0 to 45)
    for f in range(45):
        pan = int((f - 22) * 2)
        frame = generate_frame(img_a_path, (60, 65, 70), pan)
        out.write(frame)

    # Sequence 2: Person B (frames 45 to 90)
    for f in range(45):
        pan = int((f - 22) * 2)
        frame = generate_frame(img_b_path, (70, 65, 60), pan)
        out.write(frame)

    out.release()
    return video_path


# ---------------------------------------------------------
# PyTest Unit Tests for Phase 2
# ---------------------------------------------------------

def test_gallery_manager_loading_and_search():
    """Verify GalleryManager loads registered persons and performs fast vectorized 1-to-N search."""
    reg_dir = PROJECT_ROOT / "data" / "registered"
    gallery = GalleryManager(gallery_dir=reg_dir)

    assert gallery.count >= 2, f"Expected at least 2 registered persons, got {gallery.count}"

    # Search using Person A's embedding
    person_a = [p for p in gallery.persons if "person_a" in p.person_id][0]
    results = gallery.search(person_a.embedding, threshold=0.68)

    assert len(results) == gallery.count
    # Top match must be Person A with self-similarity ~ 1.0
    top_person, top_match = results[0]
    assert top_person.person_id == person_a.person_id
    assert top_match.is_match is True
    assert top_match.similarity_score >= 0.99
    assert top_match.confidence_tier == "High Similarity"


def test_gallery_cache_saving_and_loading(tmp_path):
    """Verify gallery caching mechanism saves and restores metadata without loss."""
    reg_dir = PROJECT_ROOT / "data" / "registered"
    cache_file = tmp_path / "test_gallery_cache.npz"

    # Initial load & save to temporary cache
    gallery_1 = GalleryManager(gallery_dir=reg_dir, cache_file=cache_file)
    count_1 = gallery_1.count

    # Second load directly from cache file
    gallery_2 = GalleryManager(gallery_dir=reg_dir, cache_file=cache_file)
    assert gallery_2.count == count_1
    assert len(gallery_2.persons) == count_1


def test_video_processor_screening():
    """Verify VideoProcessor processes CCTV video, records match events, and writes output."""
    sample_video = create_sample_video_clip_if_needed()
    assert sample_video.exists()

    reg_dir = PROJECT_ROOT / "data" / "registered"
    gallery = GalleryManager(gallery_dir=reg_dir)

    output_video = PROJECT_ROOT / "data" / "test_videos" / "test_annotated_feed.mp4"
    report_file = PROJECT_ROOT / "data" / "test_videos" / "test_match_report.json"

    processor = VideoProcessor(
        gallery=gallery,
        threshold=0.68,
        frame_step=5,
    )

    summary: VideoProcessingSummary = processor.process_video(
        video_source=str(sample_video),
        output_video_path=output_video,
    )

    summary.save_json(report_file)

    assert summary.total_frames == 90
    assert summary.processed_frames == 18  # 90 / 5 = 18 sampled frames
    assert summary.total_matches_detected > 0
    assert "Person A" in summary.unique_candidates_matched or "Person B" in summary.unique_candidates_matched
    assert output_video.exists()
    assert output_video.stat().st_size > 1000
    assert report_file.exists()


if __name__ == "__main__":
    print("Running Video Pipeline Tests...")
    test_gallery_manager_loading_and_search()
    print("GalleryManager test passed!")
    test_video_processor_screening()
    print("VideoProcessor test passed!")
