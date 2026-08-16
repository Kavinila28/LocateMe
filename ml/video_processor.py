"""
LocateMe — Video & CCTV Stream Processor Module
Performs frame sampling, multi-face extraction, fast gallery screening,
visual frame annotation, and structured match event logging.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from ml.embedding import FaceEmbeddingGenerator, get_default_generator
from ml.face_detector import FaceDetector, FaceDetectionResult, get_default_detector
from ml.gallery import GalleryManager, RegisteredPerson
from ml.matcher import MatchResult, DEFAULT_SIMILARITY_THRESHOLD, DISCLAIMER_TEXT

logger = logging.getLogger(__name__)


@dataclass
class MatchEvent:
    """Represents a potential missing-person match detected in a video frame."""
    frame_index: int
    timestamp_seconds: float
    person_id: str
    person_name: str
    similarity_score: float
    detection_confidence: float
    box: List[int]  # [x1, y1, x2, y2]
    confidence_tier: str
    crop_filename: Optional[str] = None

    @property
    def timestamp_formatted(self) -> str:
        """Return HH:MM:SS format of detection timestamp."""
        mins = int(self.timestamp_seconds // 60)
        secs = self.timestamp_seconds % 60
        return f"{mins:02d}:{secs:05.2f}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoProcessingSummary:
    """Comprehensive summary report of video feed screening."""
    source_video: str
    total_frames: int
    processed_frames: int
    fps: float
    duration_seconds: float
    elapsed_time_seconds: float
    processing_fps: float
    total_matches_detected: int
    unique_candidates_matched: List[str]
    matches: List[MatchEvent]
    output_video_path: Optional[str] = None
    disclaimer: str = DISCLAIMER_TEXT

    def to_dict(self) -> dict:
        data = asdict(self)
        data["matches"] = [m.to_dict() for m in self.matches]
        return data

    def save_json(self, output_path: Union[str, Path]) -> None:
        """Export summary to formatted JSON file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved video match report to {out_p}")


class VideoProcessor:
    """
    Video processor for CCTV and video feed missing person screening.
    """

    def __init__(
        self,
        gallery: GalleryManager,
        detector: Optional[FaceDetector] = None,
        generator: Optional[FaceEmbeddingGenerator] = None,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        frame_step: int = 5,
    ) -> None:
        """
        Initialize the Video Processor.

        Args:
            gallery: Initialized GalleryManager loaded with registered reference portraits.
            detector: FaceDetector instance.
            generator: FaceEmbeddingGenerator instance.
            threshold: Cosine similarity threshold for declaring a Potential Match.
            frame_step: Frame sampling interval (e.g. 5 = process every 5th frame).
        """
        self.gallery = gallery
        self.detector = detector or get_default_detector()
        self.generator = generator or get_default_generator()
        self.threshold = threshold
        self.frame_step = max(1, frame_step)

    def _draw_annotation(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        matched_person: Optional[RegisteredPerson],
        match_result: Optional[MatchResult],
        confidence: float,
    ) -> None:
        """
        Draw color-coded bounding boxes and label overlays on video frame.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if matched_person and match_result and match_result.is_match:
            # Positive candidate match -> Vibrant Green
            color = (46, 204, 113)
            label = f"MATCH: {matched_person.name} ({match_result.similarity_score:.2f})"
        else:
            # Unmatched detected face -> Neutral Light Gray
            color = (180, 180, 180)
            label = f"Detected ({confidence * 100:.0f}%)"

        # Draw bounding rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background badge
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        badge_y1 = max(0, y1 - label_h - 8)
        badge_y2 = y1
        badge_x2 = min(w, x1 + label_w + 10)

        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Draw label text
        text_color = (0, 0, 0) if (matched_person and match_result and match_result.is_match) else (255, 255, 255)
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 4),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        timestamp_sec: float,
        match_count: int,
    ) -> None:
        """
        Draw surveillance HUD banner at the top of the video frame.
        """
        h, w = frame.shape[:2]
        banner_h = 32

        # Semi-transparent HUD overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 30, 36), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        mins = int(timestamp_sec // 60)
        secs = timestamp_sec % 60
        time_str = f"{mins:02d}:{secs:05.2f}"

        hud_text = (
            f"LocateMe Video Stream | Time: {time_str} | "
            f"Frame: {frame_idx + 1}/{total_frames} | "
            f"Alerts: {match_count}"
        )

        cv2.putText(
            frame,
            hud_text,
            (12, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    def process_video(
        self,
        video_source: Union[str, Path, int],
        output_video_path: Optional[Union[str, Path]] = None,
        export_crops_dir: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> VideoProcessingSummary:
        """
        Process an input video file or webcam stream frame-by-frame.

        Args:
            video_source: Path to video file or webcam device integer index.
            output_video_path: Optional destination to save annotated MP4 video.
            export_crops_dir: Optional directory to save cropped candidate faces.
            progress_callback: Optional callback func(current_frame, total_frames, match_count).

        Returns:
            VideoProcessingSummary containing all detection events and statistics.
        """
        is_webcam = isinstance(video_source, int)
        src_str = str(video_source)
        cap = cv2.VideoCapture(video_source if is_webcam else src_str)

        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        duration_sec = total_frames / fps if total_frames > 0 else 0.0

        # Output video writer setup
        writer = None
        out_p = Path(output_video_path) if output_video_path else None
        if out_p:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, fps, (frame_width, frame_height))

        # Crop output dir setup
        crops_p = Path(export_crops_dir) if export_crops_dir else None
        if crops_p:
            crops_p.mkdir(parents=True, exist_ok=True)

        match_events: List[MatchEvent] = []
        processed_count = 0
        frame_idx = 0
        start_time = time.time()

        # Cache last detected annotations to carry forward over skipped frames
        cached_annotations: List[Tuple[np.ndarray, Optional[RegisteredPerson], Optional[MatchResult], float]] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = frame_idx / fps
                should_detect = (frame_idx % self.frame_step == 0)

                if should_detect:
                    processed_count += 1
                    cached_annotations = []

                    # Convert BGR frame to RGB PIL Image for MTCNN
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)

                    # Detect all faces in current frame
                    face_results = self.detector.detect_all_faces(pil_img)

                    for det in face_results:
                        if not det.is_detected or det.face_tensor is None or det.box is None:
                            continue

                        # Extract 512-D embedding
                        face_emb = self.generator.extract_from_tensor(det.face_tensor)
                        if face_emb is None:
                            continue

                        # Screen against registered missing person gallery
                        best_match = self.gallery.find_best_match(face_emb, threshold=self.threshold)
                        matched_person = best_match[0] if best_match else None
                        match_res = best_match[1] if best_match else None

                        # Record match event
                        if matched_person and match_res and match_res.is_match:
                            crop_fname = None
                            if crops_p:
                                crop_fname = f"match_f{frame_idx:06d}_{matched_person.person_id}.jpg"
                                x1, y1, x2, y2 = [int(v) for v in det.box]
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                                face_crop = frame[y1:y2, x1:x2]
                                if face_crop.size > 0:
                                    cv2.imwrite(str(crops_p / crop_fname), face_crop)

                            event = MatchEvent(
                                frame_index=frame_idx,
                                timestamp_seconds=round(timestamp_sec, 2),
                                person_id=matched_person.person_id,
                                person_name=matched_person.name,
                                similarity_score=match_res.similarity_score,
                                detection_confidence=round(det.probability or 0.0, 4),
                                box=[int(v) for v in det.box],
                                confidence_tier=match_res.confidence_tier,
                                crop_filename=crop_fname,
                            )
                            match_events.append(event)

                        cached_annotations.append(
                            (det.box, matched_person, match_res, det.probability or 0.0)
                        )

                # Render annotations (either freshly computed or persisted across frame steps)
                for box, person, m_res, prob in cached_annotations:
                    self._draw_annotation(frame, box, person, m_res, prob)

                # Render HUD
                self._draw_hud(
                    frame,
                    frame_idx=frame_idx,
                    total_frames=total_frames or (frame_idx + 1),
                    timestamp_sec=timestamp_sec,
                    match_count=len(match_events),
                )

                if writer:
                    writer.write(frame)

                if progress_callback and should_detect:
                    progress_callback(frame_idx, total_frames, len(match_events))

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()

        elapsed = max(0.001, time.time() - start_time)
        unique_matched = sorted(list(set(m.person_name for m in match_events)))

        return VideoProcessingSummary(
            source_video=src_str,
            total_frames=frame_idx,
            processed_frames=processed_count,
            fps=round(fps, 2),
            duration_seconds=round(duration_sec, 2),
            elapsed_time_seconds=round(elapsed, 2),
            processing_fps=round(frame_idx / elapsed, 2),
            total_matches_detected=len(match_events),
            unique_candidates_matched=unique_matched,
            matches=match_events,
            output_video_path=str(out_p) if out_p else None,
        )
