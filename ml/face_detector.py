"""
LocateMe — Face Detection Module
Uses MTCNN from facenet-pytorch to detect, align, and crop faces for embedding generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FaceDetectionResult:
    """Structured result for a face detection operation."""
    is_detected: bool
    face_tensor: Optional[torch.Tensor] = None  # Expected shape: [3, 160, 160]
    box: Optional[np.ndarray] = None           # [x1, y1, x2, y2]
    probability: Optional[float] = None        # Detection confidence [0.0 - 1.0]
    error_message: Optional[str] = None


class FaceDetector:
    """
    Modular Face Detector wrapping MTCNN.
    Handles image loading, face detection, alignment, cropping, and validation.
    """

    def __init__(
        self,
        image_size: int = 160,
        margin: int = 20,
        min_face_size: int = 20,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        post_process: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        min_probability: float = 0.80,
    ) -> None:
        """
        Initialize the MTCNN face detector.

        Args:
            image_size: Dimension to crop and scale detected faces (default: 160 for InceptionResnetV1).
            margin: Extra margin around face bounding box in pixels.
            min_face_size: Minimum face size in pixels to consider.
            thresholds: MTCNN three-stage step thresholds.
            post_process: Whether to normalize image tensors to [-1, 1] range.
            device: 'cpu', 'cuda', or torch.device. Defaults to CUDA if available, else CPU.
            min_probability: Minimum confidence threshold to consider a face valid.
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self.margin = margin
        self.min_probability = min_probability

        # Single face detector (selects the most prominent / largest face)
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=0.709,
            post_process=post_process,
            select_largest=True,
            keep_all=False,
            device=self.device,
        )

        # Multi-face detector for scenes / surveillance frames
        self.mtcnn_multi = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=0.709,
            post_process=post_process,
            keep_all=True,
            device=self.device,
        )

    def _load_and_preprocess_image(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Optional[Image.Image]:
        """
        Safely convert various input types (file path, OpenCV numpy array, PIL Image)
        into an RGB PIL Image.
        """
        if image_input is None:
            return None

        try:
            # Case 1: String path or pathlib.Path
            if isinstance(image_input, (str, Path)):
                img_path = Path(image_input)
                if not img_path.is_file():
                    logger.warning(f"Image path not found: {img_path}")
                    return None
                img = Image.open(img_path)
                return img.convert("RGB")

            # Case 2: PIL Image
            if isinstance(image_input, Image.Image):
                return image_input.convert("RGB")

            # Case 3: NumPy ndarray (OpenCV format)
            if isinstance(image_input, np.ndarray):
                if image_input.size == 0:
                    logger.warning("Empty numpy image array provided.")
                    return None

                # Handle color channels
                if len(image_input.shape) == 2:
                    # Grayscale
                    rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                elif len(image_input.shape) == 3:
                    if image_input.shape[2] == 4:
                        # BGRA -> RGB
                        rgb = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
                    elif image_input.shape[2] == 3:
                        # Assume OpenCV default BGR -> convert to RGB
                        rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                    else:
                        logger.warning(f"Unsupported channel count: {image_input.shape[2]}")
                        return None
                else:
                    logger.warning(f"Invalid image array shape: {image_input.shape}")
                    return None

                return Image.fromarray(rgb)

            logger.warning(f"Unsupported image input type: {type(image_input)}")
            return None

        except Exception as e:
            logger.error(f"Error loading image input: {e}", exc_info=True)
            return None

    def detect_face(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> FaceDetectionResult:
        """
        Detect the most prominent face in an image, crop, and align it.

        Returns:
            FaceDetectionResult containing aligned tensor [3, 160, 160], bounding box, and probability.
        """
        pil_img = self._load_and_preprocess_image(image_input)
        if pil_img is None:
            return FaceDetectionResult(
                is_detected=False,
                error_message="Failed to load or parse image input.",
            )

        try:
            # Detect bounding boxes and probabilities
            boxes, probs = self.mtcnn.detect(pil_img)

            if boxes is None or len(boxes) == 0 or probs is None or len(probs) == 0:
                return FaceDetectionResult(
                    is_detected=False,
                    error_message="No face detected in the image.",
                )

            box = boxes[0]
            prob = float(probs[0]) if probs[0] is not None else 0.0

            if prob < self.min_probability:
                return FaceDetectionResult(
                    is_detected=False,
                    box=box,
                    probability=prob,
                    error_message=f"Detected face confidence ({prob:.3f}) below threshold ({self.min_probability:.3f}).",
                )

            # Crop and align face to normalized tensor [3, 160, 160]
            face_tensor = self.mtcnn(pil_img)

            if face_tensor is None:
                return FaceDetectionResult(
                    is_detected=False,
                    box=box,
                    probability=prob,
                    error_message="Failed to extract cropped face tensor.",
                )

            return FaceDetectionResult(
                is_detected=True,
                face_tensor=face_tensor,
                box=box,
                probability=prob,
            )

        except Exception as e:
            logger.error(f"MTCNN detection error: {e}", exc_info=True)
            return FaceDetectionResult(
                is_detected=False,
                error_message=f"Detection failed due to error: {str(e)}",
            )

    def detect_all_faces(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> List[FaceDetectionResult]:
        """
        Detect all faces in an image (e.g., surveillance frame with multiple people).

        Returns:
            List of FaceDetectionResult objects for each detected face above threshold.
        """
        pil_img = self._load_and_preprocess_image(image_input)
        if pil_img is None:
            return []

        try:
            boxes, probs = self.mtcnn_multi.detect(pil_img)
            if boxes is None or len(boxes) == 0:
                return []

            face_tensors = self.mtcnn_multi(pil_img)

            results: List[FaceDetectionResult] = []
            for i, box in enumerate(boxes):
                prob = float(probs[i]) if probs is not None and probs[i] is not None else 0.0
                if prob < self.min_probability:
                    continue

                tensor = None
                if face_tensors is not None and len(face_tensors) > i:
                    tensor = face_tensors[i]

                if tensor is not None:
                    results.append(
                        FaceDetectionResult(
                            is_detected=True,
                            face_tensor=tensor,
                            box=box,
                            probability=prob,
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"Multi-face detection error: {e}", exc_info=True)
            return []


# Global singleton cache for lightweight module-level usage
_DEFAULT_DETECTOR: Optional[FaceDetector] = None


def get_default_detector() -> FaceDetector:
    """Retrieve or initialize a cached default FaceDetector instance."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = FaceDetector()
    return _DEFAULT_DETECTOR


def detect_face(
    image: Union[str, Path, Image.Image, np.ndarray],
    detector: Optional[FaceDetector] = None,
) -> FaceDetectionResult:
    """Convenience functional wrapper for face detection."""
    d = detector or get_default_detector()
    return d.detect_face(image)
