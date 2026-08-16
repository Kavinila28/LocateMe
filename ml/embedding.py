"""
LocateMe — Face Embedding Generator Module
Loads InceptionResnetV1 (pretrained on VGGFace2) via facenet-pytorch to generate
normalized 512-dimensional face embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image

from ml.face_detector import FaceDetector, detect_face, FaceDetectionResult

logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 512


class FaceEmbeddingGenerator:
    """
    Modular Face Embedding Generator using InceptionResnetV1.
    Produces unit-normalized 512-dimensional feature vectors.
    """

    def __init__(
        self,
        pretrained: str = "vggface2",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Initialize the embedding network with pretrained weights.

        Args:
            pretrained: Weights dataset name ('vggface2' or 'casia-webface').
            device: 'cpu', 'cuda', or torch.device. Auto-detects if None.
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.pretrained = pretrained
        logger.info(f"Loading InceptionResnetV1 ({pretrained}) on device: {self.device}")

        # Instantiate model in evaluation mode
        self.model = InceptionResnetV1(pretrained=pretrained, classify=False).eval().to(self.device)

    def extract_from_tensor(
        self, face_tensor: torch.Tensor, normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Generate embedding directly from a pre-cropped aligned face tensor [3, 160, 160]
        or batch [B, 3, 160, 160].

        Args:
            face_tensor: Tensor of cropped face image.
            normalize: Whether to apply L2 normalization (recommended for cosine similarity).

        Returns:
            np.ndarray of shape (512,) or (B, 512) for batches.
        """
        if face_tensor is None:
            logger.warning("Empty face tensor provided to embedding extractor.")
            return None

        try:
            # Ensure 4D shape [B, C, H, W]
            if face_tensor.dim() == 3:
                tensor_input = face_tensor.unsqueeze(0).to(self.device)
                is_batch = False
            elif face_tensor.dim() == 4:
                tensor_input = face_tensor.to(self.device)
                is_batch = True
            else:
                logger.error(f"Unexpected face tensor shape: {face_tensor.shape}")
                return None

            with torch.no_grad():
                raw_embedding = self.model(tensor_input)

            # Convert to CPU numpy array
            embedding = raw_embedding.cpu().numpy()

            # L2 normalization to unit vector
            if normalize:
                norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
                # Avoid division by zero
                norm = np.where(norm == 0, 1e-12, norm)
                embedding = embedding / norm

            # Confirm dimensions
            embedding_dim = embedding.shape[-1]
            if embedding_dim != EXPECTED_EMBEDDING_DIM:
                raise ValueError(
                    f"Invalid embedding dimension: got {embedding_dim}, expected {EXPECTED_EMBEDDING_DIM}"
                )

            if not is_batch:
                return embedding[0].astype(np.float32)

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Embedding extraction error: {e}", exc_info=True)
            return None

    def generate_embedding(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        detector: Optional[FaceDetector] = None,
        normalize: bool = True,
    ) -> Optional[np.ndarray]:
        """
        End-to-end embedding generation: accepts a tensor OR an uncropped raw image.
        If a raw image is supplied, detects and aligns the face first.

        Returns:
            1D np.ndarray of shape (512,) or None if detection/extraction fails.
        """
        if image_input is None:
            return None

        # If already a torch Tensor, extract directly
        if isinstance(image_input, torch.Tensor):
            return self.extract_from_tensor(image_input, normalize=normalize)

        # If it's an image or path, detect face first
        det_result: FaceDetectionResult = detect_face(image_input, detector=detector)
        if not det_result.is_detected or det_result.face_tensor is None:
            logger.info(f"Cannot generate embedding: {det_result.error_message}")
            return None

        return self.extract_from_tensor(det_result.face_tensor, normalize=normalize)


# Global singleton cache for lightweight module-level usage
_DEFAULT_GENERATOR: Optional[FaceEmbeddingGenerator] = None


def get_default_generator() -> FaceEmbeddingGenerator:
    """Retrieve or initialize a cached default FaceEmbeddingGenerator instance."""
    global _DEFAULT_GENERATOR
    if _DEFAULT_GENERATOR is None:
        _DEFAULT_GENERATOR = FaceEmbeddingGenerator()
    return _DEFAULT_GENERATOR


def generate_embedding(
    image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
    detector: Optional[FaceDetector] = None,
    generator: Optional[FaceEmbeddingGenerator] = None,
    normalize: bool = True,
) -> Optional[np.ndarray]:
    """
    Convenience function to generate a 512-D face embedding from an image or tensor.

    Args:
        image: File path, PIL Image, OpenCV BGR ndarray, or cropped face Tensor.
        detector: Optional FaceDetector instance.
        generator: Optional FaceEmbeddingGenerator instance.
        normalize: Whether to L2-normalize the vector (default: True).

    Returns:
        1D float32 numpy array of shape (512,) or None.
    """
    g = generator or get_default_generator()
    return g.generate_embedding(image, detector=detector, normalize=normalize)
