"""
LocateMe — Test Pipeline & Verification Suite
Verifies MTCNN face detection, InceptionResnetV1 512-D embedding extraction,
and cosine similarity matching.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

# Add project root to sys.path so ml package can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.face_detector import FaceDetector, FaceDetectionResult, detect_face
from ml.embedding import FaceEmbeddingGenerator, generate_embedding, EXPECTED_EMBEDDING_DIM
from ml.matcher import FaceMatcher, compute_cosine_similarity, match_faces, MatchResult


def create_synthetic_test_images_if_needed():
    """
    Creates lightweight synthetic test images if real sample images do not exist yet,
    guaranteeing that unit tests can run standalone in any environment.
    """
    reg_dir = PROJECT_ROOT / "data" / "registered"
    test_dir = PROJECT_ROOT / "data" / "test_images"
    reg_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Helper to generate a colored square/face placeholder if needed
    def make_image(path: Path, color: tuple):
        if not path.exists():
            img = Image.new("RGB", (200, 200), color=color)
            img.save(path)

    make_image(test_dir / "blank_image.jpg", (255, 255, 255))


# ---------------------------------------------------------
# PyTest Unit Tests
# ---------------------------------------------------------

def test_face_detector_invalid_input():
    """Test that FaceDetector handles nonexistent and invalid files gracefully without crashing."""
    detector = FaceDetector()
    res = detector.detect_face("non_existent_file_12345.jpg")
    assert not res.is_detected
    assert res.face_tensor is None
    assert res.error_message is not None


def test_face_detector_no_face():
    """Test that FaceDetector reports no face on a solid white image."""
    detector = FaceDetector()
    blank_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    res = detector.detect_face(blank_img)
    assert not res.is_detected
    assert res.face_tensor is None


def test_embedding_dimensions_and_norm():
    """Test that FaceEmbeddingGenerator generates exact 512-D L2-normalized vector."""
    generator = FaceEmbeddingGenerator()
    # Synthetic face tensor [3, 160, 160] with standard normal distribution
    import torch
    dummy_face_tensor = torch.randn(3, 160, 160)
    embedding = generator.extract_from_tensor(dummy_face_tensor, normalize=True)

    assert embedding is not None
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    # Verify L2 norm is ~1.0
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_matcher_cosine_similarity():
    """Test mathematical accuracy of cosine similarity calculation."""
    # Identical vectors -> similarity = 1.0
    v1 = np.random.randn(512).astype(np.float32)
    sim_same = compute_cosine_similarity(v1, v1)
    assert np.isclose(sim_same, 1.0, atol=1e-5)

    # Orthogonal vectors -> similarity = 0.0
    u = np.zeros(512, dtype=np.float32)
    v = np.zeros(512, dtype=np.float32)
    u[0] = 1.0
    v[1] = 1.0
    sim_ortho = compute_cosine_similarity(u, v)
    assert np.isclose(sim_ortho, 0.0, atol=1e-5)

    # Opposite vectors -> similarity = -1.0
    sim_opp = compute_cosine_similarity(u, -u)
    assert np.isclose(sim_opp, -1.0, atol=1e-5)


def test_matcher_threshold_evaluation():
    """Test matcher decision logic with configurable thresholds."""
    matcher = FaceMatcher(threshold=0.70)
    
    # Matching case
    v_ref = np.random.randn(512).astype(np.float32)
    res_match = matcher.match(v_ref, v_ref)
    assert res_match.is_match is True
    assert res_match.match_status == "Potential Match"
    assert res_match.similarity_score >= 0.70

    # Non-matching case
    v_diff = -v_ref
    res_no_match = matcher.match(v_ref, v_diff)
    assert res_no_match.is_match is False
    assert res_no_match.match_status == "No Match"


# ---------------------------------------------------------
# Standalone Pipeline Execution Script
# ---------------------------------------------------------

def run_pipeline_check():
    """
    Executes the standalone verification pipeline on available sample images.
    Prints formatted outputs as specified in the Phase 1 requirements.
    """
    print("=" * 60)
    print(" LocateMe -- Phase 1 Local ML Pipeline Verification")
    print("=" * 60)

    create_synthetic_test_images_if_needed()

    detector = FaceDetector()
    generator = FaceEmbeddingGenerator()
    matcher = FaceMatcher(threshold=0.68)

    test_images_dir = PROJECT_ROOT / "data" / "test_images"
    reg_images_dir = PROJECT_ROOT / "data" / "registered"

    # Find available sample images
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    reg_files = list(reg_images_dir.glob("*.jpg")) + list(reg_images_dir.glob("*.png"))

    # Pick primary test image
    primary_image_path = None
    for p in image_files:
        if "blank" not in p.name:
            primary_image_path = p
            break

    if not primary_image_path and image_files:
        primary_image_path = image_files[0]

    print(f"\n[1] Testing Face Detection on: {primary_image_path}")
    if primary_image_path:
        det_result = detector.detect_face(primary_image_path)
        detected_str = "YES" if det_result.is_detected else "NO"
        print(f"  Face detected: {detected_str}")

        if det_result.is_detected:
            if det_result.probability is not None:
                print(f"  Detection confidence: {det_result.probability * 100:.2f}%")
            if det_result.box is not None:
                print(f"  Bounding box: {det_result.box.astype(int).tolist()}")

            print("\n[2] Testing Embedding Extraction...")
            emb = generator.extract_from_tensor(det_result.face_tensor)
            if emb is not None:
                print(f"  Embedding shape: {emb.shape}")
                print(f"  Embedding dimension: {len(emb)}")
                print(f"  Vector sample (first 5): {np.round(emb[:5], 4)}")
            else:
                print("  Failed to extract embedding.")
        else:
            print(f"  Reason: {det_result.error_message}")
    else:
        print("  No sample image found in data/test_images/.")

    # Two-image comparison if available
    all_face_images = [p for p in (reg_files + image_files) if "blank" not in p.name]
    if len(all_face_images) >= 2:
        img_a = all_face_images[0]
        img_b = all_face_images[1]

        print(f"\n[3] Testing Two-Image Comparison:")
        print(f"  Image A: {img_a.name}")
        print(f"  Image B: {img_b.name}")

        emb_a = generator.generate_embedding(img_a, detector=detector)
        emb_b = generator.generate_embedding(img_b, detector=detector)

        if emb_a is not None and emb_b is not None:
            match_res = matcher.match(emb_a, emb_b)
            print(f"  Cosine similarity: {match_res.similarity_score:.4f}")
            print(f"  Configured threshold: {match_res.threshold:.2f}")
            print(f"  Match status: {match_res.match_status}")
            print(f"  Confidence tier: {match_res.confidence_tier}")
        else:
            print("  Could not detect faces in both comparison images.")
    else:
        print("\n[3] Note: Add two face images to data/ to run pairwise similarity comparison.")

    print("\n" + "=" * 60)
    print(" Pipeline Verification Completed.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline_check()
