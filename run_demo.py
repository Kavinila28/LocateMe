#!/usr/bin/env python3
"""
LocateMe — Interactive CLI Demonstration Script
Compares reference missing person photos against query surveillance frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output encoding across Windows/Linux terminals
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

# Ensure LocateMe project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.face_detector import FaceDetector, FaceDetectionResult
from ml.embedding import FaceEmbeddingGenerator, EXPECTED_EMBEDDING_DIM
from ml.matcher import FaceMatcher, DEFAULT_SIMILARITY_THRESHOLD, DISCLAIMER_TEXT


def format_header(title: str) -> str:
    line = "=" * 68
    return f"\n{line}\n  {title}\n{line}"


def format_card(title: str, items: list[tuple[str, str]]) -> str:
    line_width = 66
    out = [f"\n+-- [ {title} ] " + "-" * (line_width - len(title) - 8) + "+"]
    for label, val in items:
        out.append(f"|  * {label:<22}: {val:<36} |")
    out.append("+" + "-" * line_width + "+")
    return "\n".join(out)


def run_comparison(
    ref_path: Path,
    test_path: Path,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    device: Optional[str] = None,
) -> None:
    """
    Executes full face comparison pipeline between reference image and query image.
    """
    print(format_header("LocateMe — AI Missing Person Candidate Screening (Phase 1)"))
    print(f"Device: {device or 'auto'} | Experimental Similarity Threshold: {threshold:.2f}")

    if not ref_path.exists():
        print(f"\n[ERROR] Reference image not found: {ref_path}")
        return

    if not test_path.exists():
        print(f"\n[ERROR] Query test image not found: {test_path}")
        return

    print("\n[INFO] Initializing MTCNN Face Detector & InceptionResnetV1 (VGGFace2)...")
    detector = FaceDetector(device=device)
    embedding_gen = FaceEmbeddingGenerator(device=device)
    matcher = FaceMatcher(threshold=threshold)

    # 1. Process Reference Image
    print(f"\n[1/3] Processing Reference Image: {ref_path.name}...")
    ref_det: FaceDetectionResult = detector.detect_face(ref_path)

    ref_rel = str(ref_path.relative_to(PROJECT_ROOT)) if ref_path.is_relative_to(PROJECT_ROOT) else ref_path.name
    ref_items = [("File Path", ref_rel)]

    if not ref_det.is_detected:
        ref_items.extend([
            ("Detection Status", "FAILED / NO FACE DETECTED"),
            ("Details", ref_det.error_message or "Unknown"),
        ])
        print(format_card("REFERENCE IMAGE ANALYSIS", ref_items))
        print("\n[ABORT] Cannot proceed without a valid face detected in reference image.")
        return

    ref_items.extend([
        ("Detection Status", "SUCCESS (Face Detected)"),
        ("Face Confidence", f"{ref_det.probability * 100:.2f}%" if ref_det.probability else "N/A"),
        ("Bounding Box [x1,y1,x2,y2]", str(ref_det.box.astype(int).tolist()) if ref_det.box is not None else "N/A"),
    ])

    ref_emb = embedding_gen.extract_from_tensor(ref_det.face_tensor)
    if ref_emb is None:
        print("\n[ERROR] Failed to extract embedding from reference face tensor.")
        return

    ref_items.extend([
        ("Embedding Shape", str(ref_emb.shape)),
        ("Embedding Dimension", f"{len(ref_emb)} (Expected {EXPECTED_EMBEDDING_DIM})"),
        ("L2 Vector Norm", f"{np.linalg.norm(ref_emb):.4f}"),
    ])
    print(format_card("REFERENCE IMAGE ANALYSIS", ref_items))

    # 2. Process Query / Test Image
    print(f"\n[2/3] Processing Query Test Image: {test_path.name}...")
    test_det: FaceDetectionResult = detector.detect_face(test_path)

    test_rel = str(test_path.relative_to(PROJECT_ROOT)) if test_path.is_relative_to(PROJECT_ROOT) else test_path.name
    test_items = [("File Path", test_rel)]

    if not test_det.is_detected:
        test_items.extend([
            ("Detection Status", "FAILED / NO FACE DETECTED"),
            ("Details", test_det.error_message or "Unknown"),
        ])
        print(format_card("QUERY IMAGE ANALYSIS", test_items))
        print("\n[ABORT] Cannot compare because no face was detected in query image.")
        return

    test_items.extend([
        ("Detection Status", "SUCCESS (Face Detected)"),
        ("Face Confidence", f"{test_det.probability * 100:.2f}%" if test_det.probability else "N/A"),
        ("Bounding Box [x1,y1,x2,y2]", str(test_det.box.astype(int).tolist()) if test_det.box is not None else "N/A"),
    ])

    test_emb = embedding_gen.extract_from_tensor(test_det.face_tensor)
    if test_emb is None:
        print("\n[ERROR] Failed to extract embedding from query face tensor.")
        return

    test_items.extend([
        ("Embedding Shape", str(test_emb.shape)),
        ("Embedding Dimension", f"{len(test_emb)} (Expected {EXPECTED_EMBEDDING_DIM})"),
        ("L2 Vector Norm", f"{np.linalg.norm(test_emb):.4f}"),
    ])
    print(format_card("QUERY IMAGE ANALYSIS", test_items))

    # 3. Match & Compare Embeddings
    print("\n[3/3] Computing Cosine Similarity Metric...")
    match_result = matcher.match(ref_emb, test_emb, threshold=threshold)

    match_symbol = "[MATCH]" if match_result.is_match else "[NO MATCH]"
    status_label = f"{match_symbol} {match_result.match_status.upper()}"

    comparison_items = [
        ("Cosine Similarity", f"{match_result.similarity_score:.4f}"),
        ("Decision Threshold", f"{match_result.threshold:.4f}"),
        ("Confidence Category", match_result.confidence_tier),
        ("Evaluation Status", status_label),
    ]

    print(format_card("MATCH SCREENING SUMMARY", comparison_items))

    # Disclaimer banner
    print("\n" + "-" * 68)
    print(f"NOTICE: {DISCLAIMER_TEXT}")
    print("-" * 68 + "\n")


def find_default_images() -> tuple[Optional[Path], Optional[Path]]:
    """Locate sample registered and test images for quick execution."""
    reg_dir = PROJECT_ROOT / "data" / "registered"
    test_dir = PROJECT_ROOT / "data" / "test_images"

    reg_candidates = sorted(list(reg_dir.glob("*.jpg")) + list(reg_dir.glob("*.png")))
    test_candidates = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))

    # Filter out blank images
    reg_faces = [p for p in reg_candidates if "blank" not in p.name]
    test_faces = [p for p in test_candidates if "blank" not in p.name]

    ref_img = reg_faces[0] if reg_faces else None
    test_img = test_faces[0] if test_faces else None

    return ref_img, test_img


def main():
    parser = argparse.ArgumentParser(
        description="LocateMe: Face similarity comparison demo between reference and query images."
    )
    parser.add_argument(
        "-r", "--reference",
        type=str,
        help="Path to reference face image (e.g., registered missing person photo).",
    )
    parser.add_argument(
        "-t", "--test",
        type=str,
        help="Path to query image to compare against (e.g., CCTV crop or test photo).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold for match evaluation (default: {DEFAULT_SIMILARITY_THRESHOLD}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch compute device ('cpu' or 'cuda'). Auto-detected by default.",
    )

    args = parser.parse_args()

    ref_path = Path(args.reference) if args.reference else None
    test_path = Path(args.test) if args.test else None

    # Fallback to default sample images if not specified
    if ref_path is None or test_path is None:
        def_ref, def_test = find_default_images()
        if ref_path is None:
            ref_path = def_ref
        if test_path is None:
            test_path = def_test

    if ref_path is None or test_path is None:
        print("[!] No reference and/or query image specified.")
        print("Usage example:\n  python run_demo.py --reference data/registered/person_a_ref.jpg --test data/test_images/person_a_cctv.jpg")
        sys.exit(1)

    run_comparison(
        ref_path=ref_path,
        test_path=test_path,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
