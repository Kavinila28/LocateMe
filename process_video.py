#!/usr/bin/env python3
"""
LocateMe — Video Feed Screening CLI Tool
Processes CCTV / surveillance video footage against registered missing person galleries.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output encoding across Windows/Linux terminals
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.gallery import GalleryManager
from ml.video_processor import VideoProcessor, VideoProcessingSummary
from ml.matcher import DEFAULT_SIMILARITY_THRESHOLD, DISCLAIMER_TEXT


def format_header(title: str) -> str:
    line = "=" * 70
    return f"\n{line}\n  {title}\n{line}"


def format_card(title: str, items: list[tuple[str, str]]) -> str:
    line_width = 68
    out = [f"\n+-- [ {title} ] " + "-" * (line_width - len(title) - 8) + "+"]
    for label, val in items:
        out.append(f"|  * {label:<24}: {val:<36} |")
    out.append("+" + "-" * line_width + "+")
    return "\n".join(out)


def progress_indicator(current: int, total: int, matches: int) -> None:
    """Print console progress for video frames."""
    pct = (current / total * 100) if total > 0 else 0
    bar_len = 25
    filled = int(bar_len * current // total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_len - filled)
    msg = f"\r[Processing] [{bar}] {pct:5.1f}% | Frame: {current}/{total} | Matches: {matches}"
    sys.stdout.write(msg)
    sys.stdout.flush()


def run_video_screening(
    video_path: Path,
    gallery_dir: Path,
    output_video_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    frame_step: int = 5,
    export_crops: bool = True,
    device: Optional[str] = None,
) -> None:
    """
    Executes full video screening against registered missing person gallery.
    """
    print(format_header("LocateMe — Video Feed Surveillance Screening (Phase 2)"))
    print(f"Video Source : {video_path}")
    print(f"Gallery Dir  : {gallery_dir}")
    print(f"Threshold    : {threshold:.2f} | Frame Sampling: every {frame_step} frames")

    if not video_path.exists():
        print(f"\n[ERROR] Video file not found: {video_path}")
        return

    if not gallery_dir.exists():
        print(f"\n[ERROR] Gallery directory not found: {gallery_dir}")
        return

    # 1. Initialize Gallery
    print("\n[1/3] Loading Registered Missing Person Gallery...")
    gallery = GalleryManager(gallery_dir=gallery_dir)
    print(f"  -> Successfully loaded {gallery.count} registered candidates.")

    if gallery.count == 0:
        print("[ABORT] No registered persons found in gallery. Add reference photos first.")
        return

    for p in gallery.persons:
        print(f"     * [{p.person_id}] {p.name}")

    # 2. Process Video Feed
    print(f"\n[2/3] Analyzing Video Stream: {video_path.name}...")
    processor = VideoProcessor(
        gallery=gallery,
        threshold=threshold,
        frame_step=frame_step,
    )

    crops_dir = (report_path.parent / "detected_crops") if (export_crops and report_path) else None

    summary: VideoProcessingSummary = processor.process_video(
        video_source=str(video_path),
        output_video_path=output_video_path,
        export_crops_dir=crops_dir,
        progress_callback=progress_indicator,
    )
    print("\n[Done] Video analysis complete!")

    # 3. Export Report
    if report_path:
        summary.save_json(report_path)
        print(f"\n[Info] Match report saved to: {report_path}")

    # 4. Display Screening Summary Card
    stats_items = [
        ("Source Video", video_path.name),
        ("Total Frames", str(summary.total_frames)),
        ("Sampled Frames", str(summary.processed_frames)),
        ("Video Duration", f"{summary.duration_seconds:.2f}s"),
        ("Processing Time", f"{summary.elapsed_time_seconds:.2f}s ({summary.processing_fps:.1f} FPS)"),
        ("Potential Match Events", str(summary.total_matches_detected)),
        ("Matched Candidates", ", ".join(summary.unique_candidates_matched) or "None"),
        ("Annotated Video", str(output_video_path.name) if output_video_path else "Not saved"),
    ]

    print(format_card("VIDEO SCREENING SUMMARY", stats_items))

    # Print candidate detection timeline if matches found
    if summary.matches:
        print("\n+-- [ DETECTED CANDIDATE TIMELINE ] " + "-" * 34 + "+")
        for m in summary.matches[:10]:  # Show first 10 events
            print(f"|  * {m.timestamp_formatted} (F:{m.frame_index:04d}) -> {m.person_name:<16} | Sim: {m.similarity_score:.4f} ({m.confidence_tier}) |")
        if len(summary.matches) > 10:
            print(f"|  * ... and {len(summary.matches) - 10} more match events recorded. |")
        print("+" + "-" * 68 + "+")

    # Disclaimer banner
    print("\n" + "-" * 70)
    print(f"NOTICE: {DISCLAIMER_TEXT}")
    print("-" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="LocateMe: Screen video footage against registered missing person galleries."
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        help="Path to input video file (e.g. data/test_videos/cctv_sample.mp4).",
    )
    parser.add_argument(
        "-g", "--gallery",
        type=str,
        default="data/registered",
        help="Directory containing registered reference portraits (default: data/registered).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/test_videos/annotated_feed.mp4",
        help="Path to export annotated output video (default: data/test_videos/annotated_feed.mp4).",
    )
    parser.add_argument(
        "-r", "--report",
        type=str,
        default="data/test_videos/match_report.json",
        help="Path to save JSON match report (default: data/test_videos/match_report.json).",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD}).",
    )
    parser.add_argument(
        "-s", "--frame-step",
        type=int,
        default=5,
        help="Process every Nth frame for performance (default: 5).",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Disable saving candidate face crops.",
    )

    args = parser.parse_args()

    vid_path = Path(args.video) if args.video else None
    if not vid_path:
        # Check default test video directory
        test_vids = list((PROJECT_ROOT / "data" / "test_videos").glob("*.mp4"))
        if test_vids:
            vid_path = test_vids[0]

    if not vid_path:
        print("[!] No input video specified and no default test video found.")
        print("Usage:\n  python process_video.py --video data/test_videos/sample_cctv.mp4")
        sys.exit(1)

    gallery_path = PROJECT_ROOT / args.gallery
    out_video_path = PROJECT_ROOT / args.output if args.output else None
    rep_path = PROJECT_ROOT / args.report if args.report else None

    run_video_screening(
        video_path=vid_path,
        gallery_dir=gallery_path,
        output_video_path=out_video_path,
        report_path=rep_path,
        threshold=args.threshold,
        frame_step=args.frame_step,
        export_crops=not args.no_crops,
    )


if __name__ == "__main__":
    main()
