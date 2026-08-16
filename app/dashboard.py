"""
LocateMe — Streamlit Operator & Hackathon Dashboard
Interactive web interface for missing person gallery management, photo screening,
and CCTV video stream analysis.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.embedding import FaceEmbeddingGenerator, get_default_generator, EXPECTED_EMBEDDING_DIM
from ml.face_detector import FaceDetector, get_default_detector
from ml.gallery import GalleryManager, RegisteredPerson
from ml.matcher import DEFAULT_SIMILARITY_THRESHOLD, DISCLAIMER_TEXT, evaluate_confidence_tier
from ml.video_processor import VideoProcessor, VideoProcessingSummary

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------

st.set_page_config(
    page_title="LocateMe — Missing Person Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = PROJECT_ROOT / "data"
REG_DIR = DATA_DIR / "registered"
TEST_IMG_DIR = DATA_DIR / "test_images"
TEST_VID_DIR = DATA_DIR / "test_videos"

for d in [DATA_DIR, REG_DIR, TEST_IMG_DIR, TEST_VID_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Cached ML Pipeline Singletons
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Initializing MTCNN & InceptionResnetV1 (VGGFace2)...")
def load_pipeline():
    detector = FaceDetector()
    generator = FaceEmbeddingGenerator()
    gallery = GalleryManager(gallery_dir=REG_DIR, detector=detector, generator=generator)
    return detector, generator, gallery


detector, generator, gallery = load_pipeline()


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------

with st.sidebar:
    st.title("LocateMe")
    st.caption("AI-Assisted Missing Person Screening")
    st.markdown("---")

    st.subheader("⚙️ Screening Settings")
    threshold = st.slider(
        "Similarity Threshold",
        min_value=0.50,
        max_value=0.90,
        value=0.68,
        step=0.01,
        help="Cosine similarity cutoff for flagging candidate matches.",
    )
    frame_step = st.select_slider(
        "Video Frame Sampling",
        options=[1, 2, 5, 10, 15],
        value=5,
        help="Process every Nth frame for performance optimization.",
    )

    st.markdown("---")
    st.subheader("🖥 System Status")
    st.info(
        f"• **Device**: `{detector.device}`\n"
        f"• **Storage**: `{'Supabase Cloud (pgvector)' if gallery.is_cloud_mode else 'Local (.npz cache)'}`\n"
        f"• **Model**: `InceptionResnetV1`\n"
        f"• **Weights**: `VGGFace2 (512-D)`\n"
        f"• **Gallery Size**: `{gallery.count} Registered`"
    )

    st.markdown("---")
    st.caption(
        "🔒 **Controlled Prototype Notice**\n"
        "Screening scores are algorithmic feature proximity metrics for authorized test evaluation. "
        "They do not constitute positive identification."
    )


# ---------------------------------------------------------
# Main Interface Tabs
# ---------------------------------------------------------

tab_gallery, tab_image, tab_video, tab_about = st.tabs([
    "🗂 Missing Persons Gallery",
    "🔍 Photo & Snapshot Screening",
    "📹 CCTV Video Analysis",
    "ℹ️ Technical Specs & Ethics",
])


# =========================================================
# TAB 1: MISSING PERSONS GALLERY
# =========================================================
with tab_gallery:
    st.header("🗂 Registered Missing Persons Gallery")
    st.write(
        "Reference database of authorized missing-person portrait photographs. "
        "512-D facial embeddings are precomputed and cached for high-speed screening."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Active Registry ({gallery.count} Persons)")
        if gallery.count == 0:
            st.warning("No missing persons registered yet. Use the form to register reference photos.")
        else:
            cols = st.columns(3)
            for idx, person in enumerate(gallery.persons):
                with cols[idx % 3]:
                    if person.image_url and (person.image_url.startswith("http://") or person.image_url.startswith("https://")):
                        st.image(person.image_url, caption=person.name, use_container_width=True)
                    elif Path(person.image_path).exists():
                        st.image(str(person.image_path), caption=person.name, use_container_width=True)
                    else:
                        st.text(f"[{person.name}]")
                    st.caption(f"**ID**: `{person.person_id}`\n**Registered**: {person.registered_at[:10]}")

    with col2:
        st.subheader("➕ Register New Person")
        with st.form("register_form", clear_on_submit=True):
            reg_name = st.text_input("Full Name *", placeholder="e.g., Jane Doe")
            reg_file = st.file_uploader("Reference Portrait Photo *", type=["jpg", "jpeg", "png", "webp"])
            submitted = st.form_submit_button("Register in Gallery", use_container_width=True)

            if submitted:
                if not reg_name or not reg_file:
                    st.error("Please provide both name and a reference photo.")
                else:
                    # Save photo to data/registered/
                    safe_stem = reg_name.lower().replace(" ", "_")
                    suffix = Path(reg_file.name).suffix
                    save_path = REG_DIR / f"{safe_stem}_ref{suffix}"

                    with open(save_path, "wb") as f:
                        f.write(reg_file.getbuffer())

                    # Register
                    reg_person = gallery.register_person(name=reg_name, image_path=save_path, person_id=safe_stem)
                    if reg_person:
                        gallery.save_cache()
                        st.success(f"Successfully registered '{reg_name}'! Embeddings precomputed.")
                        st.rerun()
                    else:
                        if save_path.exists():
                            save_path.unlink()
                        st.error("Could not detect a clear face in the uploaded photo. Please try a clearer frontal portrait.")


# =========================================================
# TAB 2: PHOTO SCREENING
# =========================================================
with tab_image:
    st.header("🔍 Photo & Snapshot Screening")
    st.write("Upload a surveillance snapshot or photo to compare against the entire registered missing-person gallery.")

    col_up, col_res = st.columns([1, 1])

    with col_up:
        st.subheader("1. Select Query Snapshot")
        img_upload = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png", "webp"], key="query_img_up")

        # Quick sample selector
        sample_imgs = list(TEST_IMG_DIR.glob("*.jpg")) + list(TEST_IMG_DIR.glob("*.png"))
        selected_sample = st.selectbox(
            "Or select bundled sample image:",
            options=["None"] + [p.name for p in sample_imgs if "blank" not in p.name],
        )

        query_pil = None
        if img_upload is not None:
            query_pil = Image.open(img_upload).convert("RGB")
        elif selected_sample != "None":
            query_pil = Image.open(TEST_IMG_DIR / selected_sample).convert("RGB")

        if query_pil:
            st.image(query_pil, caption="Query Photo / Snapshot", use_container_width=True)

    with col_res:
        st.subheader("2. Detection & Screening Results")
        if query_pil is None:
            st.info("Upload a photo or select a sample from the left panel to begin screening.")
        else:
            with st.spinner("Detecting faces and computing cosine similarity..."):
                face_results = detector.detect_all_faces(query_pil)

            if not face_results:
                st.warning("No face detected in the query photo.")
            else:
                st.success(f"Detected **{len(face_results)} face(s)** in query image.")

                for idx, det in enumerate(face_results):
                    st.markdown(f"#### Face #{idx + 1}")
                    if det.probability:
                        st.caption(f"MTCNN Detection Confidence: **{det.probability * 100:.1f}%**")

                    emb = generator.extract_from_tensor(det.face_tensor)
                    if emb is None:
                        st.error("Failed to extract face embedding.")
                        continue

                    # Fast gallery search
                    matches = gallery.search(emb, threshold=threshold)
                    if not matches:
                        st.info("No candidates in gallery.")
                        continue

                    top_person, top_match = matches[0]

                    if top_match.is_match:
                        st.success(
                            f"🟢 **POTENTIAL MATCH: {top_person.name}**\n\n"
                            f"• **Cosine Similarity**: `{top_match.similarity_score:.4f}` (Threshold: `{threshold:.2f}`)\n"
                            f"• **Confidence Tier**: `{top_match.confidence_tier}`"
                        )
                        # Side-by-side comparison
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("Registered Reference Portrait")
                            if top_person.image_url and (top_person.image_url.startswith("http://") or top_person.image_url.startswith("https://")):
                                st.image(top_person.image_url, width=160)
                            elif Path(top_person.image_path).exists():
                                st.image(top_person.image_path, width=160)
                        with c2:
                            st.caption("Screening Status")
                            st.metric(label="Match Confidence", value=f"{top_match.similarity_score * 100:.1f}%")
                    else:
                        st.warning(
                            f"🔴 **NO MATCH (Unrecognized Individual)**\n\n"
                            f"• Highest Candidate: `{top_person.name}` (Sim: `{top_match.similarity_score:.4f}` < `{threshold:.2f}`)\n"
                            f"• Evaluation: Unrecognized / Non-Matching"
                        )


# =========================================================
# TAB 3: CCTV VIDEO ANALYSIS
# =========================================================
with tab_video:
    st.header("📹 CCTV & Surveillance Video Feed Screening")
    st.write("Process recorded surveillance footage or video clips to detect and track registered missing persons.")

    col_v_in, col_v_run = st.columns([1, 2])

    with col_v_in:
        st.subheader("1. Video Source")
        vid_upload = st.file_uploader("Upload Surveillance Video (.mp4, .avi)", type=["mp4", "avi", "mov"])

        sample_vids = list(TEST_VID_DIR.glob("*.mp4"))
        selected_vid = st.selectbox(
            "Or use pre-bundled benchmark video:",
            options=["None"] + [v.name for v in sample_vids if "annotated" not in v.name],
        )

        target_video_path: Optional[Path] = None
        if vid_upload is not None:
            target_video_path = TEST_VID_DIR / f"upload_{vid_upload.name}"
            with open(target_video_path, "wb") as f:
                f.write(vid_upload.getbuffer())
        elif selected_vid != "None":
            target_video_path = TEST_VID_DIR / selected_vid

        if target_video_path:
            st.video(str(target_video_path))
            start_btn = st.button("🚀 Start Surveillance Screening", use_container_width=True, type="primary")
        else:
            start_btn = False

    with col_v_run:
        st.subheader("2. Surveillance Analysis Output")
        if not target_video_path:
            st.info("Select or upload a video clip on the left to run feed analysis.")
        elif start_btn:
            prog_bar = st.progress(0, text="Initializing video screening...")
            stat_text = st.empty()

            processor = VideoProcessor(
                gallery=gallery,
                detector=detector,
                generator=generator,
                threshold=threshold,
                frame_step=frame_step,
            )

            out_video = TEST_VID_DIR / "dashboard_annotated_feed.mp4"
            crops_dir = TEST_VID_DIR / "detected_crops"

            def on_progress(cur, tot, matches):
                pct = cur / tot if tot > 0 else 0
                prog_bar.progress(pct, text=f"Processing Frame {cur}/{tot} | Potential Matches: {matches}")

            summary: VideoProcessingSummary = processor.process_video(
                video_source=str(target_video_path),
                output_video_path=out_video,
                export_crops_dir=crops_dir,
                progress_callback=on_progress,
            )

            prog_bar.progress(1.0, text="Screening Complete!")

            # Display Stats
            st.success(
                f"Screening completed in **{summary.elapsed_time_seconds:.2f}s** ({summary.processing_fps:.1f} FPS)!\n\n"
                f"• Processed Frames: **{summary.processed_frames}/{summary.total_frames}**\n"
                f"• Total Candidate Sighting Events: **{summary.total_matches_detected}**\n"
                f"• Candidates Identified: **{', '.join(summary.unique_candidates_matched) or 'None'}**"
            )

            if out_video.exists():
                st.subheader("Annotated Surveillance Playback")
                st.video(str(out_video))

            # Detection timeline
            if summary.matches:
                st.subheader("Detected Sighting Timeline")
                timeline_data = []
                for m in summary.matches:
                    timeline_data.append({
                        "Time": m.timestamp_formatted,
                        "Frame": m.frame_index,
                        "Candidate Name": m.person_name,
                        "Similarity Score": f"{m.similarity_score:.4f}",
                        "Face Confidence": f"{m.detection_confidence * 100:.1f}%",
                        "Tier": m.confidence_tier,
                    })
                df = pd.DataFrame(timeline_data)
                st.dataframe(df, use_container_width=True)

                # Download JSON report
                report_json = json.dumps(summary.to_dict(), indent=2)
                st.download_button(
                    label="📥 Download JSON Screening Report",
                    data=report_json,
                    file_name="locate_me_match_report.json",
                    mime="application/json",
                )


# =========================================================
# TAB 4: TECHNICAL SPECS & ETHICS
# =========================================================
with tab_about:
    st.header("ℹ️ Technical Architecture & Ethical Boundaries")

    st.markdown(r"""
    ### 1. Model Pipeline
    - **Face Localization**: Multi-task Cascaded Convolutional Networks (**MTCNN**) crops and affine-aligns faces to $160 \times 160$ RGB tensors.
    - **Feature Representation**: **InceptionResnetV1** pretrained on **VGGFace2** extracts deep facial representations and projects them onto a unit-normalized 512-dimensional hypersphere ($\|e\|_2 = 1.0$).
    - **Similarity Metric**: Cosine Similarity:
      $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
    - **Vectorized Search**: The gallery precomputes an $N \times 512$ matrix $G$. Candidate screening evaluates $S = G \cdot q$ via fast matrix-vector dot products.

    ### 2. Threshold Calibration Guide
    - **`0.85 - 1.00`**: High confidence match. Extremely strong facial similarity.
    - **`0.68 - 0.84`**: Moderate candidate match. Recommended screening range for human review.
    - **`0.55 - 0.67`**: Borderline candidate. High false-positive rate; requires cautious review.
    - **`< 0.55`**: Non-matching individuals.

    ### 3. Ethical & Controlled Prototype Notice
    LocateMe is designed as an authorized, human-in-the-loop search assistance tool for controlled missing-person operations.
    It does **not** perform mass surveillance or biometric identification without consent.
    """)
