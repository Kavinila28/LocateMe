"""
LocateMe — FastAPI REST API Test Suite
Verifies health, gallery management, and image screening endpoints.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health endpoint returns valid system diagnostics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["registered_persons_count"] >= 2
    assert "MTCNN" in data["model_architecture"]


def test_list_gallery_endpoint():
    """Verify /api/v1/gallery lists all registered persons."""
    response = client.get("/api/v1/gallery")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] >= 2
    assert len(data["persons"]) >= 2
    person_ids = [p["person_id"] for p in data["persons"]]
    assert any("person_a" in pid for pid in person_ids)


def test_screen_image_endpoint():
    """Verify /api/v1/screen/image flags Person A candidate from query photo."""
    query_path = PROJECT_ROOT / "data" / "test_images" / "person_a_cctv.jpg"
    assert query_path.exists()

    with open(query_path, "rb") as f:
        response = client.post(
            "/api/v1/screen/image",
            files={"file": ("person_a_cctv.jpg", f, "image/jpeg")},
            data={"threshold": "0.68"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total_faces_detected"] >= 1
    assert data["potential_matches_found"] >= 1

    first_det = data["detections"][0]
    assert first_det["best_match"] is not None
    assert "person_a" in first_det["best_match"]["person_id"]
    assert first_det["best_match"]["similarity_score"] >= 0.68
    assert first_det["best_match"]["is_match"] is True


def test_screen_image_blank():
    """Verify /api/v1/screen/image returns 0 faces on a blank image without error."""
    blank_path = PROJECT_ROOT / "data" / "test_images" / "blank_image.jpg"
    assert blank_path.exists()

    with open(blank_path, "rb") as f:
        response = client.post(
            "/api/v1/screen/image",
            files={"file": ("blank_image.jpg", f, "image/jpeg")},
            data={"threshold": "0.68"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total_faces_detected"] == 0
    assert data["potential_matches_found"] == 0
