"""
Unit and Mock Tests for Supabase pgvector Integration (Phase 4)
Verifies Supabase client configuration, cloud mode, and seamless local fallback without requiring live credentials.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from ml.gallery import GalleryManager, RegisteredPerson
from ml.supabase_client import SupabaseConfig, SupabaseGalleryClient, get_supabase_client


def test_supabase_config_placeholders():
    """Verify that placeholder/dummy URLs are correctly rejected."""
    cfg_empty = SupabaseConfig(url="", key="")
    assert not cfg_empty.is_configured

    cfg_placeholder = SupabaseConfig(
        url="https://your-project.supabase.co",
        key="your-supabase-anon-or-service-role-key"
    )
    assert not cfg_placeholder.is_configured

    cfg_valid = SupabaseConfig(
        url="https://xyzabcdef.supabase.co",
        key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummysecretkey"
    )
    assert cfg_valid.is_configured
    assert cfg_valid.bucket_name == "missing-persons-photos"


def test_supabase_client_mock_methods():
    """Verify SupabaseGalleryClient methods against a mocked Supabase SDK client."""
    mock_sdk_client = MagicMock()

    # Mock storage
    mock_storage = MagicMock()
    mock_storage.get_public_url.return_value = "https://cdn.supabase.co/storage/v1/object/public/missing-persons-photos/person_a.jpg"
    mock_sdk_client.storage.from_.return_value = mock_storage

    # Mock table
    mock_table = MagicMock()
    mock_table.upsert.return_value.execute.return_value.data = [{"id": "uuid-1", "name": "Jane Doe"}]
    mock_table.select.return_value.execute.return_value.data = [
        {
            "id": "uuid-1",
            "person_id": "jane_doe",
            "name": "Jane Doe",
            "image_url": "https://cdn.supabase.co/photo.jpg",
            "embedding": [0.1] * 512,
            "created_at": "2026-08-16T12:00:00Z",
            "metadata": {"age": 25},
        }
    ]
    mock_table.delete.return_value.eq.return_value.execute.return_value.data = [{"id": "uuid-1"}]
    mock_sdk_client.table.return_value = mock_table

    # Mock RPC
    mock_sdk_client.rpc.return_value.execute.return_value.data = [
        {"person_id": "jane_doe", "similarity_score": 0.92}
    ]

    client = SupabaseGalleryClient(
        config=SupabaseConfig(url="https://test.supabase.co", key="test_key"),
        client=mock_sdk_client,
    )

    assert client.is_active

    # 1. Upload photo
    url = client.upload_photo(b"image_bytes", "test.jpg")
    assert url == "https://cdn.supabase.co/storage/v1/object/public/missing-persons-photos/person_a.jpg"

    # 2. Insert missing person
    res = client.insert_missing_person(
        person_id="jane_doe",
        name="Jane Doe",
        image_url=url,
        embedding=np.zeros(512, dtype=np.float32),
    )
    assert res is not None
    assert res["name"] == "Jane Doe"

    # 3. Fetch all persons
    records = client.fetch_all_persons()
    assert len(records) == 1
    assert records[0]["person_id"] == "jane_doe"

    # 4. Match missing persons RPC
    matches = client.match_missing_persons(np.zeros(512), threshold=0.68)
    assert len(matches) == 1
    assert matches[0]["similarity_score"] == 0.92

    # 5. Delete missing person
    assert client.delete_missing_person("jane_doe")


def test_gallery_manager_local_fallback(tmp_path):
    """Verify GalleryManager defaults to local mode when Supabase is unconfigured."""
    dummy_reg = tmp_path / "registered"
    dummy_reg.mkdir()

    gallery = GalleryManager(
        gallery_dir=dummy_reg,
        supabase_client=None,
    )

    assert not gallery.is_cloud_mode
    assert gallery.storage_mode == "local"
    assert gallery.count == 0


def test_gallery_manager_cloud_mode_sync(tmp_path):
    """Verify GalleryManager syncs with Supabase records in cloud mode."""
    mock_client = MagicMock()
    mock_client.is_active = True
    mock_client.fetch_all_persons.return_value = [
        {
            "id": "uuid-1",
            "person_id": "cloud_person_1",
            "name": "Cloud Person 1",
            "image_url": "https://cdn.supabase.co/p1.jpg",
            "embedding": [0.0] * 512,
            "created_at": "2026-08-16T12:00:00Z",
            "metadata": {},
        },
        {
            "id": "uuid-2",
            "person_id": "cloud_person_2",
            "name": "Cloud Person 2",
            "image_url": "https://cdn.supabase.co/p2.jpg",
            "embedding": [0.1] * 512,
            "created_at": "2026-08-16T12:00:00Z",
            "metadata": {},
        },
    ]

    gallery = GalleryManager(
        gallery_dir=tmp_path,
        supabase_client=mock_client,
    )

    assert gallery.is_cloud_mode
    assert gallery.storage_mode == "supabase"
    assert gallery.count == 2
    assert gallery.persons[0].person_id == "cloud_person_1"
    assert gallery._gallery_matrix.shape == (2, 512)
