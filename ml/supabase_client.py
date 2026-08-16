"""
LocateMe — Supabase PostgreSQL & pgvector Client Module (Phase 4)
Provides cloud persistence for missing-person records, reference portraits (Storage),
and vectorized cosine similarity search (pgvector HNSW index).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from dotenv import load_dotenv

# Load environment variables if .env exists
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_BUCKET_NAME = "missing-persons-photos"
TABLE_NAME = "missing_persons"


class SupabaseConfig:
    """Encapsulates Supabase connection settings."""

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ) -> None:
        self.url = (url or os.getenv("SUPABASE_URL", "")).strip()
        self.key = (key or os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")).strip()
        self.bucket_name = (bucket_name or os.getenv("SUPABASE_BUCKET", DEFAULT_BUCKET_NAME)).strip()

    @property
    def is_configured(self) -> bool:
        """Check whether valid non-placeholder credentials exist."""
        if not self.url or not self.key:
            return False
        # Filter out common placeholders
        if "your-project.supabase.co" in self.url or "your-supabase" in self.key:
            return False
        return True


class SupabaseGalleryClient:
    """
    Client for interacting with Supabase PostgreSQL (pgvector) and Storage.
    """

    def __init__(self, config: Optional[SupabaseConfig] = None, client: Optional[Any] = None) -> None:
        self.config = config or SupabaseConfig()
        self._client = client

        if self._client is None and self.config.is_configured:
            try:
                from supabase import create_client
                self._client = create_client(self.config.url, self.config.key)
                logger.info("Successfully initialized Supabase client.")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase client: {e}. Falling back to local mode.")
                self._client = None

    @property
    def is_active(self) -> bool:
        """True if Supabase client is properly initialized and configured."""
        return self._client is not None

    def upload_photo(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str = "image/jpeg",
    ) -> Optional[str]:
        """
        Upload reference portrait to Supabase Storage bucket and return public URL.

        Args:
            file_bytes: Raw binary image payload.
            filename: Destination filename within bucket.
            content_type: MIME type of the image.

        Returns:
            Public CDN URL string if successful, else None.
        """
        if not self.is_active:
            return None

        try:
            storage = self._client.storage.from_(self.config.bucket_name)
            # Upsert image
            storage.upload(
                path=filename,
                file=file_bytes,
                file_options={"content-type": content_type, "upsert": "true"},
            )
            public_url = storage.get_public_url(filename)
            logger.info(f"Uploaded photo to Supabase Storage: {filename}")
            return public_url
        except Exception as e:
            logger.error(f"Failed to upload photo to Supabase Storage: {e}")
            return None

    def insert_missing_person(
        self,
        person_id: str,
        name: str,
        image_url: str,
        embedding: Union[np.ndarray, List[float]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Insert or update a missing-person record with 512-D embedding in Supabase.

        Args:
            person_id: Unique identifier for the person.
            name: Display name.
            image_url: Public image URL.
            embedding: 512-D unit-normalized feature vector.
            metadata: Optional additional JSON attributes.

        Returns:
            Inserted database record dict if successful, else None.
        """
        if not self.is_active:
            return None

        emb_list = (
            [float(x) for x in embedding.flatten()]
            if isinstance(embedding, np.ndarray)
            else [float(x) for x in embedding]
        )

        payload = {
            "person_id": person_id,
            "name": name,
            "image_url": image_url,
            "embedding": emb_list,
            "metadata": metadata or {},
        }

        try:
            response = self._client.table(TABLE_NAME).upsert(payload, on_conflict="person_id").execute()
            if response.data:
                logger.info(f"Upserted missing person into Supabase: {name} ({person_id})")
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to insert missing person into Supabase: {e}")
            return None

    def fetch_all_persons(self) -> List[Dict[str, Any]]:
        """
        Retrieve all registered missing persons and their embeddings from Supabase.

        Returns:
            List of dictionary records.
        """
        if not self.is_active:
            return []

        try:
            response = (
                self._client.table(TABLE_NAME)
                .select("id, person_id, name, image_url, embedding, metadata, created_at")
                .execute()
            )
            return response.data or []
        except Exception as e:
            logger.error(f"Failed to fetch missing persons from Supabase: {e}")
            return []

    def delete_missing_person(self, person_id: str) -> bool:
        """
        Delete a missing person from Supabase database.

        Args:
            person_id: Identifier of the person to delete.

        Returns:
            True if deletion succeeded, else False.
        """
        if not self.is_active:
            return False

        try:
            self._client.table(TABLE_NAME).delete().eq("person_id", person_id).execute()
            logger.info(f"Deleted missing person from Supabase: {person_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete missing person from Supabase: {e}")
            return False

    def match_missing_persons(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        threshold: float = 0.68,
        count: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Execute pgvector RPC search on Supabase using cosine similarity.

        Args:
            query_embedding: 512-D unit-normalized query vector.
            threshold: Cosine similarity threshold.
            count: Maximum candidate matches to return.

        Returns:
            List of candidate match records with similarity_score.
        """
        if not self.is_active:
            return []

        emb_list = (
            [float(x) for x in query_embedding.flatten()]
            if isinstance(query_embedding, np.ndarray)
            else [float(x) for x in query_embedding]
        )

        params = {
            "query_embedding": emb_list,
            "match_threshold": float(threshold),
            "match_count": int(count),
        }

        try:
            response = self._client.rpc("match_missing_persons", params).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Supabase RPC match_missing_persons failed: {e}")
            return []


# Global singleton cache
_supabase_client_singleton: Optional[SupabaseGalleryClient] = None


def get_supabase_client(force_refresh: bool = False) -> Optional[SupabaseGalleryClient]:
    """
    Get or initialize the global Supabase client singleton.
    Returns None if Supabase is not configured.
    """
    global _supabase_client_singleton
    if _supabase_client_singleton is None or force_refresh:
        cfg = SupabaseConfig()
        if cfg.is_configured:
            client = SupabaseGalleryClient(cfg)
            if client.is_active:
                _supabase_client_singleton = client
                return _supabase_client_singleton
        _supabase_client_singleton = None
    return _supabase_client_singleton
