"""
LocateMe — Registered Missing Person Gallery Manager (Phase 4 Dual-Mode)
Handles gallery registration, 512-D embedding precomputation/caching,
Supabase pgvector cloud persistence, and vectorized 1-to-N fast cosine similarity screening.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ml.embedding import FaceEmbeddingGenerator, get_default_generator
from ml.face_detector import FaceDetector, get_default_detector
from ml.matcher import (
    FaceMatcher,
    MatchResult,
    evaluate_confidence_tier,
    DEFAULT_SIMILARITY_THRESHOLD,
    DISCLAIMER_TEXT,
)
from ml.supabase_client import SupabaseGalleryClient, get_supabase_client

logger = logging.getLogger(__name__)


@dataclass
class RegisteredPerson:
    """Represents a registered missing person in the gallery."""
    person_id: str
    name: str
    image_path: str
    embedding: np.ndarray
    registered_at: str
    image_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Serialize metadata (excluding raw embedding array)."""
        return {
            "person_id": self.person_id,
            "name": self.name,
            "image_path": self.image_path,
            "image_url": self.image_url or self.image_path,
            "registered_at": self.registered_at,
            "metadata": self.metadata or {},
            "embedding_dimension": len(self.embedding) if self.embedding is not None else 0,
        }


class GalleryManager:
    """
    Manages registered missing-person galleries with dual-mode storage:
    - SUPABASE mode: Cloud PostgreSQL + pgvector + Supabase Storage CDN
    - LOCAL mode: Local filesystem (data/registered/) + NumPy cache (gallery_cache.npz)
    
    Always maintains an in-memory NumPy feature matrix for fast, zero-latency
    frame-by-frame surveillance video screening.
    """

    def __init__(
        self,
        gallery_dir: Optional[Union[str, Path]] = None,
        cache_file: Optional[Union[str, Path]] = None,
        detector: Optional[FaceDetector] = None,
        generator: Optional[FaceEmbeddingGenerator] = None,
        supabase_client: Optional[SupabaseGalleryClient] = None,
    ) -> None:
        """
        Initialize the Gallery Manager.

        Args:
            gallery_dir: Directory containing local registered reference portraits.
            cache_file: File path for caching precomputed embeddings.
            detector: Optional custom FaceDetector instance.
            generator: Optional custom FaceEmbeddingGenerator instance.
            supabase_client: Optional custom or injected SupabaseGalleryClient.
        """
        self.gallery_dir = Path(gallery_dir) if gallery_dir else None
        self.cache_file = Path(cache_file) if cache_file else None

        self.detector = detector or get_default_detector()
        self.generator = generator or get_default_generator()
        self.supabase_client = supabase_client or get_supabase_client()

        self.persons: List[RegisteredPerson] = []
        self._gallery_matrix: Optional[np.ndarray] = None  # Shape: [N, 512]

        self.load_gallery()

    @property
    def is_cloud_mode(self) -> bool:
        """True if connected to active Supabase backend."""
        return self.supabase_client is not None and self.supabase_client.is_active

    @property
    def storage_mode(self) -> str:
        """Returns active storage backend descriptor."""
        return "supabase" if self.is_cloud_mode else "local"

    @property
    def count(self) -> int:
        """Total number of registered persons currently in the gallery."""
        return len(self.persons)

    def _rebuild_matrix(self) -> None:
        """Reconstructs the N x 512 normalized feature matrix for fast vectorized search."""
        if not self.persons:
            self._gallery_matrix = None
            return

        embeddings = [p.embedding for p in self.persons]
        self._gallery_matrix = np.vstack(embeddings).astype(np.float32)  # [N, 512]

    def register_person(
        self,
        name: str,
        image_path: Union[str, Path],
        person_id: Optional[str] = None,
        image_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[RegisteredPerson]:
        """
        Register a new person into the active gallery and persistent storage.

        Args:
            name: Display name or reference code.
            image_path: Path to reference portrait photo.
            person_id: Optional unique identifier. Defaults to slugified filename.
            image_url: Optional remote public URL (e.g., from Supabase Storage).
            metadata: Optional additional metadata dict.

        Returns:
            RegisteredPerson object if registration succeeds, else None.
        """
        img_p = Path(image_path)
        if not img_p.is_file():
            logger.warning(f"Registration failed: file not found: {img_p}")
            return None

        emb = self.generator.generate_embedding(img_p, detector=self.detector, normalize=True)
        if emb is None:
            logger.warning(f"Registration failed: could not detect face in {img_p}")
            return None

        pid = person_id or img_p.stem
        public_url = image_url

        # Cloud Mode: upload to Supabase Storage & insert to pgvector table
        if self.is_cloud_mode and self.supabase_client:
            try:
                if not public_url:
                    with open(img_p, "rb") as f:
                        file_bytes = f.read()
                    filename = f"{pid}_{img_p.name}"
                    public_url = self.supabase_client.upload_photo(file_bytes, filename)

                self.supabase_client.insert_missing_person(
                    person_id=pid,
                    name=name,
                    image_url=public_url or str(img_p),
                    embedding=emb,
                    metadata=metadata,
                )
                logger.info(f"Persisted '{name}' ({pid}) to Supabase pgvector.")
            except Exception as e:
                logger.error(f"Supabase persistence error for '{name}': {e}. Falling back to local cache.")

        reg_person = RegisteredPerson(
            person_id=pid,
            name=name,
            image_path=str(img_p.resolve()),
            embedding=emb,
            registered_at=datetime.now(timezone.utc).isoformat(),
            image_url=public_url,
            metadata=metadata,
        )

        # Update in-memory registry & fast NumPy matrix
        self.persons = [p for p in self.persons if p.person_id != pid]
        self.persons.append(reg_person)
        self._rebuild_matrix()

        logger.info(f"Successfully registered: '{name}' (ID: {pid}, Storage: {self.storage_mode})")
        return reg_person

    def delete_person(self, person_id: str) -> bool:
        """
        Remove a registered person from active gallery and persistent storage.

        Args:
            person_id: Identifier of the person to delete.

        Returns:
            True if person was deleted, False if not found.
        """
        target = [p for p in self.persons if p.person_id == person_id]
        if not target:
            return False

        if self.is_cloud_mode and self.supabase_client:
            try:
                self.supabase_client.delete_missing_person(person_id)
            except Exception as e:
                logger.error(f"Failed to delete {person_id} from Supabase: {e}")

        self.persons = [p for p in self.persons if p.person_id != person_id]
        self._rebuild_matrix()
        self.save_cache()
        return True

    def load_gallery(
        self,
        gallery_dir: Optional[Union[str, Path]] = None,
        force_recompute: bool = False,
    ) -> int:
        """
        Load registered missing persons.
        If Supabase is active, syncs from PostgreSQL pgvector table.
        Otherwise, loads from local directory and .npz cache.

        Returns:
            Number of successfully registered persons.
        """
        # 1. Try Supabase Cloud Sync
        if self.is_cloud_mode and self.supabase_client:
            try:
                cloud_records = self.supabase_client.fetch_all_persons()
                if cloud_records:
                    self.persons = []
                    for rec in cloud_records:
                        emb_raw = rec.get("embedding")
                        if emb_raw is None:
                            continue
                        if isinstance(emb_raw, str):
                            # pgvector string format: "[0.123, -0.456, ...]"
                            emb_raw = [float(x.strip()) for x in emb_raw.strip("[]").split(",") if x.strip()]
                        emb_np = np.asarray(emb_raw, dtype=np.float32)

                        self.persons.append(
                            RegisteredPerson(
                                person_id=rec.get("person_id", str(rec.get("id"))),
                                name=rec.get("name", "Unknown"),
                                image_path=rec.get("image_url", ""),
                                embedding=emb_np,
                                registered_at=rec.get("created_at", datetime.now(timezone.utc).isoformat()),
                                image_url=rec.get("image_url"),
                                metadata=rec.get("metadata", {}),
                            )
                        )
                    self._rebuild_matrix()
                    logger.info(f"Loaded {len(self.persons)} missing persons from Supabase pgvector.")
                    return len(self.persons)
            except Exception as e:
                logger.warning(f"Failed to sync gallery from Supabase: {e}. Falling back to local storage.")

        # 2. Local Mode Fallback
        target_dir = Path(gallery_dir) if gallery_dir else self.gallery_dir
        if not target_dir or not target_dir.exists():
            return 0

        self.gallery_dir = target_dir
        cache_p = self.cache_file or (target_dir.parent / "gallery_cache.npz")

        # Attempt to load from npz cache
        if not force_recompute and cache_p.exists():
            try:
                data = np.load(str(cache_p), allow_pickle=True)
                cached_ids = data["person_ids"].tolist()
                cached_names = data["names"].tolist()
                cached_paths = data["image_paths"].tolist()
                cached_dates = data["registered_dates"].tolist()
                cached_matrix = data["embeddings"]

                self.persons = []
                for i in range(len(cached_ids)):
                    self.persons.append(
                        RegisteredPerson(
                            person_id=cached_ids[i],
                            name=cached_names[i],
                            image_path=cached_paths[i],
                            embedding=cached_matrix[i],
                            registered_at=cached_dates[i],
                        )
                    )

                self._rebuild_matrix()
                logger.info(f"Loaded {len(self.persons)} gallery entries from local cache ({cache_p.name}).")
                return len(self.persons)
            except Exception as e:
                logger.warning(f"Failed to read local cache {cache_p}: {e}. Recomputing...")

        # Scan local directory for reference images
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            f for f in target_dir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_exts and "blank" not in f.name.lower()
        ]

        self.persons = []
        for img_path in sorted(image_files):
            clean_name = img_path.stem.replace("_ref", "").replace("_", " ").title()
            self.register_person(name=clean_name, image_path=img_path, person_id=img_path.stem)

        self._rebuild_matrix()
        self.save_cache(cache_p)
        return len(self.persons)

    def save_cache(self, cache_file: Optional[Union[str, Path]] = None) -> None:
        """Save the current gallery embeddings and metadata to a local npz cache file."""
        if not self.persons or self._gallery_matrix is None:
            return

        cache_p = (
            Path(cache_file)
            if cache_file
            else (self.gallery_dir.parent / "gallery_cache.npz" if self.gallery_dir else None)
        )
        if not cache_p:
            return

        try:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(cache_p),
                person_ids=np.array([p.person_id for p in self.persons]),
                names=np.array([p.name for p in self.persons]),
                image_paths=np.array([p.image_path for p in self.persons]),
                registered_dates=np.array([p.registered_at for p in self.persons]),
                embeddings=self._gallery_matrix,
            )
            logger.info(f"Saved local gallery cache with {len(self.persons)} entries to {cache_p}")
        except Exception as e:
            logger.error(f"Failed to save local gallery cache: {e}", exc_info=True)

    def search(
        self,
        query_embedding: Union[np.ndarray, list],
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> List[Tuple[RegisteredPerson, MatchResult]]:
        """
        Fast vectorized 1-to-N search using the in-memory NumPy matrix.
        Zero database latency — optimal for real-time video feeds and batch screening.

        Args:
            query_embedding: 512-D normalized query vector.
            threshold: Cosine similarity threshold for declaring a Potential Match.

        Returns:
            List of (RegisteredPerson, MatchResult) tuples sorted descending by similarity score.
        """
        if self._gallery_matrix is None or len(self.persons) == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32).flatten()
        norm_q = np.linalg.norm(q)
        if norm_q == 0.0:
            return []

        # Vectorized cosine similarity: S = (G . q) / ||q|| (since G rows are unit-norm)
        scores = np.dot(self._gallery_matrix, q) / norm_q
        scores = np.clip(scores, -1.0, 1.0)

        results: List[Tuple[RegisteredPerson, MatchResult]] = []
        for i, score in enumerate(scores):
            similarity = float(score)
            is_match = similarity >= threshold
            status = "Potential Match" if is_match else "No Match"
            tier = evaluate_confidence_tier(similarity, threshold)

            res = MatchResult(
                similarity_score=round(similarity, 4),
                is_match=is_match,
                match_status=status,
                threshold=threshold,
                confidence_tier=tier,
                disclaimer=DISCLAIMER_TEXT,
            )
            results.append((self.persons[i], res))

        results.sort(key=lambda item: item[1].similarity_score, reverse=True)
        return results

    def find_best_match(
        self,
        query_embedding: Union[np.ndarray, list],
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> Optional[Tuple[RegisteredPerson, MatchResult]]:
        """
        Find the single highest-scoring candidate for a query embedding.

        Returns:
            (RegisteredPerson, MatchResult) if top candidate score >= threshold, else None.
        """
        matches = self.search(query_embedding, threshold=threshold)
        if matches and matches[0][1].is_match:
            return matches[0]
        return None
