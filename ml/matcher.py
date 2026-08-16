"""
LocateMe — Face Matcher Module
Computes cosine similarity between 512-dimensional face embeddings and evaluates
potential matches against a configurable threshold.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Default experimental threshold for InceptionResnetV1 (VGGFace2) cosine similarity
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.68"))

DISCLAIMER_TEXT = (
    "EXPERIMENTAL PROTOTYPE NOTICE: Similarity scores represent algorithmic feature "
    "proximity for authorized test screening. They do not constitute definitive identity."
)


@dataclass
class MatchResult:
    """Standardized result of a face comparison evaluation."""
    similarity_score: float
    is_match: bool
    match_status: str  # "Potential Match" or "No Match"
    threshold: float
    confidence_tier: str  # "High Similarity", "Moderate Similarity", "Low Similarity", "Non-Matching"
    disclaimer: str = DISCLAIMER_TEXT

    def to_dict(self) -> dict:
        """Convert result to serializable dictionary."""
        return asdict(self)


def compute_cosine_similarity(
    vec1: Union[np.ndarray, list], vec2: Union[np.ndarray, list]
) -> float:
    """
    Calculate the cosine similarity between two feature vectors:
        cosine_similarity = (u . v) / (||u|| * ||v||)

    Args:
        vec1: First 512-D embedding.
        vec2: Second 512-D embedding.

    Returns:
        float similarity score in the range [-1.0, 1.0].
    """
    u = np.asarray(vec1, dtype=np.float32).flatten()
    v = np.asarray(vec2, dtype=np.float32).flatten()

    if u.shape != v.shape:
        raise ValueError(
            f"Embedding shape mismatch: vec1 has shape {u.shape}, vec2 has shape {v.shape}"
        )

    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))

    if norm_u == 0.0 or norm_v == 0.0:
        logger.warning("Zero-magnitude embedding vector encountered during similarity calculation.")
        return 0.0

    # Dot product divided by norms
    similarity = float(np.dot(u, v) / (norm_u * norm_v))

    # Clamp floating point precision noise to [-1.0, 1.0]
    return max(-1.0, min(1.0, similarity))


def evaluate_confidence_tier(score: float, threshold: float) -> str:
    """Categorize candidate similarity into human-interpretable tiers."""
    if score >= 0.85:
        return "High Similarity"
    elif score >= threshold:
        return "Moderate Similarity"
    elif score >= threshold - 0.15:
        return "Low Similarity (Borderline)"
    else:
        return "Non-Matching"


class FaceMatcher:
    """
    Configurable matcher to compare facial embeddings.
    """

    def __init__(self, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> None:
        """
        Initialize the matcher.

        Args:
            threshold: Cosine similarity cutoff for declaring a 'Potential Match' (0.0 to 1.0).
        """
        if not (-1.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be between -1.0 and 1.0, got {threshold}")
        self.threshold = threshold

    def match(
        self,
        ref_embedding: Union[np.ndarray, list],
        query_embedding: Union[np.ndarray, list],
        threshold: Optional[float] = None,
    ) -> MatchResult:
        """
        Compare a reference face embedding against a query face embedding.

        Args:
            ref_embedding: 512-D reference feature vector.
            query_embedding: 512-D candidate/surveillance feature vector.
            threshold: Optional threshold override.

        Returns:
            MatchResult with similarity score, match status, and confidence tier.
        """
        effective_threshold = self.threshold if threshold is None else threshold
        similarity = compute_cosine_similarity(ref_embedding, query_embedding)
        is_match = similarity >= effective_threshold

        status = "Potential Match" if is_match else "No Match"
        tier = evaluate_confidence_tier(similarity, effective_threshold)

        return MatchResult(
            similarity_score=round(similarity, 4),
            is_match=is_match,
            match_status=status,
            threshold=effective_threshold,
            confidence_tier=tier,
        )

    def match_one_to_many(
        self,
        ref_embedding: Union[np.ndarray, list],
        candidates: Dict[str, Union[np.ndarray, list]],
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, MatchResult]]:
        """
        Screen a reference face against a gallery of registered or detected candidates.

        Args:
            ref_embedding: Reference 512-D embedding.
            candidates: Dict mapping candidate identifier to 512-D embedding.
            threshold: Optional threshold override.

        Returns:
            List of (candidate_id, MatchResult) tuples sorted by similarity descending.
        """
        results: List[Tuple[str, MatchResult]] = []
        for candidate_id, emb in candidates.items():
            res = self.match(ref_embedding, emb, threshold=threshold)
            results.append((candidate_id, res))

        # Sort descending by similarity score
        results.sort(key=lambda item: item[1].similarity_score, reverse=True)
        return results


def match_faces(
    ref_embedding: Union[np.ndarray, list],
    query_embedding: Union[np.ndarray, list],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> MatchResult:
    """Convenience function to match two face embeddings."""
    matcher = FaceMatcher(threshold=threshold)
    return matcher.match(ref_embedding, query_embedding)
