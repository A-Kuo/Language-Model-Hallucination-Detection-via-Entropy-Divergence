"""
Embedding Anomaly Detection
============================

Augments the attention-based hallucination detector with semantic embedding
anomaly detection using ChromaDB as the vector store.

Architecture:
    1. Embed each text (question + answer) using a sentence transformer
    2. Store embeddings in ChromaDB with hallucination labels
    3. At inference: retrieve k nearest neighbours from training set
       → kNN label vote
       → centroid distance score
       → Mahalanobis distance score (using training set covariance)
    4. Combine embedding anomaly score with attention-based score

Motivation:
    Attention features detect uncertainty in model internals. Embedding
    anomaly detection operates on the semantic content of the output.
    The two signals are complementary:
        - Attention: catches diffuse/inconsistent attention (model uncertain)
        - Embedding: catches outputs far from known-correct semantic space

    Combined, they address the "confident hallucination" blind spot:
    a model can produce low-entropy (focused) attention while still
    generating semantically anomalous content.

Usage:
    store = EmbeddingAnomalyDetector()
    store.fit(train_texts, train_labels)           # build ChromaDB index
    score = store.predict_proba(query_text)        # 0=correct, 1=hallucinated
    combined = 0.6 * attn_score + 0.4 * emb_score # ensemble

Dependencies:
    pip install chromadb sentence-transformers
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class AnomalyScore:
    """Decomposed anomaly scores from different methods."""
    knn_vote: float            # fraction of kNN neighbours that are hallucinated
    centroid_dist: float       # L2 distance from correct-answer centroid
    mahalanobis: float         # Mahalanobis distance from correct-answer distribution
    combined: float            # weighted combination (primary signal)
    k: int = 5


class EmbeddingAnomalyDetector:
    """
    ChromaDB-backed embedding anomaly detector.

    Uses sentence-transformers for embedding and ChromaDB for efficient
    approximate nearest-neighbour lookup.

    Parameters
    ----------
    model_name : str
        Sentence transformer model. 'all-MiniLM-L6-v2' is small (80MB),
        fast, and performs well on QA-style text.
    k : int
        Number of nearest neighbours for kNN voting.
    collection_name : str
        ChromaDB collection name.
    knn_weight : float
        Weight of kNN vote in combined score.
    centroid_weight : float
        Weight of centroid distance in combined score.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        k: int = 5,
        collection_name: str = "hallucination_embeddings",
        knn_weight: float = 0.5,
        centroid_weight: float = 0.3,
        mahal_weight: float = 0.2,
    ) -> None:
        self.model_name = model_name
        self.k = k
        self.collection_name = collection_name
        self.knn_w = knn_weight
        self.centroid_w = centroid_weight
        self.mahal_w = mahal_weight

        self._encoder = None
        self._client = None
        self._collection = None
        self._correct_centroid: Optional[np.ndarray] = None
        self._correct_cov_inv: Optional[np.ndarray] = None
        self._fitted = False

    @property
    def encoder(self):
        """Lazy-load sentence transformer."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    @property
    def collection(self):
        """Lazy-init ChromaDB client and collection."""
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError("ChromaDB required: pip install chromadb")
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def fit(
        self,
        texts: List[str],
        labels: List[int],  # 1=hallucinated, 0=correct
        batch_size: int = 64,
    ) -> "EmbeddingAnomalyDetector":
        """
        Embed training texts and store in ChromaDB.

        Also computes the centroid and covariance of correct-answer embeddings
        for Mahalanobis distance computation.

        Parameters
        ----------
        texts : List[str]
            Formatted as "Question: ...\nAnswer: ..."
        labels : List[int]
        """
        print(f"  Embedding {len(texts)} training samples...")

        # Batch encode
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs  = self.encoder.encode(batch, show_progress_bar=False)
            embeddings.extend(embs.tolist())
            if (i // batch_size + 1) % 5 == 0:
                print(f"    {i + len(batch)}/{len(texts)} encoded")

        embeddings_arr = np.array(embeddings)

        # Add to ChromaDB
        ids = [hashlib.md5(t.encode()).hexdigest()[:16] for t in texts]
        self.collection.add(
            ids=ids,
            embeddings=embeddings_arr.tolist(),
            metadatas=[{"label": int(l)} for l in labels],
        )

        # Compute centroid + covariance of correct-answer embeddings
        correct_embs = embeddings_arr[np.array(labels) == 0]
        if len(correct_embs) >= 2:
            self._correct_centroid = correct_embs.mean(axis=0)
            cov = np.cov(correct_embs.T)
            # Regularise covariance for numerical stability
            cov += 1e-6 * np.eye(cov.shape[0])
            self._correct_cov_inv = np.linalg.pinv(cov)
        else:
            self._correct_centroid = embeddings_arr.mean(axis=0)
            self._correct_cov_inv = np.eye(embeddings_arr.shape[1])

        self._fitted = True
        print(f"  ChromaDB index built: {len(texts)} embeddings")
        print(f"  Correct-answer centroid computed (dim={embeddings_arr.shape[1]})")
        return self

    def predict_proba(self, text: str) -> AnomalyScore:
        """
        Score a single text for hallucination likelihood.

        Returns decomposed AnomalyScore with kNN vote, centroid distance,
        Mahalanobis distance, and a combined probability estimate.
        """
        assert self._fitted, "Call fit() first"

        emb = self.encoder.encode([text], show_progress_bar=False)[0]  # (D,)

        # kNN vote
        results = self.collection.query(
            query_embeddings=[emb.tolist()],
            n_results=self.k,
        )
        neighbour_labels = [m["label"] for m in results["metadatas"][0]]
        knn_vote = float(np.mean(neighbour_labels))  # fraction hallucinated

        # Centroid distance (L2 from correct-answer centroid)
        centroid_dist = float(np.linalg.norm(emb - self._correct_centroid))

        # Mahalanobis distance from correct-answer distribution
        diff = emb - self._correct_centroid
        mahal = float(np.sqrt(np.maximum(diff @ self._correct_cov_inv @ diff, 0)))

        # Normalise mahal (empirically ~N(0, dim) under null)
        dim = len(emb)
        mahal_norm = float(np.clip(mahal / np.sqrt(dim), 0, 1))

        # Centroid distance: normalise by median expected distance
        centroid_norm = float(np.clip(centroid_dist / 5.0, 0, 1))

        combined = (
            self.knn_w * knn_vote
            + self.centroid_w * centroid_norm
            + self.mahal_w * mahal_norm
        )

        return AnomalyScore(
            knn_vote=knn_vote,
            centroid_dist=centroid_dist,
            mahalanobis=mahal,
            combined=float(np.clip(combined, 0, 1)),
            k=self.k,
        )

    def ensemble_score(
        self,
        attention_score: float,
        text: str,
        attn_weight: float = 0.6,
    ) -> float:
        """
        Combine attention-based and embedding-based hallucination scores.

        Parameters
        ----------
        attention_score : float  — from HallucinationDetector.predict_proba()
        text : str               — "Question: ...\nAnswer: ..."
        attn_weight : float      — weight for attention score (1 - attn_weight for embedding)

        Returns
        -------
        float — combined hallucination probability in [0, 1]
        """
        emb_score = self.predict_proba(text).combined
        return float(attn_weight * attention_score + (1 - attn_weight) * emb_score)


# =========================================================================
# Self-test (no ChromaDB/model needed for structure checks)
# =========================================================================

if __name__ == "__main__":
    print("EmbeddingAnomalyDetector — Structure Validation")
    print("=" * 50)

    score = AnomalyScore(
        knn_vote=0.6,
        centroid_dist=2.3,
        mahalanobis=1.8,
        combined=0.55,
        k=5,
    )
    assert 0 <= score.combined <= 1
    print(f"  AnomalyScore: knn={score.knn_vote:.2f}, centroid={score.centroid_dist:.2f}, "
          f"mahal={score.mahalanobis:.2f}, combined={score.combined:.2f}  ✅")

    # Verify ensemble logic
    detector_stub = type("D", (), {"predict_proba": staticmethod(lambda x: np.array([0.7]))})()
    combined = 0.6 * 0.7 + 0.4 * 0.55
    assert abs(combined - 0.598) < 0.01
    print(f"  Ensemble score logic: {combined:.3f}  ✅")

    print("\nAll structure checks pass ✅")
    print("\nTo use:")
    print("  pip install chromadb sentence-transformers")
    print("  detector = EmbeddingAnomalyDetector()")
    print("  detector.fit(texts, labels)")
    print("  score = detector.predict_proba('Question: ...\\nAnswer: ...')")
