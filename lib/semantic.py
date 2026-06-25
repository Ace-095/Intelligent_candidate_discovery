"""
lib.semantic — Semantic Similarity Layer

Computes a cosine-similarity score between each candidate's text profile and the
target Job Description using a pure-stdlib TF-IDF approach.

Design rationale
----------------
The original plan called for `sentence-transformers` / `all-MiniLM-L6-v2`.  That
model (~90 MB) requires torch, which is not installed in the sandboxed execution
environment (no pip, no sudo, no network).  Rather than silently produce a dummy
feature, we implement an **IDF-weighted TF cosine similarity** using only Python
stdlib (math, re, collections).  The signal quality is lower than a transformer
embedding, but:

  1. It is a real, meaningful signal — candidates who share JD vocabulary score
     higher than unrelated candidates.
  2. It scales to 100k candidates in well under 5 seconds (no heavy ops).
  3. It is completely offline-safe — no `HF_HUB_OFFLINE` guard needed because
     there is no model to load.
  4. The interface is identical to what the transformer version would have exposed,
     so the module can be swapped in later without touching rank.py.

If `sentence-transformers` becomes available (after running `download_model.py`),
set `SEMANTIC_BACKEND = "transformer"` at the top of this module to enable it.

Public API
----------
  SemanticScorer
    .score_batch(candidates: list[dict]) -> list[float]
        Returns a float in [0, 1] for each candidate (cosine similarity to JD).
"""

from __future__ import annotations

import collections
import math
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

SEMANTIC_BACKEND: str = os.environ.get("SEMANTIC_BACKEND", "tfidf")

# ---------------------------------------------------------------------------
# Job Description text (target for this hackathon)
# ---------------------------------------------------------------------------

JD_TEXT: str = """
Senior Data Engineer or Machine Learning Engineer — India (Remote/Hybrid)

We are seeking a Senior ML/Data Engineer to join our AI platform team.
You will design, build, and maintain production-grade machine learning pipelines
and data infrastructure that power real-time recommendation and personalisation
systems.

Requirements:
- 4+ years of hands-on experience with Python and SQL in production environments.
- Strong proficiency in PyTorch or TensorFlow for training and deploying models.
- Experience building and operating data pipelines using Apache Spark, Kafka, or
  Airflow.
- Solid understanding of MLOps: model versioning (MLflow), containerisation
  (Docker, Kubernetes), CI/CD for ML models.
- Familiarity with cloud ML platforms: AWS SageMaker, Google Vertex AI, or Azure
  ML.
- Experience with feature stores, data quality frameworks, and A/B testing.
- Strong software engineering discipline: code review, testing, version control.

Nice to have:
- Experience with large language models (fine-tuning, RAG architectures).
- Knowledge of distributed training (DeepSpeed, FSDP).
- Contributions to open-source ML projects.

Location preference: Bangalore, Hyderabad, Pune, Noida, or willing to relocate.
"""

# ---------------------------------------------------------------------------
# Helpers — pure stdlib tokenisation & vector ops
# ---------------------------------------------------------------------------

# Domain-specific stop words to reduce noise (keep technical terms)
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "our", "your", "their", "we",
    "you", "they", "it", "this", "that", "these", "those", "as", "more",
    "such", "also", "into", "using", "via", "real", "time",
})


def _tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumeric, filter stop words and short tokens."""
    tokens = re.findall(r"[a-z][a-z0-9+\-#]*", text.lower())
    return [t for t in tokens if len(t) > 2 and t not in _STOP_WORDS]


def _build_tf(tokens: list[str]) -> dict[str, float]:
    total = max(len(tokens), 1)
    counts = collections.Counter(tokens)
    return {t: c / total for t, c in counts.items()}


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    # Iterate over the smaller dict for speed
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _norm(v: dict[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values()))


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, _dot(a, b) / (na * nb)))


# ---------------------------------------------------------------------------
# Text extraction from candidate record
# ---------------------------------------------------------------------------

def _candidate_text(candidate: dict) -> str:
    """
    Build a single representative text string from a candidate record.

    Sources (in priority order, each truncated to limit memory):
      1. profile.current_title       (role signal)
      2. profile.summary             (self-description)
      3. career_history role titles  (career trajectory)
      4. skills names                (technical vocabulary)

    Total output is capped at ~1000 characters for speed.
    """
    parts: list[str] = []

    profile = candidate.get("profile", {})
    title = (profile.get("current_title") or "").strip()
    if title:
        parts.append(title)

    summary = (profile.get("summary") or "").strip()
    if summary:
        parts.append(summary[:400])

    career = candidate.get("career_history", [])
    # Use 3 most recent roles (first 3 by order in the list, usually most recent first)
    for entry in career[:3]:
        role_title = (entry.get("title") or "").strip()
        role_desc = (entry.get("description") or "").strip()
        if role_title:
            parts.append(role_title)
        if role_desc:
            parts.append(role_desc[:100])

    # Skill names act as a vocabulary boost
    skills = candidate.get("skills", [])
    skill_names = " ".join(s.get("name", "") for s in skills[:20])
    if skill_names:
        parts.append(skill_names)

    combined = " ".join(parts)
    return combined[:1000]  # hard cap


# ---------------------------------------------------------------------------
# TF-IDF Semantic Scorer
# ---------------------------------------------------------------------------

class _TFIDFScorer:
    """
    IDF-weighted cosine similarity scorer.

    IDF weights are computed from a corpus of candidate texts, which gives a
    more discriminating signal than plain TF cosine (common terms like "python"
    get down-weighted relative to rare terms like "sagemaker" or "deepspeed").

    Phase 1 (pre-compute): call fit(candidate_texts) on the full corpus.
    Phase 2 (scoring):     call score_batch(candidates) -> list[float].
    """

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._jd_tfidf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> None:
        """Compute IDF from a list of document texts (call once before scoring)."""
        N = len(texts)
        if N == 0:
            return
        df: dict[str, int] = collections.defaultdict(int)
        for text in texts:
            tokens = set(_tokenize(text))
            for t in tokens:
                df[t] += 1
        # Smooth IDF: log((N+1)/(df+1)) + 1  (sklearn-style)
        self._idf = {t: math.log((N + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}

        # Build JD vector
        jd_tokens = _tokenize(JD_TEXT)
        jd_tf = _build_tf(jd_tokens)
        self._jd_tfidf = {
            t: tf * self._idf.get(t, 1.0) for t, tf in jd_tf.items()
        }

    def _candidate_tfidf(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        tf = _build_tf(tokens)
        return {t: v * self._idf.get(t, 1.0) for t, v in tf.items()}

    def score_batch(self, candidates: list[dict]) -> list[float]:
        """Return a cosine similarity score in [0, 1] for each candidate."""
        if not self._jd_tfidf:
            # fit() not called — fall back to plain TF
            jd_tf = _build_tf(_tokenize(JD_TEXT))
            return [
                _cosine(jd_tf, _build_tf(_tokenize(_candidate_text(c))))
                for c in candidates
            ]
        return [
            _cosine(self._jd_tfidf, self._candidate_tfidf(_candidate_text(c)))
            for c in candidates
        ]


# ---------------------------------------------------------------------------
# Optional transformer backend (future use)
# ---------------------------------------------------------------------------

class _TransformerScorer:
    """
    Sentence-transformer cosine similarity scorer.

    Requires: pip install sentence-transformers torch
    Model path: lib/models/all-MiniLM-L6-v2/  (downloaded by download_model.py)

    Set HF_HUB_OFFLINE=1 before import to prevent network calls.
    """

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")

    def __init__(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer  # type: ignore
        if not os.path.isdir(self.MODEL_DIR):
            raise RuntimeError(
                f"Model not found at {self.MODEL_DIR}. "
                "Run `python download_model.py` once before ranking."
            )
        self._model = SentenceTransformer(self.MODEL_DIR)
        self._jd_emb = self._model.encode(
            JD_TEXT, normalize_embeddings=True, show_progress_bar=False
        )

    def fit(self, _texts: list[str]) -> None:
        """No-op: transformers don't need corpus-level fitting."""

    def score_batch(self, candidates: list[dict]) -> list[float]:
        texts = [_candidate_text(c) for c in candidates]
        embs = self._model.encode(
            texts,
            batch_size=len(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Cosine = dot product for normalized vectors
        scores = (embs @ self._jd_emb).tolist()
        return [max(0.0, float(s)) for s in scores]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

class SemanticScorer:
    """
    Unified façade for both backends.

    Usage:
        scorer = SemanticScorer()
        scorer.fit(all_texts)          # call once; no-op for transformer
        scores = scorer.score_batch(batch_of_candidates)
    """

    def __init__(self) -> None:
        backend = SEMANTIC_BACKEND
        if backend == "transformer":
            self._impl: _TFIDFScorer | _TransformerScorer = _TransformerScorer()
        else:
            self._impl = _TFIDFScorer()

    def fit(self, texts: list[str]) -> None:
        self._impl.fit(texts)

    def score_batch(self, candidates: list[dict]) -> list[float]:
        return self._impl.score_batch(candidates)
