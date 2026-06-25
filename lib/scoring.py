"""
lib.scoring — Candidate Scoring Against the Job Description

Responsibilities:
  - Define the Job Description (JD) requirements as a structured spec
  - Score each candidate feature vector against the JD
  - Produce a float score in [0, 1] for each candidate
  - Generate a 1–2 sentence human-readable reasoning string per candidate

Scoring components (weights sum to 1.0):
  1. Skills match              (0.30) — required skills present and practiced
  2. Career trajectory         (0.25) — product-company roles, relevant seniority
  3. Experience years          (0.15) — 5–9 years, production ML / search / ranking
  4. Behavioral signals        (0.20) — active, responsive, short notice
  5. Education                 (0.10) — tier and degree level

Public API:
  - score(features: dict) -> tuple[float, str]
      Returns (score_float, reasoning_text) for a single feature dict.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# JD Specification — derived from job_description.md
# ---------------------------------------------------------------------------

# Keywords that map to core required skills
REQUIRED_SKILL_KEYWORDS = {
    # Embeddings and retrieval
    "sentence-transformer", "sentence_transformer", "embedding", "embeddings",
    "bge", "e5", "openai embedding", "retrieval", "dense retrieval",
    "hybrid search", "hybrid retrieval", "semantic search",
    # Vector DBs
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "vector database", "vector store",
    # Ranking evaluation
    "ndcg", "mrr", "map", "ranking", "learning to rank", "ltr", "reranking",
    "a/b test", "offline eval", "evaluation framework",
    # Core ML
    "python", "machine learning", "ml", "recommendation", "search",
    "nlp", "information retrieval", "ir",
}

# Positive title signals (applied ML roles at product companies)
POSITIVE_TITLE_SIGNALS = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "data scientist", "research engineer", "applied scientist",
    "search engineer", "ranking engineer", "nlp engineer",
    "senior engineer", "staff engineer", "tech lead",
    "software engineer",
}

# Negative title signals (not technical / wrong domain)
NEGATIVE_TITLE_SIGNALS = {
    "marketing manager", "sales", "hr", "recruiter", "product manager",
    "project manager", "scrum master", "business analyst", "consultant",
    "account manager", "customer success",
}

# Disqualifying industries (pure services / outsourcing)
DISQUALIFYING_INDUSTRIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "outsourcing", "it services", "consulting only",
}

# Target experience window
EXP_MIN = 5.0
EXP_IDEAL_LOW = 6.0
EXP_IDEAL_HIGH = 8.0
EXP_MAX = 12.0

# Notice period cutoff (JD says they prefer <30 days)
NOTICE_SOFT_LIMIT_DAYS = 30
NOTICE_HARD_LIMIT_DAYS = 90

# Months-since-last-active penalty threshold
INACTIVE_SOFT_MONTHS = 3
INACTIVE_HARD_MONTHS = 6

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _skill_match_score(features: dict) -> tuple[float, list[str]]:
    """
    Score how well the candidate's skills match JD requirements.

    Semantic similarity (Phase 5) is the primary driver (60%) since it directly
    measures profile-to-JD overlap.  Proxy features (proficiency, expert count,
    endorsements, code recency) fill the remaining 40%.
    """
    notes: list[str] = []
    score = 0.0

    # ---- Semantic similarity (dominant signal, Phase 5) --------------------
    sim = features.get("semantic_similarity", 0.0)
    score += sim * 0.60  # up to 0.60
    if sim > 0.4:
        notes.append("strong semantic match to JD")

    # ---- Proficiency distribution ------------------------------------------
    prof = features.get("avg_skill_proficiency", 0.0)
    score += prof * 0.15  # up to 0.15

    # ---- Expert breadth bonus ----------------------------------------------
    expert_count = features.get("expert_skill_count", 0)
    expert_bonus = min(expert_count / 5.0, 1.0) * 0.10
    score += expert_bonus

    # ---- Code recency proxy (Phase 4): github + title signal ---------------
    code_recency = features.get("code_recency_score", 0.0)
    score += code_recency * 0.10  # up to 0.10

    # ---- Endorsement signal (social proof) ---------------------------------
    endorsements = features.get("total_endorsements", 0)
    endorsement_score = min(endorsements / 200.0, 1.0) * 0.05
    score += endorsement_score

    # Build reasoning notes
    if expert_count >= 3:
        notes.append(f"expert in {expert_count} skills")
    if code_recency > 0.5:
        notes.append("strong code recency")
    if endorsements >= 50:
        notes.append(f"{endorsements} total endorsements")

    return min(score, 1.0), notes


def _career_score(features: dict) -> tuple[float, list[str]]:
    """
    Score career trajectory quality.

    Integrates Phase 4 derived signals:
    - job_hop_frequency: roles/year — high means title-chaser risk
    - ai_production_recency: real ML depth score
    """
    notes: list[str] = []
    score = 0.0

    company_score = features.get("avg_company_size_score", 0.0)
    has_current = features.get("has_current_role", False)
    shortest_tenure = features.get("shortest_tenure_months", 0)
    n_roles = features.get("career_entry_count", 0)
    avg_role_dur = features.get("avg_role_duration_months", 0.0)

    # Larger company = more production exposure
    score += company_score * 0.35

    # AI production recency (Phase 4): rewards deep ML history over recent wrappers
    ai_recency = features.get("ai_production_recency", 0.0)
    score += ai_recency * 0.25  # up to 0.25
    if ai_recency > 0.5:
        notes.append("strong AI production history")

    # Penalise high job-hop frequency (Phase 4): > 0.8 roles/year is a red flag
    hop_freq = features.get("job_hop_frequency", 0.0)
    if hop_freq > 0.8:
        penalty = min((hop_freq - 0.8) / 0.4, 1.0) * 0.20
        score -= penalty
        notes.append(f"job-hop rate {hop_freq:.1f} roles/yr")

    # Penalise extreme job hopping (shortest tenure < 6 months)
    if shortest_tenure > 0 and shortest_tenure < 6:
        score -= 0.10
        notes.append(f"short tenure ({shortest_tenure}mo) detected")

    # Penalise fragmented career (too many short stints)
    if n_roles > 6 and avg_role_dur < 18:
        score -= 0.10

    # Current role bonus
    if has_current:
        score += 0.15
        current_size = features.get("current_company_size_score", 0.0)
        score += current_size * 0.25
        if current_size >= 0.8:
            notes.append("currently at large-scale company")
    else:
        score += 0.05  # might still be available

    return min(max(score, 0.0), 1.0), notes


def _experience_score(features: dict) -> tuple[float, list[str]]:
    """Score years-of-experience alignment with JD band (5–9 years)."""
    notes: list[str] = []
    yoe = features.get("years_of_experience", 0)

    if yoe < 2:
        score = 0.05
    elif yoe < EXP_MIN:
        # Below minimum — partial credit
        score = 0.3 + (yoe - 2) / (EXP_MIN - 2) * 0.3
    elif EXP_MIN <= yoe <= EXP_IDEAL_HIGH:
        score = 0.85 + (yoe - EXP_IDEAL_LOW) / (EXP_IDEAL_HIGH - EXP_IDEAL_LOW) * 0.15
    elif yoe <= EXP_MAX:
        # Above ideal but still relevant
        score = 0.85 - (yoe - EXP_IDEAL_HIGH) / (EXP_MAX - EXP_IDEAL_HIGH) * 0.25
    else:
        # Very senior — may be overqualified or career-changer
        score = 0.55

    if EXP_MIN <= yoe <= EXP_IDEAL_HIGH:
        notes.append(f"{yoe:.0f}y experience (ideal band)")
    else:
        notes.append(f"{yoe:.0f}y experience")

    return score, notes


def _behavioral_multiplier(features: dict) -> tuple[float, list[str]]:
    """
    Behavioral multiplier — applied multiplicatively to the base composite score.

    Per JD: inactive / unresponsive candidates should be down-weighted, not
    scored as an independent additive feature.  A candidate who is easy to
    reach and actively looking gets a small boost; one who has been dark for
    months or refuses recruiter contact is suppressed.

    Returns a multiplier in [0.1, 1.2] and a list of notes.
    """
    notes: list[str] = []
    multiplier = 1.0

    # ---- Recency penalty (strongest signal) --------------------------------
    months_inactive = features.get("months_since_last_active", 999.0)
    if months_inactive > INACTIVE_HARD_MONTHS:
        multiplier *= 0.50  # heavily inactive — real availability risk
        notes.append(f"inactive {months_inactive:.0f}mo")
    elif months_inactive > INACTIVE_SOFT_MONTHS:
        multiplier *= 0.80  # slightly stale

    # ---- Open-to-work flag -------------------------------------------------
    open_to_work = features.get("open_to_work", 0.0)
    if open_to_work:
        multiplier += 0.10
        notes.append("open to work")

    # ---- Recruiter & interview engagement ----------------------------------
    response_rate = features.get("recruiter_response_rate", 0.0)
    interview_rate = features.get("interview_completion_rate", 0.0)
    # Both rates are [0.0, 1.0]; together they can add up to +0.10
    multiplier += (response_rate * 0.05) + (interview_rate * 0.05)

    # ---- Offer acceptance rate (quality signal) ----------------------------
    offer_rate = features.get("offer_acceptance_rate", 0.0)
    if offer_rate > 0.0:
        multiplier += offer_rate * 0.05  # up to +0.05

    # Clamp to a sane range: suppress to 0.1 minimum, cap boost at 1.2
    multiplier = round(max(0.10, min(multiplier, 1.20)), 4)
    return multiplier, notes


def _education_score(features: dict) -> tuple[float, list[str]]:
    """Score education tier and degree level."""
    tier = features.get("best_tier_score", 0.0)
    degree = features.get("best_degree_level_score", 0.2)
    combined = tier * 0.6 + degree * 0.4
    notes = []
    if tier >= 0.75:
        notes.append("tier 1/2 institution")
    return combined, notes


# ---------------------------------------------------------------------------
# Title / disqualification checks
# ---------------------------------------------------------------------------


def _title_penalty(features: dict) -> float:
    """Return a penalty multiplier (0.0–1.0) based on current job title."""
    title = features.get("current_title", "").lower()
    if any(neg in title for neg in NEGATIVE_TITLE_SIGNALS):
        return 0.2  # heavy penalty — wrong role entirely
    if any(pos in title for pos in POSITIVE_TITLE_SIGNALS):
        return 1.0  # no penalty
    return 0.85  # neutral / unknown title


def _industry_penalty(features: dict) -> float:
    """Penalise candidates from pure outsourcing backgrounds."""
    industry = features.get("current_industry", "").lower()
    if any(kw in industry for kw in DISQUALIFYING_INDUSTRIES):
        return 0.7  # partial penalty (may still have prior product experience)
    # Phase 4 is_it_services flag (consulting firm or IT Services industry)
    if features.get("is_it_services"):
        return 0.7
    return 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Component weights (must sum to 1.0).
# Behavioral signals are now applied as a *multiplier* (see _behavioral_multiplier),
# so they are no longer an additive component here.
_WEIGHTS = {
    "skills":     0.35,
    "career":     0.30,
    "experience": 0.20,
    "education":  0.15,
}


def score(features: dict) -> tuple[float, str]:
    """
    Score a candidate against the JD.

    Args:
        features: Feature dict produced by lib.features.extract().

    Returns:
        A tuple of (score_float [0.0–1.0], elite_notes_list).
    """
    skill_s, skill_notes = _skill_match_score(features)
    career_s, career_notes = _career_score(features)
    exp_s, exp_notes = _experience_score(features)
    edu_s, edu_notes = _education_score(features)

    composite = (
        _WEIGHTS["skills"]     * skill_s
        + _WEIGHTS["career"]     * career_s
        + _WEIGHTS["experience"] * exp_s
        + _WEIGHTS["education"]  * edu_s
    )

    # Apply title / industry multipliers
    composite *= _title_penalty(features)
    composite *= _industry_penalty(features)

    # Apply behavioral multiplier (down-weights inactive / unresponsive candidates)
    behav_mult, behav_notes = _behavioral_multiplier(features)
    composite *= behav_mult

    # Location match penalty (Phase 4): Pune / Noida / Tier-1 or willing to relocate
    if not features.get("location_match", True):
        composite *= 0.80

    # ---- Elite Spike Bonus (NDCG@10 tuning) --------------------------------
    # NDCG@10 is 50% of competition score; getting the top 10-20 right matters
    # far more than uniform quality across all 100. Apply an additive bonus
    # *before* clamping so that genuinely elite candidates separate cleanly.
    elite_notes: list[str] = []
    sim = features.get("semantic_similarity", 0.0)
    ai_rec = features.get("ai_production_recency", 0.0)
    if sim > 0.8 and ai_rec > 0.8:
        composite += 0.15
        elite_notes.append("elite ML candidate")

    expert_count = features.get("expert_skill_count", 0)
    endorsements = features.get("total_endorsements", 0)
    if expert_count >= 5 and endorsements >= 50:
        composite += 0.05
        elite_notes.append("expert breadth + social proof")

    composite = round(min(max(composite, 0.0), 1.0), 6)

    return composite, elite_notes
