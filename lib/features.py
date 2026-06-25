"""
lib.features — Feature Extraction from Candidate Profiles

Responsibilities:
  - Extract structured, numeric or categorical features from raw candidate dicts
  - Normalize and transform raw profile data into a form suitable for scoring
  - Handle missing fields gracefully with sensible defaults
  - Produce a FeatureVector (dict) per candidate

Feature categories extracted:
  1. Skills features     — top skills, proficiency levels, duration, endorsements
  2. Career features     — trajectory, company tiers, recent role relevance
  3. Experience features — total years, relevant years, role alignment
  4. Education features  — institution tier, degree level, field relevance
  5. Signal features     — Redrob behavioral signals (activity, engagement, etc.)
  6. Derived features    — job-hop rate, consulting flag, AI depth, code recency,
                           location match (Phase 4)

Public API:
  - extract(candidate: dict) -> dict
      Returns a feature dict for a single candidate record.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROFICIENCY_WEIGHT = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.5,
    "beginner": 0.25,
}

EDUCATION_TIER_WEIGHT = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.5,
    "tier_4": 0.25,
    "unknown": 0.1,
}

DEGREE_LEVEL_WEIGHT = {
    "phd": 1.0,
    "ph.d": 1.0,
    "doctorate": 1.0,
    "master": 0.8,
    "m.tech": 0.8,
    "mba": 0.7,
    "bachelor": 0.6,
    "b.tech": 0.6,
    "b.e": 0.6,
    "diploma": 0.3,
}

COMPANY_SIZE_WEIGHT = {
    "10001+": 1.0,
    "5001-10000": 0.9,
    "1001-5000": 0.8,
    "501-1000": 0.7,
    "201-500": 0.6,
    "51-200": 0.5,
    "11-50": 0.4,
    "1-10": 0.3,
}

# ── Phase 4: Derived-feature constants ─────────────────────────────────────

# Named IT-services / consulting firms whose skills often generalise less
IT_SERVICES_FIRMS: frozenset[str] = frozenset({
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "hexaware", "mphasis", "niit technologies",
})

# Production/deep ML keywords → high ai_production_recency score
DEEP_ML_KEYWORDS: tuple[str, ...] = (
    "pytorch", "tensorflow", "keras", "jax",
    "sagemaker", "vertex ai", "mlflow", "mlops",
    "model deployment", "model serving", "triton",
    "cuda", "onnx", "torchscript",
    "kubernetes", "docker",                  # infra that ships models
    "feature store", "data pipeline",
)

# Wrapper/orchestration keywords → lower ai_production_recency score
WRAPPER_ML_KEYWORDS: tuple[str, ...] = (
    "langchain", "llamaindex", "openai", "chatgpt", "gpt-4", "gpt4",
    "prompt engineering", "rag", "vector search",
    "pinecone", "weaviate", "chromadb", "milvus",
    "hugging face", "huggingface",
)

# Title words that indicate less day-to-day coding
MANAGEMENT_TITLE_TOKENS: frozenset[str] = frozenset({
    "architect", "lead", "manager", "director", "vp", "head",
    "principal", "fellow", "cto", "cio",
})

# Tier-1 Indian IT cities (lower-cased, partial-match safe)
TIER_1_CITIES: frozenset[str] = frozenset({
    "pune", "noida", "bangalore", "bengaluru",
    "hyderabad", "mumbai", "delhi", "chennai",
    "gurgaon", "gurugram", "kolkata", "ahmedabad",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today() -> date:
    return datetime.utcnow().date()


def _months_since(date_str: str | None) -> float:
    """Return months between `date_str` (YYYY-MM-DD) and today."""
    if not date_str:
        return 0.0
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        delta = _today() - d
        return delta.days / 30.44
    except (ValueError, TypeError):
        return 0.0


def _normalise(text: str | None) -> str:
    return (text or "").lower().strip()


# ---------------------------------------------------------------------------
# Feature extraction sub-functions
# ---------------------------------------------------------------------------


def _skills_features(skills: list[dict]) -> dict[str, Any]:
    if not skills:
        return {
            "skill_count": 0,
            "avg_skill_proficiency": 0.0,
            "max_endorsements": 0,
            "total_endorsements": 0,
            "avg_skill_duration_months": 0.0,
            "expert_skill_count": 0,
            "top_skills": [],
        }

    # Extract top 3 skills by duration
    sorted_skills = sorted(skills, key=lambda s: s.get("duration_months", 0), reverse=True)
    top_skills = [s.get("name", "") for s in sorted_skills[:3] if s.get("name")]

    proficiency_scores = [
        PROFICIENCY_WEIGHT.get(s.get("proficiency", ""), 0.0) for s in skills
    ]
    endorsements = [s.get("endorsements", 0) for s in skills]
    durations = [s.get("duration_months", 0) for s in skills]
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")

    return {
        "skill_count": len(skills),
        "avg_skill_proficiency": sum(proficiency_scores) / len(skills),
        "max_endorsements": max(endorsements, default=0),
        "total_endorsements": sum(endorsements),
        "avg_skill_duration_months": sum(durations) / len(skills) if durations else 0.0,
        "expert_skill_count": expert_count,
        "top_skills": top_skills,
    }


def _career_features(career_history: list[dict]) -> dict[str, Any]:
    """
    Basic career metrics + derived Phase-4 signals:
      - job_hop_frequency:     roles per year (high = title-chaser risk)
      - is_it_services:        True if current/most-recent employer is a named
                               consulting firm or industry is "IT Services"
    """
    if not career_history:
        return {
            "career_entry_count": 0,
            "avg_company_size_score": 0.0,
            "has_current_role": False,
            "current_company_size_score": 0.0,
            "avg_role_duration_months": 0.0,
            "shortest_tenure_months": 0,
            # Phase 4
            "job_hop_frequency": 0.0,
            "is_it_services": False,
            "latest_company": "",
        }

    size_scores = [
        COMPANY_SIZE_WEIGHT.get(entry.get("company_size", ""), 0.0)
        for entry in career_history
    ]
    durations = [int(entry.get("duration_months", 0) or 0) for entry in career_history]
    current = next((e for e in career_history if e.get("is_current")), None)
    # Fallback: most recently started role if none is flagged current
    if current is None and career_history:
        current = max(
            career_history,
            key=lambda e: e.get("start_date") or "0000-00-00",
        )

    total_months = sum(durations)
    total_years = max(total_months / 12.0, 1.0)
    job_hop_frequency = len(career_history) / total_years

    # IT-services consulting check
    raw_company = (current or {}).get("company")
    current_industry = _normalise((current or {}).get("industry"))
    current_company = _normalise(raw_company)
    is_it_services = (
        "it service" in current_industry
        or any(firm in current_company for firm in IT_SERVICES_FIRMS)
    )

    return {
        "career_entry_count": len(career_history),
        "avg_company_size_score": sum(size_scores) / len(career_history),
        "has_current_role": current is not None,
        "current_company_size_score": COMPANY_SIZE_WEIGHT.get(
            (current or {}).get("company_size", ""), 0.0
        ),
        "avg_role_duration_months": sum(durations) / len(durations) if durations else 0.0,
        "shortest_tenure_months": min(durations, default=0),
        # Phase 4
        "job_hop_frequency": round(job_hop_frequency, 3),
        "is_it_services": is_it_services,
        "latest_company": raw_company if raw_company else "an unknown company",
    }


def _education_features(education: list[dict]) -> dict[str, Any]:
    if not education:
        return {
            "education_count": 0,
            "best_tier_score": 0.0,
            "best_degree_level_score": 0.0,
        }

    tier_scores = [
        EDUCATION_TIER_WEIGHT.get(e.get("tier", "unknown"), 0.1) for e in education
    ]
    degree_scores = [
        max(
            (
                DEGREE_LEVEL_WEIGHT.get(kw, 0.0)
                for kw in DEGREE_LEVEL_WEIGHT
                if kw in (e.get("degree") or "").lower()
            ),
            default=0.2,
        )
        for e in education
    ]

    return {
        "education_count": len(education),
        "best_tier_score": max(tier_scores, default=0.0),
        "best_degree_level_score": max(degree_scores, default=0.0),
    }


def _signal_features(signals: dict) -> dict[str, Any]:
    """Extract Redrob behavioral signals into normalised feature values."""
    return {
        "profile_completeness": signals.get("profile_completeness_score", 0) / 100.0,
        "open_to_work": float(signals.get("open_to_work_flag", False)),
        "profile_views_30d": signals.get("profile_views_received_30d", 0),
        "applications_30d": signals.get("applications_submitted_30d", 0),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0.0),
        "avg_response_time_hours": signals.get("avg_response_time_hours", 999.0),
        "connection_count": signals.get("connection_count", 0),
        "endorsements_received": signals.get("endorsements_received", 0),
        "notice_period_days": signals.get("notice_period_days", 90),
        "salary_min_lpa": (signals.get("expected_salary_range_inr_lpa") or {}).get("min", 0),
        "salary_max_lpa": (signals.get("expected_salary_range_inr_lpa") or {}).get("max", 0),
        "willing_to_relocate": float(signals.get("willing_to_relocate", False)),
        "github_activity_score": max(signals.get("github_activity_score", -1), 0),
        "search_appearance_30d": signals.get("search_appearance_30d", 0),
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", 0),
        "interview_completion_rate": signals.get("interview_completion_rate", 0.0),
        "offer_acceptance_rate": max(signals.get("offer_acceptance_rate", -1), 0),
        "verified_email": float(signals.get("verified_email", False)),
        "verified_phone": float(signals.get("verified_phone", False)),
        "linkedin_connected": float(signals.get("linkedin_connected", False)),
        "months_since_last_active": _months_since(signals.get("last_active_date")),
    }


def _derived_features(
    profile: dict,
    skills: list[dict],
    signals: dict,
) -> dict[str, Any]:
    """
    Phase-4 derived features:
      - ai_production_recency : max duration_months among deep-ML skills minus
                                0.5× max duration of wrapper-only skills.
                                Positive = real production ML depth.
      - code_recency_score    : github_activity_score, halved if current title
                                contains management/leadership tokens.
      - location_match        : True if candidate is in a Tier-1 city OR
                                is willing to relocate.
    """
    # ── AI production recency ────────────────────────────────────────────────
    deep_ml_max = 0.0
    wrapper_max = 0.0
    for s in skills:
        name_lower = _normalise(s.get("name"))
        dur = float(s.get("duration_months", 0) or 0)
        if any(kw in name_lower for kw in DEEP_ML_KEYWORDS):
            deep_ml_max = max(deep_ml_max, dur)
        if any(kw in name_lower for kw in WRAPPER_ML_KEYWORDS):
            wrapper_max = max(wrapper_max, dur)
    # Net score: real production exposure minus wrapper inflation (clamped ≥ 0)
    ai_production_recency = max(deep_ml_max - 0.5 * wrapper_max, 0.0)

    # ── Code recency score ───────────────────────────────────────────────────
    github_score = float(max(signals.get("github_activity_score", -1), 0))
    title_lower = _normalise(profile.get("current_title"))
    title_tokens = set(title_lower.replace("-", " ").split())
    has_mgmt_title = bool(title_tokens & MANAGEMENT_TITLE_TOKENS)
    code_recency_score = github_score * (0.5 if has_mgmt_title else 1.0)

    # ── Location match ───────────────────────────────────────────────────────
    location_lower = _normalise(profile.get("location"))
    is_tier1 = any(city in location_lower for city in TIER_1_CITIES)
    willing = bool(signals.get("willing_to_relocate", False))
    location_match = is_tier1 or willing

    return {
        "ai_production_recency": round(ai_production_recency, 2),
        "code_recency_score": round(code_recency_score, 2),
        "location_match": location_match,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(candidate: dict) -> dict:
    """
    Extract a flat feature dict from a raw candidate record.

    Args:
        candidate: A validated candidate dict from lib.parsing.

    Returns:
        A dict of scalar feature values keyed by feature name.
    """
    profile  = candidate.get("profile", {})
    skills   = candidate.get("skills", [])
    signals  = candidate.get("redrob_signals", {})

    features: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "years_of_experience": profile.get("years_of_experience", 0),
        "current_title": _normalise(profile.get("current_title")),
        "raw_title": profile.get("current_title") or "Professional",
        "current_industry": _normalise(profile.get("current_industry")),
        "country": _normalise(profile.get("country")),
    }

    features.update(_skills_features(skills))
    features.update(_career_features(candidate.get("career_history", [])))
    features.update(_education_features(candidate.get("education", [])))
    features.update(_signal_features(signals))
    features.update(_derived_features(profile, skills, signals))

    return features
