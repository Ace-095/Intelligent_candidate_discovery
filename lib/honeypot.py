"""
lib.honeypot — Honeypot Detection

Detects the ~80 candidates in this dataset with internally inconsistent profiles.

## Empirically Derived Rules (Phase 3 analysis, 2025-06-24)

The honeypot pattern in this specific dataset was discovered by scanning all
100,000 records with analyze_honeypots.py and inspecting the top outliers.
These rules are NOT copied from the spec's illustrative examples — they are
derived from the actual data.

### Primary Rule (covers all 86 confirmed honeypots, 0 false positives)

    skill_duration_ratio = max(skill.duration_months)
                           / max(total_career_months, yoe_months)
    IF skill_duration_ratio >= 4.3 → HONEYPOT

What this catches: Candidates claiming a single skill lasting 4.3× or more
their entire professional career. E.g. 1 year of work experience but claiming
a skill used for 52+ months. All 86 flagged candidates have career=12-13mo
and max_skill=52-60mo. No candidates with YOE > 3 years trigger this rule.

### Secondary Rule (data-observed, minor)

    expert_skills_with_zero_duration >= 5 → partial penalty (0.3)

Only 21 candidates in the dataset trigger this; all overlap with the primary
rule. Kept as a soft signal for scoring degradation.

### Removed checks (proven too broad in analysis)
- High skill count alone: 8,874 candidates (8.9%) — too many false positives
- Career total > YOE: only 14 candidates, inconsistent signal
- Impossible age via education dates: signal was theoretical, no empirical hits

Public API:
  - detect(candidate: dict, features: dict) -> tuple[float, str | None]
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# EMPIRICALLY-DERIVED CONSTANTS
# (derived from analyze_honeypots.py run on the full 100k dataset)
# ---------------------------------------------------------------------------

# PRIMARY RULE: The max duration_months of any claimed skill, divided by the
# candidate's total career reference (max of total_career_months and yoe_months),
# must not reach this ratio. Derived from data: 86 candidates at >= 4.3,
# zero false positives (no candidates with yoe > 3y are caught).
SKILL_TO_CAREER_RATIO_THRESHOLD: float = 4.3

# SECONDARY RULE: Number of "expert" proficiency skills with zero duration
# that triggers a soft penalty. Derived from data: only 5-skill clusters
# were observed; threshold of 5 captures true outliers without touching
# legitimate power-users.
EXPERT_ZERO_DURATION_THRESHOLD: int = 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _primary_check(candidate: dict) -> tuple[bool, str]:
    """
    PRIMARY RULE — Skill duration exceeds career timeline by factor >= 4.3.

    A person with 1 year of career history cannot have used a skill for 52+
    months. This is the strongest and cleanest honeypot signal in the dataset.
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])

    total_career_months = sum(
        int(r.get("duration_months", 0) or 0) for r in career
    )
    yoe_months = float(profile.get("years_of_experience", 0) or 0) * 12
    reference_months = max(total_career_months, yoe_months, 1)  # avoid div/0

    max_skill_duration = max(
        (int(s.get("duration_months", 0) or 0) for s in skills),
        default=0,
    )

    ratio = max_skill_duration / reference_months
    if ratio >= SKILL_TO_CAREER_RATIO_THRESHOLD:
        return True, (
            f"honeypot: max skill duration {max_skill_duration}mo "
            f"is {ratio:.1f}x career reference {reference_months:.0f}mo"
        )
    return False, ""


def _secondary_check(candidate: dict) -> tuple[bool, str]:
    """
    SECONDARY RULE — Multiple expert-level skills with zero usage duration.

    Soft signal: returns True only when >= 5 expert skills have 0 duration.
    """
    skills = candidate.get("skills", [])
    expert_zero = [
        s for s in skills
        if s.get("proficiency") == "expert"
        and int(s.get("duration_months", 0) or 0) == 0
    ]
    if len(expert_zero) >= EXPERT_ZERO_DURATION_THRESHOLD:
        names = [s.get("name", "?") for s in expert_zero[:5]]
        return True, (
            f"suspicious: {len(expert_zero)} expert skills with 0 months usage "
            f"({', '.join(names)})"
        )
    return False, ""


def _stuffer_check(candidate: dict, fv: dict) -> tuple[bool, str]:
    """
    KEYWORD STUFFER RULE — Identifies candidates cramming skills without backing.
    """
    skill_count = fv.get("skill_count", 0)
    
    # Condition 1: Lots of skills, but very little usage or validation
    avg_dur = fv.get("avg_skill_duration_months", 0.0)
    avg_end = fv.get("total_endorsements", 0) / max(skill_count, 1)
    
    if skill_count > 25 and avg_dur < 6.0 and avg_end < 2.0:
        return True, (
            f"keyword stuffer: {skill_count} skills but only {avg_dur:.1f}mo "
            f"avg duration and {avg_end:.1f} avg endorsements"
        )
        
    # Condition 2: Many skills but no career history to support them
    career_entries = fv.get("career_entry_count", 0)
    if skill_count > 15 and career_entries == 0:
        return True, (
            f"keyword stuffer: {skill_count} skills but 0 career history entries"
        )
        
    return False, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------



def detect(candidate: dict, features: dict) -> tuple[float, str | None]:
    """
    Detect honeypot profiles using empirically-derived rules.

    Args:
        candidate: Raw candidate dict from lib.parsing.
        features:  Feature dict from lib.features.extract() (reserved for
                   future use; not currently needed by the primary rule).

    Returns:
        (penalty_multiplier, reason_or_None)
        - 1.0: clean profile, no penalty.
        - 0.0: primary rule triggered → disqualified (honeypot).
        - 0.3: secondary rule only → soft downrank (suspicious but not confirmed).
    """
    primary_hit, primary_reason = _primary_check(candidate)

    if primary_hit:
        return 0.0, f"HONEYPOT: {primary_reason}"
        
    stuffer_hit, stuffer_reason = _stuffer_check(candidate, features)
    if stuffer_hit:
        return 0.1, f"STUFFER: {stuffer_reason}"

    secondary_hit, secondary_reason = _secondary_check(candidate)
    if secondary_hit:
        return 0.3, secondary_reason

    return 1.0, None
