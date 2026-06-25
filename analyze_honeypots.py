#!/usr/bin/env python3
"""
analyze_honeypots.py — Offline honeypot signature discovery

Usage:
    python3 analyze_honeypots.py --candidates <path>

Streams the full candidates.jsonl, computes 5 internal-consistency signals
per candidate, flags outliers, and writes a ranked JSON report.

DO NOT import this into rank.py. It is a one-time analysis tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root so we can import lib.parsing
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from lib.parsing import iter_candidates


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def compute_signals(c: dict) -> dict:
    """Compute all internal-consistency signals for one candidate."""
    cid = c["candidate_id"]
    profile = c.get("profile", {})
    career  = c.get("career_history", [])
    skills  = c.get("skills", [])

    yoe = float(profile.get("years_of_experience", 0) or 0)
    yoe_months = yoe * 12

    # --- Career signals ---
    total_career_months = sum(
        int(r.get("duration_months", 0) or 0) for r in career
    )
    num_roles = len(career)

    # Overlap detection: does the sum of role durations
    # massively exceed the claimed YOE?
    career_vs_yoe_excess = total_career_months - yoe_months

    # --- Skill signals ---
    skill_count = len(skills)
    skill_durations = [
        int(s.get("duration_months", 0) or 0) for s in skills
    ]
    max_skill_duration = max(skill_durations) if skill_durations else 0

    # Skill that outlasts the whole career
    skill_exceeds_career = max_skill_duration - max(total_career_months, yoe_months)

    # Expert skills with near-zero duration (0 or 1 month)
    expert_zero = [
        s for s in skills
        if s.get("proficiency") == "expert"
        and int(s.get("duration_months", 0) or 0) <= 1
    ]
    expert_zero_count = len(expert_zero)

    # Endorsement-to-duration mismatch:
    # high endorsements on a skill with 0 duration = suspicious
    high_endorse_zero_dur = [
        s for s in skills
        if s.get("endorsements", 0) > 20
        and int(s.get("duration_months", 0) or 0) == 0
    ]
    high_endorse_zero_count = len(high_endorse_zero_dur)

    # Skill count vs YOE: too many skills for too few years
    # Benchmark: roughly 3 skills per year of experience is plausible
    skill_per_yoe_ratio = skill_count / max(yoe, 1)

    # --- Flags ---
    flags = []
    flag_score = 0

    if skill_exceeds_career > 12:
        flags.append(f"skill_exceeds_career_by_{int(skill_exceeds_career)}mo")
        flag_score += 3

    if expert_zero_count >= 3:
        flags.append(f"expert_zero_dur_x{expert_zero_count}")
        flag_score += expert_zero_count

    if skill_per_yoe_ratio > 8 and skill_count > 15:
        flags.append(f"skill_count_{skill_count}_for_yoe_{yoe:.0f}y")
        flag_score += 2

    if high_endorse_zero_count >= 2:
        flags.append(f"high_endorse_zero_dur_x{high_endorse_zero_count}")
        flag_score += 2

    if career_vs_yoe_excess > 60 and career_vs_yoe_excess > yoe_months * 0.5:
        flags.append(f"career_total_{int(total_career_months)}mo_vs_yoe_{int(yoe_months)}mo")
        flag_score += 2

    # Exact YOE == 50 (maximum allowed by schema — often used as a padding value)
    if yoe >= 49.5:
        flags.append(f"yoe_maxed_at_{yoe}")
        flag_score += 1

    return {
        "candidate_id": cid,
        "flag_score": flag_score,
        "flags": flags,
        # Raw signals for threshold calibration
        "yoe": yoe,
        "yoe_months": yoe_months,
        "total_career_months": total_career_months,
        "career_vs_yoe_excess": career_vs_yoe_excess,
        "num_roles": num_roles,
        "skill_count": skill_count,
        "max_skill_duration": max_skill_duration,
        "skill_exceeds_career": skill_exceeds_career,
        "expert_zero_count": expert_zero_count,
        "expert_zero_skills": [s.get("name") for s in expert_zero],
        "high_endorse_zero_count": high_endorse_zero_count,
        "skill_per_yoe_ratio": round(skill_per_yoe_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Honeypot signature discovery — offline analysis only."
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl"
    )
    parser.add_argument(
        "--top", type=int, default=200,
        help="Number of top suspicious candidates to include in report (default: 200)"
    )
    parser.add_argument(
        "--out", default="honeypot_analysis_report.json",
        help="Output JSON report path"
    )
    parser.add_argument(
        "--threshold", type=int, default=1,
        help="Minimum flag_score to include in report (default: 1)"
    )
    args = parser.parse_args()

    print(f"[analysis] Scanning {args.candidates} ...", file=sys.stderr)

    all_signals = []
    flagged = []
    n = 0

    for candidate in iter_candidates(args.candidates):
        n += 1
        sig = compute_signals(candidate)
        if sig["flag_score"] >= args.threshold:
            flagged.append(sig)

    flagged.sort(key=lambda x: -x["flag_score"])
    top = flagged[:args.top]

    # --- Aggregate stats for calibration ---
    all_flag_scores = [s["flag_score"] for s in flagged]
    expert_zero_counts = [s["expert_zero_count"] for s in flagged]
    skill_per_yoe_ratios = [s["skill_per_yoe_ratio"] for s in flagged]
    skill_exceeds = [s["skill_exceeds_career"] for s in flagged]
    career_excess_vals = [s["career_vs_yoe_excess"] for s in flagged]

    def pct(vals, p):
        if not vals:
            return 0
        vals_sorted = sorted(vals)
        k = int(len(vals_sorted) * p / 100)
        return vals_sorted[min(k, len(vals_sorted)-1)]

    report = {
        "meta": {
            "total_scanned": n,
            "total_flagged": len(flagged),
            "threshold_used": args.threshold,
        },
        "distribution": {
            "flag_score_p50": pct(all_flag_scores, 50),
            "flag_score_p90": pct(all_flag_scores, 90),
            "flag_score_p99": pct(all_flag_scores, 99),
            "flag_score_max": max(all_flag_scores) if all_flag_scores else 0,
            "expert_zero_p90": pct(expert_zero_counts, 90),
            "expert_zero_p99": pct(expert_zero_counts, 99),
            "expert_zero_max": max(expert_zero_counts) if expert_zero_counts else 0,
            "skill_per_yoe_p90": pct(skill_per_yoe_ratios, 90),
            "skill_per_yoe_max": max(skill_per_yoe_ratios) if skill_per_yoe_ratios else 0,
            "skill_exceeds_career_p90": pct(skill_exceeds, 90),
            "skill_exceeds_career_max": max(skill_exceeds) if skill_exceeds else 0,
            "career_excess_p90": pct(career_excess_vals, 90),
            "career_excess_max": max(career_excess_vals) if career_excess_vals else 0,
        },
        "flag_frequency": {},
        "top_suspicious": top,
    }

    # Count how often each flag type fires across all flagged records
    flag_freq: dict[str, int] = {}
    for sig in flagged:
        for flag in sig["flags"]:
            # Normalise to flag type (strip numeric suffix)
            flag_type = flag.split("_x")[0].split("_by_")[0].split("_at_")[0].split("_for_")[0]
            flag_freq[flag_type] = flag_freq.get(flag_type, 0) + 1
    # Sort by frequency
    report["flag_frequency"] = dict(
        sorted(flag_freq.items(), key=lambda x: -x[1])
    )

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[analysis] {n:,} scanned | {len(flagged):,} flagged | top {len(top)} written to {out_path}")


if __name__ == "__main__":
    main()
