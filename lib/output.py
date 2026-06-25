"""
lib.output — Submission CSV Writing

Responsibilities:
  - Accept the final sorted list of top-100 scored candidates
  - Write a valid submission.csv that passes validate_submission.py
  - Enforce all format constraints from the submission spec:
    - Exactly 100 rows
    - Unique, consecutive ranks 1–100
    - Scores non-increasing by rank
    - Tie-break by candidate_id ascending
    - UTF-8 encoding

Public API:
  - write_submission(results: list[dict], out_path: str) -> None
      Writes the top-100 submission CSV.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
EXPECTED_ROWS = 100


def write_submission(results: list[dict], out_path: str) -> None:
    """
    Write the final ranked candidates to a submission CSV.

    Args:
        results: List of dicts, each containing at least:
                   - candidate_id (str)
                   - score (float)
                   - reasoning (str)
                 Must have >= 100 entries sorted by score descending,
                 with ties broken by candidate_id ascending.
        out_path: Path to write the output CSV file.

    Raises:
        ValueError: If fewer than 100 valid candidates are provided.
        IOError:    If the output path cannot be written.
    """
    if len(results) < EXPECTED_ROWS:
        raise ValueError(
            f"Need at least {EXPECTED_ROWS} candidates to write submission, "
            f"got {len(results)}."
        )

    # Take top 100
    top100 = results[:EXPECTED_ROWS]

    # Validate scores are non-increasing (sort guarantees this, but let's be explicit)
    for i in range(len(top100) - 1):
        s1 = top100[i]["score"]
        s2 = top100[i + 1]["score"]
        if s1 < s2:
            print(
                f"[output] WARNING: score not non-increasing at positions {i} ({s1}) "
                f"and {i+1} ({s2}). Output may fail validation.",
                file=sys.stderr,
            )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_HEADER)
        for rank, candidate in enumerate(top100, start=1):
            cid = candidate["candidate_id"]
            score = candidate["score"]
            reasoning = str(candidate.get("reasoning", "")).replace("\n", " ").strip()
            writer.writerow([cid, rank, score, reasoning])

    print(f"[output] Wrote {EXPECTED_ROWS} candidates to '{out_path}'.")


def sort_results(scored_candidates: list[dict]) -> list[dict]:
    """
    Sort candidates by score descending, then candidate_id ascending for ties.

    Args:
        scored_candidates: List of dicts with 'score' and 'candidate_id' keys.

    Returns:
        Sorted list.
    """
    return sorted(
        scored_candidates,
        key=lambda c: (-c["score"], c["candidate_id"]),
    )
