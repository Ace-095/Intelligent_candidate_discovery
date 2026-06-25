#!/usr/bin/env python3
"""
rank.py — Redrob Intelligent Candidate Ranking Pipeline

Entry point for the hackathon submission.

Usage:
    python rank.py --candidates <path_to_candidates.jsonl> --out <output.csv>

The script processes 100,000 candidate profiles against a target Job Description
and outputs the top 100 candidates in a spec-compliant CSV.

Constraints:
  - CPU only (no GPU)
  - Max 16GB RAM
  - Max 5 minutes for this script
  - No network access during execution
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import traceback
import subprocess
import os
from pathlib import Path
from typing import Generator

# ---------------------------------------------------------------------------
# Attempt to import psutil for memory tracking (optional but preferred)
# ---------------------------------------------------------------------------
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from lib import parsing, features, scoring, honeypot, output, semantic, reasoning


# ---------------------------------------------------------------------------
# Profiling utilities
# ---------------------------------------------------------------------------

def _get_rss_mb() -> float:
    """Return resident set size (RSS) in MB, or 0 if psutil unavailable."""
    if _PSUTIL_AVAILABLE:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    return 0.0


@contextlib.contextmanager
def stage(name: str) -> Generator[None, None, None]:
    """Context manager to log elapsed time and memory usage for a pipeline stage."""
    t0 = time.perf_counter()
    mem_before = _get_rss_mb()
    print(f"\n[{name}] Starting...", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        mem_after = _get_rss_mb()
        mem_delta = mem_after - mem_before
        print(
            f"[{name}] Done in {elapsed:.2f}s | "
            f"RSS: {mem_after:.0f} MB ({mem_delta:+.0f} MB)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Redrob Candidate Ranking Pipeline — "
            "ranks candidates.jsonl against the target JD and writes top 100 to CSV."
        )
    )
    parser.add_argument(
        "--candidates",
        required=True,
        metavar="PATH",
        help="Path to candidates.jsonl or candidates.jsonl.gz (required)",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Output path for the submission CSV (required)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        metavar="N",
        help="Number of top candidates to write to output (default: 100)",
    )
    parser.add_argument(
        "--validator",
        type=str,
        default="validate_submission.py",
        metavar="PATH",
        help="Path to validate_submission.py script (default: validate_submission.py)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Execute the full ranking pipeline."""
    pipeline_start = time.perf_counter()
    print("=" * 60)
    print(" Redrob Candidate Ranking Pipeline")
    print("=" * 60)
    print(f"  candidates: {args.candidates}")
    print(f"  output:     {args.out}")
    print(f"  top_n:      {args.top_n}")
    print(f"  validator:  {args.validator}")
    print("=" * 60)

    # ── Stage 1: Stream, extract features, and detect honeypots ──────────
    # Records are processed one at a time as they stream from disk.
    # The raw candidate JSON is discarded immediately after extraction;
    # only a lightweight flat dict is retained per candidate.
    processed_candidates: list[dict] = []
    honeypot_count = 0
    hard_disqualified = 0

    with stage("STREAM & EXTRACT"):
        # Also collect raw candidate dicts for semantic text extraction.
        # We keep only the minimal fields needed (profile, skills, career_history);
        # the full record is freed immediately after text extraction below.
        raw_for_semantic: list[dict] = []

        for candidate in parsing.iter_candidates(args.candidates):
            # Feature extraction — must happen before honeypot (needs fv)
            fv = features.extract(candidate)

            # Honeypot detection
            penalty, hp_reason = honeypot.detect(candidate, fv)
            if penalty < 1.0:
                honeypot_count += 1
            if penalty == 0.0:
                hard_disqualified += 1

            # Retain only what scoring needs; drop full raw dict
            processed_candidates.append({
                "candidate_id": candidate["candidate_id"],
                "fv": fv,
                "penalty": penalty,
                "hp_reason": hp_reason,
            })

            # Keep a slim dict for semantic text building
            raw_for_semantic.append({
                "profile": candidate.get("profile", {}),
                "career_history": candidate.get("career_history", [])[:3],
                "skills": candidate.get("skills", [])[:20],
            })

        total = len(processed_candidates)
        print(
            f"[STREAM & EXTRACT] {total:,} candidates processed | "
            f"{honeypot_count} suspicious ({hard_disqualified} disqualified).",
            flush=True,
        )

    # ── Stage 1b: Semantic similarity scoring ────────────────────────────
    # Encode the JD once, then batch-encode all candidates.
    # The IDF scorer needs a fit() pass over all texts before scoring.
    SEMANTIC_BATCH_SIZE = 512

    with stage("SEMANTIC"):
        sem_scorer = semantic.SemanticScorer()

        # Build all texts for IDF fitting
        all_texts = [semantic._candidate_text(c) for c in raw_for_semantic]
        sem_scorer.fit(all_texts)

        # Score in batches and inject into feature vectors
        for batch_start in range(0, len(processed_candidates), SEMANTIC_BATCH_SIZE):
            batch_raw = raw_for_semantic[batch_start: batch_start + SEMANTIC_BATCH_SIZE]
            batch_proc = processed_candidates[batch_start: batch_start + SEMANTIC_BATCH_SIZE]
            sim_scores = sem_scorer.score_batch(batch_raw)
            for item, sim in zip(batch_proc, sim_scores):
                item["fv"]["semantic_similarity"] = round(float(sim), 6)

        # Free the raw semantic data — no longer needed
        del raw_for_semantic, all_texts
        print(f"[SEMANTIC] Scored {len(processed_candidates):,} candidates.", flush=True)

    # ── Stage 2: Scoring ──────────────────────────────────────────────────
    with stage("SCORING"):
        scored: list[dict] = []
        for item in processed_candidates:
            raw_score, elite_notes = scoring.score(item["fv"])
            final_score = raw_score * item["penalty"]

            scored.append(
                {
                    "candidate_id": item["candidate_id"],
                    "score": round(final_score, 6),
                    "fv": item["fv"],
                    "hp_reason": item["hp_reason"],
                    "elite_notes": elite_notes,
                }
            )
        print(f"[SCORING] Scored {len(scored):,} candidates.")

    # ── Stage 3: Sort & output ────────────────────────────────────────────
    with stage("OUTPUT"):
        sorted_results = output.sort_results(scored)
        top_n = sorted_results[:args.top_n]

        # Generate reasoning strings for top candidates only
        for item in top_n:
            item["reasoning"] = reasoning.generate(
                item["candidate_id"],
                item["fv"],
                item["hp_reason"],
                item["elite_notes"]
            )

        output.write_submission(top_n, args.out)

    # ── Stage 4: Validation ───────────────────────────────────────────────
    with stage("VALIDATION"):
        if os.path.exists(args.validator):
            try:
                # Run the official validator script
                result = subprocess.run(
                    [sys.executable, args.validator, args.out],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"[VALIDATION] Output: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"[VALIDATION] FAILED!\n{e.stdout}\n{e.stderr}", file=sys.stderr)
                raise RuntimeError("Submission failed official validation.") from e
        else:
            print(f"[VALIDATION] Skipped: validator script not found at '{args.validator}'")

    # ── Summary ───────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - pipeline_start
    final_mem = _get_rss_mb()
    print("\n" + "=" * 60)
    print(f" Pipeline complete in {total_elapsed:.2f}s | Final RSS: {final_mem:.0f} MB")
    print(f" Submission written to: {args.out}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate inputs before starting the pipeline
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: candidates file not found: {args.candidates}", file=sys.stderr)
        sys.exit(1)

    try:
        run(args)
    except Exception:
        print("\n" + "=" * 60, file=sys.stderr)
        print("FATAL ERROR — pipeline aborted. No output file written.", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Ensure no partial output remains
        out_path = Path(args.out)
        if out_path.exists():
            out_path.unlink()
            print(f"Removed partial output: {args.out}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
