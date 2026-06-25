"""
lib.parsing — Candidate Data Streaming and Validation

Responsibilities:
  - Stream candidates.jsonl / candidates.jsonl.gz line by line
    (never accumulates the full dataset into a list)
  - Validate each record against the required fields from candidate_schema.json:
      Top-level: candidate_id, profile, career_history, education, skills,
                 redrob_signals
      profile:   anonymized_name, headline, summary, location, country,
                 years_of_experience, current_title, current_company,
                 current_company_size, current_industry
  - On any malformed line (bad JSON, missing keys, bad candidate_id format),
    log a warning to stderr and continue — never raise or crash
  - Yield one validated dict at a time for downstream consumption

Public API:
  - iter_candidates(path: str) -> Generator[dict, None, None]
      Yields validated candidate dicts one by one as lines are read.
      Logs summary stats on StdErr when the file is exhausted.
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path
from typing import Generator

# ---------------------------------------------------------------------------
# Validation constants (derived from candidate_schema.json)
# ---------------------------------------------------------------------------

CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")

REQUIRED_TOP_LEVEL_KEYS = frozenset({
    "candidate_id",
    "profile",
    "career_history",
    "education",
    "skills",
    "redrob_signals",
})

REQUIRED_PROFILE_KEYS = frozenset({
    "anonymized_name",
    "headline",
    "summary",
    "location",
    "country",
    "years_of_experience",
    "current_title",
    "current_company",
    "current_company_size",
    "current_industry",
})

# Log a progress heartbeat every N lines to reassure users on large files
_HEARTBEAT_EVERY = 10_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _raw_lines(path: Path) -> Generator[str, None, None]:
    """Yield raw text lines from a plain or gzipped JSONL file."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from fh
    else:
        with open(path, "r", encoding="utf-8") as fh:
            yield from fh


def _validate(record: dict, line_no: int) -> str | None:
    """
    Validate a parsed dict against schema requirements.

    Returns None if valid, or an error string describing the first failure.
    """
    # 1. Required top-level keys
    missing_top = REQUIRED_TOP_LEVEL_KEYS - record.keys()
    if missing_top:
        return f"missing top-level keys {sorted(missing_top)}"

    # 2. candidate_id format
    cid = record.get("candidate_id", "")
    if not CANDIDATE_ID_PATTERN.match(str(cid)):
        return f"invalid candidate_id '{cid}'"

    # 3. Required profile sub-keys
    profile = record.get("profile", {})
    if not isinstance(profile, dict):
        return "profile is not an object"
    missing_profile = REQUIRED_PROFILE_KEYS - profile.keys()
    if missing_profile:
        return f"missing profile keys {sorted(missing_profile)}"

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def iter_candidates(path: str) -> Generator[dict, None, None]:
    """
    Stream and validate candidates from a JSONL file, one record at a time.

    Yields each valid candidate dict as soon as it is parsed — no full-file
    accumulation. Malformed or invalid lines are logged to stderr and skipped.

    Args:
        path: Path to candidates.jsonl or candidates.jsonl.gz.

    Yields:
        Validated candidate dicts.

    Raises:
        FileNotFoundError: If the file does not exist (fast-fail before stream).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    yielded = 0
    skipped = 0

    for line_no, raw_line in enumerate(_raw_lines(p), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # Heartbeat for large files
        if line_no % _HEARTBEAT_EVERY == 0:
            print(
                f"[parsing] ...{line_no:,} lines read, "
                f"{yielded:,} valid, {skipped} skipped",
                file=sys.stderr,
            )

        # JSON decode
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            print(
                f"[parsing] SKIP line {line_no}: JSON decode error — {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        # Schema validation
        error = _validate(record, line_no)
        if error:
            cid = record.get("candidate_id", "<no id>")
            print(
                f"[parsing] SKIP line {line_no} ({cid}): {error}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        yielded += 1
        yield record

    # Final summary
    print(
        f"[parsing] Done — {yielded:,} valid candidates, {skipped} skipped.",
        flush=True,
    )
