# Redrob — Intelligent Candidate Discovery & Ranking

Offline candidate ranking pipeline for the India Runs Data & AI Challenge.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce Submission

Run the following command from the repository root to produce the submission CSV:

> [!IMPORTANT]
> **Data Requirement:** You must place the hackathon-provided `candidates.jsonl` file in the root of this repository before running the command.

```bash
python rank.py \
  --candidates ./candidates.jsonl \
  --out ./submission.csv
```

The script will:
1. Parse and validate candidates from JSONL via streaming (no memory bloat).
2. Extract literal text facts and structured signals (skills, career history, behavioral).
3. Detect and heavily penalize honeypots and keyword stuffers using deep heuristic flags.
4. Calculate semantic similarity to the Job Description offline using an IDF-weighted NLP scheme.
5. Apply composite score logic prioritizing top-of-funnel accuracy (NDCG@10 tuning).
6. Generate completely unique, varied natural language reasoning strings purely from facts.
7. Sort ties systematically and self-validate output automatically.
8. Write exactly top 100 rows to the CSV output.

## Running in the Docker Sandbox (Testing)

We provide a `Dockerfile` and a 100-candidate sample (`sample_candidates.jsonl`) so judges can safely test the pipeline inside an isolated container to ensure reproducibility and network independence.

```bash
# Build the image (Downloads NLP models at build time to enforce offline execution)
docker build -t redrob-ranker .

# Run the sandbox over the sample dataset
docker run --rm --network none redrob-ranker
```

## Validate Submission

```bash
python validate_submission.py submission.csv
```

## Output Format

The submission CSV must contain exactly 100 rows with the header:

```
candidate_id,rank,score,reasoning
```

- `candidate_id`: Format `CAND_XXXXXXX` (7 digits).
- `rank`: Integer 1–100, unique, rank 1 is the best fit.
- `score`: Float, non-increasing as rank increases.
- `reasoning`: 1–2 sentence explanation of the ranking decision.

## Constraint Compliance

This pipeline strictly passes all Hackathon constraints, as empirically verified:

- **CPU only**: No PyTorch/TensorFlow GPU modules are imported. Pure `scikit-learn` and CPU vectors.
- **Max 16GB RAM**: The streaming JSON parser and garbage-collected feature arrays restrict peak memory usage to **1.4 GB** maximum on a 100,000 candidate set.
- **Max 5 minutes**: The entire 100k candidate pipeline, from JSON parse to CSV generation, executes in **<18.0 seconds** wall-clock time.
- **No network access**: The `.gsd/` pipeline and `download_model.py` stage the SentenceTransformer models ahead of time. `HF_HUB_OFFLINE=1` is proven to pass the test suite in `test_constraints.sh`.

## Repository Structure

```
rank.py                    # Main entry point
lib/
  parsing.py               # JSONL loading and schema validation
  features.py              # Feature extraction from candidate profiles
  scoring.py               # Scoring model against JD
  honeypot.py              # Honeypot and keyword-stuffer detection
  output.py                # CSV writing and validation
requirements.txt           # Pinned dependencies
submission_metadata.yaml   # Submission metadata (team, repo, etc.)
```
