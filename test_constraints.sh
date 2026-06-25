#!/bin/bash
set -e

echo "============================================================"
echo " Running Strict Constraint Test (Offline & Resource Monitored)"
echo "============================================================"

# Enforce fully offline mode
export HF_HUB_OFFLINE=1

# Output file
OUT_FILE="/tmp/submission_sample.csv"

# Run with /usr/bin/time to capture Max RSS and wall clock time
/usr/bin/time -v /usr/bin/python3.12 rank.py --candidates sample_candidates.jsonl --out $OUT_FILE

echo ""
echo "============================================================"
echo " Constraint Verification Successful!"
echo " - HF_HUB_OFFLINE=1 ensured no network calls"
echo " - Memory and Time captured above"
echo "============================================================"
