FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY lib/ lib/
COPY rank.py download_model.py validate_submission.py ./

# Download model at build time to ensure zero network calls at runtime
RUN python3 download_model.py

# Create a data directory for the container volume
RUN mkdir -p /data
COPY sample_candidates.jsonl /data/candidates.jsonl

# Enforce offline mode
ENV HF_HUB_OFFLINE=1

# Entrypoint to run the ranking script
ENTRYPOINT ["python3", "rank.py", "--candidates", "/data/candidates.jsonl", "--out", "/data/submission.csv"]
