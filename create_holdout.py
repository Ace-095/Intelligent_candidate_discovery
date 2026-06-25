import json
import random
import csv

CANDIDATES_PATH = "AI_DATA/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
HOLDOUT_JSONL = "holdout.jsonl"
HOLDOUT_CSV = "holdout_human.csv"
SEED = 42
SAMPLE_SIZE = 30

def create_holdout():
    # Read all candidates
    candidates = []
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
            
    print(f"Loaded {len(candidates)} candidates.")
    
    # Shuffle and sample
    random.seed(SEED)
    sampled = random.sample(candidates, SAMPLE_SIZE)
    
    # Write to holdout.jsonl
    with open(HOLDOUT_JSONL, "w", encoding="utf-8") as f:
        for c in sampled:
            f.write(json.dumps(c) + "\n")
            
    print(f"Wrote {SAMPLE_SIZE} candidates to {HOLDOUT_JSONL}.")
    
    # Write a human readable CSV
    with open(HOLDOUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "title", "yoe", "skills", "latest_company"])
        for c in sampled:
            prof = c.get("profile", {})
            title = prof.get("current_title", "")
            yoe = prof.get("years_of_experience", "")
            skills = ", ".join(s.get("name", "") for s in c.get("skills", [])[:5])
            
            career = c.get("career_history", [])
            company = ""
            if career:
                current = next((e for e in career if e.get("is_current")), None)
                if not current:
                    current = max(career, key=lambda e: e.get("start_date", ""))
                company = current.get("company", "")
                
            writer.writerow([c.get("candidate_id"), title, yoe, skills, company])
            
    print(f"Wrote human readable summary to {HOLDOUT_CSV}.")

if __name__ == "__main__":
    create_holdout()
