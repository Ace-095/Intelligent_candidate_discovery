import csv
import json
from lib import parsing, features, scoring, honeypot, semantic

HOLDOUT_JSONL = "holdout.jsonl"
HOLDOUT_LABELS = "holdout_labels.csv"

def spearman_rank_correlation(y_true, y_pred):
    n = len(y_true)
    if n <= 1:
        return 0.0
        
    def get_ranks(values):
        sorted_indices = sorted(range(n), key=lambda i: values[i])
        ranks = [0] * n
        for rank, index in enumerate(sorted_indices):
            ranks[index] = rank + 1
        return ranks
        
    rank_true = get_ranks(y_true)
    rank_pred = get_ranks(y_pred)
    
    d_squared = sum((rank_true[i] - rank_pred[i]) ** 2 for i in range(n))
    rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
    return rho

def evaluate():
    # 1. Read ground truth
    ground_truth = {}
    with open(HOLDOUT_LABELS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth[row["candidate_id"]] = float(row["true_score"])
            
    # 2. Score with pipeline
    pipeline_scores = {}
    
    raw_for_semantic = []
    processed_candidates = []
    
    for parsed in parsing.iter_candidates(HOLDOUT_JSONL):
        fv = features.extract(parsed)
        penalty, hp_reason = honeypot.detect(parsed, fv)
        
        raw_for_semantic.append({
            "profile": parsed.get("profile", {}),
            "career_history": parsed.get("career_history", [])[:3],
            "skills": parsed.get("skills", [])[:15],
        })
        
        processed_candidates.append({
            "candidate_id": parsed["candidate_id"],
            "fv": fv,
            "penalty": penalty
        })
        
    sem_scorer = semantic.SemanticScorer()
    all_texts = [semantic._candidate_text(c) for c in raw_for_semantic]
    sem_scorer.fit(all_texts)
    sim_scores = sem_scorer.score_batch(raw_for_semantic)
    
    for item, sim in zip(processed_candidates, sim_scores):
        fv = item["fv"]
        fv["semantic_similarity"] = float(sim)
        raw_score, _ = scoring.score(fv)
        final_score = raw_score * item["penalty"]
        pipeline_scores[item["candidate_id"]] = final_score
            
    # 3. Align and calculate rank correlation
    y_true = []
    y_pred = []
    
    for cid, true_score in ground_truth.items():
        if cid in pipeline_scores:
            y_true.append(true_score)
            y_pred.append(pipeline_scores[cid])
            
    if not y_true:
        print("No intersecting candidates found.")
        return
        
    correlation = spearman_rank_correlation(y_true, y_pred)
    
    print("=== Evaluation against Internal Holdout ===")
    print(f"Total evaluated: {len(y_true)}")
    print(f"Spearman Rank Correlation: {correlation:.4f}")
    
    if correlation > 0.6:
        print("Result: EXCELLENT alignment with human judgment.")
    elif correlation > 0.4:
        print("Result: GOOD alignment, but could be improved.")
    else:
        print("Result: POOR alignment. Review scoring weights.")

if __name__ == "__main__":
    evaluate()
