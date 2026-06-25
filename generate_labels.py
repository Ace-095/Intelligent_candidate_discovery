import csv

labels = {
    "CAND_0083811": 0.0,
    "CAND_0014593": 0.0,
    "CAND_0003279": 0.0,
    "CAND_0097197": 0.0,
    "CAND_0036049": 0.0,
    "CAND_0032099": 0.0,
    "CAND_0029257": 0.0,
    "CAND_0018290": 0.0,
    "CAND_0096531": 0.0,
    "CAND_0013435": 0.0,
    "CAND_0088697": 0.2,
    "CAND_0097081": 0.0,
    "CAND_0071483": 0.0,
    "CAND_0011396": 0.0,
    "CAND_0077398": 0.0,
    "CAND_0055303": 0.0,
    "CAND_0004166": 0.0,
    "CAND_0003906": 0.0,
    "CAND_0012281": 0.0,
    "CAND_0028658": 0.1,
    "CAND_0030496": 0.0,
    "CAND_0066238": 0.0,
    "CAND_0078908": 0.1,
    "CAND_0003479": 0.0,
    "CAND_0073564": 0.0,
    "CAND_0026063": 0.0,
    "CAND_0093851": 0.0,
    "CAND_0085182": 0.0,
    "CAND_0091925": 0.0,
    "CAND_0071427": 0.4,
}

with open("holdout_labels.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["candidate_id", "true_score"])
    for cid, score in labels.items():
        writer.writerow([cid, score])

print("Wrote holdout_labels.csv")
