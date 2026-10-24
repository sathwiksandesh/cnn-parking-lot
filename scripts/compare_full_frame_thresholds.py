
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "cnr_ext_fast_demo" / "predictions.csv"
OUTPUT = ROOT / "outputs" / "cnr_ext_fast_demo" / "full_frame_threshold_comparison.csv"

df = pd.read_csv(INPUT)

y_true = df["true_label"].astype(int)
probabilities = df["busy_probability"].astype(float)

rows = []

for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    y_pred = (probabilities >= threshold).astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    rows.append({
        "threshold": threshold,
        "accuracy": (tp + tn) / len(df),
        "busy_precision": precision,
        "busy_recall": recall,
        "busy_f1": f1,
        "free_predicted_busy": fp,
        "busy_predicted_free": fn,
        "correct_free": tn,
        "correct_busy": tp,
    })

results = pd.DataFrame(rows)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

print(results.round(4).to_string(index=False))
print(f"\nSaved to: {OUTPUT}")
print("\nReminder: use this comparison diagnostically; select the final")
print("threshold on validation data, not by optimizing the test set.")
