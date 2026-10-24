
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "cnr_ext_fast_demo" / "predictions.csv"

df = pd.read_csv(CSV_PATH)

errors = df[df["true_label"] != df["predicted_label"]].copy()
errors["error_type"] = errors.apply(
    lambda row: (
        "Occupied predicted FREE"
        if row["true_label"] == 1
        else "Free predicted OCCUPIED"
    ),
    axis=1,
)

print(f"Total evaluated: {len(df)}")
print(f"Total errors: {len(errors)}")
print("\nErrors by type:")
print(errors["error_type"].value_counts())

print("\nMisclassified slots:")
print(
    errors[
        [
            "date",
            "camera",
            "slot_id",
            "true_label",
            "predicted_label",
            "busy_probability",
            "error_type",
        ]
    ].to_string(index=False)
)

output_path = ROOT / "outputs" / "cnr_ext_fast_demo" / "misclassified_slots.csv"
errors.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
