import csv
from pathlib import Path


def main():
    source = Path(__file__).resolve().parents[1] / "public_data" / "evaluation_predictions_212_sanitized.csv"
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 212, f"Expected 212 rows, found {len(rows)}"
    assert len({row["evaluation_id"] for row in rows}) == 212, "Duplicate evaluation_id detected"

    tp = sum(row["ground_truth"] == "AD" and row["prediction"] == "AD" for row in rows)
    tn = sum(row["ground_truth"] == "CN" and row["prediction"] == "CN" for row in rows)
    fp = sum(row["ground_truth"] == "CN" and row["prediction"] == "AD" for row in rows)
    fn = sum(row["ground_truth"] == "AD" and row["prediction"] == "CN" for row in rows)

    accuracy = (tp + tn) / len(rows)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    brier = sum(
        (float(row["probability_positive"]) - (1 if row["ground_truth"] == "AD" else 0)) ** 2
        for row in rows
    ) / len(rows)

    expected = {
        "TP": 73,
        "TN": 36,
        "FP": 88,
        "FN": 15,
    }
    actual = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    assert actual == expected, f"Confusion matrix mismatch: {actual}"

    print(f"Subjects: {len(rows)}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy={accuracy:.6f}")
    print(f"Precision={precision:.6f}")
    print(f"Recall={recall:.6f}")
    print(f"Specificity={specificity:.6f}")
    print(f"Brier score={brier:.6f}")
    print("Verification: PASS")


if __name__ == "__main__":
    main()
