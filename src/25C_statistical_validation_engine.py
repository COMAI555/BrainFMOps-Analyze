#!/usr/bin/env python3
"""
25C_statistical_validation_engine.py

BrainFMOps-Analyze — STEP 25C
Statistical Validation Engine for subject-level binary MRI classification.

Inputs
------
- evaluation_records_with_labels.csv from STEP 25B
  or evaluation_summary.csv plus --labels-csv

Outputs
-------
25C_Statistical_Validation/
├── statistical_validation_report.json
├── statistical_metrics.csv
├── bootstrap_intervals.csv
├── calibration_table.csv
├── decision_curve_table.csv
├── statistical_summary.txt
├── Fig_confusion_matrix_normalized.png
├── Fig_roc_curve_ci.png
├── Fig_precision_recall_curve.png
├── Fig_calibration_curve.png
├── Fig_decision_curve.png
├── Fig_bootstrap_metric_distributions.png
└── Fig_probability_by_ground_truth.png

Implemented statistics
----------------------
- Accuracy
- Balanced accuracy
- Precision
- Sensitivity / Recall
- Specificity
- Negative predictive value
- F1-score
- Matthews correlation coefficient (MCC)
- Cohen's kappa
- ROC-AUC
- PR-AUC
- Brier score
- Log loss
- Wilson confidence intervals for proportions
- Stratified bootstrap confidence intervals
- Exact binomial-style calibration table
- Decision curve analysis
- McNemar test versus a majority-class baseline

Critical limitations
--------------------
- Five cases are enough only for software verification, not scientific claims.
- DeLong comparison requires at least two competing models; it is not faked here.
- McNemar against a majority baseline is descriptive and does not replace
  comparison with a serious benchmark model.
- External clinical validity is not established.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_VERSION = "1.0.0"


@dataclass
class LabeledRecord:
    case_id: str
    ground_truth: str
    prediction: str
    probability_positive: float
    locked_threshold: float
    total_runtime_seconds: Optional[float] = None


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def load_labels(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {path}")

    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if not reader.fieldnames or not {"case_id", "label"}.issubset(reader.fieldnames):
            raise ValueError("Labels CSV must contain columns: case_id,label")
        for row in reader:
            case_id = row["case_id"].strip()
            label = row["label"].strip()
            if case_id and label:
                labels[case_id] = label
    return labels


def load_records(
    records_csv: Path,
    external_labels: dict[str, str],
) -> list[LabeledRecord]:
    if not records_csv.exists():
        raise FileNotFoundError(f"Records CSV not found: {records_csv}")

    records: list[LabeledRecord] = []
    with records_csv.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if not reader.fieldnames:
            raise ValueError("CSV has no header.")

        required = {"case_id", "prediction", "probability_positive"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            case_id = row["case_id"].strip()
            prediction = row.get("prediction", "").strip()
            probability = parse_float(row.get("probability_positive"))
            threshold = parse_float(row.get("locked_threshold"))
            ground_truth = (
                external_labels.get(case_id)
                or row.get("ground_truth", "").strip()
            )

            status = row.get("status", "COMPLETED").strip().upper()
            inference_success_text = row.get("inference_success", "True")
            inference_success = str(inference_success_text).strip().lower() in {
                "true", "1", "yes", "y"
            }

            if status not in {"COMPLETED", "COMPLETED_RESUMED"}:
                continue
            if not inference_success:
                continue
            if not case_id or not prediction or probability is None or not ground_truth:
                continue

            records.append(
                LabeledRecord(
                    case_id=case_id,
                    ground_truth=ground_truth,
                    prediction=prediction,
                    probability_positive=probability,
                    locked_threshold=threshold if threshold is not None else 0.32,
                    total_runtime_seconds=parse_float(
                        row.get("total_runtime_seconds")
                    ),
                )
            )

    if not records:
        raise ValueError(
            "No completed labeled records were available after filtering."
        )
    return records


def safe_divide(a: float, b: float) -> Optional[float]:
    return a / b if b else None


def confusion_counts(
    records: list[LabeledRecord],
    positive_label: str,
) -> dict[str, int]:
    positive = positive_label.upper()
    tp = tn = fp = fn = 0

    for record in records:
        truth_positive = record.ground_truth.upper() == positive
        pred_positive = record.prediction.upper() == positive

        if truth_positive and pred_positive:
            tp += 1
        elif not truth_positive and not pred_positive:
            tn += 1
        elif not truth_positive and pred_positive:
            fp += 1
        else:
            fn += 1

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def roc_curve_manual(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Optional[float]]:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return np.array([]), np.array([]), None

    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    fpr_values, tpr_values = [], []

    for threshold in thresholds:
        pred = scores >= threshold
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = positives - tp
        tn = negatives - fp
        tpr_values.append(tp / (tp + fn))
        fpr_values.append(fp / (fp + tn))

    fpr = np.asarray(fpr_values, dtype=float)
    tpr = np.asarray(tpr_values, dtype=float)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def pr_curve_manual(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Optional[float]]:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return np.array([]), np.array([]), None

    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    recalls, precisions = [], []

    for threshold in thresholds:
        pred = scores >= threshold
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = positives - tp
        precisions.append(tp / (tp + fp) if tp + fp else 1.0)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)

    recall = np.asarray(recalls, dtype=float)
    precision = np.asarray(precisions, dtype=float)
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    auc = float(np.trapezoid(precision, recall))
    return recall, precision, auc


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[Optional[float], Optional[float]]:
    if total <= 0:
        return None, None

    # 1.959963984540054 for 95%; use normal approximation for configured CI.
    # Rational approximation via inverse error function is avoided here.
    z_table = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    z = z_table.get(round(confidence, 2), 1.959963984540054)

    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            p * (1 - p) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def calculate_metrics(
    records: list[LabeledRecord],
    positive_label: str,
) -> dict[str, Any]:
    c = confusion_counts(records, positive_label)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = tp + tn + fp + fn

    accuracy = safe_divide(tp + tn, n)
    precision = safe_divide(tp, tp + fp)
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    npv = safe_divide(tn, tn + fn)
    balanced = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision is not None
        and sensitivity is not None
        and precision + sensitivity > 0
        else None
    )

    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = (tp * tn - fp * fn) / denominator if denominator else None

    observed_agreement = accuracy
    truth_pos = safe_divide(tp + fn, n)
    truth_neg = safe_divide(tn + fp, n)
    pred_pos = safe_divide(tp + fp, n)
    pred_neg = safe_divide(tn + fn, n)

    expected_agreement = (
        truth_pos * pred_pos + truth_neg * pred_neg
        if None not in (truth_pos, truth_neg, pred_pos, pred_neg)
        else None
    )
    kappa = (
        (observed_agreement - expected_agreement) / (1 - expected_agreement)
        if observed_agreement is not None
        and expected_agreement is not None
        and expected_agreement < 1
        else None
    )

    positive = positive_label.upper()
    y_true = np.asarray(
        [1 if r.ground_truth.upper() == positive else 0 for r in records],
        dtype=int,
    )
    scores = np.asarray(
        [float(r.probability_positive) for r in records],
        dtype=float,
    )

    _, _, roc_auc = roc_curve_manual(y_true, scores)
    _, _, pr_auc = pr_curve_manual(y_true, scores)

    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    brier = float(np.mean((scores - y_true) ** 2))
    log_loss = float(
        -np.mean(
            y_true * np.log(clipped)
            + (1 - y_true) * np.log(1 - clipped)
        )
    )

    return {
        "n": n,
        **c,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "f1_score": f1,
        "matthews_correlation_coefficient": mcc,
        "cohens_kappa": kappa,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": log_loss,
    }


def stratified_bootstrap_ci(
    records: list[LabeledRecord],
    positive_label: str,
    metric_name: str,
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    positive = [
        r for r in records if r.ground_truth.upper() == positive_label.upper()
    ]
    negative = [
        r for r in records if r.ground_truth.upper() != positive_label.upper()
    ]

    estimate = calculate_metrics(records, positive_label).get(metric_name)
    if not positive or not negative:
        return {
            "metric": metric_name,
            "estimate": estimate,
            "lower": None,
            "upper": None,
            "valid_samples": 0,
            "confidence_level": confidence,
        }

    rng = random.Random(seed)
    values: list[float] = []

    for _ in range(iterations):
        sample_positive = [
            positive[rng.randrange(len(positive))]
            for _ in range(len(positive))
        ]
        sample_negative = [
            negative[rng.randrange(len(negative))]
            for _ in range(len(negative))
        ]
        sample = sample_positive + sample_negative
        value = calculate_metrics(sample, positive_label).get(metric_name)

        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))

    if not values:
        return {
            "metric": metric_name,
            "estimate": estimate,
            "lower": None,
            "upper": None,
            "valid_samples": 0,
            "confidence_level": confidence,
        }

    alpha = 1 - confidence
    return {
        "metric": metric_name,
        "estimate": estimate,
        "lower": float(np.percentile(values, 100 * alpha / 2)),
        "upper": float(np.percentile(values, 100 * (1 - alpha / 2))),
        "valid_samples": len(values),
        "confidence_level": confidence,
        "bootstrap_values": values,
    }


def exact_mcnemar_p_value(b: int, c: int) -> Optional[float]:
    """
    Two-sided exact McNemar p-value using Binomial(n=b+c, p=0.5).
    """
    n = b + c
    if n == 0:
        return None

    k = min(b, c)
    cumulative = sum(
        math.comb(n, i) * (0.5 ** n)
        for i in range(k + 1)
    )
    return min(1.0, 2 * cumulative)


def majority_baseline_comparison(
    records: list[LabeledRecord],
    positive_label: str,
    negative_label: str,
) -> dict[str, Any]:
    truth_counts = Counter(r.ground_truth.upper() for r in records)
    majority = truth_counts.most_common(1)[0][0]

    model_correct = [
        r.prediction.upper() == r.ground_truth.upper()
        for r in records
    ]
    baseline_correct = [
        majority == r.ground_truth.upper()
        for r in records
    ]

    b = sum(
        model and not baseline
        for model, baseline in zip(model_correct, baseline_correct)
    )
    c = sum(
        baseline and not model
        for model, baseline in zip(model_correct, baseline_correct)
    )

    baseline_records = [
        LabeledRecord(
            case_id=r.case_id,
            ground_truth=r.ground_truth,
            prediction=majority,
            probability_positive=(
                1.0 if majority == positive_label.upper() else 0.0
            ),
            locked_threshold=r.locked_threshold,
        )
        for r in records
    ]

    return {
        "baseline_label": majority,
        "baseline_metrics": calculate_metrics(
            baseline_records,
            positive_label,
        ),
        "mcnemar_discordant_model_correct_baseline_wrong": b,
        "mcnemar_discordant_baseline_correct_model_wrong": c,
        "mcnemar_exact_p_value": exact_mcnemar_p_value(b, c),
        "warning": (
            "This majority baseline is weak and should not be presented as "
            "the primary benchmark in the final paper."
        ),
    }


def calibration_table(
    records: list[LabeledRecord],
    positive_label: str,
    bins: int,
) -> list[dict[str, Any]]:
    positive = positive_label.upper()
    probabilities = np.asarray(
        [r.probability_positive for r in records],
        dtype=float,
    )
    outcomes = np.asarray(
        [1 if r.ground_truth.upper() == positive else 0 for r in records],
        dtype=float,
    )

    edges = np.linspace(0, 1, bins + 1)
    rows: list[dict[str, Any]] = []

    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        count = int(np.sum(mask))
        rows.append(
            {
                "bin_index": index + 1,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "count": count,
                "mean_predicted_probability": (
                    float(np.mean(probabilities[mask])) if count else None
                ),
                "observed_positive_rate": (
                    float(np.mean(outcomes[mask])) if count else None
                ),
            }
        )

    return rows


def decision_curve(
    records: list[LabeledRecord],
    positive_label: str,
    thresholds: np.ndarray,
) -> list[dict[str, Any]]:
    positive = positive_label.upper()
    y_true = np.asarray(
        [1 if r.ground_truth.upper() == positive else 0 for r in records],
        dtype=int,
    )
    scores = np.asarray(
        [r.probability_positive for r in records],
        dtype=float,
    )
    n = len(records)
    prevalence = float(np.mean(y_true))

    rows = []
    for threshold in thresholds:
        pred = scores >= threshold
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))

        odds = threshold / (1 - threshold)
        model_nb = tp / n - fp / n * odds
        treat_all_nb = prevalence - (1 - prevalence) * odds

        rows.append(
            {
                "threshold": float(threshold),
                "model_net_benefit": float(model_nb),
                "treat_all_net_benefit": float(treat_all_nb),
                "treat_none_net_benefit": 0.0,
            }
        )
    return rows


def write_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_normalized(
    metrics: dict[str, Any],
    positive_label: str,
    negative_label: str,
    path: Path,
    dpi: int,
) -> None:
    matrix = np.asarray(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        dtype=float,
    )
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(normalized, vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Row-normalized proportion")
    ax.set_xticks([0, 1], labels=[negative_label, positive_label])
    ax.set_yticks([0, 1], labels=[negative_label, positive_label])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground-Truth Label")
    ax.set_title("Normalized Subject-Level Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{int(matrix[i, j])}\n({normalized[i, j]:.1%})",
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.5 else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_roc_with_ci(
    records: list[LabeledRecord],
    positive_label: str,
    auc_ci: dict[str, Any],
    path: Path,
    dpi: int,
) -> None:
    positive = positive_label.upper()
    y_true = np.asarray(
        [1 if r.ground_truth.upper() == positive else 0 for r in records],
        dtype=int,
    )
    scores = np.asarray(
        [r.probability_positive for r in records],
        dtype=float,
    )
    fpr, tpr, auc = roc_curve_manual(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    label = f"ROC-AUC = {auc:.3f}" if auc is not None else "ROC-AUC unavailable"
    if auc_ci.get("lower") is not None:
        label += (
            f" (95% CI {auc_ci['lower']:.3f}–"
            f"{auc_ci['upper']:.3f})"
        )
    ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Subject-Level ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(
    records: list[LabeledRecord],
    positive_label: str,
    path: Path,
    dpi: int,
) -> None:
    positive = positive_label.upper()
    y_true = np.asarray(
        [1 if r.ground_truth.upper() == positive else 0 for r in records],
        dtype=int,
    )
    scores = np.asarray(
        [r.probability_positive for r in records],
        dtype=float,
    )
    recall, precision, auc = pr_curve_manual(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, label=f"PR-AUC = {auc:.3f}")
    ax.axhline(np.mean(y_true), linestyle="--", linewidth=1, label="Prevalence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Subject-Level Precision–Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(
    calibration_rows: list[dict[str, Any]],
    path: Path,
    dpi: int,
) -> None:
    valid = [row for row in calibration_rows if row["count"] > 0]
    predicted = [row["mean_predicted_probability"] for row in valid]
    observed = [row["observed_positive_rate"] for row in valid]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(predicted, observed, marker="o", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_decision_curve(
    rows: list[dict[str, Any]],
    path: Path,
    dpi: int,
) -> None:
    thresholds = [row["threshold"] for row in rows]
    model = [row["model_net_benefit"] for row in rows]
    all_benefit = [row["treat_all_net_benefit"] for row in rows]
    none = [0.0 for _ in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, model, linewidth=2, label="Model")
    ax.plot(thresholds, all_benefit, linestyle="--", label="Treat all")
    ax.plot(thresholds, none, linestyle=":", label="Treat none")
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_probability_by_truth(
    records: list[LabeledRecord],
    positive_label: str,
    negative_label: str,
    path: Path,
    dpi: int,
) -> None:
    positive = positive_label.upper()
    positive_values = [
        r.probability_positive
        for r in records
        if r.ground_truth.upper() == positive
    ]
    negative_values = [
        r.probability_positive
        for r in records
        if r.ground_truth.upper() != positive
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        [negative_values, positive_values],
        labels=[negative_label, positive_label],
        showmeans=True,
    )
    threshold = float(np.median([r.locked_threshold for r in records]))
    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"Locked threshold = {threshold:.2f}",
    )
    ax.set_ylabel("Positive-Class Probability")
    ax.set_title("Probability Distribution by Ground Truth")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_distributions(
    bootstrap_results: dict[str, dict[str, Any]],
    path: Path,
    dpi: int,
) -> None:
    metrics = [
        name
        for name in ("accuracy", "sensitivity_recall", "specificity", "f1_score", "roc_auc")
        if bootstrap_results.get(name, {}).get("bootstrap_values")
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    data = [
        bootstrap_results[name]["bootstrap_values"]
        for name in metrics
    ]
    ax.boxplot(data, labels=metrics, showmeans=True)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Bootstrap Metric Value")
    ax.set_title("Stratified Bootstrap Metric Distributions")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    report: dict[str, Any],
    path: Path,
) -> None:
    metrics = report["metrics"]
    warnings = report["validity_warnings"]

    lines = [
        "=" * 84,
        "BrainFMOps-Analyze — Statistical Validation Summary",
        "=" * 84,
        f"Labeled subjects          : {metrics['n']}",
        f"Positive class            : {report['positive_label']}",
        f"Negative class            : {report['negative_label']}",
        "",
        "Core Metrics",
        "-" * 84,
    ]

    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity_recall",
        "specificity",
        "negative_predictive_value",
        "f1_score",
        "matthews_correlation_coefficient",
        "cohens_kappa",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
    ):
        value = metrics.get(key)
        lines.append(
            f"{key:36s}: "
            + (f"{value:.6f}" if value is not None else "N/A")
        )

    lines.extend(["", "Validity Warnings", "-" * 84])
    lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(
        [
            "",
            "Research-use notice",
            "-" * 84,
            "These statistics describe a research prototype.",
            "They are not clinical performance claims.",
            "=" * 84,
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run statistical validation on labeled STEP 25B records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--records-csv", required=True, type=Path)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--positive-label", default="AD")
    parser.add_argument("--negative-label", default="CN")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--calibration-bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    if args.bootstrap_iterations <= 0:
        raise SystemExit("--bootstrap-iterations must be greater than zero.")
    if not 0 < args.confidence_level < 1:
        raise SystemExit("--confidence-level must be between 0 and 1.")
    if args.calibration_bins <= 1:
        raise SystemExit("--calibration-bins must be greater than one.")

    records_csv = args.records_csv.expanduser().resolve()
    labels_csv = (
        args.labels_csv.expanduser().resolve()
        if args.labels_csv is not None else None
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else records_csv.parent / "25C_Statistical_Validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        records = load_records(records_csv, load_labels(labels_csv))
        metrics = calculate_metrics(records, args.positive_label)

        metric_names = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "sensitivity_recall",
            "specificity",
            "negative_predictive_value",
            "f1_score",
            "matthews_correlation_coefficient",
            "cohens_kappa",
            "roc_auc",
            "pr_auc",
            "brier_score",
        ]

        bootstrap_results: dict[str, dict[str, Any]] = {}
        for metric_name in metric_names:
            bootstrap_results[metric_name] = stratified_bootstrap_ci(
                records=records,
                positive_label=args.positive_label,
                metric_name=metric_name,
                iterations=args.bootstrap_iterations,
                confidence=args.confidence_level,
                seed=args.seed,
            )

        accuracy_wilson = wilson_interval(
            metrics["tp"] + metrics["tn"],
            metrics["n"],
            args.confidence_level,
        )
        sensitivity_wilson = wilson_interval(
            metrics["tp"],
            metrics["tp"] + metrics["fn"],
            args.confidence_level,
        )
        specificity_wilson = wilson_interval(
            metrics["tn"],
            metrics["tn"] + metrics["fp"],
            args.confidence_level,
        )

        calibration_rows = calibration_table(
            records,
            args.positive_label,
            args.calibration_bins,
        )
        decision_rows = decision_curve(
            records,
            args.positive_label,
            thresholds=np.linspace(0.05, 0.95, 19),
        )
        baseline_comparison = majority_baseline_comparison(
            records,
            args.positive_label,
            args.negative_label,
        )

        class_counts = Counter(r.ground_truth for r in records)
        validity_warnings = []

        if len(records) < 30:
            validity_warnings.append(
                "Sample size is below 30; all performance estimates are unstable."
            )
        if len(records) < 100:
            validity_warnings.append(
                "Sample size is below 100; results should not support a strong paper claim."
            )
        if min(class_counts.values()) < 10:
            validity_warnings.append(
                "At least one class has fewer than 10 cases; sensitivity/specificity are unreliable."
            )
        if len(records) == 5:
            validity_warnings.append(
                "Five cases validate software execution only, not model performance."
            )

        metric_rows = []
        for name in metric_names + ["log_loss"]:
            interval = bootstrap_results.get(name, {})
            metric_rows.append(
                {
                    "metric": name,
                    "estimate": metrics.get(name),
                    "bootstrap_ci_lower": interval.get("lower"),
                    "bootstrap_ci_upper": interval.get("upper"),
                    "confidence_level": args.confidence_level,
                }
            )

        interval_rows = [
            {
                "metric": "accuracy_wilson",
                "estimate": metrics["accuracy"],
                "lower": accuracy_wilson[0],
                "upper": accuracy_wilson[1],
            },
            {
                "metric": "sensitivity_wilson",
                "estimate": metrics["sensitivity_recall"],
                "lower": sensitivity_wilson[0],
                "upper": sensitivity_wilson[1],
            },
            {
                "metric": "specificity_wilson",
                "estimate": metrics["specificity"],
                "lower": specificity_wilson[0],
                "upper": specificity_wilson[1],
            },
        ]

        write_csv_rows(metric_rows, output_dir / "statistical_metrics.csv")
        write_csv_rows(interval_rows, output_dir / "bootstrap_intervals.csv")
        write_csv_rows(calibration_rows, output_dir / "calibration_table.csv")
        write_csv_rows(decision_rows, output_dir / "decision_curve_table.csv")

        plot_confusion_normalized(
            metrics,
            args.positive_label,
            args.negative_label,
            output_dir / "Fig_confusion_matrix_normalized.png",
            args.dpi,
        )
        plot_roc_with_ci(
            records,
            args.positive_label,
            bootstrap_results["roc_auc"],
            output_dir / "Fig_roc_curve_ci.png",
            args.dpi,
        )
        plot_pr_curve(
            records,
            args.positive_label,
            output_dir / "Fig_precision_recall_curve.png",
            args.dpi,
        )
        plot_calibration(
            calibration_rows,
            output_dir / "Fig_calibration_curve.png",
            args.dpi,
        )
        plot_decision_curve(
            decision_rows,
            output_dir / "Fig_decision_curve.png",
            args.dpi,
        )
        plot_probability_by_truth(
            records,
            args.positive_label,
            args.negative_label,
            output_dir / "Fig_probability_by_ground_truth.png",
            args.dpi,
        )
        plot_bootstrap_distributions(
            bootstrap_results,
            output_dir / "Fig_bootstrap_metric_distributions.png",
            args.dpi,
        )

        report = {
            "schema_version": "1.0",
            "script_version": SCRIPT_VERSION,
            "generated_at_utc": utc_now_iso(),
            "records_csv": str(records_csv),
            "labels_csv": str(labels_csv) if labels_csv else None,
            "positive_label": args.positive_label,
            "negative_label": args.negative_label,
            "class_counts": dict(class_counts),
            "metrics": metrics,
            "wilson_intervals": {
                "accuracy": {
                    "lower": accuracy_wilson[0],
                    "upper": accuracy_wilson[1],
                },
                "sensitivity": {
                    "lower": sensitivity_wilson[0],
                    "upper": sensitivity_wilson[1],
                },
                "specificity": {
                    "lower": specificity_wilson[0],
                    "upper": specificity_wilson[1],
                },
            },
            "stratified_bootstrap": {
                name: {
                    key: value
                    for key, value in result.items()
                    if key != "bootstrap_values"
                }
                for name, result in bootstrap_results.items()
            },
            "majority_baseline_comparison": baseline_comparison,
            "validity_warnings": validity_warnings,
            "research_use_only": True,
            "clinical_diagnosis": False,
            "delong_comparison": {
                "available": False,
                "reason": (
                    "DeLong comparison requires predictions from at least "
                    "two competing models on the same labeled subjects."
                ),
            },
        }

        (output_dir / "statistical_validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_summary(
            report,
            output_dir / "statistical_summary.txt",
        )

    except Exception as exc:
        logging.exception("Statistical validation failed: %s", exc)
        return 1

    print("\n" + "=" * 84)
    print(f"LABELED SUBJECTS : {metrics['n']}")
    print(f"ACCURACY         : {metrics['accuracy']}")
    print(f"SENSITIVITY      : {metrics['sensitivity_recall']}")
    print(f"SPECIFICITY      : {metrics['specificity']}")
    print(f"F1-SCORE         : {metrics['f1_score']}")
    print(f"ROC-AUC          : {metrics['roc_auc']}")
    print(f"MCC              : {metrics['matthews_correlation_coefficient']}")
    print(f"COHEN KAPPA      : {metrics['cohens_kappa']}")
    print(f"OUTPUT           : {output_dir}")
    print("=" * 84)

    if len(records) < 30:
        print(
            "WARNING: This sample is too small for a credible scientific "
            "performance claim."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
