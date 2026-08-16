#!/usr/bin/env python3
"""
25B_performance_evaluator.py

BrainFMOps-Analyze — STEP 25B
Performance and Operational Evaluation Engine.

Reads STEP 25A evaluation_summary.csv and creates:
- operational metrics for every batch
- classification metrics only when valid ground-truth labels exist
- bootstrap 95% confidence intervals
- confusion matrix, ROC, precision-recall, probability, runtime, readiness plots

Optional labels CSV:
case_id,label
OAS1_0001_MR1,AD
OAS1_0002_MR1,CN

Without ground truth, the script intentionally does NOT claim accuracy,
sensitivity, specificity, F1-score, ROC-AUC, or confusion-matrix performance.
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
class Record:
    case_id: str
    status: str
    readiness_status: str
    prediction: str
    probability_positive: Optional[float]
    locked_threshold: Optional[float]
    ground_truth: str
    total_runtime_seconds: Optional[float]
    inference_success: bool
    gradcam_success: bool
    report_success: bool

    @property
    def correct(self) -> Optional[bool]:
        if not self.prediction or not self.ground_truth:
            return None
        return self.prediction.upper() == self.ground_truth.upper()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_float(value: Any) -> Optional[float]:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_labels(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {path}")

    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if not reader.fieldnames or not {"case_id", "label"}.issubset(reader.fieldnames):
            raise ValueError("Labels CSV must contain case_id,label")
        for row in reader:
            case_id = row["case_id"].strip()
            label = row["label"].strip()
            if case_id and label:
                labels[case_id] = label
    return labels


def load_records(path: Path, external_labels: dict[str, str]) -> list[Record]:
    if not path.exists():
        raise FileNotFoundError(f"evaluation_summary.csv not found: {path}")

    records: list[Record] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {
            "case_id", "status", "readiness_status", "prediction",
            "probability_positive", "total_runtime_seconds",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            case_id = row["case_id"].strip()
            records.append(
                Record(
                    case_id=case_id,
                    status=row.get("status", "").strip(),
                    readiness_status=row.get("readiness_status", "").strip(),
                    prediction=row.get("prediction", "").strip(),
                    probability_positive=parse_float(row.get("probability_positive")),
                    locked_threshold=parse_float(row.get("locked_threshold")),
                    ground_truth=(
                        external_labels.get(case_id)
                        or row.get("ground_truth", "").strip()
                    ),
                    total_runtime_seconds=parse_float(row.get("total_runtime_seconds")),
                    inference_success=parse_bool(row.get("inference_success")),
                    gradcam_success=parse_bool(row.get("gradcam_success")),
                    report_success=parse_bool(row.get("report_success")),
                )
            )
    return records


def completed(records: list[Record]) -> list[Record]:
    valid_status = {"COMPLETED", "COMPLETED_RESUMED"}
    return [
        r for r in records
        if r.status in valid_status
        and r.inference_success
        and r.prediction
        and r.probability_positive is not None
    ]


def labeled(records: list[Record]) -> list[Record]:
    return [r for r in completed(records) if r.ground_truth]


def safe_div(a: float, b: float) -> Optional[float]:
    return a / b if b else None


def confusion(records: list[Record], positive_label: str) -> dict[str, int]:
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


def binary_metrics(records: list[Record], positive_label: str) -> dict[str, Any]:
    c = confusion(records, positive_label)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    total = tp + tn + fp + fn

    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
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

    return {
        "positive_label": positive_label,
        "n": total,
        **c,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "balanced_accuracy": balanced,
        "f1_score": f1,
    }


def roc_manual(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Optional[float]]:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return np.array([]), np.array([]), None

    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    fpr, tpr = [], []

    for threshold in thresholds:
        pred = scores >= threshold
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = positives - tp
        tn = negatives - fp
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    fpr_arr = np.asarray(fpr)
    tpr_arr = np.asarray(tpr)
    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    tpr_arr = tpr_arr[order]
    auc = float(np.trapezoid(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, auc


def pr_manual(
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

    recall_arr = np.asarray(recalls)
    precision_arr = np.asarray(precisions)
    order = np.argsort(recall_arr)
    recall_arr = recall_arr[order]
    precision_arr = precision_arr[order]
    auc = float(np.trapezoid(precision_arr, recall_arr))
    return recall_arr, precision_arr, auc


def bootstrap_ci(
    records: list[Record],
    statistic: Callable[[list[Record]], Optional[float]],
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    estimate = statistic(records)
    rng = random.Random(seed)
    values: list[float] = []
    n = len(records)

    for _ in range(iterations):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(float(value))

    if not values:
        return {
            "estimate": estimate,
            "lower": None,
            "upper": None,
            "valid_bootstrap_samples": 0,
            "confidence_level": confidence,
        }

    alpha = 1 - confidence
    return {
        "estimate": estimate,
        "lower": float(np.percentile(values, 100 * alpha / 2)),
        "upper": float(np.percentile(values, 100 * (1 - alpha / 2))),
        "valid_bootstrap_samples": len(values),
        "confidence_level": confidence,
    }


def operational_metrics(records: list[Record]) -> dict[str, Any]:
    completed_records = completed(records)
    runtimes = [
        float(r.total_runtime_seconds)
        for r in completed_records
        if r.total_runtime_seconds is not None
    ]

    runtime = {
        "count": len(runtimes),
        "mean": float(np.mean(runtimes)) if runtimes else None,
        "std": float(np.std(runtimes)) if runtimes else None,
        "min": float(np.min(runtimes)) if runtimes else None,
        "p25": float(np.percentile(runtimes, 25)) if runtimes else None,
        "median": float(np.median(runtimes)) if runtimes else None,
        "p75": float(np.percentile(runtimes, 75)) if runtimes else None,
        "max": float(np.max(runtimes)) if runtimes else None,
    }

    total = len(records)
    return {
        "total_records": total,
        "completed_records": len(completed_records),
        "completion_rate": len(completed_records) / total if total else 0.0,
        "inference_success_rate": (
            sum(r.inference_success for r in records) / total if total else 0.0
        ),
        "gradcam_success_rate": (
            sum(r.gradcam_success for r in records) / total if total else 0.0
        ),
        "report_success_rate": (
            sum(r.report_success for r in records) / total if total else 0.0
        ),
        "status_counts": dict(Counter(r.status for r in records)),
        "readiness_counts": dict(
            Counter(r.readiness_status or "UNKNOWN" for r in records)
        ),
        "prediction_counts": dict(
            Counter(r.prediction for r in completed_records)
        ),
        "runtime_seconds": runtime,
    }


def write_records(records: list[Record], path: Path) -> None:
    rows = []
    for record in records:
        row = asdict(record)
        row["correct"] = record.correct
        rows.append(row)

    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metric_table(
    metrics: dict[str, Any],
    cis: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    names = [
        "accuracy", "precision", "sensitivity_recall", "specificity",
        "negative_predictive_value", "balanced_accuracy", "f1_score",
        "roc_auc", "pr_auc",
    ]
    rows = [
        {
            "metric": name,
            "estimate": metrics.get(name),
            "ci_lower": cis.get(name, {}).get("lower"),
            "ci_upper": cis.get(name, {}).get("upper"),
            "confidence_level": cis.get(name, {}).get("confidence_level"),
            "valid_bootstrap_samples": cis.get(name, {}).get(
                "valid_bootstrap_samples"
            ),
        }
        for name in names
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_operational_table(metrics: dict[str, Any], path: Path) -> None:
    rows = [
        {"metric": "total_records", "value": metrics["total_records"]},
        {"metric": "completed_records", "value": metrics["completed_records"]},
        {"metric": "completion_rate", "value": metrics["completion_rate"]},
        {"metric": "inference_success_rate", "value": metrics["inference_success_rate"]},
        {"metric": "gradcam_success_rate", "value": metrics["gradcam_success_rate"]},
        {"metric": "report_success_rate", "value": metrics["report_success_rate"]},
    ]
    for key, value in metrics["runtime_seconds"].items():
        rows.append({"metric": f"runtime_seconds_{key}", "value": value})

    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion(
    metrics: dict[str, Any],
    positive_label: str,
    negative_label: str,
    path: Path,
    dpi: int,
) -> None:
    matrix = np.array(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    )
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=[negative_label, positive_label])
    ax.set_yticks([0, 1], labels=[negative_label, positive_label])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground-Truth Label")
    ax.set_title("Subject-Level Confusion Matrix")

    midpoint = matrix.max() / 2 if matrix.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > midpoint else "black",
                fontsize=13,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {auc:.3f}")
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


def plot_pr(
    recall: np.ndarray,
    precision: np.ndarray,
    auc: float,
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, label=f"PR-AUC = {auc:.3f}")
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


def plot_probability(records: list[Record], path: Path, dpi: int) -> None:
    values = [float(r.probability_positive) for r in completed(records)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=min(20, max(5, len(values))))
    thresholds = [
        float(r.locked_threshold)
        for r in completed(records)
        if r.locked_threshold is not None
    ]
    if thresholds:
        threshold = float(np.median(thresholds))
        ax.axvline(
            threshold, linestyle="--", linewidth=1.5,
            label=f"Locked threshold = {threshold:.2f}"
        )
        ax.legend()
    ax.set_xlabel("Positive-Class Probability")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Subject-Level Probability Distribution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(records: list[Record], path: Path, dpi: int) -> None:
    values = [
        float(r.total_runtime_seconds)
        for r in completed(records)
        if r.total_runtime_seconds is not None
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=min(20, max(5, len(values))))
    if values:
        mean_value = float(np.mean(values))
        ax.axvline(
            mean_value, linestyle="--", linewidth=1.5,
            label=f"Mean = {mean_value:.2f} s"
        )
        ax.legend()
    ax.set_xlabel("Total Runtime per Subject (s)")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Pipeline Runtime Distribution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_readiness(records: list[Record], path: Path, dpi: int) -> None:
    counts = Counter(r.readiness_status or "UNKNOWN" for r in records)
    labels = list(counts)
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = np.arange(len(labels))
    ax.bar(positions, values)
    ax.set_xticks(positions, labels=labels, rotation=20, ha="right")
    ax.set_ylabel("Number of Subjects")
    ax.set_title("Case Readiness Status Summary")
    for position, value in zip(positions, values):
        ax.text(position, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary(report: dict[str, Any], path: Path) -> None:
    operational = report["operational_metrics"]
    classification = report["classification_evaluation"]
    runtime = operational["runtime_seconds"]

    lines = [
        "=" * 82,
        "BrainFMOps-Analyze — Performance Evaluation Summary",
        "=" * 82,
        f"Total records            : {operational['total_records']}",
        f"Completed records        : {operational['completed_records']}",
        f"Completion rate          : {operational['completion_rate']:.2%}",
        f"Inference success rate   : {operational['inference_success_rate']:.2%}",
        f"Grad-CAM success rate    : {operational['gradcam_success_rate']:.2%}",
        f"Report success rate      : {operational['report_success_rate']:.2%}",
        "",
        "Runtime Statistics",
        "-" * 82,
    ]

    for name in ("mean", "std", "min", "p25", "median", "p75", "max"):
        value = runtime.get(name)
        lines.append(
            f"{name:24s}: "
            + (f"{value:.4f} s" if value is not None else "N/A")
        )

    lines.extend(["", "Classification Evaluation", "-" * 82])

    if not classification["available"]:
        lines.extend(
            [
                "NOT AVAILABLE",
                classification["reason"],
                "",
                "Do not report accuracy, sensitivity, specificity, F1-score,",
                "ROC-AUC, PR-AUC, or a confusion matrix until valid",
                "subject-level ground-truth labels are integrated.",
            ]
        )
    else:
        metrics = classification["metrics"]
        lines.append(
            f"Labeled completed cases  : {classification['labeled_case_count']}"
        )
        for name in (
            "accuracy", "precision", "sensitivity_recall", "specificity",
            "balanced_accuracy", "f1_score", "roc_auc", "pr_auc",
        ):
            value = metrics.get(name)
            lines.append(
                f"{name:24s}: "
                + (f"{value:.6f}" if value is not None else "N/A")
            )

    lines.extend(
        [
            "",
            "Research-use notice",
            "-" * 82,
            "These results describe a research prototype.",
            "They are not clinical performance claims.",
            "=" * 82,
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate BrainFMOps-Analyze STEP 25A batch results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--evaluation-csv", required=True, type=Path)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--positive-label", default="AD")
    parser.add_argument("--negative-label", default="CN")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
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
    if args.dpi <= 0:
        raise SystemExit("--dpi must be greater than zero.")

    evaluation_csv = args.evaluation_csv.expanduser().resolve()
    labels_csv = (
        args.labels_csv.expanduser().resolve()
        if args.labels_csv is not None else None
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else evaluation_csv.parent / "25B_Performance_Evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        records = load_records(evaluation_csv, load_labels(labels_csv))
        operational = operational_metrics(records)
        labeled_set = labeled(records)

        classification: dict[str, Any]

        if labeled_set:
            metrics = binary_metrics(labeled_set, args.positive_label)
            positive = args.positive_label.upper()
            y_true = np.asarray(
                [1 if r.ground_truth.upper() == positive else 0 for r in labeled_set],
                dtype=int,
            )
            scores = np.asarray(
                [float(r.probability_positive) for r in labeled_set],
                dtype=float,
            )
            fpr, tpr, roc_auc = roc_manual(y_true, scores)
            recall, precision_curve, pr_auc = pr_manual(y_true, scores)
            metrics["roc_auc"] = roc_auc
            metrics["pr_auc"] = pr_auc

            cis: dict[str, dict[str, Any]] = {}
            for metric_name in (
                "accuracy", "precision", "sensitivity_recall", "specificity",
                "negative_predictive_value", "balanced_accuracy", "f1_score",
            ):
                cis[metric_name] = bootstrap_ci(
                    labeled_set,
                    lambda sample, name=metric_name: binary_metrics(
                        sample, args.positive_label
                    ).get(name),
                    args.bootstrap_iterations,
                    args.confidence_level,
                    args.seed,
                )

            cis["roc_auc"] = bootstrap_ci(
                labeled_set,
                lambda sample: roc_manual(
                    np.asarray(
                        [1 if r.ground_truth.upper() == positive else 0 for r in sample],
                        dtype=int,
                    ),
                    np.asarray(
                        [float(r.probability_positive) for r in sample],
                        dtype=float,
                    ),
                )[2],
                args.bootstrap_iterations,
                args.confidence_level,
                args.seed,
            )
            cis["pr_auc"] = bootstrap_ci(
                labeled_set,
                lambda sample: pr_manual(
                    np.asarray(
                        [1 if r.ground_truth.upper() == positive else 0 for r in sample],
                        dtype=int,
                    ),
                    np.asarray(
                        [float(r.probability_positive) for r in sample],
                        dtype=float,
                    ),
                )[2],
                args.bootstrap_iterations,
                args.confidence_level,
                args.seed,
            )

            classification = {
                "available": True,
                "labeled_case_count": len(labeled_set),
                "metrics": metrics,
                "bootstrap_confidence_intervals": cis,
                "bootstrap_iterations": args.bootstrap_iterations,
                "confidence_level": args.confidence_level,
            }

            write_metric_table(
                metrics, cis, output_dir / "performance_metrics.csv"
            )
            plot_confusion(
                metrics, args.positive_label, args.negative_label,
                output_dir / "Fig_confusion_matrix.png", args.dpi
            )
            if roc_auc is not None:
                plot_roc(
                    fpr, tpr, roc_auc,
                    output_dir / "Fig_roc_curve.png", args.dpi
                )
            if pr_auc is not None:
                plot_pr(
                    recall, precision_curve, pr_auc,
                    output_dir / "Fig_precision_recall_curve.png", args.dpi
                )
        else:
            classification = {
                "available": False,
                "labeled_case_count": 0,
                "reason": (
                    "No completed subject has a valid ground-truth label. "
                    "Classification performance cannot be computed honestly."
                ),
                "metrics": {},
                "bootstrap_confidence_intervals": {},
            }

        plot_probability(
            records, output_dir / "Fig_probability_distribution.png", args.dpi
        )
        plot_runtime(
            records, output_dir / "Fig_runtime_distribution.png", args.dpi
        )
        plot_readiness(
            records, output_dir / "Fig_readiness_summary.png", args.dpi
        )

        write_records(
            records, output_dir / "evaluation_records_with_labels.csv"
        )
        write_operational_table(
            operational, output_dir / "operational_metrics.csv"
        )

        report = {
            "schema_version": "1.0",
            "script_version": SCRIPT_VERSION,
            "generated_at_utc": now_utc(),
            "evaluation_csv": str(evaluation_csv),
            "labels_csv": str(labels_csv) if labels_csv else None,
            "positive_label": args.positive_label,
            "negative_label": args.negative_label,
            "operational_metrics": operational,
            "classification_evaluation": classification,
            "research_use_only": True,
            "clinical_diagnosis": False,
            "limitations": [
                "Classification metrics require correct subject-level labels.",
                "Five pilot cases are insufficient for a credible performance claim.",
                "External clinical validity has not been established.",
            ],
        }

        (output_dir / "performance_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_summary(report, output_dir / "performance_summary.txt")

    except Exception as exc:
        logging.exception("Performance evaluation failed: %s", exc)
        return 1

    print("\n" + "=" * 82)
    print(f"TOTAL RECORDS      : {operational['total_records']}")
    print(f"COMPLETED RECORDS  : {operational['completed_records']}")
    print(f"COMPLETION RATE    : {operational['completion_rate']:.2%}")
    print(f"LABELED CASES      : {classification['labeled_case_count']}")
    print(
        "CLASSIFICATION     : "
        + ("AVAILABLE" if classification["available"] else "NOT AVAILABLE")
    )
    print(f"OUTPUT             : {output_dir}")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    sys.exit(main())
