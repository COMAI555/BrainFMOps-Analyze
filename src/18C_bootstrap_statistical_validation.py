#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18C_bootstrap_statistical_validation.py

BrainFMOps Phase 2
Publication-grade Subject-level Bootstrap Statistical Validation

Input
-----
subject_level_predictions.csv produced by STEP 18A.

Primary analysis
----------------
- Resampling unit: subject (one row per subject)
- Locked decision threshold: supplied from STEP 18B validation analysis
- Default threshold: 0.32
- Bootstrap repetitions: 2,000
- Confidence interval: percentile 95% CI
- Random seed: 42

Metrics
-------
Accuracy
Balanced accuracy
Sensitivity
Specificity
Precision / PPV
Negative predictive value / NPV
F1-score
Matthews correlation coefficient / MCC
ROC-AUC
PR-AUC

Scientific rule
---------------
The threshold must have been selected using validation data only.
Do not optimize the threshold using the independent test predictions.

Example
-------
python 18C_bootstrap_statistical_validation.py ^
  --prediction-file "outputs/independent_test_evaluation_v1/subject_level_predictions.csv" ^
  --output-dir "outputs/bootstrap_statistical_validation_v1" ^
  --threshold 0.32 ^
  --bootstrap 2000 ^
  --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)


SCRIPT_VERSION = "1.0.0"

REQUIRED_COLUMNS = {
    "subject_id",
    "true_label",
    "mean_probability_dementia",
}

METRIC_ORDER = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision_ppv",
    "negative_predictive_value",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
]

DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision_ppv": "Precision (PPV)",
    "negative_predictive_value": "Negative predictive value",
    "f1": "F1-score",
    "mcc": "Matthews correlation coefficient",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


@dataclass
class BootstrapConfig:
    threshold: float = 0.32
    bootstrap_repetitions: int = 2000
    confidence_level: float = 0.95
    random_seed: int = 42
    maximum_attempt_multiplier: int = 20


@dataclass
class MetricSummary:
    metric: str
    display_name: str
    estimate: float
    bootstrap_mean: float
    bootstrap_median: float
    bootstrap_standard_deviation: float
    bootstrap_standard_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    valid_bootstrap_repetitions: int


def configure_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                output_dir / "bootstrap_console.log",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Prediction file is missing required columns: "
            + ", ".join(missing)
        )

    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["true_label"] = pd.to_numeric(df["true_label"], errors="coerce")
    df["mean_probability_dementia"] = pd.to_numeric(
        df["mean_probability_dementia"],
        errors="coerce",
    )

    invalid = (
        ~df["true_label"].isin([0, 1])
        | df["mean_probability_dementia"].isna()
        | ~df["mean_probability_dementia"].between(0.0, 1.0)
        | df["subject_id"].eq("")
    )

    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum())} row(s) contain invalid labels, probabilities, "
            "or subject identifiers."
        )

    if df["subject_id"].duplicated().any():
        duplicated = int(df["subject_id"].duplicated().sum())
        raise ValueError(
            f"Prediction file contains {duplicated} duplicate subject row(s). "
            "Bootstrap input must contain exactly one row per subject."
        )

    df["true_label"] = df["true_label"].astype(int)

    if df["true_label"].nunique() < 2:
        raise ValueError("Both binary classes are required.")

    return df.reset_index(drop=True)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_predicted = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_predicted,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_predicted)
        ),
        "sensitivity": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "precision_ppv": float(
            precision_score(y_true, y_predicted, zero_division=0)
        ),
        "negative_predictive_value": safe_divide(tn, tn + fn),
        "f1": float(f1_score(y_true, y_predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_predicted)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(
            average_precision_score(y_true, y_probability)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def run_bootstrap(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    config: BootstrapConfig,
) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(config.random_seed)
    subject_count = len(y_true)
    rows: list[dict[str, float]] = []

    attempts = 0
    maximum_attempts = (
        config.bootstrap_repetitions
        * config.maximum_attempt_multiplier
    )

    while (
        len(rows) < config.bootstrap_repetitions
        and attempts < maximum_attempts
    ):
        attempts += 1
        indices = rng.integers(
            low=0,
            high=subject_count,
            size=subject_count,
        )
        sampled_true = y_true[indices]
        sampled_probability = y_probability[indices]

        # ROC-AUC and PR-AUC require both classes.
        if np.unique(sampled_true).size < 2:
            continue

        values = compute_metrics(
            sampled_true,
            sampled_probability,
            config.threshold,
        )

        if any(
            not np.isfinite(values[metric])
            for metric in METRIC_ORDER
        ):
            continue

        row = {
            "bootstrap_iteration": len(rows) + 1,
            **{metric: values[metric] for metric in METRIC_ORDER},
        }
        rows.append(row)

        if len(rows) % 250 == 0:
            logging.info(
                "Completed %d/%d valid bootstrap repetitions",
                len(rows),
                config.bootstrap_repetitions,
            )

    if len(rows) < config.bootstrap_repetitions:
        raise RuntimeError(
            f"Only {len(rows)} valid bootstrap repetitions were obtained "
            f"after {attempts} attempts."
        )

    discarded = attempts - len(rows)
    return pd.DataFrame(rows), discarded


def summarize_bootstrap(
    point_estimates: dict[str, float],
    bootstrap_df: pd.DataFrame,
    config: BootstrapConfig,
) -> pd.DataFrame:
    alpha = 1.0 - config.confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)

    summaries: list[MetricSummary] = []

    for metric in METRIC_ORDER:
        values = bootstrap_df[metric].to_numpy(dtype=float)
        summaries.append(
            MetricSummary(
                metric=metric,
                display_name=DISPLAY_NAMES[metric],
                estimate=float(point_estimates[metric]),
                bootstrap_mean=float(np.mean(values)),
                bootstrap_median=float(np.median(values)),
                bootstrap_standard_deviation=float(
                    np.std(values, ddof=1)
                ),
                bootstrap_standard_error=float(
                    np.std(values, ddof=1)
                ),
                ci_lower=float(
                    np.percentile(values, lower_percentile)
                ),
                ci_upper=float(
                    np.percentile(values, upper_percentile)
                ),
                confidence_level=config.confidence_level,
                valid_bootstrap_repetitions=len(values),
            )
        )

    return pd.DataFrame([asdict(item) for item in summaries])


def create_paper_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    paper = summary_df[
        [
            "display_name",
            "estimate",
            "ci_lower",
            "ci_upper",
        ]
    ].copy()

    paper["Estimate"] = paper["estimate"].map(
        lambda value: f"{value:.3f}"
    )
    paper["95% CI"] = paper.apply(
        lambda row: f"{row['ci_lower']:.3f}–{row['ci_upper']:.3f}",
        axis=1,
    )

    return paper[
        ["display_name", "Estimate", "95% CI"]
    ].rename(columns={"display_name": "Metric"})


def save_forest_plot(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = summary_df.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(plot_df))
    estimates = plot_df["estimate"].to_numpy()
    lower = plot_df["ci_lower"].to_numpy()
    upper = plot_df["ci_upper"].to_numpy()

    errors = np.vstack([estimates - lower, upper - estimates])

    plt.figure(figsize=(9, 7))
    plt.errorbar(
        estimates,
        positions,
        xerr=errors,
        fmt="o",
        capsize=4,
    )
    plt.yticks(positions, plot_df["display_name"])
    plt.xlim(0.0, 1.02)
    plt.xlabel("Metric estimate with 95% bootstrap CI")
    plt.title("Subject-level Bootstrap Confidence Intervals")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_selected_distribution_plots(
    bootstrap_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    selected = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    summary_lookup = summary_df.set_index("metric")

    for metric in selected:
        values = bootstrap_df[metric].to_numpy()
        summary = summary_lookup.loc[metric]

        plt.figure(figsize=(7, 5))
        plt.hist(values, bins=30)
        plt.axvline(
            summary["estimate"],
            linestyle="-",
            label=f"Estimate = {summary['estimate']:.3f}",
        )
        plt.axvline(
            summary["ci_lower"],
            linestyle="--",
            label=f"Lower CI = {summary['ci_lower']:.3f}",
        )
        plt.axvline(
            summary["ci_upper"],
            linestyle="--",
            label=f"Upper CI = {summary['ci_upper']:.3f}",
        )
        plt.xlabel(DISPLAY_NAMES[metric])
        plt.ylabel("Bootstrap repetitions")
        plt.title(
            f"Bootstrap Distribution: {DISPLAY_NAMES[metric]}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / f"bootstrap_distribution_{metric}.png",
            dpi=300,
        )
        plt.close()


def save_boxplot(
    bootstrap_df: pd.DataFrame,
    output_path: Path,
) -> None:
    selected = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    values = [
        bootstrap_df[metric].to_numpy()
        for metric in selected
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        values,
        tick_labels=[DISPLAY_NAMES[m] for m in selected],
        showfliers=False,
    )
    plt.ylabel("Bootstrap metric value")
    plt.title("Bootstrap Metric Distributions")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_text_summary(
    path: Path,
    prediction_file: Path,
    subject_count: int,
    positive_count: int,
    negative_count: int,
    point_estimates: dict[str, float],
    summary_df: pd.DataFrame,
    config: BootstrapConfig,
    discarded_repetitions: int,
) -> None:
    lines = [
        "=" * 94,
        "BrainFMOps Publication-grade Bootstrap Statistical Validation",
        "=" * 94,
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Prediction file: {prediction_file}",
        f"Subjects: {subject_count}",
        f"Dementia subjects: {positive_count}",
        f"Normal subjects: {negative_count}",
        f"Locked threshold: {config.threshold:.4f}",
        f"Bootstrap repetitions: {config.bootstrap_repetitions}",
        f"Confidence level: {config.confidence_level:.1%}",
        f"Random seed: {config.random_seed}",
        f"Discarded invalid bootstrap samples: {discarded_repetitions}",
        "",
        "POINT CONFUSION MATRIX",
        "-" * 94,
        f"TN: {point_estimates['tn']}",
        f"FP: {point_estimates['fp']}",
        f"FN: {point_estimates['fn']}",
        f"TP: {point_estimates['tp']}",
        "",
        "SUBJECT-LEVEL METRICS WITH PERCENTILE BOOTSTRAP CI",
        "-" * 94,
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['display_name']}: "
            f"{row['estimate']:.4f} "
            f"({config.confidence_level:.0%} CI "
            f"{row['ci_lower']:.4f}–{row['ci_upper']:.4f})"
        )

    lines.extend(
        [
            "",
            "METHOD NOTE",
            "-" * 94,
            (
                "Nonparametric bootstrap resampling was performed at the subject "
                "level. Each bootstrap sample contained the same number of subjects "
                "as the original independent test cohort and was sampled with replacement."
            ),
            (
                "The decision threshold was fixed before this analysis using validation "
                "data only. The independent test set was not used to optimize the threshold."
            ),
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate subject-level bootstrap confidence intervals and "
            "publication-ready statistical outputs."
        )
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.32,
        help="Threshold locked using validation data only.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    prediction_file = args.prediction_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir, args.verbose)

    if not prediction_file.is_file():
        logging.error("Prediction file not found: %s", prediction_file)
        return 2

    if not 0.0 < args.threshold < 1.0:
        logging.error("Threshold must be between 0 and 1.")
        return 2

    if args.bootstrap < 100:
        logging.error("At least 100 bootstrap repetitions are required.")
        return 2

    if not 0.0 < args.confidence_level < 1.0:
        logging.error("Confidence level must be between 0 and 1.")
        return 2

    config = BootstrapConfig(
        threshold=args.threshold,
        bootstrap_repetitions=args.bootstrap,
        confidence_level=args.confidence_level,
        random_seed=args.seed,
    )

    logging.info("Loading subject-level predictions")

    try:
        predictions = load_predictions(prediction_file)
    except Exception as exc:
        logging.exception("Prediction validation failed: %s", exc)
        return 2

    y_true = predictions["true_label"].to_numpy(dtype=int)
    y_probability = predictions[
        "mean_probability_dementia"
    ].to_numpy(dtype=float)

    point_estimates = compute_metrics(
        y_true,
        y_probability,
        config.threshold,
    )

    logging.info(
        "Subjects=%d | Normal=%d | Dementia=%d | Threshold=%.4f",
        len(predictions),
        int((y_true == 0).sum()),
        int((y_true == 1).sum()),
        config.threshold,
    )
    logging.info(
        "Running %d subject-level bootstrap repetitions",
        config.bootstrap_repetitions,
    )

    try:
        bootstrap_df, discarded = run_bootstrap(
            y_true,
            y_probability,
            config,
        )
    except Exception as exc:
        logging.exception("Bootstrap analysis failed: %s", exc)
        return 2

    summary_df = summarize_bootstrap(
        point_estimates,
        bootstrap_df,
        config,
    )
    paper_table = create_paper_table(summary_df)

    bootstrap_path = output_dir / "bootstrap_metric_distribution.csv"
    summary_path = output_dir / "bootstrap_summary.csv"
    ci_path = output_dir / "confidence_interval_table.csv"
    paper_path = output_dir / "paper_table_statistics.csv"
    json_path = output_dir / "statistics_report.json"
    text_path = output_dir / "statistics_summary.txt"

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary_df[
        [
            "metric",
            "display_name",
            "estimate",
            "ci_lower",
            "ci_upper",
            "confidence_level",
        ]
    ].to_csv(
        ci_path,
        index=False,
        encoding="utf-8-sig",
    )
    paper_table.to_csv(
        paper_path,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_level": "subject",
        "scientific_rule": (
            "Decision threshold was locked on validation data only."
        ),
        "prediction_file": str(prediction_file),
        "prediction_file_sha256": sha256_file(prediction_file),
        "configuration": asdict(config),
        "cohort": {
            "subjects": len(predictions),
            "normal_subjects": int((y_true == 0).sum()),
            "dementia_subjects": int((y_true == 1).sum()),
        },
        "point_confusion_matrix": {
            "tn": point_estimates["tn"],
            "fp": point_estimates["fp"],
            "fn": point_estimates["fn"],
            "tp": point_estimates["tp"],
        },
        "point_estimates": {
            metric: point_estimates[metric]
            for metric in METRIC_ORDER
        },
        "bootstrap": {
            "requested_repetitions": config.bootstrap_repetitions,
            "valid_repetitions": len(bootstrap_df),
            "discarded_invalid_repetitions": discarded,
            "method": "nonparametric subject-level percentile bootstrap",
        },
        "metric_summaries": summary_df.to_dict(orient="records"),
        "status": "COMPLETED",
    }

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_text_summary(
        text_path,
        prediction_file,
        len(predictions),
        int((y_true == 1).sum()),
        int((y_true == 0).sum()),
        point_estimates,
        summary_df,
        config,
        discarded,
    )

    save_forest_plot(
        summary_df,
        output_dir / "forest_plot.png",
    )
    save_boxplot(
        bootstrap_df,
        output_dir / "boxplot_metrics.png",
    )
    save_selected_distribution_plots(
        bootstrap_df,
        summary_df,
        output_dir,
    )

    print()
    print("=" * 94)
    print("BRAINF MOPS BOOTSTRAP STATISTICAL VALIDATION COMPLETE")
    print("=" * 94)
    print(f"Subjects                 : {len(predictions)}")
    print(f"Normal subjects          : {int((y_true == 0).sum())}")
    print(f"Dementia subjects        : {int((y_true == 1).sum())}")
    print(f"Locked threshold         : {config.threshold:.4f}")
    print(f"Bootstrap repetitions    : {len(bootstrap_df)}")
    print(f"Discarded repetitions    : {discarded}")
    print()

    summary_lookup = summary_df.set_index("metric")
    for metric in [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:
        row = summary_lookup.loc[metric]
        print(
            f"{DISPLAY_NAMES[metric]:26s}: "
            f"{row['estimate']:.4f} "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        )

    print()
    print(f"Output directory         : {output_dir}")
    print(f"Paper table              : {paper_path}")
    print(f"Forest plot              : {output_dir / 'forest_plot.png'}")
    print(f"Summary                  : {text_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
