#!/usr/bin/env python3
"""
28A_dataset_manifest_generator.py

BrainFMOps-Analyze — STEP 28A
Dataset Manifest Generator

Purpose
-------
Create a reproducible dataset manifest from the full BrainFMOps evaluation
outputs and optional OASIS ground-truth labels.

Inputs
------
- STEP 25A evaluation_summary.csv
- Optional oasis_ground_truth.csv
- Optional batch_configuration.json
- Optional provenance audit report

Outputs
-------
28A_Dataset_Manifest/
├── dataset_manifest.json
├── dataset_summary.csv
├── dataset_statistics.json
├── dataset_subjects.csv
├── dataset_quality_summary.txt
├── Fig_dataset_class_distribution.png
├── Fig_dataset_readiness_distribution.png
└── Fig_dataset_runtime_distribution.png

Important
---------
This manifest describes the evaluated dataset and pipeline outputs.
It does not prove leakage-free evaluation unless provenance manifests are
complete and independently verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import statistics
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_VERSION = "1.0.0"


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: Any) -> Optional[float]:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def load_json_optional(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        logging.warning("Optional JSON not found: %s", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")

    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if not reader.fieldnames or not {"case_id", "label"}.issubset(reader.fieldnames):
            raise ValueError("Ground-truth CSV must contain case_id,label")

        for row in reader:
            case_id = row["case_id"].strip()
            label = row["label"].strip()
            if case_id:
                labels[case_id] = label
    return labels


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_uuid(
    evaluation_csv_hash: str,
    labels_csv_hash: Optional[str],
    dataset_name: str,
) -> str:
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    payload = f"{dataset_name}|{evaluation_csv_hash}|{labels_csv_hash or 'NO_LABELS'}"
    return str(uuid.uuid5(namespace, payload))


def load_evaluation_records(
    evaluation_csv: Path,
    labels: dict[str, str],
) -> list[dict[str, Any]]:
    if not evaluation_csv.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {evaluation_csv}")

    records: list[dict[str, Any]] = []
    with evaluation_csv.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if not reader.fieldnames:
            raise ValueError("Evaluation CSV has no header.")

        required = {"case_id", "status", "readiness_status"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Evaluation CSV missing columns: {sorted(missing)}")

        for row in reader:
            case_id = row.get("case_id", "").strip()
            ground_truth = labels.get(case_id) or row.get("ground_truth", "").strip()

            records.append(
                {
                    "case_id": case_id,
                    "status": row.get("status", "").strip(),
                    "readiness_status": row.get("readiness_status", "").strip(),
                    "prediction": row.get("prediction", "").strip(),
                    "probability_positive": parse_float(
                        row.get("probability_positive")
                    ),
                    "locked_threshold": parse_float(
                        row.get("locked_threshold")
                    ),
                    "ground_truth": ground_truth,
                    "correct": (
                        row.get("prediction", "").strip().upper()
                        == ground_truth.strip().upper()
                        if row.get("prediction", "").strip() and ground_truth
                        else None
                    ),
                    "readiness_success": parse_bool(
                        row.get("readiness_success")
                    ),
                    "volume_selection_success": parse_bool(
                        row.get("volume_selection_success")
                    ),
                    "inference_success": parse_bool(
                        row.get("inference_success")
                    ),
                    "gradcam_success": parse_bool(
                        row.get("gradcam_success")
                    ),
                    "report_success": parse_bool(
                        row.get("report_success")
                    ),
                    "selected_volume_index": parse_float(
                        row.get("selected_volume_index")
                    ),
                    "selected_volume_score": parse_float(
                        row.get("selected_volume_score")
                    ),
                    "total_runtime_seconds": parse_float(
                        row.get("total_runtime_seconds")
                    ),
                    "failure_step": row.get("failure_step", "").strip(),
                    "error_message": row.get("error_message", "").strip(),
                    "output_dir": row.get("output_dir", "").strip(),
                    "html_report": row.get("html_report", "").strip(),
                }
            )

    return records


def numeric_summary(values: list[float]) -> dict[str, Optional[float] | int]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }

    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def build_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    status_counts = Counter(record["status"] or "UNKNOWN" for record in records)
    readiness_counts = Counter(
        record["readiness_status"] or "UNKNOWN" for record in records
    )
    prediction_counts = Counter(
        record["prediction"] or "UNAVAILABLE" for record in records
    )
    ground_truth_counts = Counter(
        record["ground_truth"] or "UNLABELED" for record in records
    )

    probabilities = [
        float(record["probability_positive"])
        for record in records
        if record["probability_positive"] is not None
    ]
    runtimes = [
        float(record["total_runtime_seconds"])
        for record in records
        if record["total_runtime_seconds"] is not None
    ]
    volume_scores = [
        float(record["selected_volume_score"])
        for record in records
        if record["selected_volume_score"] is not None
    ]

    labeled = [
        record for record in records if record["ground_truth"]
    ]
    correctly_classified = [
        record for record in labeled if record["correct"] is True
    ]

    return {
        "total_subjects": total,
        "completed_subjects": sum(
            record["status"] in {"COMPLETED", "COMPLETED_RESUMED"}
            for record in records
        ),
        "failed_subjects": sum(record["status"] == "FAILED" for record in records),
        "not_ready_subjects": sum(
            record["status"] == "SKIPPED_NOT_READY" for record in records
        ),
        "labeled_subjects": len(labeled),
        "unlabeled_subjects": total - len(labeled),
        "correctly_classified_labeled_subjects": len(correctly_classified),
        "status_counts": dict(status_counts),
        "readiness_counts": dict(readiness_counts),
        "prediction_counts": dict(prediction_counts),
        "ground_truth_counts": dict(ground_truth_counts),
        "pipeline_success_rates": {
            "readiness_success_rate": (
                sum(record["readiness_success"] for record in records) / total
                if total else 0.0
            ),
            "volume_selection_success_rate": (
                sum(record["volume_selection_success"] for record in records) / total
                if total else 0.0
            ),
            "inference_success_rate": (
                sum(record["inference_success"] for record in records) / total
                if total else 0.0
            ),
            "gradcam_success_rate": (
                sum(record["gradcam_success"] for record in records) / total
                if total else 0.0
            ),
            "report_success_rate": (
                sum(record["report_success"] for record in records) / total
                if total else 0.0
            ),
        },
        "probability_positive_summary": numeric_summary(probabilities),
        "runtime_seconds_summary": numeric_summary(runtimes),
        "volume_selection_score_summary": numeric_summary(volume_scores),
    }


def quality_score(statistics_data: dict[str, Any]) -> tuple[float, list[str]]:
    rates = statistics_data["pipeline_success_rates"]
    score = 100.0 * float(np.mean(list(rates.values())))
    warnings: list[str] = []

    if statistics_data["failed_subjects"] > 0:
        warnings.append("Some subjects failed during full pipeline execution.")
    if statistics_data["not_ready_subjects"] > 0:
        warnings.append("Some subjects were rejected by the readiness gate.")
    if statistics_data["unlabeled_subjects"] > 0:
        warnings.append(
            "Some evaluated subjects do not have ground-truth labels."
        )
    if statistics_data["total_subjects"] < 100:
        warnings.append(
            "Dataset contains fewer than 100 evaluated subjects."
        )

    return round(score, 2), warnings


def write_subjects_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def flatten_dict(
    data: dict[str, Any],
    prefix: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(flatten_dict(value, full_key))
        else:
            rows.append({"metric": full_key, "value": value})
    return rows


def write_summary_csv(data: dict[str, Any], path: Path) -> None:
    rows = flatten_dict(data)
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def plot_class_distribution(
    statistics_data: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    counts = statistics_data["ground_truth_counts"]
    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = np.arange(len(labels))
    ax.bar(positions, values)
    ax.set_xticks(positions, labels=labels, rotation=20, ha="right")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Ground-Truth Class Distribution")

    for position, value in zip(positions, values):
        ax.text(position, value, str(value), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_readiness_distribution(
    statistics_data: dict[str, Any],
    output_path: Path,
    dpi: int,
) -> None:
    counts = statistics_data["readiness_counts"]
    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = np.arange(len(labels))
    ax.bar(positions, values)
    ax.set_xticks(positions, labels=labels, rotation=20, ha="right")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Readiness Status Distribution")

    for position, value in zip(positions, values):
        ax.text(position, value, str(value), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_distribution(
    records: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    runtimes = [
        record["total_runtime_seconds"]
        for record in records
        if record["total_runtime_seconds"] is not None
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(runtimes, bins=min(30, max(10, int(math.sqrt(len(runtimes))))))
    if runtimes:
        mean_value = statistics.mean(runtimes)
        ax.axvline(
            mean_value,
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {mean_value:.2f} s",
        )
        ax.legend()

    ax.set_xlabel("Total runtime per subject (s)")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Pipeline Runtime Distribution")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_quality_summary(
    manifest: dict[str, Any],
    output_path: Path,
) -> None:
    stats = manifest["dataset_statistics"]
    lines = [
        "=" * 84,
        "BrainFMOps-Analyze — Dataset Manifest Summary",
        "=" * 84,
        f"Dataset name              : {manifest['dataset_name']}",
        f"Dataset manifest UUID     : {manifest['manifest_uuid']}",
        f"Generated at (UTC)        : {manifest['generated_at_utc']}",
        f"Total evaluated subjects  : {stats['total_subjects']}",
        f"Completed subjects        : {stats['completed_subjects']}",
        f"Failed subjects           : {stats['failed_subjects']}",
        f"Not-ready subjects        : {stats['not_ready_subjects']}",
        f"Labeled subjects          : {stats['labeled_subjects']}",
        f"Unlabeled subjects        : {stats['unlabeled_subjects']}",
        f"Dataset quality score     : {manifest['dataset_quality_score']:.2f}/100",
        "",
        "Ground-Truth Distribution",
        "-" * 84,
    ]

    for label, count in stats["ground_truth_counts"].items():
        lines.append(f"{label:24s}: {count}")

    lines.extend(["", "Pipeline Success Rates", "-" * 84])
    for metric, value in stats["pipeline_success_rates"].items():
        lines.append(f"{metric:36s}: {value:.2%}")

    lines.extend(["", "Warnings", "-" * 84])
    warnings = manifest["warnings"]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- None")

    lines.extend(
        [
            "",
            "Interpretation Boundary",
            "-" * 84,
            "This manifest describes the evaluated dataset and generated outputs.",
            "It does not prove that the evaluation is leakage-free.",
            "Subject-level provenance must be verified separately.",
            "=" * 84,
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a reproducible dataset manifest from BrainFMOps full "
            "evaluation outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--evaluation-csv",
        required=True,
        type=Path,
        help="STEP 25A evaluation_summary.csv.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional oasis_ground_truth.csv.",
    )
    parser.add_argument(
        "--batch-configuration",
        type=Path,
        default=None,
        help="Optional STEP 25A batch_configuration.json.",
    )
    parser.add_argument(
        "--provenance-report",
        type=Path,
        default=None,
        help="Optional STEP 27A subject_overlap_report.json.",
    )
    parser.add_argument(
        "--dataset-name",
        default="OASIS Cross-Sectional Brain MRI — BrainFMOps Evaluation Cohort",
    )
    parser.add_argument(
        "--dataset-version",
        default="BrainFMOps-Analyze-v1.0",
    )
    parser.add_argument(
        "--source-dataset",
        default="OASIS Cross-Sectional MRI Dataset",
    )
    parser.add_argument(
        "--license-note",
        default=(
            "Use is subject to the original OASIS data-use terms and citation "
            "requirements."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to 28A_Dataset_Manifest beside evaluation CSV.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    evaluation_csv = args.evaluation_csv.expanduser().resolve()
    labels_csv = (
        args.labels_csv.expanduser().resolve()
        if args.labels_csv is not None
        else None
    )
    batch_configuration_path = (
        args.batch_configuration.expanduser().resolve()
        if args.batch_configuration is not None
        else None
    )
    provenance_report_path = (
        args.provenance_report.expanduser().resolve()
        if args.provenance_report is not None
        else None
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else evaluation_csv.parent / "28A_Dataset_Manifest"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        labels = load_labels(labels_csv)
        records = load_evaluation_records(evaluation_csv, labels)
        statistics_data = build_statistics(records)
        dataset_quality_score, warnings = quality_score(statistics_data)

        evaluation_hash = file_sha256(evaluation_csv)
        labels_hash = file_sha256(labels_csv) if labels_csv else None

        batch_configuration = load_json_optional(batch_configuration_path)
        provenance_report = load_json_optional(provenance_report_path)

        provenance_decision = provenance_report.get(
            "decision",
            "NOT_PROVIDED",
        )
        if provenance_decision != "LEAKAGE_FREE":
            warnings.append(
                f"Provenance decision is {provenance_decision}; "
                "do not claim independent leakage-free performance."
            )

        manifest_uuid = canonical_manifest_uuid(
            evaluation_hash,
            labels_hash,
            args.dataset_name,
        )

        manifest = {
            "schema_version": "1.0",
            "script_version": SCRIPT_VERSION,
            "manifest_uuid": manifest_uuid,
            "generated_at_utc": utc_now_iso(),
            "dataset_name": args.dataset_name,
            "dataset_version": args.dataset_version,
            "source_dataset": args.source_dataset,
            "license_note": args.license_note,
            "evaluation_csv": str(evaluation_csv),
            "labels_csv": str(labels_csv) if labels_csv else None,
            "input_hashes": {
                "evaluation_csv_sha256": evaluation_hash,
                "labels_csv_sha256": labels_hash,
            },
            "dataset_statistics": statistics_data,
            "dataset_quality_score": dataset_quality_score,
            "warnings": warnings,
            "provenance": {
                "report_path": (
                    str(provenance_report_path)
                    if provenance_report_path
                    else None
                ),
                "decision": provenance_decision,
                "complete": provenance_decision == "LEAKAGE_FREE",
            },
            "batch_configuration": batch_configuration,
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "operating_system": os.name,
            },
            "research_use_only": True,
            "clinical_diagnosis": False,
            "interpretation_boundary": (
                "This manifest documents evaluated subjects and pipeline "
                "outputs. It does not establish clinical validity."
            ),
        }

        (output_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "dataset_statistics.json").write_text(
            json.dumps(statistics_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        write_subjects_csv(
            records,
            output_dir / "dataset_subjects.csv",
        )
        write_summary_csv(
            statistics_data,
            output_dir / "dataset_summary.csv",
        )
        write_quality_summary(
            manifest,
            output_dir / "dataset_quality_summary.txt",
        )

        plot_class_distribution(
            statistics_data,
            output_dir / "Fig_dataset_class_distribution.png",
            args.dpi,
        )
        plot_readiness_distribution(
            statistics_data,
            output_dir / "Fig_dataset_readiness_distribution.png",
            args.dpi,
        )
        plot_runtime_distribution(
            records,
            output_dir / "Fig_dataset_runtime_distribution.png",
            args.dpi,
        )

    except Exception as exc:
        logging.exception("Dataset manifest generation failed: %s", exc)
        return 1

    print("\n" + "=" * 84)
    print(f"DATASET UUID      : {manifest_uuid}")
    print(f"TOTAL SUBJECTS    : {statistics_data['total_subjects']}")
    print(f"LABELED SUBJECTS  : {statistics_data['labeled_subjects']}")
    print(f"COMPLETED         : {statistics_data['completed_subjects']}")
    print(f"QUALITY SCORE     : {dataset_quality_score:.2f}/100")
    print(f"PROVENANCE        : {provenance_decision}")
    print(f"OUTPUT            : {output_dir}")
    print("=" * 84)

    return 0


if __name__ == "__main__":
    sys.exit(main())
