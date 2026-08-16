#!/usr/bin/env python3
"""
25A_batch_evaluation_framework.py

BrainFMOps-Analyze — STEP 25A
Batch Evaluation Framework for multiple OASIS subject folders.

Pipeline per case
-----------------
1. STEP 24A  : MRI case readiness validation
2. STEP 24C.5: deterministic volume selection
3. STEP 24B  : subject-level inference with locked threshold
4. STEP 24C  : representative Grad-CAM generation
5. STEP 24D  : self-contained HTML case report
6. Optional  : STEP 24C.6 publication figures

Key features
------------
- Discovers OASIS subjects recursively (e.g. OAS1_0001_MR1)
- Centralized output folder per case
- Resume completed cases
- Per-step timeout and error capture
- Batch summary CSV and JSON
- Optional ground-truth CSV integration
- Preliminary accuracy/precision/recall/F1 when labels are available
- Reproducible command manifest

Example
-------
python 25A_batch_evaluation_framework.py ^
  --dataset-root "data/OASIS" ^
  --checkpoint "outputs/efficientnetb0_binary_gpu_v1/best_model.pth" ^
  --output-root "outputs/batch_evaluation" ^
  --limit 50 ^
  --device cuda

Optional labels CSV format
--------------------------
case_id,label
OAS1_0001_MR1,AD
OAS1_0002_MR1,CN

Important
---------
This software is for research evaluation only.
It is not a clinical diagnosis or medical-device system.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCRIPT_VERSION = "1.0.0"
SUBJECT_PATTERN = re.compile(r"^OAS\d+_\d+_MR\d+$", re.IGNORECASE)


@dataclass
class StepResult:
    step: str
    status: str
    return_code: Optional[int]
    runtime_seconds: float
    command: list[str]
    stdout_log: str
    stderr_log: str
    error_message: str = ""


@dataclass
class CaseResult:
    case_id: str
    case_dir: str
    status: str = "PENDING"
    readiness_status: str = ""
    prediction: str = ""
    probability_positive: Optional[float] = None
    locked_threshold: Optional[float] = None
    ground_truth: str = ""
    correct: Optional[bool] = None

    readiness_success: bool = False
    volume_selection_success: bool = False
    inference_success: bool = False
    gradcam_success: bool = False
    report_success: bool = False
    publication_figure_success: bool = False

    selected_volume_index: Optional[int] = None
    selected_volume_score: Optional[float] = None
    selected_volume_file: str = ""

    total_runtime_seconds: float = 0.0
    failure_step: str = ""
    error_message: str = ""
    output_dir: str = ""
    html_report: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_case_id(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    return cleaned or "UNKNOWN_CASE"


def discover_subjects(
    dataset_root: Path,
    subject_regex: re.Pattern[str],
) -> list[Path]:
    subjects: list[Path] = []

    for path in dataset_root.rglob("*"):
        if not path.is_dir():
            continue
        if subject_regex.match(path.name):
            subjects.append(path)

    unique = sorted(
        {path.resolve() for path in subjects},
        key=lambda p: p.name.lower(),
    )
    return unique


def load_ground_truth(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth CSV does not exist: {path}")

    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"case_id", "label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Ground-truth CSV must contain columns: case_id,label"
            )

        for row in reader:
            case_id = row["case_id"].strip()
            label = row["label"].strip()
            if case_id:
                mapping[case_id] = label

    return mapping


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_command(
    step_name: str,
    command: list[str],
    log_dir: Path,
    timeout_seconds: int,
    env: dict[str, str],
) -> StepResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{step_name}_stdout.log"
    stderr_path = log_dir / f"{step_name}_stderr.log"

    logging.info("Running %s", step_name)
    logging.debug("Command: %s", subprocess.list2cmdline(command))

    started = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
        runtime = time.perf_counter() - started

        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")

        status = "SUCCESS" if completed.returncode == 0 else "FAILED"
        error_message = ""
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip()
            error_message = tail[-2000:]

        return StepResult(
            step=step_name,
            status=status,
            return_code=completed.returncode,
            runtime_seconds=round(runtime, 4),
            command=command,
            stdout_log=str(stdout_path.resolve()),
            stderr_log=str(stderr_path.resolve()),
            error_message=error_message,
        )

    except subprocess.TimeoutExpired as exc:
        runtime = time.perf_counter() - started
        stdout_text = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr_text = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )

        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")

        return StepResult(
            step=step_name,
            status="TIMEOUT",
            return_code=None,
            runtime_seconds=round(runtime, 4),
            command=command,
            stdout_log=str(stdout_path.resolve()),
            stderr_log=str(stderr_path.resolve()),
            error_message=f"Step exceeded timeout of {timeout_seconds} seconds.",
        )

    except Exception as exc:
        runtime = time.perf_counter() - started
        stderr_path.write_text(str(exc), encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")

        return StepResult(
            step=step_name,
            status="ERROR",
            return_code=None,
            runtime_seconds=round(runtime, 4),
            command=command,
            stdout_log=str(stdout_path.resolve()),
            stderr_log=str(stderr_path.resolve()),
            error_message=f"{type(exc).__name__}: {exc}",
        )


def step_succeeded(step: StepResult) -> bool:
    return step.status == "SUCCESS" and step.return_code == 0


def append_step(case_result: CaseResult, step_result: StepResult) -> None:
    case_result.steps.append(asdict(step_result))
    case_result.total_runtime_seconds = round(
        case_result.total_runtime_seconds + step_result.runtime_seconds,
        4,
    )


def mark_failure(
    case_result: CaseResult,
    step_result: StepResult,
) -> CaseResult:
    case_result.status = "FAILED"
    case_result.failure_step = step_result.step
    case_result.error_message = step_result.error_message
    return case_result


def is_completed_case(case_output_dir: Path) -> bool:
    subject_result = case_output_dir / "inference" / "subject_result.json"
    html_report = case_output_dir / f"{case_output_dir.name}_report.html"
    return subject_result.exists() and html_report.exists()


def evaluate_case(
    case_dir: Path,
    output_root: Path,
    scripts_dir: Path,
    checkpoint: Path,
    python_executable: Path,
    device: str,
    threshold: float,
    aggregation: str,
    num_slices: int,
    timeout_seconds: int,
    labels: dict[str, str],
    generate_publication_figures: bool,
    resume: bool,
    force_not_ready: bool,
) -> CaseResult:
    case_id = safe_case_id(case_dir.name)
    case_output = output_root / case_id
    logs_dir = case_output / "logs"

    readiness_dir = case_output / "readiness"
    volume_dir = case_output / "volume_selection"
    inference_dir = case_output / "inference"
    gradcam_dir = case_output / "gradcam"
    figures_dir = case_output / "publication_figures"
    html_path = case_output / f"{case_id}_report.html"

    result = CaseResult(
        case_id=case_id,
        case_dir=str(case_dir.resolve()),
        ground_truth=labels.get(case_id, ""),
        output_dir=str(case_output.resolve()),
        html_report=str(html_path.resolve()),
    )

    if resume and is_completed_case(case_output):
        try:
            readiness = load_json(readiness_dir / "readiness_report.json")
            subject = load_json(inference_dir / "subject_result.json")
            volume_report = load_json(volume_dir / "volume_selection_report.json")

            result.status = "COMPLETED_RESUMED"
            result.readiness_status = str(readiness.get("status", ""))
            result.prediction = str(subject.get("predicted_class_name", ""))
            result.probability_positive = subject.get(
                "subject_probability_positive"
            )
            result.locked_threshold = subject.get("locked_threshold")
            result.selected_volume_index = volume_report.get(
                "selected_volume_index"
            )
            result.selected_volume_score = volume_report.get("selected_score")
            result.selected_volume_file = str(
                volume_report.get("selected_volume_file", "")
            )
            result.readiness_success = True
            result.volume_selection_success = True
            result.inference_success = True
            result.gradcam_success = gradcam_dir.exists()
            result.report_success = html_path.exists()
            result.publication_figure_success = figures_dir.exists()

            if result.ground_truth:
                result.correct = (
                    result.prediction.strip().upper()
                    == result.ground_truth.strip().upper()
                )
            return result
        except Exception as exc:
            logging.warning(
                "Resume metadata could not be loaded for %s: %s. Re-running.",
                case_id,
                exc,
            )

    case_output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "42"

    py = str(python_executable)
    script_24a = scripts_dir / "24A_case_readiness_checker_v2.py"
    script_24b = scripts_dir / "24B_subject_inference_engine.py"
    script_24c = scripts_dir / "24C_case_gradcam_generator.py"
    script_24c5 = scripts_dir / "24C5_mri_volume_selector.py"
    script_24c6 = scripts_dir / "24C6_publication_figure_generator.py"
    script_24d = scripts_dir / "24D_case_report_generator.py"

    # STEP 24A
    command_24a = [
        py,
        str(script_24a),
        "--case-dir",
        str(case_dir),
        "--output-dir",
        str(readiness_dir),
    ]
    step = run_command(
        "24A_readiness",
        command_24a,
        logs_dir,
        timeout_seconds,
        env,
    )
    append_step(result, step)
    if not step_succeeded(step):
        return mark_failure(result, step)

    readiness_path = readiness_dir / "readiness_report.json"
    readiness = load_json(readiness_path)
    result.readiness_status = str(readiness.get("status", ""))
    result.readiness_success = True

    if (
        result.readiness_status == "NOT_READY"
        and not force_not_ready
    ):
        result.status = "SKIPPED_NOT_READY"
        result.failure_step = "24A_readiness_gate"
        result.error_message = (
            "Inference skipped because readiness status is NOT_READY."
        )
        return result

    primary_relative = readiness.get("statistics", {}).get("primary_volume")
    if not primary_relative:
        result.status = "FAILED"
        result.failure_step = "24A_primary_volume"
        result.error_message = "Readiness report did not select a primary volume."
        return result

    primary_volume = case_dir / primary_relative

    # STEP 24C.5
    command_24c5 = [
        py,
        str(script_24c5),
        "--source-mri",
        str(primary_volume),
        "--case-id",
        case_id,
        "--output-dir",
        str(volume_dir),
    ]
    step = run_command(
        "24C5_volume_selection",
        command_24c5,
        logs_dir,
        timeout_seconds,
        env,
    )
    append_step(result, step)
    if not step_succeeded(step):
        return mark_failure(result, step)

    volume_report_path = volume_dir / "volume_selection_report.json"
    volume_report = load_json(volume_report_path)
    selected_volume = Path(volume_report["selected_volume_file"])

    result.volume_selection_success = True
    result.selected_volume_index = volume_report.get("selected_volume_index")
    result.selected_volume_score = volume_report.get("selected_score")
    result.selected_volume_file = str(selected_volume)

    # STEP 24B
    command_24b = [
        py,
        str(script_24b),
        "--readiness-report",
        str(readiness_path),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(inference_dir),
        "--volume",
        str(selected_volume),
        "--device",
        device,
        "--threshold",
        str(threshold),
        "--aggregation",
        aggregation,
        "--num-slices",
        str(num_slices),
    ]
    if force_not_ready:
        command_24b.append("--force")

    step = run_command(
        "24B_inference",
        command_24b,
        logs_dir,
        timeout_seconds,
        env,
    )
    append_step(result, step)
    if not step_succeeded(step):
        return mark_failure(result, step)

    subject_path = inference_dir / "subject_result.json"
    slice_predictions_path = inference_dir / "slice_predictions.csv"
    subject = load_json(subject_path)

    result.inference_success = True
    result.prediction = str(subject.get("predicted_class_name", ""))
    result.probability_positive = subject.get(
        "subject_probability_positive"
    )
    result.locked_threshold = subject.get("locked_threshold")

    if result.ground_truth:
        result.correct = (
            result.prediction.strip().upper()
            == result.ground_truth.strip().upper()
        )

    # STEP 24C
    command_24c = [
        py,
        str(script_24c),
        "--subject-result",
        str(subject_path),
        "--slice-predictions",
        str(slice_predictions_path),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(gradcam_dir),
        "--device",
        device,
    ]
    step = run_command(
        "24C_gradcam",
        command_24c,
        logs_dir,
        timeout_seconds,
        env,
    )
    append_step(result, step)
    if not step_succeeded(step):
        return mark_failure(result, step)

    result.gradcam_success = True
    gradcam_summary_path = gradcam_dir / "gradcam_summary.json"
    representative_csv = gradcam_dir / "representative_slices.csv"

    # Optional STEP 24C.6
    if generate_publication_figures:
        command_24c6 = [
            py,
            str(script_24c6),
            "--subject-result",
            str(subject_path),
            "--gradcam-summary",
            str(gradcam_summary_path),
            "--representative-slices",
            str(representative_csv),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(figures_dir),
            "--device",
            device,
        ]
        step = run_command(
            "24C6_publication_figures",
            command_24c6,
            logs_dir,
            timeout_seconds,
            env,
        )
        append_step(result, step)
        if not step_succeeded(step):
            return mark_failure(result, step)
        result.publication_figure_success = True

    # STEP 24D
    command_24d = [
        py,
        str(script_24d),
        "--readiness-report",
        str(readiness_path),
        "--subject-result",
        str(subject_path),
        "--gradcam-summary",
        str(gradcam_summary_path),
        "--volume-selection-report",
        str(volume_report_path),
        "--representative-slices",
        str(representative_csv),
        "--gradcam-dir",
        str(gradcam_dir),
        "--output",
        str(html_path),
    ]
    step = run_command(
        "24D_html_report",
        command_24d,
        logs_dir,
        timeout_seconds,
        env,
    )
    append_step(result, step)
    if not step_succeeded(step):
        return mark_failure(result, step)

    result.report_success = True
    result.status = "COMPLETED"
    return result


def case_to_csv_row(result: CaseResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "case_dir": result.case_dir,
        "status": result.status,
        "readiness_status": result.readiness_status,
        "prediction": result.prediction,
        "probability_positive": result.probability_positive,
        "locked_threshold": result.locked_threshold,
        "ground_truth": result.ground_truth,
        "correct": result.correct,
        "readiness_success": result.readiness_success,
        "volume_selection_success": result.volume_selection_success,
        "inference_success": result.inference_success,
        "gradcam_success": result.gradcam_success,
        "report_success": result.report_success,
        "publication_figure_success": result.publication_figure_success,
        "selected_volume_index": result.selected_volume_index,
        "selected_volume_score": result.selected_volume_score,
        "selected_volume_file": result.selected_volume_file,
        "total_runtime_seconds": result.total_runtime_seconds,
        "failure_step": result.failure_step,
        "error_message": result.error_message,
        "output_dir": result.output_dir,
        "html_report": result.html_report,
    }


def write_summary_csv(results: list[CaseResult], path: Path) -> None:
    rows = [case_to_csv_row(result) for result in results]
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else [
        "case_id",
        "status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_binary_metrics(
    results: list[CaseResult],
    positive_label: str,
) -> dict[str, Any]:
    labeled = [
        result
        for result in results
        if result.ground_truth
        and result.prediction
        and result.status in {"COMPLETED", "COMPLETED_RESUMED"}
    ]

    if not labeled:
        return {
            "labeled_case_count": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    positive = positive_label.strip().upper()

    tp = tn = fp = fn = 0
    for result in labeled:
        truth_pos = result.ground_truth.strip().upper() == positive
        pred_pos = result.prediction.strip().upper() == positive

        if truth_pos and pred_pos:
            tp += 1
        elif not truth_pos and not pred_pos:
            tn += 1
        elif not truth_pos and pred_pos:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and (precision + recall) > 0
        else None
    )

    return {
        "labeled_case_count": len(labeled),
        "positive_label": positive_label,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def build_batch_summary(
    results: list[CaseResult],
    discovered_count: int,
    positive_label: str,
    started_at: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    completed = [
        r for r in results
        if r.status in {"COMPLETED", "COMPLETED_RESUMED"}
    ]
    failed = [r for r in results if r.status == "FAILED"]
    not_ready = [r for r in results if r.status == "SKIPPED_NOT_READY"]

    runtimes = [
        r.total_runtime_seconds
        for r in completed
        if r.total_runtime_seconds > 0
    ]

    readiness_counts: dict[str, int] = {}
    prediction_counts: dict[str, int] = {}

    for result in results:
        if result.readiness_status:
            readiness_counts[result.readiness_status] = (
                readiness_counts.get(result.readiness_status, 0) + 1
            )
        if result.prediction:
            prediction_counts[result.prediction] = (
                prediction_counts.get(result.prediction, 0) + 1
            )

    return {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now_iso(),
        "batch_runtime_seconds": round(runtime_seconds, 4),
        "discovered_subject_count": discovered_count,
        "processed_subject_count": len(results),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "skipped_not_ready_count": len(not_ready),
        "completion_rate": (
            len(completed) / len(results) if results else 0.0
        ),
        "readiness_status_counts": readiness_counts,
        "prediction_counts": prediction_counts,
        "runtime_seconds": {
            "mean": (
                sum(runtimes) / len(runtimes) if runtimes else None
            ),
            "min": min(runtimes) if runtimes else None,
            "max": max(runtimes) if runtimes else None,
        },
        "classification_metrics": compute_binary_metrics(
            results,
            positive_label=positive_label,
        ),
        "research_use_only": True,
        "clinical_diagnosis": False,
    }


def validate_required_files(
    scripts_dir: Path,
    checkpoint: Path,
) -> None:
    required = [
        "24A_case_readiness_checker_v2.py",
        "24B_subject_inference_engine.py",
        "24C_case_gradcam_generator.py",
        "24C5_mri_volume_selector.py",
        "24D_case_report_generator.py",
    ]

    missing = [
        str(scripts_dir / name)
        for name in required
        if not (scripts_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required pipeline script(s) missing:\n" + "\n".join(missing)
        )

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")


def ratio_0_to_1(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("Value must be between 0 and 1.")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run BrainFMOps-Analyze over multiple OASIS subjects and "
            "collect batch evaluation results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root containing OASIS disc folders.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Trained EfficientNet-B0 checkpoint.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Central output directory for all evaluated cases.",
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing STEP 24 scripts.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used for child processes.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns case_id,label.",
    )
    parser.add_argument(
        "--positive-label",
        default="AD",
        help="Positive label used for preliminary binary metrics.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Maximum number of cases to process.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based index into sorted discovered subjects.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--threshold",
        type=ratio_0_to_1,
        default=0.32,
    )
    parser.add_argument(
        "--aggregation",
        choices=("mean", "median", "max", "topk_mean"),
        default="mean",
    )
    parser.add_argument(
        "--num-slices",
        type=positive_int,
        default=16,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=positive_int,
        default=900,
        help="Maximum runtime for each individual pipeline step.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases with existing subject_result.json and HTML report.",
    )
    parser.add_argument(
        "--force-not-ready",
        action="store_true",
        help="Continue inference for NOT_READY cases for controlled research.",
    )
    parser.add_argument(
        "--generate-publication-figures",
        action="store_true",
        help="Run STEP 24C.6 for every completed case. This is expensive.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    started_at = utc_now_iso()
    batch_started = time.perf_counter()

    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    scripts_dir = args.scripts_dir.expanduser().resolve()
    python_executable = args.python.expanduser().resolve()

    try:
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist: {dataset_root}"
            )

        validate_required_files(scripts_dir, checkpoint)
        labels = load_ground_truth(
            args.labels_csv.expanduser().resolve()
            if args.labels_csv is not None
            else None
        )

        subjects = discover_subjects(dataset_root, SUBJECT_PATTERN)
        discovered_count = len(subjects)

        if args.start_index < 0:
            parser.error("--start-index must be non-negative.")

        subjects = subjects[args.start_index :]
        if args.limit is not None:
            subjects = subjects[: args.limit]

        if not subjects:
            raise RuntimeError(
                "No OASIS subject folders were discovered. "
                "Expected names such as OAS1_0001_MR1."
            )

        output_root.mkdir(parents=True, exist_ok=True)

        configuration = {
            "script_version": SCRIPT_VERSION,
            "started_at_utc": started_at,
            "dataset_root": str(dataset_root),
            "checkpoint": str(checkpoint),
            "output_root": str(output_root),
            "scripts_dir": str(scripts_dir),
            "python_executable": str(python_executable),
            "device": args.device,
            "threshold": args.threshold,
            "aggregation": args.aggregation,
            "num_slices": args.num_slices,
            "timeout_seconds": args.timeout_seconds,
            "resume": args.resume,
            "force_not_ready": args.force_not_ready,
            "generate_publication_figures": (
                args.generate_publication_figures
            ),
            "discovered_subject_count": discovered_count,
            "selected_subject_count": len(subjects),
            "start_index": args.start_index,
            "limit": args.limit,
        }
        write_json(
            configuration,
            output_root / "batch_configuration.json",
        )

        results: list[CaseResult] = []

        for index, case_dir in enumerate(subjects, start=1):
            logging.info(
                "[%d/%d] Processing %s",
                index,
                len(subjects),
                case_dir.name,
            )

            try:
                result = evaluate_case(
                    case_dir=case_dir,
                    output_root=output_root,
                    scripts_dir=scripts_dir,
                    checkpoint=checkpoint,
                    python_executable=python_executable,
                    device=args.device,
                    threshold=args.threshold,
                    aggregation=args.aggregation,
                    num_slices=args.num_slices,
                    timeout_seconds=args.timeout_seconds,
                    labels=labels,
                    generate_publication_figures=(
                        args.generate_publication_figures
                    ),
                    resume=args.resume,
                    force_not_ready=args.force_not_ready,
                )
            except Exception as exc:
                logging.exception(
                    "Unhandled case failure for %s: %s",
                    case_dir.name,
                    exc,
                )
                result = CaseResult(
                    case_id=safe_case_id(case_dir.name),
                    case_dir=str(case_dir.resolve()),
                    status="FAILED",
                    failure_step="FRAMEWORK",
                    error_message=f"{type(exc).__name__}: {exc}",
                    output_dir=str(
                        (output_root / safe_case_id(case_dir.name)).resolve()
                    ),
                )

            results.append(result)

            # Write incrementally to preserve progress after interruption.
            write_summary_csv(
                results,
                output_root / "evaluation_summary.csv",
            )
            write_json(
                [asdict(item) for item in results],
                output_root / "evaluation_case_details.json",
            )

            logging.info(
                "%s -> %s",
                result.case_id,
                result.status,
            )

        batch_runtime = time.perf_counter() - batch_started
        summary = build_batch_summary(
            results=results,
            discovered_count=discovered_count,
            positive_label=args.positive_label,
            started_at=started_at,
            runtime_seconds=batch_runtime,
        )
        write_json(
            summary,
            output_root / "batch_evaluation_summary.json",
        )

    except Exception as exc:
        logging.exception("Batch evaluation failed: %s", exc)
        return 1

    print("\n" + "=" * 82)
    print(f"DISCOVERED CASES : {discovered_count}")
    print(f"PROCESSED CASES  : {len(results)}")
    print(f"COMPLETED        : {summary['completed_count']}")
    print(f"FAILED           : {summary['failed_count']}")
    print(f"NOT READY        : {summary['skipped_not_ready_count']}")
    print(f"COMPLETION RATE  : {summary['completion_rate']:.2%}")
    print(f"OUTPUT           : {output_root}")
    print("=" * 82)

    return 0


if __name__ == "__main__":
    sys.exit(main())
