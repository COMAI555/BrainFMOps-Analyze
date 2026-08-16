#!/usr/bin/env python3
"""
24B_subject_inference_engine.py

BrainFMOps-Analyze — STEP 24B
Subject-Level MRI Inference Engine

Pipeline
--------
1. Read STEP 24A readiness_report.json
2. Reject NOT_READY cases unless --force is used
3. Load the selected primary MRI volume
4. Select deterministic informative slices
5. Apply the same image preprocessing used for model inference
6. Run slice-level probabilities
7. Aggregate to one subject-level probability
8. Apply the locked decision threshold (default = 0.32)
9. Save reproducible outputs

Outputs
-------
- slice_predictions.csv
- subject_result.json
- inference_summary.txt

Important
---------
This script is for research use only. It is not a clinical diagnosis system.

Expected model
--------------
Default architecture: torchvision EfficientNet-B0, binary classification.

Supported checkpoint layouts include:
- raw state_dict
- {"state_dict": ...}
- {"model_state_dict": ...}
- {"model": ...}

The final classifier may have either:
- 1 output logit: sigmoid probability
- 2 output logits: softmax probability of --positive-class-index
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


SCRIPT_VERSION = "1.0.0"
LOCKED_THRESHOLD_DEFAULT = 0.32
READY_STATUSES = {"READY", "READY_WITH_WARNINGS"}


@dataclass(frozen=True)
class InferenceConfig:
    architecture: str
    image_size: int
    num_selected_slices: int
    min_foreground_ratio: float
    threshold: float
    aggregation: str
    positive_class_index: int
    negative_class_name: str
    positive_class_name: str
    seed: int


@dataclass
class SlicePrediction:
    rank: int
    slice_index: int
    axis: int
    foreground_ratio: float
    intensity_mean: float
    intensity_std: float
    probability_positive: float
    predicted_class_index: int
    predicted_class_name: str


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("nibabel").setLevel(logging.ERROR)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_case_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "UNKNOWN_CASE"


def load_readiness_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Readiness report does not exist: {path}")

    with path.open("r", encoding="utf-8") as file_obj:
        report = json.load(file_obj)

    required = {"case_id", "case_dir", "status", "statistics"}
    missing = required.difference(report)
    if missing:
        raise ValueError(
            f"Readiness report is missing required field(s): {sorted(missing)}"
        )

    return report


def resolve_primary_volume(
    readiness_report: dict[str, Any],
    override_volume: Optional[Path],
) -> Path:
    case_dir = Path(readiness_report["case_dir"])

    if override_volume is not None:
        path = override_volume.expanduser().resolve()
    else:
        relative = readiness_report["statistics"].get("primary_volume")
        if not relative:
            raise ValueError(
                "STEP 24A report does not contain a selected primary volume."
            )
        path = (case_dir / relative).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Primary MRI volume does not exist: {path}")

    return path


def build_model(
    architecture: str,
    num_outputs: int,
) -> nn.Module:
    architecture = architecture.lower()

    if architecture != "efficientnet_b0":
        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            "Current STEP 24B supports efficientnet_b0."
        )

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_outputs)
    return model


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint does not contain a valid state_dict.")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue

        clean_key = key
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        state_dict[clean_key] = value

    if not state_dict:
        raise ValueError("No tensor parameters were found in the checkpoint.")

    return state_dict


def infer_num_outputs(state_dict: dict[str, torch.Tensor]) -> int:
    candidate_keys = (
        "classifier.1.weight",
        "classifier.weight",
        "fc.weight",
    )

    for key in candidate_keys:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.ndim == 2:
            return int(tensor.shape[0])

    for key, tensor in state_dict.items():
        if key.endswith("classifier.1.weight") and tensor.ndim == 2:
            return int(tensor.shape[0])

    raise ValueError(
        "Could not infer classifier output size from checkpoint. "
        "Expected an EfficientNet-B0 classifier weight."
    )


def load_model(
    checkpoint_path: Path,
    architecture: str,
    device: torch.device,
) -> tuple[nn.Module, int, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = extract_state_dict(checkpoint)
    num_outputs = infer_num_outputs(state_dict)

    if num_outputs not in (1, 2):
        raise ValueError(
            f"Binary inference expects 1 or 2 model outputs, found {num_outputs}."
        )

    model = build_model(architecture, num_outputs=num_outputs)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning("Missing checkpoint keys: %s", missing)
    if unexpected:
        logging.warning("Unexpected checkpoint keys: %s", unexpected)

    critical_missing = [
        key for key in missing
        if key.startswith("features.") or key.startswith("classifier.")
    ]
    if critical_missing:
        raise ValueError(
            "Checkpoint is incompatible with EfficientNet-B0. "
            f"Critical missing keys: {critical_missing[:10]}"
        )

    model.to(device)
    model.eval()

    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return model, num_outputs, metadata


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("MRI volume contains no finite voxel values.")

    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))

    if high <= low:
        raise ValueError(
            "MRI intensity range is degenerate after percentile clipping."
        )

    clipped = np.clip(volume, low, high)
    normalized = (clipped - low) / (high - low)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def choose_axis(shape: tuple[int, ...], requested_axis: int) -> int:
    if requested_axis in (0, 1, 2):
        return requested_axis

    # Auto mode: use the axis with the largest number of slices.
    spatial = shape[:3]
    return int(np.argmax(spatial))


def slice_from_volume(
    volume: np.ndarray,
    axis: int,
    index: int,
) -> np.ndarray:
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def informative_slice_candidates(
    volume: np.ndarray,
    axis: int,
    min_foreground_ratio: float,
) -> list[dict[str, float]]:
    num_slices = volume.shape[axis]
    candidates: list[dict[str, float]] = []

    for index in range(num_slices):
        image = slice_from_volume(volume, axis, index)
        finite = image[np.isfinite(image)]

        if finite.size == 0:
            continue

        foreground_ratio = float(np.mean(finite > 0.02))
        if foreground_ratio < min_foreground_ratio:
            continue

        candidates.append(
            {
                "slice_index": index,
                "foreground_ratio": foreground_ratio,
                "intensity_mean": float(np.mean(finite)),
                "intensity_std": float(np.std(finite)),
            }
        )

    return candidates


def select_slices(
    candidates: list[dict[str, float]],
    num_selected: int,
) -> list[dict[str, float]]:
    if not candidates:
        raise ValueError(
            "No informative MRI slices passed the foreground-ratio criterion."
        )

    # Keep candidates in anatomical order and select evenly spaced positions.
    candidates = sorted(candidates, key=lambda item: item["slice_index"])

    if len(candidates) <= num_selected:
        return candidates

    positions = np.linspace(
        0,
        len(candidates) - 1,
        num=num_selected,
        dtype=int,
    )
    unique_positions = sorted(set(int(pos) for pos in positions))
    return [candidates[pos] for pos in unique_positions]


def build_transform(image_size: int) -> transforms.Compose:
    """
    Standard ImageNet-compatible preprocessing.

    This must match the preprocessing used during training. If the training
    pipeline used different normalization, change these values deliberately.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def slice_to_pil(slice_array: np.ndarray) -> Image.Image:
    array = np.clip(slice_array, 0.0, 1.0)
    uint8 = np.round(array * 255.0).astype(np.uint8)
    grayscale = Image.fromarray(uint8, mode="L")
    return grayscale.convert("RGB")


def probability_from_output(
    output: torch.Tensor,
    num_outputs: int,
    positive_class_index: int,
) -> torch.Tensor:
    if num_outputs == 1:
        return torch.sigmoid(output.reshape(-1))

    if positive_class_index not in (0, 1):
        raise ValueError(
            "--positive-class-index must be 0 or 1 for a two-output model."
        )

    return torch.softmax(output, dim=1)[:, positive_class_index]


def run_slice_inference(
    model: nn.Module,
    num_outputs: int,
    volume: np.ndarray,
    selected: list[dict[str, float]],
    axis: int,
    transform: transforms.Compose,
    batch_size: int,
    device: torch.device,
    threshold: float,
    positive_class_index: int,
    negative_class_name: str,
    positive_class_name: str,
) -> list[SlicePrediction]:
    predictions: list[SlicePrediction] = []

    tensors: list[torch.Tensor] = []
    for item in selected:
        image = slice_from_volume(
            volume,
            axis,
            int(item["slice_index"]),
        )
        tensors.append(transform(slice_to_pil(image)))

    for start in range(0, len(tensors), batch_size):
        batch_tensors = torch.stack(
            tensors[start : start + batch_size]
        ).to(device)

        with torch.inference_mode():
            output = model(batch_tensors)
            probabilities = probability_from_output(
                output=output,
                num_outputs=num_outputs,
                positive_class_index=positive_class_index,
            )

        for local_index, probability in enumerate(
            probabilities.detach().cpu().numpy().tolist()
        ):
            global_index = start + local_index
            item = selected[global_index]
            probability = float(probability)
            predicted_positive = probability >= threshold

            predictions.append(
                SlicePrediction(
                    rank=global_index + 1,
                    slice_index=int(item["slice_index"]),
                    axis=axis,
                    foreground_ratio=float(item["foreground_ratio"]),
                    intensity_mean=float(item["intensity_mean"]),
                    intensity_std=float(item["intensity_std"]),
                    probability_positive=probability,
                    predicted_class_index=(
                        positive_class_index
                        if predicted_positive
                        else 1 - positive_class_index
                    ),
                    predicted_class_name=(
                        positive_class_name
                        if predicted_positive
                        else negative_class_name
                    ),
                )
            )

    return predictions


def aggregate_probabilities(
    probabilities: list[float],
    method: str,
) -> float:
    if not probabilities:
        raise ValueError("No slice probabilities are available for aggregation.")

    values = np.asarray(probabilities, dtype=np.float64)

    if method == "mean":
        return float(np.mean(values))
    if method == "median":
        return float(np.median(values))
    if method == "max":
        return float(np.max(values))
    if method == "topk_mean":
        k = min(5, len(values))
        return float(np.mean(np.sort(values)[-k:]))

    raise ValueError(f"Unsupported aggregation method: {method}")


def write_slice_csv(
    predictions: list[SlicePrediction],
    path: Path,
) -> None:
    rows = [asdict(item) for item in predictions]
    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_summary(result: dict[str, Any], path: Path) -> None:
    lines = [
        "=" * 78,
        "BrainFMOps-Analyze — Subject-Level Inference Summary",
        "=" * 78,
        f"Case ID                  : {result['case_id']}",
        f"Generated at (UTC)       : {result['generated_at_utc']}",
        f"Script version           : {result['script_version']}",
        "",
        "Input",
        "-" * 78,
        f"Readiness status         : {result['readiness_status']}",
        f"Primary volume           : {result['primary_volume']}",
        f"Model checkpoint         : {result['model_checkpoint']}",
        f"Device                   : {result['device']}",
        "",
        "Inference Configuration",
        "-" * 78,
        f"Architecture             : {result['config']['architecture']}",
        f"Selected slice axis      : {result['selected_axis']}",
        f"Selected slices          : {result['num_selected_slices']}",
        f"Aggregation              : {result['config']['aggregation']}",
        f"Locked threshold         : {result['config']['threshold']:.4f}",
        "",
        "Subject-Level Result",
        "-" * 78,
        f"Probability positive     : {result['subject_probability_positive']:.6f}",
        f"Predicted class index    : {result['predicted_class_index']}",
        f"Predicted class name     : {result['predicted_class_name']}",
        f"Decision margin          : {result['decision_margin']:.6f}",
        f"Runtime seconds          : {result['runtime_seconds']:.3f}",
        "",
        "Research-use notice",
        "-" * 78,
        "This result is produced by a research prototype.",
        "It is not a clinical diagnosis or medical-device decision.",
        "=" * 78,
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_subject_inference(
    readiness_report_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    axis: int,
    batch_size: int,
    device_name: str,
    override_volume: Optional[Path],
    force: bool,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    set_reproducibility(config.seed)

    readiness = load_readiness_report(readiness_report_path)
    readiness_status = str(readiness["status"])

    if readiness_status not in READY_STATUSES and not force:
        raise RuntimeError(
            f"Case readiness status is {readiness_status}. "
            "Inference is blocked. Use --force only for controlled research."
        )

    primary_volume_path = resolve_primary_volume(
        readiness_report=readiness,
        override_volume=override_volume,
    )

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    logging.info("Device: %s", device)
    logging.info("Primary MRI volume: %s", primary_volume_path)
    logging.info("Loading model checkpoint: %s", checkpoint_path)

    model, num_outputs, checkpoint_metadata = load_model(
        checkpoint_path=checkpoint_path,
        architecture=config.architecture,
        device=device,
    )

    image = nib.load(str(primary_volume_path))
    raw_volume = np.asarray(image.dataobj, dtype=np.float32)

    if raw_volume.ndim > 3:
        logging.warning(
            "Volume has %d dimensions; first volume/channel will be used.",
            raw_volume.ndim,
        )
        while raw_volume.ndim > 3:
            raw_volume = raw_volume[..., 0]

    if raw_volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI volume after reduction, found shape "
            f"{raw_volume.shape}."
        )

    volume = normalize_volume(raw_volume)
    selected_axis = choose_axis(volume.shape, axis)

    candidates = informative_slice_candidates(
        volume=volume,
        axis=selected_axis,
        min_foreground_ratio=config.min_foreground_ratio,
    )
    selected = select_slices(
        candidates=candidates,
        num_selected=config.num_selected_slices,
    )

    logging.info(
        "Selected %d informative slices from %d candidates on axis %d.",
        len(selected),
        len(candidates),
        selected_axis,
    )

    predictions = run_slice_inference(
        model=model,
        num_outputs=num_outputs,
        volume=volume,
        selected=selected,
        axis=selected_axis,
        transform=build_transform(config.image_size),
        batch_size=batch_size,
        device=device,
        threshold=config.threshold,
        positive_class_index=config.positive_class_index,
        negative_class_name=config.negative_class_name,
        positive_class_name=config.positive_class_name,
    )

    probabilities = [
        item.probability_positive for item in predictions
    ]
    subject_probability = aggregate_probabilities(
        probabilities=probabilities,
        method=config.aggregation,
    )
    predicted_positive = subject_probability >= config.threshold

    predicted_class_index = (
        config.positive_class_index
        if predicted_positive
        else 1 - config.positive_class_index
    )
    predicted_class_name = (
        config.positive_class_name
        if predicted_positive
        else config.negative_class_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = time.perf_counter() - start_time

    result = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "case_id": sanitize_case_id(str(readiness["case_id"])),
        "case_dir": str(readiness["case_dir"]),
        "readiness_report": str(readiness_report_path.resolve()),
        "readiness_status": readiness_status,
        "primary_volume": str(primary_volume_path),
        "model_checkpoint": str(checkpoint_path.resolve()),
        "model_num_outputs": num_outputs,
        "device": str(device),
        "selected_axis": selected_axis,
        "volume_shape": list(volume.shape),
        "candidate_slice_count": len(candidates),
        "num_selected_slices": len(predictions),
        "subject_probability_positive": round(subject_probability, 8),
        "locked_threshold": config.threshold,
        "decision_margin": round(subject_probability - config.threshold, 8),
        "predicted_class_index": predicted_class_index,
        "predicted_class_name": predicted_class_name,
        "runtime_seconds": round(runtime, 4),
        "config": asdict(config),
        "checkpoint_metadata_keys": (
            sorted(checkpoint_metadata.keys())
            if isinstance(checkpoint_metadata, dict)
            else []
        ),
        "slice_probability_summary": {
            "mean": round(float(np.mean(probabilities)), 8),
            "std": round(float(np.std(probabilities)), 8),
            "min": round(float(np.min(probabilities)), 8),
            "median": round(float(np.median(probabilities)), 8),
            "max": round(float(np.max(probabilities)), 8),
        },
        "research_use_only": True,
        "clinical_diagnosis": False,
    }

    write_slice_csv(
        predictions,
        output_dir / "slice_predictions.csv",
    )
    write_json(
        result,
        output_dir / "subject_result.json",
    )
    write_summary(
        result,
        output_dir / "inference_summary.txt",
    )

    return result


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
            "Run locked-threshold subject-level MRI inference after "
            "STEP 24A readiness validation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--readiness-report",
        required=True,
        type=Path,
        help="Path to STEP 24A readiness_report.json.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the trained EfficientNet-B0 checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Inference output directory. Defaults to a sibling folder named "
            "brainfmops_inference."
        ),
    )
    parser.add_argument(
        "--volume",
        type=Path,
        default=None,
        help="Optional explicit MRI volume override.",
    )
    parser.add_argument(
        "--architecture",
        choices=("efficientnet_b0",),
        default="efficientnet_b0",
    )
    parser.add_argument(
        "--threshold",
        type=ratio_0_to_1,
        default=LOCKED_THRESHOLD_DEFAULT,
        help="Locked subject classification threshold.",
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
        "--axis",
        type=int,
        choices=(-1, 0, 1, 2),
        default=-1,
        help="-1 selects the spatial axis with the most slices.",
    )
    parser.add_argument(
        "--image-size",
        type=positive_int,
        default=224,
    )
    parser.add_argument(
        "--min-foreground-ratio",
        type=ratio_0_to_1,
        default=0.05,
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=16,
    )
    parser.add_argument(
        "--positive-class-index",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument(
        "--negative-class-name",
        default="CN",
        help="Name of the negative class.",
    )
    parser.add_argument(
        "--positive-class-name",
        default="AD",
        help="Name of the positive class.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow inference when STEP 24A status is NOT_READY.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    readiness_report = args.readiness_report.expanduser().resolve()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else readiness_report.parent.parent / "brainfmops_inference"
    )

    config = InferenceConfig(
        architecture=args.architecture,
        image_size=args.image_size,
        num_selected_slices=args.num_slices,
        min_foreground_ratio=args.min_foreground_ratio,
        threshold=args.threshold,
        aggregation=args.aggregation,
        positive_class_index=args.positive_class_index,
        negative_class_name=args.negative_class_name,
        positive_class_name=args.positive_class_name,
        seed=args.seed,
    )

    try:
        result = run_subject_inference(
            readiness_report_path=readiness_report,
            checkpoint_path=args.checkpoint.expanduser().resolve(),
            output_dir=output_dir,
            config=config,
            axis=args.axis,
            batch_size=args.batch_size,
            device_name=args.device,
            override_volume=args.volume,
            force=args.force,
        )
    except Exception as exc:
        logging.exception("Subject-level inference failed: %s", exc)
        return 1

    print("\n" + "=" * 78)
    print(f"CASE ID     : {result['case_id']}")
    print(f"PROBABILITY : {result['subject_probability_positive']:.6f}")
    print(f"THRESHOLD   : {result['locked_threshold']:.4f}")
    print(f"PREDICTION  : {result['predicted_class_name']}")
    print(f"DEVICE      : {result['device']}")
    print(f"OUTPUT      : {output_dir}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
