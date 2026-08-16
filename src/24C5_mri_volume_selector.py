#!/usr/bin/env python3
"""
24C5_mri_volume_selector.py

BrainFMOps-Analyze — STEP 24C.5
Deterministic MRI Volume/Channel Selector for 4D medical images.

Purpose
-------
When an MRI file contains more than 3 dimensions, do not blindly use
volume[..., 0]. This module evaluates every 3D volume/channel and selects
one primary volume using transparent quality criteria.

Selection criteria
------------------
Each candidate 3D volume is evaluated using:
- finite voxel ratio
- NaN and infinity ratio
- nonzero voxel ratio
- robust intensity standard deviation
- foreground occupancy
- edge-energy sharpness proxy
- saturation ratio
- central-slice information score

The final selection score is a weighted engineering score. It is intended
for deterministic research preprocessing, not clinical quality assessment.

Outputs
-------
- volume_selection_report.json
- volume_selection_candidates.csv
- volume_selection_summary.txt
- selected_volume.nii.gz

Important
---------
The selected volume must subsequently be used by STEP 24B and STEP 24C
via their explicit volume override arguments. This prevents hidden and
irreproducible channel selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np


SCRIPT_VERSION = "1.0.0"


@dataclass(frozen=True)
class SelectorWeights:
    finite_ratio: float = 0.20
    intensity_variation: float = 0.20
    foreground_balance: float = 0.20
    edge_energy: float = 0.20
    central_information: float = 0.15
    saturation_penalty: float = 0.05


@dataclass
class VolumeCandidate:
    volume_index: int
    shape: list[int]
    dtype: str

    finite_ratio: float
    nan_ratio: float
    inf_ratio: float
    nonzero_ratio: float
    foreground_ratio: float
    saturation_ratio: float

    intensity_min: float
    intensity_p01: float
    intensity_mean: float
    intensity_std: float
    intensity_p99: float
    intensity_max: float

    edge_energy: float
    central_information_score: float

    normalized_finite_score: float = 0.0
    normalized_variation_score: float = 0.0
    normalized_foreground_score: float = 0.0
    normalized_edge_score: float = 0.0
    normalized_central_score: float = 0.0
    normalized_saturation_penalty: float = 0.0

    final_selection_score: float = 0.0
    selected: bool = False
    warnings: list[str] = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("nibabel").setLevel(logging.ERROR)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_subject_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Subject result does not exist: {path}")

    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    if "primary_volume" not in data:
        raise ValueError("subject_result.json does not contain primary_volume.")

    return data


def robust_normalize(volume: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("Volume contains no finite voxels.")

    p01 = float(np.percentile(finite, 1))
    p99 = float(np.percentile(finite, 99))

    if p99 <= p01:
        normalized = np.zeros_like(volume, dtype=np.float32)
    else:
        normalized = (
            np.clip(volume, p01, p99) - p01
        ) / (p99 - p01)
        normalized = normalized.astype(np.float32)

    normalized[~np.isfinite(normalized)] = 0.0
    return normalized, {"p01": p01, "p99": p99}


def gradient_energy_3d(volume: np.ndarray) -> float:
    """
    Mean spatial gradient magnitude as a dependency-free sharpness proxy.
    """
    gradients = np.gradient(volume.astype(np.float32))
    magnitude_sq = np.zeros_like(volume, dtype=np.float32)

    for gradient in gradients:
        magnitude_sq += gradient * gradient

    magnitude = np.sqrt(magnitude_sq)
    return float(np.mean(magnitude))


def central_information_score(volume: np.ndarray) -> float:
    """
    Evaluate the middle 20% of slices along each spatial axis.

    This reduces the chance that mostly empty boundary slices dominate
    the selection score.
    """
    scores: list[float] = []

    for axis in range(3):
        axis_size = volume.shape[axis]
        center = axis_size // 2
        radius = max(1, int(round(axis_size * 0.10)))
        start = max(0, center - radius)
        stop = min(axis_size, center + radius + 1)

        slab = np.take(
            volume,
            indices=range(start, stop),
            axis=axis,
        )

        foreground = slab[slab > 0.02]
        if foreground.size == 0:
            scores.append(0.0)
        else:
            occupancy = float(np.mean(slab > 0.02))
            variation = float(np.std(foreground))
            scores.append(occupancy * variation)

    return float(np.mean(scores))


def inspect_candidate(
    volume: np.ndarray,
    volume_index: int,
) -> VolumeCandidate:
    flat = volume.reshape(-1)
    finite_mask = np.isfinite(flat)
    nan_mask = np.isnan(flat)
    inf_mask = np.isinf(flat)

    finite = flat[finite_mask]
    if finite.size == 0:
        raise ValueError(f"Volume {volume_index} contains no finite voxels.")

    normalized, percentiles = robust_normalize(volume)
    normalized_flat = normalized.reshape(-1)

    nonzero_ratio = float(np.mean(finite != 0))
    foreground_ratio = float(np.mean(normalized_flat > 0.02))
    saturation_ratio = float(
        np.mean(
            (normalized_flat <= 0.001)
            | (normalized_flat >= 0.999)
        )
    )

    warnings: list[str] = []
    finite_ratio = float(np.mean(finite_mask))
    nan_ratio = float(np.mean(nan_mask))
    inf_ratio = float(np.mean(inf_mask))

    if finite_ratio < 1.0:
        warnings.append(f"NONFINITE_VOXELS(finite_ratio={finite_ratio:.6f})")
    if foreground_ratio < 0.01:
        warnings.append(
            f"LOW_FOREGROUND(foreground_ratio={foreground_ratio:.6f})"
        )
    if foreground_ratio > 0.95:
        warnings.append(
            f"UNUSUALLY_DENSE(foreground_ratio={foreground_ratio:.6f})"
        )
    if float(np.std(finite)) <= 1e-8:
        warnings.append("NEAR_CONSTANT_INTENSITY")
    if saturation_ratio > 0.98:
        warnings.append(
            f"HIGH_SATURATION(saturation_ratio={saturation_ratio:.6f})"
        )

    return VolumeCandidate(
        volume_index=volume_index,
        shape=list(volume.shape),
        dtype=str(volume.dtype),
        finite_ratio=finite_ratio,
        nan_ratio=nan_ratio,
        inf_ratio=inf_ratio,
        nonzero_ratio=nonzero_ratio,
        foreground_ratio=foreground_ratio,
        saturation_ratio=saturation_ratio,
        intensity_min=float(np.min(finite)),
        intensity_p01=percentiles["p01"],
        intensity_mean=float(np.mean(finite)),
        intensity_std=float(np.std(finite)),
        intensity_p99=percentiles["p99"],
        intensity_max=float(np.max(finite)),
        edge_energy=gradient_energy_3d(normalized),
        central_information_score=central_information_score(normalized),
        warnings=warnings,
    )


def minmax_scores(values: list[float], neutral_if_constant: float = 1.0) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(array))
    maximum = float(np.max(array))

    if maximum - minimum <= 1e-12:
        return [neutral_if_constant] * len(values)

    return [
        float((value - minimum) / (maximum - minimum))
        for value in values
    ]


def foreground_balance_score(ratio: float, target: float = 0.25) -> float:
    """
    Reward plausible foreground occupancy around a neutral engineering target.

    This is not an anatomical ground truth. It only penalizes volumes that are
    almost empty or almost completely filled.
    """
    distance = abs(ratio - target)
    return max(0.0, 1.0 - distance / max(target, 1.0 - target))


def score_candidates(
    candidates: list[VolumeCandidate],
    weights: SelectorWeights,
) -> None:
    variation_scores = minmax_scores(
        [candidate.intensity_std for candidate in candidates]
    )
    edge_scores = minmax_scores(
        [candidate.edge_energy for candidate in candidates]
    )
    central_scores = minmax_scores(
        [candidate.central_information_score for candidate in candidates]
    )

    for index, candidate in enumerate(candidates):
        candidate.normalized_finite_score = candidate.finite_ratio
        candidate.normalized_variation_score = variation_scores[index]
        candidate.normalized_foreground_score = foreground_balance_score(
            candidate.foreground_ratio
        )
        candidate.normalized_edge_score = edge_scores[index]
        candidate.normalized_central_score = central_scores[index]
        candidate.normalized_saturation_penalty = candidate.saturation_ratio

        score = (
            weights.finite_ratio * candidate.normalized_finite_score
            + weights.intensity_variation
            * candidate.normalized_variation_score
            + weights.foreground_balance
            * candidate.normalized_foreground_score
            + weights.edge_energy * candidate.normalized_edge_score
            + weights.central_information
            * candidate.normalized_central_score
            - weights.saturation_penalty
            * candidate.normalized_saturation_penalty
        )

        severe_penalty = 0.0
        if candidate.nan_ratio > 0:
            severe_penalty += 0.25
        if candidate.inf_ratio > 0:
            severe_penalty += 0.25
        if candidate.intensity_std <= 1e-8:
            severe_penalty += 0.50

        candidate.final_selection_score = float(score - severe_penalty)


def select_best_candidate(
    candidates: list[VolumeCandidate],
) -> VolumeCandidate:
    if not candidates:
        raise ValueError("No volume candidates were evaluated.")

    # Deterministic tie-breaking: highest score, then lowest warning count,
    # then lowest volume index.
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate.final_selection_score,
            -len(candidate.warnings),
            -candidate.volume_index,
        ),
    )
    selected.selected = True
    return selected


def write_candidates_csv(
    candidates: list[VolumeCandidate],
    path: Path,
) -> None:
    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        row = asdict(candidate)
        row["shape"] = "x".join(map(str, candidate.shape))
        row["warnings"] = " | ".join(candidate.warnings)
        rows.append(row)

    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    report: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "=" * 78,
        "BrainFMOps-Analyze — MRI Volume Selection Summary",
        "=" * 78,
        f"Case ID                  : {report['case_id']}",
        f"Source MRI               : {report['source_mri']}",
        f"Original shape           : {report['original_shape']}",
        f"Candidate volume count   : {report['candidate_volume_count']}",
        f"Selected volume index    : {report['selected_volume_index']}",
        f"Selected score           : {report['selected_score']:.6f}",
        f"Selected output          : {report['selected_volume_file']}",
        "",
        "Selection Rationale",
        "-" * 78,
        "The selected 3D volume maximized a deterministic engineering score",
        "based on finite voxels, intensity variation, foreground balance,",
        "edge energy, central-slice information, and saturation penalty.",
        "",
        "Important Limitation",
        "-" * 78,
        "This selector does not identify clinical sequence semantics.",
        "When acquisition metadata defines the intended volume explicitly,",
        "metadata-based selection should override this heuristic.",
        "",
        "Research-use notice",
        "-" * 78,
        "This output is for reproducible research preprocessing only.",
        "It is not a clinical diagnosis or medical-device decision.",
        "=" * 78,
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_selector(
    source_mri: Path,
    output_dir: Path,
    case_id: str,
    weights: SelectorWeights,
) -> dict[str, Any]:
    if not source_mri.exists():
        raise FileNotFoundError(f"Source MRI does not exist: {source_mri}")

    image = nib.load(str(source_mri))
    data = np.asarray(image.dataobj, dtype=np.float32)

    if data.ndim < 3:
        raise ValueError(f"MRI must be at least 3D, found shape {data.shape}.")

    if data.ndim == 3:
        logging.info("MRI is already 3D; volume index 0 will be retained.")
        volumes = [data]
    else:
        trailing_shape = data.shape[3:]
        candidate_count = int(np.prod(trailing_shape))
        reshaped = data.reshape(data.shape[:3] + (candidate_count,))
        volumes = [reshaped[..., index] for index in range(candidate_count)]

    logging.info("Evaluating %d candidate volume(s).", len(volumes))

    candidates: list[VolumeCandidate] = []
    for index, volume in enumerate(volumes):
        logging.info("Inspecting volume %d/%d", index + 1, len(volumes))
        candidates.append(inspect_candidate(volume, index))

    score_candidates(candidates, weights)
    selected = select_best_candidate(candidates)

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_volume.nii.gz"

    selected_volume = volumes[selected.volume_index]
    selected_image = nib.Nifti1Image(
        selected_volume.astype(np.float32),
        affine=image.affine,
    )

    # Preserve spatial voxel sizes for the first three axes when available.
    original_zooms = image.header.get_zooms()
    if len(original_zooms) >= 3:
        selected_image.header.set_zooms(tuple(original_zooms[:3]))

    nib.save(selected_image, str(selected_path))

    report = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "case_id": case_id,
        "source_mri": str(source_mri.resolve()),
        "original_shape": list(data.shape),
        "candidate_volume_count": len(candidates),
        "selected_volume_index": selected.volume_index,
        "selected_score": round(selected.final_selection_score, 8),
        "selected_volume_file": str(selected_path.resolve()),
        "weights": asdict(weights),
        "selection_method": (
            "weighted deterministic engineering quality score"
        ),
        "candidate_results": [asdict(candidate) for candidate in candidates],
        "limitations": [
            (
                "The selector does not infer acquisition semantics or diagnose "
                "which sequence is clinically preferred."
            ),
            (
                "Metadata-based sequence selection should take precedence "
                "when explicit acquisition labels are available."
            ),
        ],
        "research_use_only": True,
        "clinical_diagnosis": False,
    }

    (output_dir / "volume_selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_candidates_csv(
        candidates,
        output_dir / "volume_selection_candidates.csv",
    )
    write_summary(
        report,
        output_dir / "volume_selection_summary.txt",
    )

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and select one deterministic 3D volume from a 4D MRI."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--subject-result",
        type=Path,
        help=(
            "Path to STEP 24B subject_result.json; its primary_volume is used."
        ),
    )
    source_group.add_argument(
        "--source-mri",
        type=Path,
        help="Explicit source MRI path.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<subject inference directory>/volume_selection."
        ),
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Optional case identifier when --source-mri is used.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.subject_result is not None:
        subject_result_path = args.subject_result.expanduser().resolve()
        subject_result = load_subject_result(subject_result_path)
        source_mri = Path(subject_result["primary_volume"]).expanduser().resolve()
        case_id = str(subject_result.get("case_id", source_mri.parent.name))
        default_output_dir = subject_result_path.parent / "volume_selection"
    else:
        source_mri = args.source_mri.expanduser().resolve()
        case_id = args.case_id or source_mri.stem
        default_output_dir = source_mri.parent / "volume_selection"

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir
    )

    try:
        report = run_selector(
            source_mri=source_mri,
            output_dir=output_dir,
            case_id=case_id,
            weights=SelectorWeights(),
        )
    except Exception as exc:
        logging.exception("MRI volume selection failed: %s", exc)
        return 1

    print("\n" + "=" * 78)
    print(f"CASE ID        : {report['case_id']}")
    print(f"ORIGINAL SHAPE : {report['original_shape']}")
    print(f"CANDIDATES     : {report['candidate_volume_count']}")
    print(f"SELECTED INDEX : {report['selected_volume_index']}")
    print(f"SCORE          : {report['selected_score']:.6f}")
    print(f"OUTPUT         : {report['selected_volume_file']}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
