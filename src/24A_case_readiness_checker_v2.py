#!/usr/bin/env python3
"""
24A_case_readiness_checker_v2.py
BrainFMOps-Analyze — STEP 24A (Medical MRI Edition)

Purpose
-------
Validate one subject-level MRI case folder before subject-level inference.

Supported medical image formats
-------------------------------
- Analyze 7.5 / NIfTI pair: .hdr + .img
- NIfTI single file: .nii
- Compressed NIfTI: .nii.gz

Primary target
--------------
OASIS cross-sectional subject folders such as:

OAS1_0001_MR1/
├── RAW/
├── PROCESSED/
├── FSL_SEG/
├── OAS1_0001_MR1.txt
└── OAS1_0001_MR1.xml

Outputs
-------
- readiness_report.json
- readiness_report.csv
- readiness_summary.txt

Status
------
- READY
- READY_WITH_WARNINGS
- NOT_READY

Important
---------
This script is for research pipeline quality control only.
It is not a diagnostic tool or a clinical decision-support system.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np


SCRIPT_VERSION = "2.0.0"
STATUS_READY = "READY"
STATUS_WARNING = "READY_WITH_WARNINGS"
STATUS_NOT_READY = "NOT_READY"

PREFERRED_FOLDER_ORDER = ("PROCESSED", "RAW", "FSL_SEG")


@dataclass(frozen=True)
class MRIThresholds:
    min_dimensions: int = 3
    min_axis_size: int = 32
    min_slices: int = 32
    min_nonzero_ratio: float = 0.01
    max_nonzero_ratio: float = 0.99
    min_intensity_std: float = 1e-6
    min_voxel_spacing_mm: float = 0.20
    max_voxel_spacing_mm: float = 10.0
    max_nan_ratio: float = 0.0
    max_inf_ratio: float = 0.0
    max_affine_condition_number: float = 1e8


@dataclass
class VolumeRecord:
    index: int
    relative_path: str
    source_folder: str
    format: str

    readable: bool = False
    accepted: bool = False
    rejection_reason: str = ""

    paired_header: str = ""
    paired_image: str = ""

    shape: list[int] = field(default_factory=list)
    ndim: Optional[int] = None
    dtype: str = ""
    voxel_spacing_mm: list[float] = field(default_factory=list)
    orientation: list[str] = field(default_factory=list)

    finite_ratio: Optional[float] = None
    nan_ratio: Optional[float] = None
    inf_ratio: Optional[float] = None
    nonzero_ratio: Optional[float] = None

    intensity_min: Optional[float] = None
    intensity_p01: Optional[float] = None
    intensity_mean: Optional[float] = None
    intensity_std: Optional[float] = None
    intensity_p99: Optional[float] = None
    intensity_max: Optional[float] = None

    affine_determinant: Optional[float] = None
    affine_condition_number: Optional[float] = None

    file_size_bytes: int = 0
    sha256: str = ""
    duplicate_of: str = ""

    warnings: list[str] = field(default_factory=list)
    warning_count: int = 0


@dataclass
class ReadinessReport:
    schema_version: str
    script_version: str
    generated_at_utc: str

    case_id: str
    case_dir: str
    output_dir: str

    status: str
    decision_reasons: list[str]
    warnings: list[str]
    errors: list[str]

    folder_validation: dict[str, Any]
    thresholds: dict[str, Any]
    statistics: dict[str, Any]
    volumes: list[dict[str, Any]]


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_case_id(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    return cleaned or "UNKNOWN_CASE"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_nii_gz(path: Path) -> bool:
    return path.name.lower().endswith(".nii.gz")


def medical_format(path: Path) -> Optional[str]:
    lower = path.name.lower()
    if lower.endswith(".nii.gz"):
        return "NIFTI_GZ"
    if lower.endswith(".nii"):
        return "NIFTI"
    if lower.endswith(".hdr"):
        return "ANALYZE_OR_NIFTI_PAIR"
    return None


def discover_medical_volumes(case_dir: Path) -> tuple[list[Path], list[str]]:
    """
    Discover loadable medical image entry points.

    For paired .hdr/.img data, only .hdr is returned because nibabel loads
    the pair from the header path.
    """
    candidates: list[Path] = []
    orphan_pair_files: list[str] = []

    for path in case_dir.rglob("*"):
        if not path.is_file():
            continue

        lower = path.name.lower()

        if lower.endswith(".nii") or lower.endswith(".nii.gz"):
            candidates.append(path)

        elif lower.endswith(".hdr"):
            img_path = path.with_suffix(".img")
            img_gz_path = path.with_suffix(".img.gz")
            if img_path.exists() or img_gz_path.exists():
                candidates.append(path)
            else:
                orphan_pair_files.append(str(path.relative_to(case_dir)))

        elif lower.endswith(".img"):
            hdr_path = path.with_suffix(".hdr")
            if not hdr_path.exists():
                orphan_pair_files.append(str(path.relative_to(case_dir)))

    candidates.sort(
        key=lambda p: (
            PREFERRED_FOLDER_ORDER.index(p.parent.name.upper())
            if p.parent.name.upper() in PREFERRED_FOLDER_ORDER
            else len(PREFERRED_FOLDER_ORDER),
            str(p.relative_to(case_dir)).lower(),
        )
    )
    return candidates, sorted(set(orphan_pair_files))


def source_folder_name(path: Path, case_dir: Path) -> str:
    try:
        relative = path.relative_to(case_dir)
        return relative.parts[0] if len(relative.parts) > 1 else "CASE_ROOT"
    except ValueError:
        return "UNKNOWN"


def paired_paths(path: Path) -> tuple[str, str]:
    if path.suffix.lower() != ".hdr":
        return "", ""

    img_path = path.with_suffix(".img")
    img_gz_path = path.with_suffix(".img.gz")
    chosen_img = img_path if img_path.exists() else img_gz_path

    return str(path), str(chosen_img) if chosen_img.exists() else ""


def safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def sample_array(dataobj: Any, shape: tuple[int, ...], max_voxels: int = 2_000_000) -> np.ndarray:
    """
    Read a deterministic subsample for quality statistics.

    The entire volume is loaded when it is reasonably small. For a larger
    volume, evenly spaced strides are applied to reduce memory usage.
    """
    total = int(np.prod(shape))
    if total <= max_voxels:
        return np.asarray(dataobj, dtype=np.float32)

    stride = max(1, int(round((total / max_voxels) ** (1 / max(len(shape), 1)))))
    slicing = tuple(slice(None, None, stride) for _ in shape)
    return np.asarray(dataobj[slicing], dtype=np.float32)


def inspect_volume(
    index: int,
    path: Path,
    case_dir: Path,
    thresholds: MRIThresholds,
) -> VolumeRecord:
    relative = str(path.relative_to(case_dir))
    fmt = medical_format(path) or "UNKNOWN"

    record = VolumeRecord(
        index=index,
        relative_path=relative,
        source_folder=source_folder_name(path, case_dir),
        format=fmt,
    )

    try:
        record.file_size_bytes = path.stat().st_size
        record.sha256 = sha256_file(path)

        header_path, image_path = paired_paths(path)
        if header_path:
            record.paired_header = str(Path(header_path).relative_to(case_dir))
        if image_path:
            record.paired_image = str(Path(image_path).relative_to(case_dir))
            record.file_size_bytes += Path(image_path).stat().st_size

        image = nib.load(str(path))
        shape = tuple(int(x) for x in image.shape)

        record.shape = list(shape)
        record.ndim = len(shape)
        record.dtype = str(image.get_data_dtype())

        zooms = tuple(float(z) for z in image.header.get_zooms()[: min(3, len(shape))])
        record.voxel_spacing_mm = list(zooms)

        try:
            record.orientation = list(nib.aff2axcodes(image.affine))
        except Exception:
            record.orientation = []

        affine = np.asarray(image.affine, dtype=np.float64)
        record.affine_determinant = safe_float(np.linalg.det(affine[:3, :3]))
        record.affine_condition_number = safe_float(np.linalg.cond(affine))

        data = sample_array(image.dataobj, shape)
        flat = np.asarray(data, dtype=np.float32).reshape(-1)

        finite_mask = np.isfinite(flat)
        nan_mask = np.isnan(flat)
        inf_mask = np.isinf(flat)

        record.finite_ratio = float(np.mean(finite_mask))
        record.nan_ratio = float(np.mean(nan_mask))
        record.inf_ratio = float(np.mean(inf_mask))

        finite = flat[finite_mask]
        if finite.size == 0:
            raise ValueError("Volume contains no finite voxel values.")

        record.nonzero_ratio = float(np.mean(finite != 0))

        record.intensity_min = float(np.min(finite))
        record.intensity_p01 = float(np.percentile(finite, 1))
        record.intensity_mean = float(np.mean(finite))
        record.intensity_std = float(np.std(finite))
        record.intensity_p99 = float(np.percentile(finite, 99))
        record.intensity_max = float(np.max(finite))

        # Dimensional checks
        if record.ndim < thresholds.min_dimensions:
            record.warnings.append(
                f"LOW_DIMENSIONALITY(ndim={record.ndim})"
            )

        spatial_shape = shape[:3]
        if any(axis < thresholds.min_axis_size for axis in spatial_shape):
            record.warnings.append(
                f"SMALL_SPATIAL_AXIS(shape={shape})"
            )

        if len(spatial_shape) >= 3 and min(spatial_shape) < thresholds.min_slices:
            record.warnings.append(
                f"LOW_SLICE_COUNT(min_axis={min(spatial_shape)})"
            )

        # Voxel spacing checks
        if zooms:
            if any(z < thresholds.min_voxel_spacing_mm for z in zooms):
                record.warnings.append(
                    f"VOXEL_SPACING_TOO_SMALL({zooms})"
                )
            if any(z > thresholds.max_voxel_spacing_mm for z in zooms):
                record.warnings.append(
                    f"VOXEL_SPACING_TOO_LARGE({zooms})"
                )
        else:
            record.warnings.append("MISSING_VOXEL_SPACING")

        # Numeric checks
        if record.nan_ratio > thresholds.max_nan_ratio:
            record.warnings.append(
                f"NAN_VALUES(ratio={record.nan_ratio:.6f})"
            )

        if record.inf_ratio > thresholds.max_inf_ratio:
            record.warnings.append(
                f"INFINITE_VALUES(ratio={record.inf_ratio:.6f})"
            )

        if record.nonzero_ratio < thresholds.min_nonzero_ratio:
            record.warnings.append(
                f"TOO_MANY_ZERO_VOXELS(nonzero_ratio={record.nonzero_ratio:.6f})"
            )

        if record.nonzero_ratio > thresholds.max_nonzero_ratio:
            record.warnings.append(
                f"UNUSUALLY_DENSE_VOLUME(nonzero_ratio={record.nonzero_ratio:.6f})"
            )

        if record.intensity_std < thresholds.min_intensity_std:
            record.warnings.append(
                f"NEAR_CONSTANT_INTENSITY(std={record.intensity_std:.8f})"
            )

        if not record.orientation or any(code is None for code in record.orientation):
            record.warnings.append("UNKNOWN_ORIENTATION")

        if (
            record.affine_condition_number is None
            or record.affine_condition_number > thresholds.max_affine_condition_number
        ):
            record.warnings.append(
                f"UNSTABLE_AFFINE(condition={record.affine_condition_number})"
            )

        record.readable = True
        record.accepted = True
        record.warning_count = len(record.warnings)
        return record

    except Exception as exc:
        record.rejection_reason = f"{type(exc).__name__}: {exc}"
        logging.debug("Failed to inspect %s: %s", path, exc)
        return record


def mark_duplicates(records: list[VolumeRecord]) -> dict[str, list[str]]:
    by_hash: dict[str, list[VolumeRecord]] = {}
    for record in records:
        if record.readable and record.sha256:
            by_hash.setdefault(record.sha256, []).append(record)

    groups: dict[str, list[str]] = {}
    group_index = 0

    for members in by_hash.values():
        if len(members) < 2:
            continue

        group_index += 1
        key = f"DUPLICATE_GROUP_{group_index:03d}"
        groups[key] = [member.relative_path for member in members]
        canonical = members[0]

        for duplicate in members[1:]:
            duplicate.duplicate_of = canonical.relative_path
            duplicate.warnings.append(
                f"EXACT_DUPLICATE_OF({canonical.relative_path})"
            )
            duplicate.warning_count = len(duplicate.warnings)

    return groups


def validate_folder_structure(case_dir: Path) -> dict[str, Any]:
    present = {
        folder: (case_dir / folder).is_dir()
        for folder in PREFERRED_FOLDER_ORDER
    }

    metadata_txt = sorted(
        str(p.relative_to(case_dir)) for p in case_dir.glob("*.txt")
    )
    metadata_xml = sorted(
        str(p.relative_to(case_dir)) for p in case_dir.glob("*.xml")
    )

    return {
        "expected_oasis_folders": list(PREFERRED_FOLDER_ORDER),
        "folder_presence": present,
        "has_processed_folder": present["PROCESSED"],
        "has_raw_folder": present["RAW"],
        "has_fsl_seg_folder": present["FSL_SEG"],
        "metadata_txt_files": metadata_txt,
        "metadata_xml_files": metadata_xml,
    }


def choose_primary_volume(records: list[VolumeRecord]) -> Optional[str]:
    readable = [record for record in records if record.readable]
    if not readable:
        return None

    def rank(record: VolumeRecord) -> tuple[int, int, str]:
        folder = record.source_folder.upper()
        folder_rank = (
            PREFERRED_FOLDER_ORDER.index(folder)
            if folder in PREFERRED_FOLDER_ORDER
            else len(PREFERRED_FOLDER_ORDER)
        )
        warning_rank = record.warning_count
        return folder_rank, warning_rank, record.relative_path.lower()

    return sorted(readable, key=rank)[0].relative_path


def summarize_statistics(
    records: list[VolumeRecord],
    orphan_pair_files: list[str],
    duplicate_groups: dict[str, list[str]],
) -> dict[str, Any]:
    total = len(records)
    readable = sum(record.readable for record in records)
    rejected = total - readable
    warned = sum(record.warning_count > 0 for record in records)
    duplicates = sum(bool(record.duplicate_of) for record in records)

    return {
        "total_volume_candidates": total,
        "readable_volumes": readable,
        "rejected_volumes": rejected,
        "readable_ratio": round(readable / total, 6) if total else 0.0,
        "volumes_with_warnings": warned,
        "duplicate_volume_count": duplicates,
        "duplicate_groups": duplicate_groups,
        "orphan_pair_file_count": len(orphan_pair_files),
        "orphan_pair_files": orphan_pair_files,
        "primary_volume": choose_primary_volume(records),
        "format_counts": {
            fmt: sum(record.format == fmt for record in records)
            for fmt in sorted({record.format for record in records})
        },
        "source_folder_counts": {
            folder: sum(record.source_folder == folder for record in records)
            for folder in sorted({record.source_folder for record in records})
        },
    }


def decide_readiness(
    folder_validation: dict[str, Any],
    statistics: dict[str, Any],
    records: list[VolumeRecord],
) -> tuple[str, list[str], list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    total = statistics["total_volume_candidates"]
    readable = statistics["readable_volumes"]
    rejected = statistics["rejected_volumes"]

    if total == 0:
        errors.append(
            "No supported MRI volume was found (.hdr/.img, .nii, or .nii.gz)."
        )

    if readable == 0:
        errors.append("No MRI volume could be loaded successfully.")

    if statistics["primary_volume"] is None:
        errors.append("A primary MRI volume could not be selected.")

    if statistics["orphan_pair_file_count"] > 0:
        errors.append(
            f"{statistics['orphan_pair_file_count']} orphan .hdr/.img pair file(s) "
            "were detected."
        )

    severe_warning_codes = (
        "NAN_VALUES",
        "INFINITE_VALUES",
        "NEAR_CONSTANT_INTENSITY",
        "UNSTABLE_AFFINE",
    )

    severe_records = [
        record.relative_path
        for record in records
        if record.readable
        and any(
            warning.startswith(severe_warning_codes)
            for warning in record.warnings
        )
    ]

    if readable > 0 and len(severe_records) == readable:
        errors.append(
            "All readable volumes contain severe numerical or affine warnings."
        )

    if errors:
        reasons.extend(errors)
        return STATUS_NOT_READY, reasons, warnings, errors

    reasons.append(
        f"{readable}/{total} MRI volume candidate(s) were loaded successfully."
    )
    reasons.append(
        f"Primary volume selected: {statistics['primary_volume']}."
    )

    if not folder_validation["has_processed_folder"]:
        warnings.append(
            "PROCESSED folder is missing; the pipeline may fall back to RAW data."
        )

    if not folder_validation["metadata_txt_files"]:
        warnings.append("No subject-level TXT metadata file was found.")

    if not folder_validation["metadata_xml_files"]:
        warnings.append("No subject-level XML metadata file was found.")

    if rejected > 0:
        warnings.append(f"{rejected} MRI volume candidate(s) were rejected.")

    if statistics["volumes_with_warnings"] > 0:
        warnings.append(
            f"{statistics['volumes_with_warnings']} readable volume(s) "
            "contain quality warnings."
        )

    if statistics["duplicate_volume_count"] > 0:
        warnings.append(
            f"{statistics['duplicate_volume_count']} duplicate volume(s) "
            "were detected."
        )

    if warnings:
        reasons.extend(warnings)
        return STATUS_WARNING, reasons, warnings, errors

    reasons.append(
        "Folder structure, volume integrity, geometry, and intensity checks passed."
    )
    return STATUS_READY, reasons, warnings, errors


def rounded(value: Optional[float], digits: int = 6) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def csv_rows(records: list[VolumeRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = asdict(record)
        for key in (
            "finite_ratio",
            "nan_ratio",
            "inf_ratio",
            "nonzero_ratio",
            "intensity_min",
            "intensity_p01",
            "intensity_mean",
            "intensity_std",
            "intensity_p99",
            "intensity_max",
            "affine_determinant",
            "affine_condition_number",
        ):
            row[key] = rounded(row[key])
        row["shape"] = "x".join(map(str, record.shape))
        row["voxel_spacing_mm"] = "x".join(
            f"{value:.6g}" for value in record.voxel_spacing_mm
        )
        row["orientation"] = "".join(record.orientation)
        row["warnings"] = " | ".join(record.warnings)
        rows.append(row)
    return rows


def write_json(report: ReadinessReport, path: Path) -> None:
    path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(records: list[VolumeRecord], path: Path) -> None:
    rows = csv_rows(records)
    fieldnames = list(rows[0].keys()) if rows else [
        "index",
        "relative_path",
        "source_folder",
        "format",
        "readable",
        "accepted",
        "rejection_reason",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(report: ReadinessReport, path: Path) -> None:
    stats = report.statistics
    folders = report.folder_validation

    lines = [
        "=" * 78,
        "BrainFMOps-Analyze — MRI Case Readiness Summary",
        "=" * 78,
        f"Case ID              : {report.case_id}",
        f"Case directory       : {report.case_dir}",
        f"Generated at (UTC)   : {report.generated_at_utc}",
        f"Script version       : {report.script_version}",
        "",
        f"FINAL STATUS         : {report.status}",
        "",
        "OASIS Folder Structure",
        "-" * 78,
        f"PROCESSED present    : {folders['has_processed_folder']}",
        f"RAW present          : {folders['has_raw_folder']}",
        f"FSL_SEG present      : {folders['has_fsl_seg_folder']}",
        f"TXT metadata files   : {len(folders['metadata_txt_files'])}",
        f"XML metadata files   : {len(folders['metadata_xml_files'])}",
        "",
        "MRI Volume Statistics",
        "-" * 78,
        f"Volume candidates    : {stats['total_volume_candidates']}",
        f"Readable volumes     : {stats['readable_volumes']}",
        f"Rejected volumes     : {stats['rejected_volumes']}",
        f"Readable ratio       : {stats['readable_ratio']:.2%}",
        f"Volumes with warning : {stats['volumes_with_warnings']}",
        f"Duplicate volumes    : {stats['duplicate_volume_count']}",
        f"Orphan pair files    : {stats['orphan_pair_file_count']}",
        f"Primary volume       : {stats['primary_volume']}",
        "",
        "Decision Reasons",
        "-" * 78,
    ]
    lines.extend(f"- {reason}" for reason in report.decision_reasons)

    if report.errors:
        lines.extend(["", "Errors", "-" * 78])
        lines.extend(f"- {item}" for item in report.errors)

    if report.warnings:
        lines.extend(["", "Warnings", "-" * 78])
        lines.extend(f"- {item}" for item in report.warnings)

    lines.extend([
        "",
        "Generated Files",
        "-" * 78,
        "- readiness_report.json",
        "- readiness_report.csv",
        "- readiness_summary.txt",
        "",
        "Research-use notice",
        "-" * 78,
        "This output is a technical data-readiness assessment.",
        "It is not a clinical diagnosis or medical-device decision.",
        "=" * 78,
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_case(
    case_dir: Path,
    output_dir: Path,
    thresholds: MRIThresholds,
) -> ReadinessReport:
    case_dir = case_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    if not case_dir.is_dir():
        raise NotADirectoryError(f"Case path is not a directory: {case_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    case_id = sanitize_case_id(case_dir.name)
    folder_validation = validate_folder_structure(case_dir)
    candidates, orphan_pair_files = discover_medical_volumes(case_dir)

    logging.info("Case ID: %s", case_id)
    logging.info("MRI volume candidates found: %d", len(candidates))

    records = [
        inspect_volume(
            index=index,
            path=path,
            case_dir=case_dir,
            thresholds=thresholds,
        )
        for index, path in enumerate(candidates, start=1)
    ]

    duplicate_groups = mark_duplicates(records)
    statistics = summarize_statistics(
        records=records,
        orphan_pair_files=orphan_pair_files,
        duplicate_groups=duplicate_groups,
    )

    status, reasons, warnings, errors = decide_readiness(
        folder_validation=folder_validation,
        statistics=statistics,
        records=records,
    )

    report = ReadinessReport(
        schema_version="2.0",
        script_version=SCRIPT_VERSION,
        generated_at_utc=utc_now_iso(),
        case_id=case_id,
        case_dir=str(case_dir),
        output_dir=str(output_dir),
        status=status,
        decision_reasons=reasons,
        warnings=warnings,
        errors=errors,
        folder_validation=folder_validation,
        thresholds=asdict(thresholds),
        statistics=statistics,
        volumes=[asdict(record) for record in records],
    )

    write_json(report, output_dir / "readiness_report.json")
    write_csv(records, output_dir / "readiness_report.csv")
    write_summary(report, output_dir / "readiness_summary.txt")
    return report


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def ratio_0_to_1(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("Value must be between 0 and 1.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one OASIS/medical MRI subject folder before "
            "BrainFMOps-Analyze inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--case-dir",
        required=True,
        type=Path,
        help="Path to one subject-level MRI case folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<case-dir>/brainfmops_readiness_v2."
        ),
    )
    parser.add_argument("--min-axis-size", type=positive_int, default=32)
    parser.add_argument("--min-slices", type=positive_int, default=32)
    parser.add_argument("--min-nonzero-ratio", type=ratio_0_to_1, default=0.01)
    parser.add_argument("--max-nonzero-ratio", type=ratio_0_to_1, default=0.99)
    parser.add_argument("--min-voxel-spacing-mm", type=float, default=0.20)
    parser.add_argument("--max-voxel-spacing-mm", type=float, default=10.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.min_nonzero_ratio >= args.max_nonzero_ratio:
        parser.error(
            "--min-nonzero-ratio must be lower than --max-nonzero-ratio."
        )

    if args.min_voxel_spacing_mm >= args.max_voxel_spacing_mm:
        parser.error(
            "--min-voxel-spacing-mm must be lower than "
            "--max-voxel-spacing-mm."
        )

    thresholds = MRIThresholds(
        min_axis_size=args.min_axis_size,
        min_slices=args.min_slices,
        min_nonzero_ratio=args.min_nonzero_ratio,
        max_nonzero_ratio=args.max_nonzero_ratio,
        min_voxel_spacing_mm=args.min_voxel_spacing_mm,
        max_voxel_spacing_mm=args.max_voxel_spacing_mm,
    )

    case_dir: Path = args.case_dir
    output_dir: Path = (
        args.output_dir
        if args.output_dir is not None
        else case_dir / "brainfmops_readiness_v2"
    )

    try:
        report = validate_case(
            case_dir=case_dir,
            output_dir=output_dir,
            thresholds=thresholds,
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        logging.error("%s", exc)
        return 2
    except Exception as exc:
        logging.exception("MRI case readiness validation failed: %s", exc)
        return 3

    print("\n" + "=" * 78)
    print(f"CASE ID : {report.case_id}")
    print(f"STATUS  : {report.status}")
    print(f"PRIMARY : {report.statistics['primary_volume']}")
    print(f"OUTPUT  : {report.output_dir}")
    print("=" * 78)

    if report.status == STATUS_NOT_READY:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
