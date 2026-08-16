#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16A_oasis_readiness_checker.py

OASIS Dataset Readiness Checker for BrainFMOps Phase 2

Purpose
-------
1. Inspect OASIS dataset directory structure.
2. Detect MRI files (NIfTI and DICOM).
3. Detect and load clinical metadata (CSV/XLS/XLSX/TSV).
4. Extract subject identifiers from MRI paths and metadata.
5. Check label availability and class distribution.
6. Detect duplicate subjects, missing labels, missing MRI files, and likely leakage risks.
7. Generate a subject-level manifest.
8. Produce a machine-readable readiness report and a human-readable summary.

This script is intentionally configurable because OASIS releases and local
download layouts may differ.

Example
-------
python 16A_oasis_readiness_checker.py ^
    --dataset-root "data/OASIS" ^
    --output-dir "outputs/16A_readiness"

Optional explicit metadata:
python 16A_oasis_readiness_checker.py ^
    --dataset-root "data/OASIS" ^
    --metadata-file "data/OASIS/clinical_data.xlsx" ^
    --subject-column "Subject" ^
    --label-column "CDR" ^
    --output-dir "outputs/16A_readiness"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


SCRIPT_VERSION = "1.0.0"

MRI_EXTENSIONS = (
    ".nii",
    ".nii.gz",
    ".dcm",
    ".ima",
)

METADATA_EXTENSIONS = (
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
)

DEFAULT_SUBJECT_COLUMN_CANDIDATES = (
    "subject",
    "subject_id",
    "subjectid",
    "id",
    "participant_id",
    "participant",
    "oasis_id",
    "mr_id",
)

DEFAULT_LABEL_COLUMN_CANDIDATES = (
    "label",
    "diagnosis",
    "dx",
    "group",
    "class",
    "cdr",
    "clinical_dementia_rating",
    "dementia",
    "status",
)

SUBJECT_PATTERNS = (
    re.compile(r"(OAS1_\d{4}_MR\d+)", re.IGNORECASE),
    re.compile(r"(OAS2_\d{4}_MR\d+)", re.IGNORECASE),
    re.compile(r"(OAS3\d{4})", re.IGNORECASE),
    re.compile(r"(OASIS[-_]?\d+)", re.IGNORECASE),
    re.compile(r"(sub-[A-Za-z0-9]+)", re.IGNORECASE),
)


@dataclass
class ReadinessThresholds:
    minimum_subjects: int = 30
    minimum_labeled_subject_fraction: float = 0.90
    minimum_mri_subject_fraction: float = 0.90
    maximum_duplicate_subject_fraction: float = 0.05
    minimum_classes: int = 2


@dataclass
class ReadinessResult:
    script_version: str
    generated_at_utc: str
    dataset_root: str
    metadata_file: Optional[str]
    subject_column: Optional[str]
    label_column: Optional[str]
    total_mri_files: int
    total_subjects_from_mri: int
    total_metadata_rows: int
    total_subjects_from_metadata: int
    total_manifest_subjects: int
    subjects_with_mri: int
    subjects_with_metadata: int
    subjects_with_labels: int
    subjects_missing_mri: int
    subjects_missing_metadata: int
    subjects_missing_labels: int
    duplicate_metadata_subjects: int
    label_distribution: dict[str, int]
    readiness_status: str
    critical_issues: list[str]
    warnings: list[str]
    recommendations: list[str]
    outputs: dict[str, str]


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_column_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalize_subject_id(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None

    text = text.replace("\\", "/")
    text = Path(text).name

    for pattern in SUBJECT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper().replace("-", "_")

    cleaned = re.sub(r"\.(nii(\.gz)?|dcm|ima)$", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", cleaned).strip("_")
    return cleaned.upper() if cleaned else None


def extract_subject_id_from_path(path: Path, root: Path) -> Optional[str]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path

    candidates = list(relative.parts) + [relative.name]
    for candidate in candidates:
        subject_id = normalize_subject_id(candidate)
        if subject_id:
            for pattern in SUBJECT_PATTERNS:
                if pattern.search(candidate):
                    return subject_id

    # Conservative fallback: use nearest non-generic parent directory.
    generic_names = {
        "anat", "mri", "raw", "rawdata", "dicom", "nifti", "images",
        "scans", "session", "sessions", "t1", "t1w", "mprage",
    }
    for parent in reversed(relative.parents):
        if parent.name and parent.name.lower() not in generic_names:
            fallback = normalize_subject_id(parent.name)
            if fallback:
                return fallback

    return None


def is_mri_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(MRI_EXTENSIONS)


def find_mri_files(dataset_root: Path) -> list[Path]:
    logging.info("Scanning MRI files under: %s", dataset_root)
    files = [p for p in dataset_root.rglob("*") if p.is_file() and is_mri_file(p)]
    files.sort()
    logging.info("Detected %d MRI files", len(files))
    return files


def find_metadata_candidates(dataset_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if any(lower.endswith(ext) for ext in METADATA_EXTENSIONS):
            candidates.append(path)

    def score(path: Path) -> tuple[int, int]:
        name = path.name.lower()
        keywords = (
            "clinical", "demographic", "metadata", "subject", "participant",
            "diagnosis", "cdr", "oasis", "cross-sectional", "longitudinal",
        )
        keyword_score = sum(1 for keyword in keywords if keyword in name)
        return (-keyword_score, len(path.parts))

    candidates.sort(key=score)
    return candidates


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported metadata format: {path}")


def choose_metadata_file(
    candidates: list[Path],
    explicit_file: Optional[Path],
) -> Optional[Path]:
    if explicit_file:
        if not explicit_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {explicit_file}")
        return explicit_file

    for candidate in candidates:
        try:
            df = read_table(candidate)
        except Exception as exc:
            logging.debug("Cannot read metadata candidate %s: %s", candidate, exc)
            continue
        if len(df) > 0 and len(df.columns) >= 2:
            return candidate
    return None


def detect_column(
    columns: Iterable[Any],
    explicit_name: Optional[str],
    candidates: Iterable[str],
) -> Optional[str]:
    original_columns = [str(c) for c in columns]
    normalized_map = {normalize_column_name(c): c for c in original_columns}

    if explicit_name:
        if explicit_name in original_columns:
            return explicit_name
        normalized_explicit = normalize_column_name(explicit_name)
        return normalized_map.get(normalized_explicit)

    for candidate in candidates:
        normalized_candidate = normalize_column_name(candidate)
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    # Partial match, but only for reasonably specific names.
    for normalized, original in normalized_map.items():
        for candidate in candidates:
            normalized_candidate = normalize_column_name(candidate)
            if len(normalized_candidate) >= 4 and normalized_candidate in normalized:
                return original

    return None


def clean_label(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a", "unknown"}:
        return None
    return text


def build_mri_index(
    mri_files: list[Path],
    dataset_root: Path,
) -> tuple[dict[str, list[Path]], list[Path]]:
    subject_to_files: dict[str, list[Path]] = defaultdict(list)
    unidentified_files: list[Path] = []

    for path in mri_files:
        subject_id = extract_subject_id_from_path(path, dataset_root)
        if subject_id:
            subject_to_files[subject_id].append(path)
        else:
            unidentified_files.append(path)

    return dict(subject_to_files), unidentified_files


def prepare_metadata(
    metadata_df: pd.DataFrame,
    subject_column: str,
    label_column: Optional[str],
) -> pd.DataFrame:
    df = metadata_df.copy()
    df["_subject_id"] = df[subject_column].map(normalize_subject_id)

    if label_column:
        df["_label"] = df[label_column].map(clean_label)
    else:
        df["_label"] = None

    return df


def create_manifest(
    dataset_root: Path,
    mri_index: dict[str, list[Path]],
    metadata_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    metadata_subjects: set[str] = set()
    metadata_first_rows: dict[str, pd.Series] = {}
    metadata_row_counts: Counter[str] = Counter()

    if metadata_df is not None and "_subject_id" in metadata_df.columns:
        valid_metadata = metadata_df[metadata_df["_subject_id"].notna()].copy()
        metadata_subjects = set(valid_metadata["_subject_id"].astype(str))
        metadata_row_counts = Counter(valid_metadata["_subject_id"].astype(str))

        for subject_id, group in valid_metadata.groupby("_subject_id", sort=False):
            metadata_first_rows[str(subject_id)] = group.iloc[0]

    all_subjects = sorted(set(mri_index) | metadata_subjects)
    rows: list[dict[str, Any]] = []

    for subject_id in all_subjects:
        files = mri_index.get(subject_id, [])
        metadata_row = metadata_first_rows.get(subject_id)
        label = None
        if metadata_row is not None and "_label" in metadata_row.index:
            label = metadata_row["_label"]

        relative_files = []
        for file_path in files:
            try:
                relative_files.append(str(file_path.relative_to(dataset_root)))
            except ValueError:
                relative_files.append(str(file_path))

        suffix_counts = Counter(
            ".nii.gz" if p.name.lower().endswith(".nii.gz") else p.suffix.lower()
            for p in files
        )

        rows.append(
            {
                "subject_id": subject_id,
                "has_mri": bool(files),
                "mri_file_count": len(files),
                "nifti_file_count": suffix_counts.get(".nii", 0)
                + suffix_counts.get(".nii.gz", 0),
                "dicom_file_count": suffix_counts.get(".dcm", 0)
                + suffix_counts.get(".ima", 0),
                "has_metadata": metadata_row is not None,
                "metadata_row_count": metadata_row_counts.get(subject_id, 0),
                "duplicate_metadata_subject": metadata_row_counts.get(subject_id, 0) > 1,
                "label": label,
                "has_label": label is not None,
                "mri_files": json.dumps(relative_files, ensure_ascii=False),
            }
        )

    return pd.DataFrame(rows)


def evaluate_readiness(
    manifest: pd.DataFrame,
    total_mri_files: int,
    unidentified_mri_files: int,
    metadata_file: Optional[Path],
    subject_column: Optional[str],
    label_column: Optional[str],
    thresholds: ReadinessThresholds,
) -> tuple[str, list[str], list[str], list[str]]:
    critical: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []

    n_subjects = len(manifest)
    n_with_mri = int(manifest["has_mri"].sum()) if n_subjects else 0
    n_with_labels = int(manifest["has_label"].sum()) if n_subjects else 0
    n_duplicates = int(manifest["duplicate_metadata_subject"].sum()) if n_subjects else 0
    n_classes = int(manifest["label"].dropna().nunique()) if n_subjects else 0

    labeled_fraction = n_with_labels / n_subjects if n_subjects else 0.0
    mri_fraction = n_with_mri / n_subjects if n_subjects else 0.0
    duplicate_fraction = n_duplicates / n_subjects if n_subjects else 0.0

    if total_mri_files == 0:
        critical.append("No MRI files were detected.")
        recommendations.append(
            "Verify the dataset root and confirm that NIfTI (.nii/.nii.gz) "
            "or DICOM (.dcm/.ima) files were fully extracted."
        )

    if n_subjects < thresholds.minimum_subjects:
        critical.append(
            f"Only {n_subjects} subjects were identified; "
            f"minimum configured threshold is {thresholds.minimum_subjects}."
        )
        recommendations.append(
            "Check whether the dataset archive was completely downloaded and extracted."
        )

    if metadata_file is None:
        critical.append("No readable clinical metadata file was detected.")
        recommendations.append(
            "Provide --metadata-file explicitly and verify CSV/XLSX formatting."
        )

    if metadata_file is not None and subject_column is None:
        critical.append("A subject identifier column could not be detected in metadata.")
        recommendations.append(
            "Run again with --subject-column using the exact metadata column name."
        )

    if metadata_file is not None and label_column is None:
        critical.append("A classification label column could not be detected.")
        recommendations.append(
            "Run again with --label-column, for example CDR, diagnosis, group, or label."
        )

    if n_subjects and mri_fraction < thresholds.minimum_mri_subject_fraction:
        critical.append(
            f"Only {mri_fraction:.1%} of manifest subjects have MRI files; "
            f"required threshold is {thresholds.minimum_mri_subject_fraction:.1%}."
        )
        recommendations.append(
            "Reconcile subject identifiers between image folders and metadata."
        )

    if n_subjects and labeled_fraction < thresholds.minimum_labeled_subject_fraction:
        critical.append(
            f"Only {labeled_fraction:.1%} of subjects have labels; "
            f"required threshold is {thresholds.minimum_labeled_subject_fraction:.1%}."
        )
        recommendations.append(
            "Resolve missing labels before leakage-free dataset splitting."
        )

    if label_column is not None and n_classes < thresholds.minimum_classes:
        critical.append(
            f"Only {n_classes} distinct labeled class(es) were detected; "
            f"at least {thresholds.minimum_classes} are required."
        )
        recommendations.append(
            "Verify label mapping and define the intended binary or multiclass endpoint."
        )

    if duplicate_fraction > thresholds.maximum_duplicate_subject_fraction:
        critical.append(
            f"Duplicate metadata records affect {duplicate_fraction:.1%} of subjects; "
            f"maximum configured threshold is "
            f"{thresholds.maximum_duplicate_subject_fraction:.1%}."
        )
        recommendations.append(
            "Aggregate repeated visits at subject level or define a visit-selection rule."
        )
    elif n_duplicates > 0:
        warnings.append(
            f"{n_duplicates} subject(s) have multiple metadata rows. "
            "This may represent longitudinal visits and must be handled explicitly."
        )

    if unidentified_mri_files > 0:
        warnings.append(
            f"{unidentified_mri_files} MRI file(s) could not be assigned to a subject."
        )
        recommendations.append(
            "Review unidentified_mri_files.csv and adjust subject-ID parsing if necessary."
        )

    if n_subjects:
        multi_scan_subjects = int((manifest["mri_file_count"] > 1).sum())
        if multi_scan_subjects > 0:
            warnings.append(
                f"{multi_scan_subjects} subject(s) contain multiple MRI files. "
                "Use subject-level splitting and define scan selection or aggregation."
            )

    label_counts = manifest["label"].dropna().value_counts()
    if len(label_counts) >= 2:
        minority = int(label_counts.min())
        majority = int(label_counts.max())
        imbalance_ratio = majority / minority if minority else float("inf")
        if imbalance_ratio >= 3:
            warnings.append(
                f"Class imbalance ratio is approximately {imbalance_ratio:.2f}:1."
            )
            recommendations.append(
                "Use stratified subject-level splitting and report balanced metrics."
            )

    status = "READY"
    if critical:
        status = "NOT_READY"
    elif warnings:
        status = "READY_WITH_WARNINGS"

    if status == "READY":
        recommendations.append(
            "Proceed to STEP 16B: deterministic MRI preprocessing and QC."
        )
    elif status == "READY_WITH_WARNINGS":
        recommendations.append(
            "Resolve or document warnings before preprocessing and model evaluation."
        )

    return status, critical, warnings, recommendations


def write_text_summary(
    output_path: Path,
    result: ReadinessResult,
    manifest: pd.DataFrame,
) -> None:
    lines = [
        "=" * 80,
        "BrainFMOps - OASIS Dataset Readiness Report",
        "=" * 80,
        f"Status: {result.readiness_status}",
        f"Generated (UTC): {result.generated_at_utc}",
        f"Script version: {result.script_version}",
        f"Dataset root: {result.dataset_root}",
        f"Metadata file: {result.metadata_file or 'NOT FOUND'}",
        f"Subject column: {result.subject_column or 'NOT DETECTED'}",
        f"Label column: {result.label_column or 'NOT DETECTED'}",
        "",
        "Dataset counts",
        "-" * 80,
        f"Total MRI files: {result.total_mri_files}",
        f"Subjects identified from MRI: {result.total_subjects_from_mri}",
        f"Metadata rows: {result.total_metadata_rows}",
        f"Subjects identified from metadata: {result.total_subjects_from_metadata}",
        f"Manifest subjects: {result.total_manifest_subjects}",
        f"Subjects with MRI: {result.subjects_with_mri}",
        f"Subjects with metadata: {result.subjects_with_metadata}",
        f"Subjects with labels: {result.subjects_with_labels}",
        f"Subjects missing MRI: {result.subjects_missing_mri}",
        f"Subjects missing metadata: {result.subjects_missing_metadata}",
        f"Subjects missing labels: {result.subjects_missing_labels}",
        f"Subjects with duplicate metadata rows: {result.duplicate_metadata_subjects}",
        "",
        "Label distribution",
        "-" * 80,
    ]

    if result.label_distribution:
        for label, count in result.label_distribution.items():
            lines.append(f"{label}: {count}")
    else:
        lines.append("No usable labels detected.")

    lines.extend(["", "Critical issues", "-" * 80])
    lines.extend(
        [f"- {issue}" for issue in result.critical_issues]
        or ["- None"]
    )

    lines.extend(["", "Warnings", "-" * 80])
    lines.extend([f"- {warning}" for warning in result.warnings] or ["- None"])

    lines.extend(["", "Recommendations", "-" * 80])
    lines.extend(
        [f"- {recommendation}" for recommendation in result.recommendations]
        or ["- None"]
    )

    if not manifest.empty:
        lines.extend(
            [
                "",
                "Readiness fractions",
                "-" * 80,
                f"MRI coverage: {manifest['has_mri'].mean():.2%}",
                f"Metadata coverage: {manifest['has_metadata'].mean():.2%}",
                f"Label coverage: {manifest['has_label'].mean():.2%}",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check OASIS dataset readiness and generate a subject manifest."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing the extracted OASIS dataset.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Optional explicit clinical metadata CSV/TSV/XLS/XLSX file.",
    )
    parser.add_argument(
        "--subject-column",
        type=str,
        default=None,
        help="Optional exact subject identifier column in metadata.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Optional exact target label column in metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/16A_oasis_readiness"),
        help="Directory for manifest and readiness reports.",
    )
    parser.add_argument(
        "--minimum-subjects",
        type=int,
        default=30,
        help="Minimum number of identifiable subjects.",
    )
    parser.add_argument(
        "--minimum-labeled-fraction",
        type=float,
        default=0.90,
        help="Minimum fraction of subjects with labels.",
    )
    parser.add_argument(
        "--minimum-mri-fraction",
        type=float,
        default=0.90,
        help="Minimum fraction of subjects with MRI data.",
    )
    parser.add_argument(
        "--maximum-duplicate-fraction",
        type=float,
        default=0.05,
        help="Maximum fraction of subjects with duplicate metadata rows.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists() or not dataset_root.is_dir():
        logging.error("Dataset root does not exist or is not a directory: %s", dataset_root)
        return 2

    logging.info("BrainFMOps OASIS Readiness Checker v%s", SCRIPT_VERSION)

    mri_files = find_mri_files(dataset_root)
    mri_index, unidentified_files = build_mri_index(mri_files, dataset_root)

    metadata_candidates = find_metadata_candidates(dataset_root)
    metadata_file = choose_metadata_file(metadata_candidates, args.metadata_file)

    metadata_df: Optional[pd.DataFrame] = None
    subject_column: Optional[str] = None
    label_column: Optional[str] = None

    if metadata_file is not None:
        logging.info("Using metadata file: %s", metadata_file)
        metadata_df = read_table(metadata_file)

        subject_column = detect_column(
            metadata_df.columns,
            args.subject_column,
            DEFAULT_SUBJECT_COLUMN_CANDIDATES,
        )
        label_column = detect_column(
            metadata_df.columns,
            args.label_column,
            DEFAULT_LABEL_COLUMN_CANDIDATES,
        )

        if subject_column:
            metadata_df = prepare_metadata(
                metadata_df=metadata_df,
                subject_column=subject_column,
                label_column=label_column,
            )
        else:
            logging.warning("Could not detect a metadata subject column.")
    else:
        logging.warning("No readable metadata file was detected.")

    manifest = create_manifest(
        dataset_root=dataset_root,
        mri_index=mri_index,
        metadata_df=metadata_df if subject_column else None,
    )

    thresholds = ReadinessThresholds(
        minimum_subjects=args.minimum_subjects,
        minimum_labeled_subject_fraction=args.minimum_labeled_fraction,
        minimum_mri_subject_fraction=args.minimum_mri_fraction,
        maximum_duplicate_subject_fraction=args.maximum_duplicate_fraction,
    )

    status, critical, warnings, recommendations = evaluate_readiness(
        manifest=manifest,
        total_mri_files=len(mri_files),
        unidentified_mri_files=len(unidentified_files),
        metadata_file=metadata_file,
        subject_column=subject_column,
        label_column=label_column,
        thresholds=thresholds,
    )

    manifest_path = output_dir / "oasis_subject_manifest.csv"
    report_json_path = output_dir / "oasis_readiness_report.json"
    report_text_path = output_dir / "oasis_readiness_summary.txt"
    unidentified_path = output_dir / "unidentified_mri_files.csv"
    metadata_candidates_path = output_dir / "metadata_candidates.csv"

    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "mri_file": [
                str(path.relative_to(dataset_root))
                if path.is_relative_to(dataset_root)
                else str(path)
                for path in unidentified_files
            ]
        }
    ).to_csv(unidentified_path, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "candidate_path": [str(p) for p in metadata_candidates],
            "selected": [bool(metadata_file and p.resolve() == metadata_file.resolve())
                         for p in metadata_candidates],
        }
    ).to_csv(metadata_candidates_path, index=False, encoding="utf-8-sig")

    total_metadata_rows = len(metadata_df) if metadata_df is not None else 0
    total_metadata_subjects = (
        int(metadata_df["_subject_id"].dropna().nunique())
        if metadata_df is not None and "_subject_id" in metadata_df.columns
        else 0
    )

    label_distribution = (
        {
            str(label): int(count)
            for label, count in manifest["label"].dropna().value_counts().items()
        }
        if not manifest.empty
        else {}
    )

    result = ReadinessResult(
        script_version=SCRIPT_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_root=str(dataset_root),
        metadata_file=str(metadata_file) if metadata_file else None,
        subject_column=subject_column,
        label_column=label_column,
        total_mri_files=len(mri_files),
        total_subjects_from_mri=len(mri_index),
        total_metadata_rows=total_metadata_rows,
        total_subjects_from_metadata=total_metadata_subjects,
        total_manifest_subjects=len(manifest),
        subjects_with_mri=int(manifest["has_mri"].sum()) if not manifest.empty else 0,
        subjects_with_metadata=int(manifest["has_metadata"].sum()) if not manifest.empty else 0,
        subjects_with_labels=int(manifest["has_label"].sum()) if not manifest.empty else 0,
        subjects_missing_mri=int((~manifest["has_mri"]).sum()) if not manifest.empty else 0,
        subjects_missing_metadata=int((~manifest["has_metadata"]).sum())
        if not manifest.empty
        else 0,
        subjects_missing_labels=int((~manifest["has_label"]).sum())
        if not manifest.empty
        else 0,
        duplicate_metadata_subjects=int(
            manifest["duplicate_metadata_subject"].sum()
        ) if not manifest.empty else 0,
        label_distribution=label_distribution,
        readiness_status=status,
        critical_issues=critical,
        warnings=warnings,
        recommendations=recommendations,
        outputs={
            "subject_manifest_csv": str(manifest_path),
            "readiness_report_json": str(report_json_path),
            "readiness_summary_txt": str(report_text_path),
            "unidentified_mri_files_csv": str(unidentified_path),
            "metadata_candidates_csv": str(metadata_candidates_path),
        },
    )

    report_json_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_text_summary(report_text_path, result, manifest)

    print()
    print("=" * 80)
    print(f"OASIS READINESS STATUS: {status}")
    print("=" * 80)
    print(f"Subjects in manifest : {len(manifest)}")
    print(f"MRI files            : {len(mri_files)}")
    print(f"Metadata file        : {metadata_file or 'NOT FOUND'}")
    print(f"Subject column       : {subject_column or 'NOT DETECTED'}")
    print(f"Label column         : {label_column or 'NOT DETECTED'}")
    print(f"Manifest             : {manifest_path}")
    print(f"Summary              : {report_text_path}")
    print(f"JSON report          : {report_json_path}")

    if critical:
        print("\nCritical issues:")
        for issue in critical:
            print(f"  - {issue}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    # Exit codes support automation:
    # 0 = ready, 1 = ready with warnings, 2 = not ready / execution problem
    if status == "READY":
        return 0
    if status == "READY_WITH_WARNINGS":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
