#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16B_leakage_free_subject_split_binary.py

BrainFMOps Phase 2
Leakage-Free Subject-Level Split Generator (Binary Edition)

Input
-----
subject_manifest.csv
image_manifest.csv

Binary label mapping
--------------------
Non Demented -> 0 (Normal)
Very mild Dementia -> 1 (Dementia)
Mild Dementia -> 1 (Dementia)
Moderate Dementia -> 1 (Dementia)

Core guarantees
---------------
1. Subjects, not images, are split.
2. No subject appears in more than one partition.
3. Original four-class labels are preserved for audit.
4. Binary class proportions are approximately stratified.
5. Output files are deterministic using a fixed random seed.
6. Image-level manifests inherit subject-level partitions.
7. Leakage audit and reproducibility metadata are generated.

Example
-------
python 16B_leakage_free_subject_split_binary.py ^
  --subject-manifest "workspace/dataset_certification/subject_manifest.csv" ^
  --image-manifest "workspace/dataset_certification/image_manifest.csv" ^
  --output-dir "outputs/subject_split_binary_v1"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SCRIPT_VERSION = "1.0.0"

BINARY_MAPPING = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 1,
    "Moderate Dementia": 1,
}

BINARY_NAME_MAPPING = {
    0: "Normal",
    1: "Dementia",
}


@dataclass
class SplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    random_seed: int = 42


@dataclass
class SplitReport:
    script_version: str
    generated_at_utc: str
    subject_manifest: str
    image_manifest: str
    output_dir: str
    random_seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    total_subjects: int
    total_images: int
    train_subjects: int
    validation_subjects: int
    test_subjects: int
    train_images: int
    validation_images: int
    test_images: int
    subject_overlap_count: int
    image_overlap_count: int
    unmapped_subject_count: int
    unmapped_image_count: int
    binary_subject_distribution: dict[str, int]
    partition_subject_distribution: dict[str, dict[str, int]]
    partition_image_distribution: dict[str, dict[str, int]]
    manifest_sha256: dict[str, str]
    status: str
    warnings: list[str]
    outputs: dict[str, str]


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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


def validate_split_fractions(config: SplitConfig) -> None:
    fractions = [
        config.train_fraction,
        config.validation_fraction,
        config.test_fraction,
    ]
    if any(x <= 0 or x >= 1 for x in fractions):
        raise ValueError("All split fractions must be between 0 and 1.")

    total = sum(fractions)
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"Split fractions must sum to 1.0; received {total:.8f}."
        )


def load_and_prepare_subjects(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)

    required = {
        "subject_id",
        "primary_class",
        "cross_class_subject",
        "image_count",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "subject_manifest.csv is missing required columns: "
            + ", ".join(missing)
        )

    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["primary_class"] = df["primary_class"].astype(str).str.strip()

    cross_class_mask = df["cross_class_subject"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    if cross_class_mask.any():
        raise ValueError(
            f"{int(cross_class_mask.sum())} cross-class subject(s) detected. "
            "Resolve label conflicts before splitting."
        )

    df["binary_label"] = df["primary_class"].map(BINARY_MAPPING)
    df["binary_label_name"] = df["binary_label"].map(BINARY_NAME_MAPPING)

    unmapped = df[df["binary_label"].isna()].copy()
    mapped = df[df["binary_label"].notna()].copy()

    mapped["binary_label"] = mapped["binary_label"].astype(int)
    mapped["image_count"] = pd.to_numeric(
        mapped["image_count"], errors="coerce"
    ).fillna(0).astype(int)

    if mapped["subject_id"].duplicated().any():
        duplicates = mapped.loc[
            mapped["subject_id"].duplicated(keep=False), "subject_id"
        ].unique()
        raise ValueError(
            f"Duplicate subject rows detected: {len(duplicates)} unique subjects."
        )

    return mapped, unmapped


def load_and_prepare_images(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "image_id",
        "subject_id",
        "class_label",
        "relative_path",
        "is_valid_image",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "image_manifest.csv is missing required columns: "
            + ", ".join(missing)
        )

    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    df["class_label"] = df["class_label"].astype(str).str.strip()
    df["binary_label"] = df["class_label"].map(BINARY_MAPPING)
    df["binary_label_name"] = df["binary_label"].map(BINARY_NAME_MAPPING)

    return df


def stratified_subject_split(
    subjects: pd.DataFrame,
    config: SplitConfig,
) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        subjects,
        test_size=(1.0 - config.train_fraction),
        random_state=config.random_seed,
        stratify=subjects["binary_label"],
        shuffle=True,
    )

    relative_test_fraction = (
        config.test_fraction
        / (config.validation_fraction + config.test_fraction)
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_fraction,
        random_state=config.random_seed,
        stratify=temp_df["binary_label"],
        shuffle=True,
    )

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df["partition"] = "train"
    validation_df["partition"] = "validation"
    test_df["partition"] = "test"

    split_df = pd.concat(
        [train_df, validation_df, test_df],
        ignore_index=True,
    )

    split_df = split_df.sort_values(
        ["partition", "binary_label", "subject_id"],
        kind="stable",
    ).reset_index(drop=True)

    return split_df


def attach_partitions_to_images(
    images: pd.DataFrame,
    subject_splits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_map = subject_splits[
        ["subject_id", "partition", "binary_label", "binary_label_name"]
    ].rename(
        columns={
            "binary_label": "subject_binary_label",
            "binary_label_name": "subject_binary_label_name",
        }
    )

    merged = images.merge(
        subject_map,
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    unmapped = merged[merged["partition"].isna()].copy()
    mapped = merged[merged["partition"].notna()].copy()

    label_mismatch = (
        mapped["binary_label"].notna()
        & (
            mapped["binary_label"].astype(int)
            != mapped["subject_binary_label"].astype(int)
        )
    )
    if label_mismatch.any():
        bad = mapped[label_mismatch]
        raise ValueError(
            f"{len(bad)} image row(s) disagree with the subject-level binary label."
        )

    mapped["binary_label"] = mapped["subject_binary_label"].astype(int)
    mapped["binary_label_name"] = mapped[
        "subject_binary_label_name"
    ].astype(str)

    mapped = mapped.drop(
        columns=["subject_binary_label", "subject_binary_label_name"]
    )

    return mapped, unmapped


def audit_subject_overlap(
    train_subjects: pd.DataFrame,
    validation_subjects: pd.DataFrame,
    test_subjects: pd.DataFrame,
) -> pd.DataFrame:
    sets = {
        "train": set(train_subjects["subject_id"]),
        "validation": set(validation_subjects["subject_id"]),
        "test": set(test_subjects["subject_id"]),
    }

    rows: list[dict[str, Any]] = []
    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for left, right in pairs:
        overlap = sorted(sets[left] & sets[right])
        for subject_id in overlap:
            rows.append(
                {
                    "subject_id": subject_id,
                    "partition_a": left,
                    "partition_b": right,
                }
            )

    return pd.DataFrame(rows)


def audit_image_overlap(
    train_images: pd.DataFrame,
    validation_images: pd.DataFrame,
    test_images: pd.DataFrame,
) -> pd.DataFrame:
    key = "sha256" if "sha256" in train_images.columns else "image_id"

    sets = {
        "train": set(train_images[key].dropna().astype(str)),
        "validation": set(validation_images[key].dropna().astype(str)),
        "test": set(test_images[key].dropna().astype(str)),
    }

    rows: list[dict[str, Any]] = []
    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for left, right in pairs:
        overlap = sorted(sets[left] & sets[right])
        for value in overlap:
            rows.append(
                {
                    "overlap_key": key,
                    "overlap_value": value,
                    "partition_a": left,
                    "partition_b": right,
                }
            )

    return pd.DataFrame(rows)


def distribution_dict(
    df: pd.DataFrame,
    label_column: str = "binary_label_name",
) -> dict[str, int]:
    if df.empty:
        return {}
    return {
        str(k): int(v)
        for k, v in df[label_column].value_counts().sort_index().items()
    }


def write_summary(
    path: Path,
    report: SplitReport,
) -> None:
    lines = [
        "=" * 88,
        "BrainFMOps Leakage-Free Subject-Level Split Report (Binary Edition)",
        "=" * 88,
        f"Status: {report.status}",
        f"Generated UTC: {report.generated_at_utc}",
        f"Script version: {report.script_version}",
        f"Random seed: {report.random_seed}",
        "",
        "Split configuration",
        "-" * 88,
        f"Train fraction: {report.train_fraction:.2%}",
        f"Validation fraction: {report.validation_fraction:.2%}",
        f"Test fraction: {report.test_fraction:.2%}",
        "",
        "Dataset totals",
        "-" * 88,
        f"Subjects: {report.total_subjects}",
        f"Images: {report.total_images}",
        f"Unmapped subjects: {report.unmapped_subject_count}",
        f"Unmapped images: {report.unmapped_image_count}",
        "",
        "Partition counts",
        "-" * 88,
        f"Train subjects: {report.train_subjects}",
        f"Validation subjects: {report.validation_subjects}",
        f"Test subjects: {report.test_subjects}",
        f"Train images: {report.train_images}",
        f"Validation images: {report.validation_images}",
        f"Test images: {report.test_images}",
        "",
        "Leakage audit",
        "-" * 88,
        f"Subject overlap count: {report.subject_overlap_count}",
        f"Image overlap count: {report.image_overlap_count}",
        "",
        "Overall binary subject distribution",
        "-" * 88,
    ]

    lines.extend(
        [
            f"{label}: {count}"
            for label, count in report.binary_subject_distribution.items()
        ]
        or ["No distribution available."]
    )

    lines.extend(["", "Partition subject distributions", "-" * 88])
    for partition, values in report.partition_subject_distribution.items():
        lines.append(f"{partition}: {values}")

    lines.extend(["", "Partition image distributions", "-" * 88])
    for partition, values in report.partition_image_distribution.items():
        lines.append(f"{partition}: {values}")

    lines.extend(["", "Warnings", "-" * 88])
    lines.extend([f"- {item}" for item in report.warnings] or ["- None"])

    lines.extend(
        [
            "",
            "Leakage-control statement",
            "-" * 88,
            (
                "All partitions were generated at the subject level. Every image inherited "
                "the partition assigned to its subject. No subject was intentionally allowed "
                "to appear in more than one partition."
            ),
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic leakage-free binary train/validation/test splits "
            "from BrainFMOps subject and image manifests."
        )
    )
    parser.add_argument(
        "--subject-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--image-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/16B_subject_split_binary"),
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    subject_manifest_path = args.subject_manifest.expanduser().resolve()
    image_manifest_path = args.image_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SplitConfig(
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )
    validate_split_fractions(config)

    if not subject_manifest_path.exists():
        logging.error("Subject manifest not found: %s", subject_manifest_path)
        return 2
    if not image_manifest_path.exists():
        logging.error("Image manifest not found: %s", image_manifest_path)
        return 2

    logging.info("Loading subject manifest")
    subjects, unmapped_subjects = load_and_prepare_subjects(
        subject_manifest_path
    )

    logging.info("Loading image manifest")
    images = load_and_prepare_images(image_manifest_path)

    logging.info("Generating stratified subject-level split")
    subject_splits = stratified_subject_split(subjects, config)

    logging.info("Assigning image partitions from subject partitions")
    image_splits, unmapped_images = attach_partitions_to_images(
        images,
        subject_splits,
    )

    train_subjects = subject_splits[
        subject_splits["partition"] == "train"
    ].copy()
    validation_subjects = subject_splits[
        subject_splits["partition"] == "validation"
    ].copy()
    test_subjects = subject_splits[
        subject_splits["partition"] == "test"
    ].copy()

    train_images = image_splits[
        image_splits["partition"] == "train"
    ].copy()
    validation_images = image_splits[
        image_splits["partition"] == "validation"
    ].copy()
    test_images = image_splits[
        image_splits["partition"] == "test"
    ].copy()

    subject_overlap = audit_subject_overlap(
        train_subjects,
        validation_subjects,
        test_subjects,
    )
    image_overlap = audit_image_overlap(
        train_images,
        validation_images,
        test_images,
    )

    output_paths = {
        "all_subject_splits_csv": output_dir / "all_subject_splits.csv",
        "train_subjects_csv": output_dir / "train_subjects.csv",
        "validation_subjects_csv": output_dir / "validation_subjects.csv",
        "test_subjects_csv": output_dir / "test_subjects.csv",
        "all_image_splits_csv": output_dir / "all_image_splits.csv",
        "train_manifest_csv": output_dir / "train_manifest.csv",
        "validation_manifest_csv": output_dir / "validation_manifest.csv",
        "test_manifest_csv": output_dir / "test_manifest.csv",
        "subject_overlap_audit_csv": output_dir / "subject_overlap_audit.csv",
        "image_overlap_audit_csv": output_dir / "image_overlap_audit.csv",
        "unmapped_subjects_csv": output_dir / "unmapped_subjects.csv",
        "unmapped_images_csv": output_dir / "unmapped_images.csv",
        "split_report_json": output_dir / "split_report.json",
        "split_summary_txt": output_dir / "split_summary.txt",
    }

    subject_splits.to_csv(
        output_paths["all_subject_splits_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    train_subjects.to_csv(
        output_paths["train_subjects_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    validation_subjects.to_csv(
        output_paths["validation_subjects_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    test_subjects.to_csv(
        output_paths["test_subjects_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    image_splits.to_csv(
        output_paths["all_image_splits_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    train_images.to_csv(
        output_paths["train_manifest_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    validation_images.to_csv(
        output_paths["validation_manifest_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    test_images.to_csv(
        output_paths["test_manifest_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    subject_overlap.to_csv(
        output_paths["subject_overlap_audit_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    image_overlap.to_csv(
        output_paths["image_overlap_audit_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    unmapped_subjects.to_csv(
        output_paths["unmapped_subjects_csv"],
        index=False,
        encoding="utf-8-sig",
    )
    unmapped_images.to_csv(
        output_paths["unmapped_images_csv"],
        index=False,
        encoding="utf-8-sig",
    )

    warnings: list[str] = []

    if len(unmapped_subjects) > 0:
        warnings.append(
            f"{len(unmapped_subjects)} subject row(s) had unmapped original classes."
        )

    if len(unmapped_images) > 0:
        warnings.append(
            f"{len(unmapped_images)} image row(s) could not be assigned to a split."
        )

    if len(subject_overlap) > 0:
        warnings.append(
            f"Subject overlap audit detected {len(subject_overlap)} overlap row(s)."
        )

    if len(image_overlap) > 0:
        warnings.append(
            f"Image overlap audit detected {len(image_overlap)} overlap row(s)."
        )

    status = "PASS"
    if len(subject_overlap) > 0 or len(image_overlap) > 0:
        status = "FAIL"
    elif len(unmapped_subjects) > 0 or len(unmapped_images) > 0:
        status = "PASS_WITH_WARNINGS"

    partition_subject_distribution = {
        "train": distribution_dict(train_subjects),
        "validation": distribution_dict(validation_subjects),
        "test": distribution_dict(test_subjects),
    }
    partition_image_distribution = {
        "train": distribution_dict(train_images),
        "validation": distribution_dict(validation_images),
        "test": distribution_dict(test_images),
    }

    manifest_hashes = {
        "subject_manifest_sha256": sha256_file(subject_manifest_path),
        "image_manifest_sha256": sha256_file(image_manifest_path),
    }

    report = SplitReport(
        script_version=SCRIPT_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        subject_manifest=str(subject_manifest_path),
        image_manifest=str(image_manifest_path),
        output_dir=str(output_dir),
        random_seed=config.random_seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        total_subjects=len(subject_splits),
        total_images=len(image_splits),
        train_subjects=len(train_subjects),
        validation_subjects=len(validation_subjects),
        test_subjects=len(test_subjects),
        train_images=len(train_images),
        validation_images=len(validation_images),
        test_images=len(test_images),
        subject_overlap_count=len(subject_overlap),
        image_overlap_count=len(image_overlap),
        unmapped_subject_count=len(unmapped_subjects),
        unmapped_image_count=len(unmapped_images),
        binary_subject_distribution=distribution_dict(subject_splits),
        partition_subject_distribution=partition_subject_distribution,
        partition_image_distribution=partition_image_distribution,
        manifest_sha256=manifest_hashes,
        status=status,
        warnings=warnings,
        outputs={key: str(value) for key, value in output_paths.items()},
    )

    output_paths["split_report_json"].write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(output_paths["split_summary_txt"], report)

    print()
    print("=" * 88)
    print("BRAINF MOPS LEAKAGE-FREE SUBJECT SPLIT - BINARY EDITION")
    print("=" * 88)
    print(f"Status                 : {status}")
    print(f"Random seed            : {config.random_seed}")
    print(f"Total subjects         : {len(subject_splits)}")
    print(f"Train subjects         : {len(train_subjects)}")
    print(f"Validation subjects    : {len(validation_subjects)}")
    print(f"Test subjects          : {len(test_subjects)}")
    print(f"Total images           : {len(image_splits):,}")
    print(f"Train images           : {len(train_images):,}")
    print(f"Validation images      : {len(validation_images):,}")
    print(f"Test images            : {len(test_images):,}")
    print(f"Subject overlap count  : {len(subject_overlap)}")
    print(f"Image overlap count    : {len(image_overlap)}")
    print(f"Output directory       : {output_dir}")
    print(f"Summary report         : {output_paths['split_summary_txt']}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if status == "PASS":
        return 0
    if status == "PASS_WITH_WARNINGS":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
