#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18A_independent_test_evaluation.py

BrainFMOps Phase 2
Independent Test Evaluation for EfficientNetB0 (Research Edition)

Purpose
-------
1. Load the frozen best_model.pth checkpoint from STEP 17B.
2. Evaluate once on the independent test manifest from STEP 16B.
3. Generate image-level predictions and metrics.
4. Aggregate predictions to subject level by mean probability.
5. Compute image-level and subject-level:
   - Accuracy
   - Balanced accuracy
   - Precision
   - Sensitivity/Recall
   - Specificity
   - F1-score
   - ROC-AUC
   - PR-AUC
   - Confusion matrix
6. Save ROC, PR, and confusion-matrix figures.
7. Export reproducibility metadata and prediction tables.

Scientific note
---------------
The smoke-test checkpoint must not be used for publication results.
Run STEP 17B full training first, then evaluate the resulting best_model.pth.

Example
-------
python 18A_independent_test_evaluation.py ^
  --dataset-root "data/processed" ^
  --test-manifest "outputs/subject_split_binary_v1/test_manifest.csv" ^
  --checkpoint "outputs/efficientnetb0_binary_v1/best_model.pth" ^
  --output-dir "outputs/independent_test_evaluation_v1" ^
  --batch-size 32 ^
  --num-workers 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0


SCRIPT_VERSION = "1.0.0"

REQUIRED_COLUMNS = {
    "image_id",
    "subject_id",
    "relative_path",
    "class_label",
    "binary_label",
    "binary_label_name",
    "partition",
    "is_valid_image",
}


@dataclass
class EvaluationConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    random_seed: int = 42
    threshold: float = 0.50


@dataclass
class MetricBundle:
    level: str
    n_samples: int
    n_positive: int
    n_negative: int
    threshold: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    sensitivity: float
    specificity: float
    f1: float
    roc_auc: float
    pr_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


def configure_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            output_dir / "evaluation_console.log",
            encoding="utf-8",
        ),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def load_test_manifest(
    path: Path,
    dataset_root: Path,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"Test manifest is missing required columns: {', '.join(missing)}"
        )

    df = df.copy()
    df["partition"] = df["partition"].astype(str).str.strip().str.lower()
    df["binary_label"] = pd.to_numeric(df["binary_label"], errors="coerce")
    df["is_valid_image"] = parse_bool(df["is_valid_image"])

    if (df["partition"] != "test").any():
        bad = sorted(df.loc[df["partition"] != "test", "partition"].unique())
        raise ValueError(
            f"Test manifest contains non-test partitions: {bad}"
        )

    valid = df["binary_label"].isin([0, 1]) & df["is_valid_image"]
    df = df.loc[valid].copy()
    df["binary_label"] = df["binary_label"].astype(int)

    def resolve(relative_path: str) -> str:
        p = Path(str(relative_path))
        full = p if p.is_absolute() else dataset_root / p
        return str(full.resolve())

    df["absolute_path"] = df["relative_path"].map(resolve)
    missing_files = ~df["absolute_path"].map(lambda p: Path(p).is_file())

    if missing_files.any():
        raise FileNotFoundError(
            f"Test manifest references {int(missing_files.sum())} missing image file(s)."
        )

    if df["image_id"].duplicated().any():
        raise ValueError("Test manifest contains duplicate image IDs.")

    return df.reset_index(drop=True)


class TestManifestDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        image_size: int,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        image_path = Path(row["absolute_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        return {
            "image": tensor,
            "label": torch.tensor(
                int(row["binary_label"]),
                dtype=torch.long,
            ),
            "image_id": str(row["image_id"]),
            "subject_id": str(row["subject_id"]),
            "class_label": str(row["class_label"]),
            "binary_label_name": str(row["binary_label_name"]),
            "relative_path": str(row["relative_path"]),
        }


def build_model(dropout: float) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 2),
    )
    return model


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    training_config = checkpoint.get("training_config", {})
    dropout = float(training_config.get("dropout", 0.30))

    model = build_model(dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def specificity(cm: np.ndarray) -> float:
    if cm.shape != (2, 2):
        return float("nan")

    tn, fp, _, _ = cm.ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    level: str,
) -> MetricBundle:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")

    tn, fp, fn, tp = cm.ravel()

    return MetricBundle(
        level=level,
        n_samples=int(len(y_true)),
        n_positive=int((y_true == 1).sum()),
        n_negative=int((y_true == 0).sum()),
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        precision=float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        sensitivity=float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        specificity=specificity(cm),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = (probabilities >= 0.5).long()

            batch_size = images.shape[0]
            for i in range(batch_size):
                rows.append(
                    {
                        "image_id": batch["image_id"][i],
                        "subject_id": batch["subject_id"][i],
                        "original_class": batch["class_label"][i],
                        "binary_label_name": batch["binary_label_name"][i],
                        "true_label": int(batch["label"][i].item()),
                        "predicted_label_at_0_5": int(
                            predictions[i].cpu().item()
                        ),
                        "probability_dementia": float(
                            probabilities[i].cpu().item()
                        ),
                        "relative_path": batch["relative_path"][i],
                    }
                )

            if batch_index % 100 == 0:
                logging.info(
                    "Processed %d batches (%d predictions)",
                    batch_index,
                    len(rows),
                )

    return pd.DataFrame(rows)


def aggregate_subject_predictions(
    image_predictions: pd.DataFrame,
) -> pd.DataFrame:
    label_consistency = (
        image_predictions.groupby("subject_id")["true_label"].nunique()
    )

    inconsistent = label_consistency[label_consistency > 1]
    if len(inconsistent) > 0:
        raise ValueError(
            f"{len(inconsistent)} subject(s) have inconsistent true labels."
        )

    subject_df = (
        image_predictions.groupby("subject_id", as_index=False)
        .agg(
            true_label=("true_label", "first"),
            binary_label_name=("binary_label_name", "first"),
            original_class=("original_class", "first"),
            image_count=("image_id", "count"),
            mean_probability_dementia=(
                "probability_dementia",
                "mean",
            ),
            median_probability_dementia=(
                "probability_dementia",
                "median",
            ),
            minimum_probability_dementia=(
                "probability_dementia",
                "min",
            ),
            maximum_probability_dementia=(
                "probability_dementia",
                "max",
            ),
        )
    )

    subject_df["predicted_label_at_0_5"] = (
        subject_df["mean_probability_dementia"] >= 0.5
    ).astype(int)

    return subject_df


def save_confusion_matrix(
    metrics: MetricBundle,
    output_path: Path,
    title: str,
) -> None:
    cm = np.array(
        [
            [metrics.tn, metrics.fp],
            [metrics.fn, metrics.tp],
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Dementia"])
    ax.set_yticklabels(["Normal", "Dementia"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def write_text_summary(
    path: Path,
    image_metrics: MetricBundle,
    subject_metrics: MetricBundle,
    checkpoint: dict[str, Any],
    config: EvaluationConfig,
) -> None:
    lines = [
        "=" * 92,
        "BrainFMOps Independent Test Evaluation",
        "=" * 92,
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Evaluation threshold: {config.threshold:.4f}",
        f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}",
        "",
        "IMAGE-LEVEL METRICS",
        "-" * 92,
    ]

    for key, value in asdict(image_metrics).items():
        lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "SUBJECT-LEVEL METRICS",
            "-" * 92,
        ]
    )

    for key, value in asdict(subject_metrics).items():
        lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "Interpretation note",
            "-" * 92,
            (
                "Subject-level results are the primary clinically relevant endpoint "
                "because multiple MRI slices originate from the same participant. "
                "Image-level metrics are secondary and should not be interpreted as "
                "independent patient observations."
            ),
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen EfficientNetB0 checkpoint on the independent test set."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/18A_independent_test_evaluation"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    test_manifest_path = args.test_manifest.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir, args.verbose)

    config = EvaluationConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        threshold=args.threshold,
    )

    if not 0 < config.threshold < 1:
        logging.error("Threshold must be between 0 and 1.")
        return 2

    if not dataset_root.is_dir():
        logging.error("Dataset root is invalid: %s", dataset_root)
        return 2

    if not test_manifest_path.is_file():
        logging.error("Test manifest not found: %s", test_manifest_path)
        return 2

    if not checkpoint_path.is_file():
        logging.error("Checkpoint not found: %s", checkpoint_path)
        return 2

    seed_everything(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Device: %s", device)
    logging.info("Loading independent test manifest")

    try:
        test_df = load_test_manifest(
            test_manifest_path,
            dataset_root,
        )
    except Exception as exc:
        logging.exception("Test manifest validation failed: %s", exc)
        return 2

    test_subject_count = test_df["subject_id"].nunique()
    logging.info("Test images: %d", len(test_df))
    logging.info("Test subjects: %d", test_subject_count)

    dataset = TestManifestDataset(
        manifest=test_df,
        image_size=config.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=False,
    )

    try:
        model, checkpoint = load_checkpoint_model(
            checkpoint_path,
            device,
        )
    except Exception as exc:
        logging.exception("Checkpoint loading failed: %s", exc)
        return 2

    logging.info(
        "Loaded checkpoint from epoch %s",
        checkpoint.get("epoch", "unknown"),
    )

    try:
        image_predictions = run_inference(
            model=model,
            loader=loader,
            device=device,
        )
        subject_predictions = aggregate_subject_predictions(
            image_predictions
        )
    except Exception as exc:
        logging.exception("Inference failed: %s", exc)
        return 2

    image_true = image_predictions["true_label"].to_numpy(dtype=int)
    image_prob = image_predictions[
        "probability_dementia"
    ].to_numpy(dtype=float)

    subject_true = subject_predictions["true_label"].to_numpy(dtype=int)
    subject_prob = subject_predictions[
        "mean_probability_dementia"
    ].to_numpy(dtype=float)

    image_metrics = compute_metrics(
        image_true,
        image_prob,
        config.threshold,
        "image",
    )
    subject_metrics = compute_metrics(
        subject_true,
        subject_prob,
        config.threshold,
        "subject",
    )

    image_predictions["predicted_label"] = (
        image_predictions["probability_dementia"]
        >= config.threshold
    ).astype(int)
    image_predictions["correct"] = (
        image_predictions["predicted_label"]
        == image_predictions["true_label"]
    )

    subject_predictions["predicted_label"] = (
        subject_predictions["mean_probability_dementia"]
        >= config.threshold
    ).astype(int)
    subject_predictions["correct"] = (
        subject_predictions["predicted_label"]
        == subject_predictions["true_label"]
    )

    image_predictions_path = output_dir / "image_level_predictions.csv"
    subject_predictions_path = output_dir / "subject_level_predictions.csv"
    metrics_csv_path = output_dir / "test_metrics.csv"
    metrics_json_path = output_dir / "test_metrics.json"
    summary_path = output_dir / "independent_test_summary.txt"

    image_predictions.to_csv(
        image_predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    subject_predictions.to_csv(
        subject_predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    metrics_df = pd.DataFrame(
        [asdict(image_metrics), asdict(subject_metrics)]
    )
    metrics_df.to_csv(
        metrics_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "config": asdict(config),
        "dataset_root": str(dataset_root),
        "test_manifest": str(test_manifest_path),
        "test_manifest_sha256": sha256_file(test_manifest_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_metrics": checkpoint.get(
            "validation_metrics",
            {},
        ),
        "test_images": len(test_df),
        "test_subjects": int(test_subject_count),
        "image_level_metrics": asdict(image_metrics),
        "subject_level_metrics": asdict(subject_metrics),
        "status": "COMPLETED",
    }

    metrics_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_text_summary(
        summary_path,
        image_metrics,
        subject_metrics,
        checkpoint,
        config,
    )

    save_confusion_matrix(
        image_metrics,
        output_dir / "image_level_confusion_matrix.png",
        "Image-level Confusion Matrix",
    )
    save_confusion_matrix(
        subject_metrics,
        output_dir / "subject_level_confusion_matrix.png",
        "Subject-level Confusion Matrix",
    )
    save_roc_curve(
        image_true,
        image_prob,
        output_dir / "image_level_roc_curve.png",
        "Image-level ROC Curve",
    )
    save_roc_curve(
        subject_true,
        subject_prob,
        output_dir / "subject_level_roc_curve.png",
        "Subject-level ROC Curve",
    )
    save_pr_curve(
        image_true,
        image_prob,
        output_dir / "image_level_pr_curve.png",
        "Image-level Precision–Recall Curve",
    )
    save_pr_curve(
        subject_true,
        subject_prob,
        output_dir / "subject_level_pr_curve.png",
        "Subject-level Precision–Recall Curve",
    )

    print()
    print("=" * 92)
    print("BRAINF MOPS INDEPENDENT TEST EVALUATION COMPLETE")
    print("=" * 92)
    print(f"Device                     : {device}")
    print(f"Test images                : {len(test_df):,}")
    print(f"Test subjects              : {test_subject_count}")
    print()
    print("IMAGE LEVEL")
    print(f"Accuracy                   : {image_metrics.accuracy:.4f}")
    print(f"Balanced accuracy          : {image_metrics.balanced_accuracy:.4f}")
    print(f"Sensitivity                : {image_metrics.sensitivity:.4f}")
    print(f"Specificity                : {image_metrics.specificity:.4f}")
    print(f"F1                         : {image_metrics.f1:.4f}")
    print(f"ROC-AUC                    : {image_metrics.roc_auc:.4f}")
    print(f"PR-AUC                     : {image_metrics.pr_auc:.4f}")
    print()
    print("SUBJECT LEVEL")
    print(f"Accuracy                   : {subject_metrics.accuracy:.4f}")
    print(f"Balanced accuracy          : {subject_metrics.balanced_accuracy:.4f}")
    print(f"Sensitivity                : {subject_metrics.sensitivity:.4f}")
    print(f"Specificity                : {subject_metrics.specificity:.4f}")
    print(f"F1                         : {subject_metrics.f1:.4f}")
    print(f"ROC-AUC                    : {subject_metrics.roc_auc:.4f}")
    print(f"PR-AUC                     : {subject_metrics.pr_auc:.4f}")
    print()
    print(f"Output directory           : {output_dir}")
    print(f"Summary report             : {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
