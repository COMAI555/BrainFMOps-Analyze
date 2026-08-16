#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17B_efficientnetb0_trainer.py

BrainFMOps Phase 2
EfficientNetB0 Trainer (Research Edition)

This script trains a binary classifier using leakage-free manifests created by
STEP 16B and the manifest-driven loader design validated in STEP 17A.

Key features
------------
1. Subject-level leakage protection through fixed manifests.
2. ImageNet-pretrained EfficientNetB0.
3. Weighted cross-entropy for class imbalance.
4. Reproducible training.
5. Early stopping and learning-rate scheduling.
6. Mixed precision when CUDA is available.
7. Best-checkpoint saving.
8. Epoch-level CSV logging.
9. Validation metrics: accuracy, balanced accuracy, precision, recall,
   specificity, F1, ROC-AUC, PR-AUC, and confusion matrix.
10. Training curves and experiment metadata.
11. Safe CPU smoke-test mode.

Important scientific note
-------------------------
EfficientNetB0 is a pretrained CNN baseline. It is not, by itself, a brain
foundation model. Use it as a strong reproducible baseline in BrainFMOps.

Example full training
---------------------
python 17B_efficientnetb0_trainer.py ^
  --dataset-root "data/processed" ^
  --split-dir "outputs/subject_split_binary_v1" ^
  --output-dir "outputs/efficientnetb0_binary_v1" ^
  --epochs 20 ^
  --batch-size 32 ^
  --num-workers 0

Example CPU smoke test
----------------------
python 17B_efficientnetb0_trainer.py ^
  --dataset-root "data/processed" ^
  --split-dir "outputs/subject_split_binary_v1" ^
  --output-dir "outputs/efficientnetb0_smoke_test" ^
  --epochs 1 ^
  --batch-size 16 ^
  --num-workers 0 ^
  --max-train-batches 20 ^
  --max-validation-batches 10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


SCRIPT_VERSION = "1.0.0"

EXPECTED_MANIFESTS = {
    "train": "train_manifest.csv",
    "validation": "validation_manifest.csv",
}

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
class TrainingConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.30
    patience: int = 5
    min_delta: float = 1e-4
    random_seed: int = 42
    freeze_backbone_epochs: int = 2
    use_class_weights: bool = True
    use_amp: bool = True
    max_train_batches: int = 0
    max_validation_batches: int = 0
    require_cuda: bool = False
    performance_mode: bool = False
    channels_last: bool = True
    prefetch_factor: int = 2


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_root: Path,
        transform: transforms.Compose,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.dataset_root = dataset_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        path = Path(str(row["relative_path"]))
        image_path = path if path.is_absolute() else self.dataset_root / path

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        return {
            "image": tensor,
            "label": torch.tensor(int(row["binary_label"]), dtype=torch.long),
            "subject_id": str(row["subject_id"]),
            "image_id": str(row["image_id"]),
            "relative_path": str(row["relative_path"]),
        }


def configure_logging(output_dir: Path, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_console.log"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
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

    # Reproducible mode is the default. Performance mode is configured in main().
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def load_manifest(path: Path, partition: str, dataset_root: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {', '.join(missing)}"
        )

    df = df.copy()
    df["partition"] = df["partition"].astype(str).str.strip().str.lower()
    df["binary_label"] = pd.to_numeric(df["binary_label"], errors="coerce")
    df["is_valid_image"] = parse_bool(df["is_valid_image"])

    wrong_partition = df["partition"] != partition
    if wrong_partition.any():
        raise ValueError(
            f"{path.name} contains rows outside partition '{partition}'."
        )

    valid_mask = df["binary_label"].isin([0, 1]) & df["is_valid_image"]
    df = df.loc[valid_mask].copy()
    df["binary_label"] = df["binary_label"].astype(int)

    def exists(relative_path: str) -> bool:
        p = Path(str(relative_path))
        full = p if p.is_absolute() else dataset_root / p
        return full.is_file()

    file_exists = df["relative_path"].map(exists)
    if not file_exists.all():
        missing_count = int((~file_exists).sum())
        raise FileNotFoundError(
            f"{path.name} references {missing_count} missing image file(s)."
        )

    if df["image_id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate image IDs.")

    return df.reset_index(drop=True)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    return train_transform, evaluation_transform


def build_dataloaders(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    dataset_root: Path,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    train_transform, validation_transform = build_transforms(config.image_size)

    train_dataset = ManifestDataset(train_df, dataset_root, train_transform)
    validation_dataset = ManifestDataset(
        validation_df,
        dataset_root,
        validation_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(config.random_seed)

    pin_memory = torch.cuda.is_available()
    persistent_workers = config.num_workers > 0

    loader_common = {
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": seed_worker,
        "drop_last": False,
    }
    if config.num_workers > 0:
        loader_common["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        **loader_common,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_common,
    )

    return train_loader, validation_loader


def build_model(dropout: float, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 2),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def calculate_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df["binary_label"].value_counts().sort_index()
    if set(counts.index.tolist()) != {0, 1}:
        raise ValueError("Training data must contain both binary classes.")

    total = counts.sum()
    weights = total / (2 * counts)
    return torch.tensor(
        [float(weights.loc[0]), float(weights.loc[1])],
        dtype=torch.float32,
    )


def specificity_from_confusion(cm: np.ndarray) -> float:
    if cm.shape != (2, 2):
        return float("nan")
    tn, fp, fn, tp = cm.ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    loss: float,
) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")

    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_sensitivity": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "specificity": specificity_from_confusion(cm),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: GradScaler,
    use_amp: bool,
    max_batches: int,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    sample_count = 0
    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob: list[float] = []

    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        labels = batch["label"].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with autocast(
                device_type=device.type,
                enabled=use_amp and device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = torch.argmax(logits, dim=1)

        batch_size = labels.shape[0]
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size

        all_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        all_pred.extend(predictions.detach().cpu().numpy().astype(int).tolist())
        all_prob.extend(
            probabilities.detach().cpu().numpy().astype(float).tolist()
        )

    if sample_count == 0:
        raise RuntimeError("No samples were processed in the epoch.")

    average_loss = running_loss / sample_count
    return compute_metrics(
        y_true=np.asarray(all_true),
        y_pred=np.asarray(all_pred),
        y_prob=np.asarray(all_prob),
        loss=average_loss,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    validation_metrics: dict[str, Any],
    config: TrainingConfig,
    class_weights: torch.Tensor,
) -> None:
    payload = {
        "script_version": SCRIPT_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "validation_metrics": validation_metrics,
        "training_config": asdict(config),
        "class_weights": class_weights.cpu().tolist(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, path)


def save_history_plot(history: pd.DataFrame, output_dir: Path) -> None:
    if history.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train loss")
    plt.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        history["epoch"],
        history["train_balanced_accuracy"],
        label="Train balanced accuracy",
    )
    plt.plot(
        history["epoch"],
        history["validation_balanced_accuracy"],
        label="Validation balanced accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Balanced accuracy")
    plt.title("Balanced Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "balanced_accuracy_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        history["epoch"],
        history["validation_roc_auc"],
        label="Validation ROC-AUC",
    )
    plt.plot(
        history["epoch"],
        history["validation_f1"],
        label="Validation F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation ROC-AUC and F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "validation_metrics_curve.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a reproducible EfficientNetB0 binary MRI classifier."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/17B_efficientnetb0"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Stop with an error instead of silently falling back to CPU.",
    )
    parser.add_argument(
        "--performance-mode",
        action="store_true",
        help="Enable cuDNN benchmarking and TF32 for faster NVIDIA GPU training; less strictly deterministic.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir, args.verbose)

    if not dataset_root.is_dir():
        logging.error("Dataset root is invalid: %s", dataset_root)
        return 2

    if not split_dir.is_dir():
        logging.error("Split directory is invalid: %s", split_dir)
        return 2

    config = TrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        random_seed=args.random_seed,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        use_class_weights=not args.disable_class_weights,
        use_amp=not args.disable_amp,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        require_cuda=args.require_cuda,
        performance_mode=args.performance_mode,
        prefetch_factor=args.prefetch_factor,
    )

    seed_everything(config.random_seed)
    if config.performance_mode and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config_path = output_dir / "training_config.json"
    config_path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_manifest_path = split_dir / EXPECTED_MANIFESTS["train"]
    validation_manifest_path = split_dir / EXPECTED_MANIFESTS["validation"]

    try:
        train_df = load_manifest(
            train_manifest_path,
            "train",
            dataset_root,
        )
        validation_df = load_manifest(
            validation_manifest_path,
            "validation",
            dataset_root,
        )
    except Exception as exc:
        logging.exception("Manifest loading failed: %s", exc)
        return 2

    train_subjects = set(train_df["subject_id"].astype(str))
    validation_subjects = set(validation_df["subject_id"].astype(str))
    overlap = train_subjects & validation_subjects
    if overlap:
        logging.error(
            "Subject leakage detected between train and validation: %d subjects",
            len(overlap),
        )
        return 2

    cuda_available = torch.cuda.is_available()
    if config.require_cuda and not cuda_available:
        logging.error("CUDA was required but torch.cuda.is_available() returned False.")
        logging.error("Installed torch version: %s", torch.__version__)
        logging.error("PyTorch CUDA runtime: %s", torch.version.cuda)
        logging.error("Reinstall a CUDA-enabled PyTorch build and verify the NVIDIA driver.")
        return 2

    device = torch.device("cuda:0" if cuda_available else "cpu")
    logging.info("Device: %s", device)
    logging.info("PyTorch version: %s", torch.__version__)
    logging.info("PyTorch CUDA runtime: %s", torch.version.cuda)
    if cuda_available:
        logging.info("GPU: %s", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        logging.info("GPU memory: %.2f GB", props.total_memory / (1024 ** 3))
        logging.info("Performance mode: %s", config.performance_mode)
    logging.info("Train images: %d", len(train_df))
    logging.info("Validation images: %d", len(validation_df))
    logging.info("Train subjects: %d", len(train_subjects))
    logging.info("Validation subjects: %d", len(validation_subjects))

    train_loader, validation_loader = build_dataloaders(
        train_df,
        validation_df,
        dataset_root,
        config,
    )

    try:
        model = build_model(
            dropout=config.dropout,
            pretrained=not args.no_pretrained,
        ).to(device)
        if device.type == "cuda" and config.channels_last:
            model = model.to(memory_format=torch.channels_last)
    except Exception as exc:
        logging.exception(
            "Model initialization failed. Internet may be required once to "
            "download pretrained weights: %s",
            exc,
        )
        return 2

    if config.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
        logging.info(
            "Backbone frozen for first %d epoch(s)",
            config.freeze_backbone_epochs,
        )

    class_weights = calculate_class_weights(train_df)
    logging.info(
        "Class weights: Normal=%.4f, Dementia=%.4f",
        class_weights[0].item(),
        class_weights[1].item(),
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
        if config.use_class_weights
        else None
    )

    optimizer = AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    scaler = GradScaler(
        device=device.type,
        enabled=config.use_amp and device.type == "cuda",
    )

    history_rows: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    best_checkpoint_path = output_dir / "best_model.pth"
    last_checkpoint_path = output_dir / "last_model.pth"
    history_csv_path = output_dir / "training_history.csv"

    training_start = time.time()

    for epoch in range(1, config.epochs + 1):
        if (
            config.freeze_backbone_epochs > 0
            and epoch == config.freeze_backbone_epochs + 1
        ):
            set_backbone_trainable(model, True)
            optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate * 0.1,
                weight_decay=config.weight_decay,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
            )
            logging.info("Backbone unfrozen at epoch %d", epoch)

        epoch_start = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=config.use_amp,
            max_batches=config.max_train_batches,
        )

        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=scaler,
            use_amp=config.use_amp,
            max_batches=config.max_validation_batches,
        )

        validation_score = validation_metrics["roc_auc"]
        if math.isnan(validation_score):
            validation_score = validation_metrics["balanced_accuracy"]

        scheduler.step(validation_score)

        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "epoch_seconds": elapsed,
        }
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update(
            {f"validation_{k}": v for k, v in validation_metrics.items()}
        )
        history_rows.append(row)

        pd.DataFrame(history_rows).to_csv(
            history_csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        logging.info(
            "Epoch %d/%d | train loss %.4f | val loss %.4f | "
            "val BA %.4f | val F1 %.4f | val AUC %.4f | %.1fs",
            epoch,
            config.epochs,
            train_metrics["loss"],
            validation_metrics["loss"],
            validation_metrics["balanced_accuracy"],
            validation_metrics["f1"],
            validation_metrics["roc_auc"],
            elapsed,
        )

        improved = validation_score > best_score + config.min_delta
        if improved:
            best_score = validation_score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                validation_metrics,
                config,
                class_weights,
            )
            logging.info(
                "New best checkpoint saved at epoch %d (score %.5f)",
                epoch,
                validation_score,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            validation_metrics,
            config,
            class_weights,
        )

        if epochs_without_improvement >= config.patience:
            logging.info(
                "Early stopping triggered after %d epoch(s) without improvement",
                epochs_without_improvement,
            )
            break

    total_training_seconds = time.time() - training_start
    history_df = pd.DataFrame(history_rows)
    save_history_plot(history_df, output_dir)

    summary = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "config": asdict(config),
        "pretrained_weights": not args.no_pretrained,
        "train_manifest": str(train_manifest_path),
        "validation_manifest": str(validation_manifest_path),
        "train_manifest_sha256": sha256_file(train_manifest_path),
        "validation_manifest_sha256": sha256_file(
            validation_manifest_path
        ),
        "train_images": len(train_df),
        "validation_images": len(validation_df),
        "train_subjects": len(train_subjects),
        "validation_subjects": len(validation_subjects),
        "subject_overlap_count": len(overlap),
        "class_weights": {
            "Normal_0": float(class_weights[0].item()),
            "Dementia_1": float(class_weights[1].item()),
        },
        "epochs_completed": len(history_rows),
        "best_epoch": best_epoch,
        "best_validation_score": float(best_score),
        "total_training_seconds": total_training_seconds,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "status": "COMPLETED",
    }

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("BRAINF MOPS EFFICIENTNETB0 TRAINING COMPLETE")
    print("=" * 90)
    print(f"Device                 : {device}")
    print(f"Epochs completed       : {len(history_rows)}")
    print(f"Best epoch             : {best_epoch}")
    print(f"Best validation score  : {best_score:.5f}")
    print(f"Training time          : {total_training_seconds / 60:.2f} minutes")
    print(f"Best checkpoint        : {best_checkpoint_path}")
    print(f"Training history       : {history_csv_path}")
    print(f"Output directory       : {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
