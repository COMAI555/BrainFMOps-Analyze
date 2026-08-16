#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19A_gradcam_professional.py

BrainFMOps Phase 2
Professional Grad-CAM Generator for EfficientNetB0

Purpose
-------
1. Load the trained EfficientNetB0 checkpoint.
2. Read the independent test manifest.
3. Generate predictions for all test images.
4. Select representative TP, TN, FP, and FN cases.
5. Generate:
   - Original image
   - Grad-CAM heatmap
   - Overlay image
   - Publication-ready composite figure
6. Export quantitative explainability statistics:
   - Activation ratio
   - Activation area in pixels
   - Heatmap mean
   - Heatmap maximum
   - Activation centroid
   - Prediction confidence
   - Predictive entropy
7. Produce CSV, JSON, TXT, and PNG outputs.

Scientific caution
------------------
Grad-CAM is a post-hoc explanation method. It indicates image regions associated
with the model decision but does not prove causal or clinically valid reasoning.

Example
-------
python 19A_gradcam_professional.py ^
  --dataset-root "data/processed" ^
  --test-manifest "outputs/subject_split_binary_v1/test_manifest.csv" ^
  --checkpoint "outputs/efficientnetb0_binary_gpu_v1/best_model.pth" ^
  --output-dir "outputs/gradcam_professional_v1" ^
  --threshold 0.32 ^
  --cases-per-category 10 ^
  --batch-size 32 ^
  --num-workers 4
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
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import Tensor, nn
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
    "partition",
    "is_valid_image",
}

CATEGORY_ORDER = ["TN", "TP", "FP", "FN"]


@dataclass
class GradCAMConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    random_seed: int = 42
    threshold: float = 0.32
    cases_per_category: int = 10
    activation_threshold: float = 0.60
    overlay_alpha: float = 0.45
    publication_dpi: int = 300


class ManifestDataset(Dataset):
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
        path = Path(row["absolute_path"])

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)

        return {
            "image": tensor,
            "label": torch.tensor(
                int(row["binary_label"]),
                dtype=torch.long,
            ),
            "image_id": str(row["image_id"]),
            "subject_id": str(row["subject_id"]),
            "class_label": str(row["class_label"]),
            "relative_path": str(row["relative_path"]),
            "absolute_path": str(row["absolute_path"]),
        }


class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Tensor,
    ) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[Any, ...],
        grad_output: tuple[Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: Tensor,
        target_class: int,
    ) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        score = logits[:, target_class].sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations or gradients."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy().astype(np.float32)
        minimum = float(cam.min())
        maximum = float(cam.max())

        if maximum > minimum:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        return cam

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def configure_logging(
    output_dir: Path,
    verbose: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                output_dir / "gradcam_console.log",
                encoding="utf-8",
            ),
        ],
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


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
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


def load_manifest(
    path: Path,
    dataset_root: Path,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Test manifest is missing required columns: "
            + ", ".join(missing)
        )

    df = df.copy()
    df["partition"] = (
        df["partition"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df["binary_label"] = pd.to_numeric(
        df["binary_label"],
        errors="coerce",
    )
    df["is_valid_image"] = parse_bool(df["is_valid_image"])

    if (df["partition"] != "test").any():
        raise ValueError(
            "Test manifest contains rows outside the test partition."
        )

    df = df[
        df["binary_label"].isin([0, 1])
        & df["is_valid_image"]
    ].copy()

    df["binary_label"] = df["binary_label"].astype(int)

    def resolve(relative_path: str) -> str:
        candidate = Path(str(relative_path))
        full_path = (
            candidate
            if candidate.is_absolute()
            else dataset_root / candidate
        )
        return str(full_path.resolve())

    df["absolute_path"] = df["relative_path"].map(resolve)

    missing_files = ~df["absolute_path"].map(
        lambda value: Path(value).is_file()
    )
    if missing_files.any():
        raise FileNotFoundError(
            f"{int(missing_files.sum())} test image file(s) are missing."
        )

    if df["image_id"].duplicated().any():
        raise ValueError(
            "Duplicate image IDs were detected in the test manifest."
        )

    return df.reset_index(drop=True)


def build_model(dropout: float) -> nn.Module:
    model = efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(input_features, 2),
    )
    return model


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    dropout = float(
        checkpoint.get(
            "training_config",
            {},
        ).get(
            "dropout",
            0.30,
        )
    )

    model = build_model(dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def entropy_binary(probability: float) -> float:
    epsilon = 1e-12
    p = min(max(probability, epsilon), 1.0 - epsilon)
    return float(
        -p * math.log2(p)
        - (1.0 - p) * math.log2(1.0 - p)
    )


def run_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(
                device,
                non_blocking=True,
            )
            logits = model(images)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            for index in range(images.shape[0]):
                probability = float(
                    probabilities[index]
                    .detach()
                    .cpu()
                    .item()
                )
                true_label = int(
                    batch["label"][index].item()
                )
                predicted_label = int(
                    probability >= threshold
                )

                if true_label == 0 and predicted_label == 0:
                    category = "TN"
                elif true_label == 1 and predicted_label == 1:
                    category = "TP"
                elif true_label == 0 and predicted_label == 1:
                    category = "FP"
                else:
                    category = "FN"

                confidence = (
                    probability
                    if predicted_label == 1
                    else 1.0 - probability
                )

                rows.append(
                    {
                        "image_id": batch["image_id"][index],
                        "subject_id": batch["subject_id"][index],
                        "original_class": batch["class_label"][index],
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "probability_dementia": probability,
                        "prediction_confidence": confidence,
                        "predictive_entropy": entropy_binary(
                            probability
                        ),
                        "category": category,
                        "correct": bool(
                            true_label == predicted_label
                        ),
                        "relative_path": batch[
                            "relative_path"
                        ][index],
                        "absolute_path": batch[
                            "absolute_path"
                        ][index],
                    }
                )

            if batch_index % 100 == 0:
                logging.info(
                    "Processed %d batches (%d predictions)",
                    batch_index,
                    len(rows),
                )

    return pd.DataFrame(rows)


def select_representative_cases(
    predictions: pd.DataFrame,
    cases_per_category: int,
) -> pd.DataFrame:
    selected_frames: list[pd.DataFrame] = []

    for category in CATEGORY_ORDER:
        subset = predictions[
            predictions["category"] == category
        ].copy()

        if subset.empty:
            logging.warning(
                "No cases available for category %s",
                category,
            )
            continue

        if category in {"TP", "TN"}:
            subset = subset.sort_values(
                [
                    "prediction_confidence",
                    "predictive_entropy",
                ],
                ascending=[False, True],
            )
        else:
            subset = subset.sort_values(
                [
                    "prediction_confidence",
                    "predictive_entropy",
                ],
                ascending=[False, True],
            )

        # Avoid selecting many slices from the same subject.
        subset = subset.drop_duplicates(
            subset=["subject_id"],
            keep="first",
        )

        selected = subset.head(cases_per_category).copy()
        selected["case_rank"] = np.arange(
            1,
            len(selected) + 1,
        )
        selected_frames.append(selected)

    if not selected_frames:
        raise RuntimeError(
            "No Grad-CAM cases could be selected."
        )

    selected_cases = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    category_rank = {
        category: index
        for index, category in enumerate(CATEGORY_ORDER)
    }
    selected_cases["_category_rank"] = (
        selected_cases["category"].map(category_rank)
    )

    return (
        selected_cases.sort_values(
            ["_category_rank", "case_rank"]
        )
        .drop(columns="_category_rank")
        .reset_index(drop=True)
    )


def load_original_resized(
    path: Path,
    image_size: int,
) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb = rgb.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )

    return np.asarray(rgb).astype(np.float32) / 255.0


def prepare_input_tensor(
    path: Path,
    image_size: int,
    device: torch.device,
) -> Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        tensor = transform(rgb)

    return tensor.unsqueeze(0).to(device)


def calculate_activation_statistics(
    heatmap: np.ndarray,
    activation_threshold: float,
) -> dict[str, float]:
    mask = heatmap >= activation_threshold
    activation_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    activation_ratio = (
        activation_pixels / total_pixels
        if total_pixels
        else float("nan")
    )

    if activation_pixels > 0:
        y_coordinates, x_coordinates = np.nonzero(mask)
        centroid_x = float(x_coordinates.mean())
        centroid_y = float(y_coordinates.mean())
    else:
        centroid_x = float("nan")
        centroid_y = float("nan")

    return {
        "activation_area_pixels": activation_pixels,
        "activation_ratio": float(activation_ratio),
        "heatmap_mean": float(np.mean(heatmap)),
        "heatmap_maximum": float(np.max(heatmap)),
        "heatmap_standard_deviation": float(
            np.std(heatmap)
        ),
        "activation_centroid_x": centroid_x,
        "activation_centroid_y": centroid_y,
    }


def save_case_outputs(
    row: pd.Series,
    original: np.ndarray,
    heatmap: np.ndarray,
    output_dir: Path,
    config: GradCAMConfig,
) -> dict[str, str]:
    category = str(row["category"])
    rank = int(row["case_rank"])
    subject_id = str(row["subject_id"])
    image_id = str(row["image_id"])

    safe_subject = subject_id.replace("/", "_")
    safe_image = image_id.replace("/", "_")
    stem = (
        f"{category}_{rank:02d}_"
        f"{safe_subject}_{safe_image}"
    )

    category_dir = output_dir / category
    original_dir = category_dir / "original"
    heatmap_dir = category_dir / "heatmaps"
    overlay_dir = category_dir / "overlays"
    composite_dir = category_dir / "composites"

    for directory in (
        original_dir,
        heatmap_dir,
        overlay_dir,
        composite_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    original_path = original_dir / f"{stem}_original.png"
    heatmap_path = heatmap_dir / f"{stem}_heatmap.png"
    overlay_path = overlay_dir / f"{stem}_overlay.png"
    composite_path = (
        composite_dir / f"{stem}_composite.png"
    )

    plt.imsave(
        original_path,
        np.clip(original, 0.0, 1.0),
    )
    plt.imsave(
        heatmap_path,
        heatmap,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    colormap = plt.get_cmap("jet")
    colored_heatmap = colormap(heatmap)[..., :3]
    overlay = (
        (1.0 - config.overlay_alpha) * original
        + config.overlay_alpha * colored_heatmap
    )
    overlay = np.clip(overlay, 0.0, 1.0)

    plt.imsave(
        overlay_path,
        overlay,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(11, 4),
    )

    axes[0].imshow(original)
    axes[0].set_title("Original MRI")
    axes[0].axis("off")

    heat_image = axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    figure.colorbar(
        heat_image,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    true_name = (
        "Dementia"
        if int(row["true_label"]) == 1
        else "Normal"
    )
    predicted_name = (
        "Dementia"
        if int(row["predicted_label"]) == 1
        else "Normal"
    )

    figure.suptitle(
        (
            f"{category} | Subject: {subject_id} | "
            f"True: {true_name} | Predicted: {predicted_name} | "
            f"P(Dementia): {float(row['probability_dementia']):.3f}"
        ),
        fontsize=10,
    )

    figure.tight_layout()
    figure.savefig(
        composite_path,
        dpi=config.publication_dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    return {
        "original_file": str(original_path),
        "heatmap_file": str(heatmap_path),
        "overlay_file": str(overlay_path),
        "composite_file": str(composite_path),
    }


def create_publication_panel(
    summary_df: pd.DataFrame,
    output_dir: Path,
    config: GradCAMConfig,
) -> Optional[Path]:
    representative_rows: list[pd.Series] = []

    for category in CATEGORY_ORDER:
        subset = summary_df[
            summary_df["category"] == category
        ]
        if not subset.empty:
            representative_rows.append(
                subset.iloc[0]
            )

    if not representative_rows:
        return None

    column_count = len(representative_rows)
    figure, axes = plt.subplots(
        3,
        column_count,
        figsize=(4 * column_count, 10),
        squeeze=False,
    )

    for column, row in enumerate(representative_rows):
        original = plt.imread(row["original_file"])
        heatmap = plt.imread(row["heatmap_file"])
        overlay = plt.imread(row["overlay_file"])

        axes[0, column].imshow(original)
        axes[0, column].set_title(
            f"{row['category']} — Original"
        )
        axes[0, column].axis("off")

        axes[1, column].imshow(
            heatmap,
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, column].set_title("Grad-CAM")
        axes[1, column].axis("off")

        axes[2, column].imshow(overlay)
        axes[2, column].set_title(
            (
                f"Overlay\n"
                f"P(Dementia)={row['probability_dementia']:.3f}"
            )
        )
        axes[2, column].axis("off")

    figure.suptitle(
        "Representative Grad-CAM Explanations",
        fontsize=14,
    )
    figure.tight_layout()

    publication_dir = output_dir / "publication"
    publication_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    publication_path = (
        publication_dir
        / "Fig_GradCAM_representative_cases_300dpi.png"
    )
    figure.savefig(
        publication_path,
        dpi=config.publication_dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    return publication_path


def write_report(
    path: Path,
    summary_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    checkpoint: dict[str, Any],
    config: GradCAMConfig,
) -> None:
    lines = [
        "=" * 96,
        "BrainFMOps Professional Grad-CAM Report",
        "=" * 96,
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}",
        f"Decision threshold: {config.threshold:.4f}",
        f"Activation threshold: {config.activation_threshold:.4f}",
        f"Selected cases: {len(summary_df)}",
        "",
        "TEST PREDICTION COUNTS",
        "-" * 96,
    ]

    for category in CATEGORY_ORDER:
        count = int(
            (predictions_df["category"] == category).sum()
        )
        lines.append(f"{category}: {count}")

    lines.extend(
        [
            "",
            "GRAD-CAM SUMMARY",
            "-" * 96,
            (
                f"Mean activation ratio: "
                f"{summary_df['activation_ratio'].mean():.4f}"
            ),
            (
                f"Mean heatmap intensity: "
                f"{summary_df['heatmap_mean'].mean():.4f}"
            ),
            (
                f"Mean prediction confidence: "
                f"{summary_df['prediction_confidence'].mean():.4f}"
            ),
            (
                f"Mean predictive entropy: "
                f"{summary_df['predictive_entropy'].mean():.4f}"
            ),
            "",
            "CATEGORY-LEVEL SUMMARY",
            "-" * 96,
        ]
    )

    category_summary = (
        summary_df.groupby("category")
        .agg(
            cases=("image_id", "count"),
            mean_activation_ratio=(
                "activation_ratio",
                "mean",
            ),
            mean_heatmap=(
                "heatmap_mean",
                "mean",
            ),
            mean_confidence=(
                "prediction_confidence",
                "mean",
            ),
            mean_entropy=(
                "predictive_entropy",
                "mean",
            ),
        )
        .reset_index()
    )

    for _, row in category_summary.iterrows():
        lines.append(
            (
                f"{row['category']}: "
                f"n={int(row['cases'])}, "
                f"activation_ratio={row['mean_activation_ratio']:.4f}, "
                f"heatmap_mean={row['mean_heatmap']:.4f}, "
                f"confidence={row['mean_confidence']:.4f}, "
                f"entropy={row['mean_entropy']:.4f}"
            )
        )

    lines.extend(
        [
            "",
            "INTERPRETATION CAUTION",
            "-" * 96,
            (
                "Grad-CAM highlights spatial regions associated with the model output. "
                "It does not establish anatomical validity, causal reasoning, or clinical utility."
            ),
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-grade Grad-CAM explanations "
            "for EfficientNetB0 MRI classification."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.32,
    )
    parser.add_argument(
        "--cases-per-category",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--activation-threshold",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
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

    dataset_root = args.dataset_root.expanduser().resolve()
    test_manifest = args.test_manifest.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_logging(
        output_dir,
        args.verbose,
    )

    config = GradCAMConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        threshold=args.threshold,
        cases_per_category=args.cases_per_category,
        activation_threshold=args.activation_threshold,
        overlay_alpha=args.overlay_alpha,
    )

    if not dataset_root.is_dir():
        logging.error(
            "Invalid dataset root: %s",
            dataset_root,
        )
        return 2

    if not test_manifest.is_file():
        logging.error(
            "Test manifest not found: %s",
            test_manifest,
        )
        return 2

    if not checkpoint_path.is_file():
        logging.error(
            "Checkpoint not found: %s",
            checkpoint_path,
        )
        return 2

    if not 0.0 < config.threshold < 1.0:
        logging.error(
            "Decision threshold must be between 0 and 1."
        )
        return 2

    if not 0.0 < config.activation_threshold < 1.0:
        logging.error(
            "Activation threshold must be between 0 and 1."
        )
        return 2

    seed_everything(config.random_seed)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    logging.info("Device: %s", device)

    try:
        manifest = load_manifest(
            test_manifest,
            dataset_root,
        )

        dataset = ManifestDataset(
            manifest,
            config.image_size,
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

        model, checkpoint = load_model(
            checkpoint_path,
            device,
        )

        logging.info(
            "Loaded checkpoint from epoch %s",
            checkpoint.get(
                "epoch",
                "unknown",
            ),
        )

        predictions = run_predictions(
            model,
            loader,
            device,
            config.threshold,
        )

        selected_cases = select_representative_cases(
            predictions,
            config.cases_per_category,
        )

    except Exception as exc:
        logging.exception(
            "Grad-CAM preparation failed: %s",
            exc,
        )
        return 2

    predictions_path = (
        output_dir / "test_image_predictions_gradcam.csv"
    )
    selected_path = (
        output_dir / "selected_gradcam_cases.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )
    selected_cases.to_csv(
        selected_path,
        index=False,
        encoding="utf-8-sig",
    )

    target_layer = model.features[-1]
    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    result_rows: list[dict[str, Any]] = []

    try:
        for case_number, (_, row) in enumerate(
            selected_cases.iterrows(),
            start=1,
        ):
            logging.info(
                "Generating Grad-CAM %d/%d: %s %s",
                case_number,
                len(selected_cases),
                row["category"],
                row["subject_id"],
            )

            image_path = Path(
                row["absolute_path"]
            )
            original = load_original_resized(
                image_path,
                config.image_size,
            )
            input_tensor = prepare_input_tensor(
                image_path,
                config.image_size,
                device,
            )

            target_class = int(
                row["predicted_label"]
            )
            heatmap = gradcam.generate(
                input_tensor,
                target_class=target_class,
            )

            statistics = (
                calculate_activation_statistics(
                    heatmap,
                    config.activation_threshold,
                )
            )

            files = save_case_outputs(
                row=row,
                original=original,
                heatmap=heatmap,
                output_dir=output_dir,
                config=config,
            )

            result_rows.append(
                {
                    **row.to_dict(),
                    **statistics,
                    **files,
                }
            )

    except Exception as exc:
        logging.exception(
            "Grad-CAM generation failed: %s",
            exc,
        )
        gradcam.close()
        return 2
    finally:
        gradcam.close()

    summary_df = pd.DataFrame(result_rows)
    summary_path = output_dir / "gradcam_summary.csv"
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    category_summary = (
        summary_df.groupby("category")
        .agg(
            cases=("image_id", "count"),
            mean_activation_ratio=(
                "activation_ratio",
                "mean",
            ),
            standard_deviation_activation_ratio=(
                "activation_ratio",
                "std",
            ),
            mean_heatmap_intensity=(
                "heatmap_mean",
                "mean",
            ),
            mean_prediction_confidence=(
                "prediction_confidence",
                "mean",
            ),
            mean_predictive_entropy=(
                "predictive_entropy",
                "mean",
            ),
        )
        .reset_index()
    )
    category_summary.to_csv(
        output_dir / "gradcam_category_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    publication_figure = create_publication_panel(
        summary_df,
        output_dir,
        config,
    )

    report_path = output_dir / "gradcam_report.txt"
    write_report(
        report_path,
        summary_df,
        predictions,
        checkpoint,
        config,
    )

    report_json = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "device": str(device),
        "dataset_root": str(dataset_root),
        "test_manifest": str(test_manifest),
        "test_manifest_sha256": sha256_file(
            test_manifest
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),
        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),
        "configuration": asdict(config),
        "test_predictions": {
            category: int(
                (predictions["category"] == category).sum()
            )
            for category in CATEGORY_ORDER
        },
        "selected_cases": len(summary_df),
        "mean_activation_ratio": float(
            summary_df["activation_ratio"].mean()
        ),
        "mean_heatmap_intensity": float(
            summary_df["heatmap_mean"].mean()
        ),
        "mean_prediction_confidence": float(
            summary_df[
                "prediction_confidence"
            ].mean()
        ),
        "mean_predictive_entropy": float(
            summary_df[
                "predictive_entropy"
            ].mean()
        ),
        "publication_figure": (
            str(publication_figure)
            if publication_figure
            else None
        ),
        "status": "COMPLETED",
        "scientific_caution": (
            "Grad-CAM is a post-hoc association map and "
            "does not establish causal or clinical validity."
        ),
    }

    (
        output_dir / "gradcam_report.json"
    ).write_text(
        json.dumps(
            report_json,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print(
        "BRAINF MOPS PROFESSIONAL GRAD-CAM COMPLETE"
    )
    print("=" * 96)
    print(f"Device                    : {device}")
    print(
        f"Checkpoint epoch          : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    print(
        f"Decision threshold        : "
        f"{config.threshold:.4f}"
    )
    print(
        f"Selected cases            : "
        f"{len(summary_df)}"
    )

    for category in CATEGORY_ORDER:
        available = int(
            (
                predictions["category"]
                == category
            ).sum()
        )
        selected = int(
            (
                summary_df["category"]
                == category
            ).sum()
        )
        print(
            f"{category:26s}: "
            f"available={available}, "
            f"selected={selected}"
        )

    print(
        f"Mean activation ratio     : "
        f"{summary_df['activation_ratio'].mean():.4f}"
    )
    print(
        f"Mean heatmap intensity    : "
        f"{summary_df['heatmap_mean'].mean():.4f}"
    )
    print(
        f"Output directory          : "
        f"{output_dir}"
    )
    print(
        f"Grad-CAM summary          : "
        f"{summary_path}"
    )
    print(
        f"Publication figure        : "
        f"{publication_figure}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
