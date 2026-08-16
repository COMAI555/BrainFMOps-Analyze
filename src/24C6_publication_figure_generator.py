#!/usr/bin/env python3
"""
24C6_publication_figure_generator.py

BrainFMOps-Analyze — STEP 24C.6
Publication Figure Generator for IEEE-style paper figures.

Purpose
-------
Create high-resolution publication-ready figures from STEP 24B and STEP 24C
outputs.

Inputs
------
- subject_result.json
- gradcam_summary.json
- representative_slices.csv
- Primary MRI volume
- Trained EfficientNet-B0 checkpoint

Outputs
-------
publication_figures/
├── Fig_case_explainability_highest_probability.png
├── Fig_case_explainability_closest_to_subject_probability.png
├── Fig_case_explainability_median_probability.png
├── Fig_case_explainability_montage.png
└── publication_figure_manifest.json

Figure content
--------------
Each single-case figure includes:
A. Original MRI slice
B. Grad-CAM heatmap
C. Grad-CAM overlay
D. Probability panel

Default resolution: 600 dpi

Important
---------
This script is for research visualization only.
Grad-CAM does not establish clinical causality or diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


SCRIPT_VERSION = "1.0.0"


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        num_outputs: int,
        positive_class_index: int,
    ) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)

        if num_outputs == 1:
            logit = output.reshape(-1)[0]
            probability = torch.sigmoid(logit)
            score = logit
        else:
            probability = torch.softmax(output, dim=1)[0, positive_class_index]
            score = output[0, positive_class_index]

        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed.")

        activations = self.activations[0]
        gradients = self.gradients[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu(torch.sum(weights * activations, dim=0))

        minimum = cam.min()
        maximum = cam.max()
        if float(maximum - minimum) > 1e-12:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy(), float(probability.detach().cpu())

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_representative_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(
                {
                    "strategy": row["strategy"],
                    "slice_index": int(row["slice_index"]),
                    "axis": int(row["axis"]),
                    "probability_positive": float(row["probability_positive"]),
                    "subject_probability_positive": float(
                        row["subject_probability_positive"]
                    ),
                    "predicted_class_name": row["predicted_class_name"],
                }
            )

    if not rows:
        raise ValueError("representative_slices.csv is empty.")
    return rows


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint structure.")

    state_dict = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        state_dict[clean_key] = value

    if not state_dict:
        raise ValueError("No tensor weights found.")
    return state_dict


def infer_num_outputs(state_dict: dict[str, torch.Tensor]) -> int:
    for key in ("classifier.1.weight", "classifier.weight", "fc.weight"):
        tensor = state_dict.get(key)
        if tensor is not None and tensor.ndim == 2:
            return int(tensor.shape[0])

    for key, tensor in state_dict.items():
        if key.endswith("classifier.1.weight") and tensor.ndim == 2:
            return int(tensor.shape[0])

    raise ValueError("Cannot infer classifier output dimension.")


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, int]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = extract_state_dict(checkpoint)
    num_outputs = infer_num_outputs(state_dict)

    if num_outputs not in (1, 2):
        raise ValueError(
            f"Binary model must have 1 or 2 outputs, found {num_outputs}."
        )

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_outputs)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical_missing = [
        key for key in missing
        if key.startswith("features.") or key.startswith("classifier.")
    ]
    if critical_missing:
        raise ValueError(
            f"Checkpoint incompatible with EfficientNet-B0: "
            f"{critical_missing[:10]}"
        )
    if unexpected:
        logging.warning("Unexpected keys: %s", unexpected)

    model.to(device)
    model.eval()
    return model, num_outputs


def reduce_to_3d(volume: np.ndarray) -> np.ndarray:
    while volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D MRI volume, found {volume.shape}.")
    return volume


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("MRI volume contains no finite voxels.")

    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))
    if high <= low:
        raise ValueError("Degenerate MRI intensity range.")

    normalized = (np.clip(volume, low, high) - low) / (high - low)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def get_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    if axis == 2:
        return volume[:, :, index]
    raise ValueError(f"Invalid axis: {axis}")


def build_transform(image_size: int) -> transforms.Compose:
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
    uint8 = np.round(np.clip(slice_array, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(uint8, mode="L").convert("RGB")


def resize_cam(cam: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    cam_uint8 = np.round(np.clip(cam, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(cam_uint8, mode="L")
    resized = image.resize(
        (shape[1], shape[0]),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def save_single_figure(
    original: np.ndarray,
    cam: np.ndarray,
    strategy: str,
    axis: int,
    slice_index: int,
    slice_probability: float,
    subject_probability: float,
    threshold: float,
    predicted_class_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    resized_cam = resize_cam(cam, original.shape)

    fig = plt.figure(figsize=(11.5, 3.4))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(original, cmap="gray")
    ax1.set_title("(a) MRI Slice", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 4, 2)
    heat = ax2.imshow(resized_cam, cmap="jet", vmin=0, vmax=1)
    ax2.set_title("(b) Grad-CAM", fontsize=10)
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(original, cmap="gray")
    ax3.imshow(resized_cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax3.set_title("(c) Overlay", fontsize=10)
    ax3.axis("off")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.barh(["Positive"], [subject_probability])
    ax4.axvline(threshold, linestyle="--", linewidth=1.5)
    ax4.set_xlim(0, 1)
    ax4.set_xlabel("Probability")
    ax4.set_title("(d) Subject Decision", fontsize=10)
    ax4.text(
        min(subject_probability + 0.02, 0.95),
        0,
        f"{subject_probability:.3f}",
        va="center",
        fontsize=9,
    )
    ax4.text(
        threshold,
        -0.38,
        f"Threshold={threshold:.2f}",
        ha="center",
        fontsize=8,
    )

    strategy_label = strategy.replace("_", " ").title()
    fig.suptitle(
        f"{strategy_label} | Axis={axis}, Slice={slice_index} | "
        f"Slice P={slice_probability:.4f} | "
        f"Subject P={subject_probability:.4f} | "
        f"Prediction={predicted_class_name}",
        fontsize=11,
    )

    colorbar = fig.colorbar(
        heat,
        ax=[ax1, ax2, ax3],
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Normalized activation", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_montage(
    generated: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(12, 9))

    for index, item in enumerate(generated, start=1):
        original = item["original"]
        cam = item["cam"]
        overlay_cam = resize_cam(cam, original.shape)

        ax = fig.add_subplot(len(generated), 3, (index - 1) * 3 + 1)
        ax.imshow(original, cmap="gray")
        ax.set_ylabel(
            item["strategy"].replace("_", " ").title(),
            fontsize=9,
        )
        ax.set_title("MRI", fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(len(generated), 3, (index - 1) * 3 + 2)
        ax.imshow(overlay_cam, cmap="jet", vmin=0, vmax=1)
        ax.set_title("Grad-CAM", fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(len(generated), 3, (index - 1) * 3 + 3)
        ax.imshow(original, cmap="gray")
        ax.imshow(overlay_cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(
            f"Overlay, P={item['probability']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(
        "Representative MRI Grad-CAM Explanations",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def run_generator(
    subject_result_path: Path,
    gradcam_summary_path: Path,
    representative_csv_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    image_size: int,
    positive_class_index: int,
    device_name: str,
    dpi: int,
    seed: int,
) -> dict[str, Any]:
    set_reproducibility(seed)

    subject = load_json(subject_result_path)
    gradcam_summary = load_json(gradcam_summary_path)
    representatives = load_representative_rows(representative_csv_path)

    primary_volume = Path(subject["primary_volume"]).expanduser().resolve()
    if not primary_volume.exists():
        raise FileNotFoundError(f"Primary volume not found: {primary_volume}")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    model, num_outputs = load_model(checkpoint_path, device)
    gradcam = GradCAM(model, model.features[-1])

    volume = normalize_volume(
        reduce_to_3d(
            np.asarray(
                nib.load(str(primary_volume)).dataobj,
                dtype=np.float32,
            )
        )
    )

    transform = build_transform(image_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_items = []

    try:
        for row in representatives:
            original = get_slice(
                volume,
                row["axis"],
                row["slice_index"],
            )

            input_tensor = transform(
                slice_to_pil(original)
            ).unsqueeze(0).to(device)

            cam, probability = gradcam.generate(
                input_tensor,
                num_outputs=num_outputs,
                positive_class_index=positive_class_index,
            )

            filename = (
                f"Fig_case_explainability_{row['strategy']}.png"
            )
            output_path = output_dir / filename

            save_single_figure(
                original=original,
                cam=cam,
                strategy=row["strategy"],
                axis=row["axis"],
                slice_index=row["slice_index"],
                slice_probability=probability,
                subject_probability=float(
                    subject["subject_probability_positive"]
                ),
                threshold=float(subject["locked_threshold"]),
                predicted_class_name=str(
                    subject["predicted_class_name"]
                ),
                output_path=output_path,
                dpi=dpi,
            )

            generated_items.append(
                {
                    "strategy": row["strategy"],
                    "axis": row["axis"],
                    "slice_index": row["slice_index"],
                    "probability": probability,
                    "output_file": str(output_path.resolve()),
                    "original": original,
                    "cam": cam,
                }
            )

            logging.info(
                "Created %s",
                output_path.name,
            )
    finally:
        gradcam.close()

    montage_path = output_dir / "Fig_case_explainability_montage.png"
    save_montage(
        generated_items,
        montage_path,
        dpi=dpi,
    )

    manifest = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "case_id": subject.get("case_id"),
        "subject_result": str(subject_result_path.resolve()),
        "gradcam_summary": str(gradcam_summary_path.resolve()),
        "representative_slices": str(
            representative_csv_path.resolve()
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "primary_volume": str(primary_volume),
        "device": str(device),
        "dpi": dpi,
        "single_figures": [
            {
                key: value
                for key, value in item.items()
                if key not in ("original", "cam")
            }
            for item in generated_items
        ],
        "montage_figure": str(montage_path.resolve()),
        "research_use_only": True,
        "clinical_diagnosis": False,
    }

    (output_dir / "publication_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return manifest


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 600-dpi publication figures from BrainFMOps-Analyze "
            "Grad-CAM outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--subject-result",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--gradcam-summary",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--representative-slices",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Defaults to <inference-dir>/publication_figures."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=positive_int,
        default=224,
    )
    parser.add_argument(
        "--positive-class-index",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=600,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    subject_result_path = args.subject_result.expanduser().resolve()
    gradcam_summary_path = args.gradcam_summary.expanduser().resolve()
    representative_csv_path = (
        args.representative_slices.expanduser().resolve()
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else subject_result_path.parent / "publication_figures"
    )

    try:
        manifest = run_generator(
            subject_result_path=subject_result_path,
            gradcam_summary_path=gradcam_summary_path,
            representative_csv_path=representative_csv_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            image_size=args.image_size,
            positive_class_index=args.positive_class_index,
            device_name=args.device,
            dpi=args.dpi,
            seed=args.seed,
        )
    except Exception as exc:
        logging.exception("Publication figure generation failed: %s", exc)
        return 1

    print("\n" + "=" * 78)
    print(f"CASE ID        : {manifest['case_id']}")
    print(f"SINGLE FIGURES : {len(manifest['single_figures'])}")
    print(f"MONTAGE        : {manifest['montage_figure']}")
    print(f"DPI            : {manifest['dpi']}")
    print(f"OUTPUT         : {output_dir}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
