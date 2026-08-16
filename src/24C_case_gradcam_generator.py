#!/usr/bin/env python3
"""
24C_case_gradcam_generator.py

BrainFMOps-Analyze — STEP 24C
Representative Grad-CAM Generator for subject-level brain MRI classification.

Inputs
------
- STEP 24B subject_result.json
- STEP 24B slice_predictions.csv
- Trained EfficientNet-B0 checkpoint
- Primary MRI volume selected by STEP 24A/24B

Outputs
-------
gradcam/
├── gradcam_highest_probability.png
├── gradcam_closest_to_subject_probability.png
├── gradcam_median_probability.png
├── representative_slices.csv
└── gradcam_summary.json

Selection strategies
--------------------
1. highest_probability
2. closest_to_subject_probability
3. median_probability

Important
---------
This script is for research explainability only.
Grad-CAM is not proof of causal reasoning and is not a clinical diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


SCRIPT_VERSION = "1.0.0"


@dataclass
class RepresentativeSlice:
    strategy: str
    rank: int
    slice_index: int
    axis: int
    probability_positive: float
    subject_probability_positive: float
    absolute_distance_to_subject_probability: float
    predicted_class_name: str
    output_file: str


class GradCAM:
    """Minimal Grad-CAM implementation for CNN classification models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: tuple[Optional[torch.Tensor], ...],
        grad_output: tuple[Optional[torch.Tensor], ...],
    ) -> None:
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
            target_score = logit
        else:
            if positive_class_index not in (0, 1):
                raise ValueError("positive_class_index must be 0 or 1.")
            probability = torch.softmax(output, dim=1)[0, positive_class_index]
            target_score = output[0, positive_class_index]

        target_score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations or gradients."
            )

        activations = self.activations[0]
        gradients = self.gradients[0]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = torch.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        if float(cam_max - cam_min) > 1e-12:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.detach().cpu().numpy(), float(probability.detach().cpu())

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
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_slice_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Slice predictions file does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(
                {
                    "rank": int(row["rank"]),
                    "slice_index": int(row["slice_index"]),
                    "axis": int(row["axis"]),
                    "probability_positive": float(row["probability_positive"]),
                    "predicted_class_name": row["predicted_class_name"],
                }
            )

    if not rows:
        raise ValueError("slice_predictions.csv contains no rows.")
    return rows


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
        raise ValueError("No tensor parameters were found in checkpoint.")
    return state_dict


def infer_num_outputs(state_dict: dict[str, torch.Tensor]) -> int:
    for key in (
        "classifier.1.weight",
        "classifier.weight",
        "fc.weight",
    ):
        tensor = state_dict.get(key)
        if tensor is not None and tensor.ndim == 2:
            return int(tensor.shape[0])

    for key, tensor in state_dict.items():
        if key.endswith("classifier.1.weight") and tensor.ndim == 2:
            return int(tensor.shape[0])

    raise ValueError("Cannot infer classifier output size from checkpoint.")


def build_model(num_outputs: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_outputs)
    return model


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
            f"Expected binary model with 1 or 2 outputs, found {num_outputs}."
        )

    model = build_model(num_outputs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    critical_missing = [
        key for key in missing
        if key.startswith("features.") or key.startswith("classifier.")
    ]
    if critical_missing:
        raise ValueError(
            "Checkpoint incompatible with EfficientNet-B0. "
            f"Critical missing keys: {critical_missing[:10]}"
        )

    if unexpected:
        logging.warning("Unexpected checkpoint keys: %s", unexpected)

    model.to(device)
    model.eval()
    return model, num_outputs


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("MRI volume has no finite voxel values.")

    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))
    if high <= low:
        raise ValueError("Degenerate MRI intensity range.")

    normalized = (np.clip(volume, low, high) - low) / (high - low)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def reduce_to_3d(volume: np.ndarray) -> np.ndarray:
    while volume.ndim > 3:
        logging.warning(
            "Volume has %d dimensions; first volume/channel is used.",
            volume.ndim,
        )
        volume = volume[..., 0]

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D MRI volume, found shape {volume.shape}.")
    return volume


def slice_from_volume(
    volume: np.ndarray,
    axis: int,
    index: int,
) -> np.ndarray:
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
    array = np.clip(slice_array, 0.0, 1.0)
    uint8 = np.round(array * 255.0).astype(np.uint8)
    return Image.fromarray(uint8, mode="L").convert("RGB")


def select_representative_rows(
    rows: list[dict[str, Any]],
    subject_probability: float,
) -> dict[str, dict[str, Any]]:
    highest = max(rows, key=lambda item: item["probability_positive"])

    closest = min(
        rows,
        key=lambda item: abs(
            item["probability_positive"] - subject_probability
        ),
    )

    sorted_rows = sorted(
        rows,
        key=lambda item: item["probability_positive"],
    )
    median_row = sorted_rows[len(sorted_rows) // 2]

    return {
        "highest_probability": highest,
        "closest_to_subject_probability": closest,
        "median_probability": median_row,
    }


def resize_cam(cam: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    cam_uint8 = np.round(np.clip(cam, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(cam_uint8, mode="L")
    resized = image.resize(
        (target_shape[1], target_shape[0]),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def save_gradcam_figure(
    original: np.ndarray,
    cam: np.ndarray,
    output_path: Path,
    title: str,
    probability: float,
    subject_probability: float,
    threshold: float,
    predicted_class_name: str,
) -> None:
    original = np.clip(original, 0.0, 1.0)
    resized_cam = resize_cam(cam, original.shape)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(original, cmap="gray")
    ax1.set_title("MRI Slice")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(resized_cam, cmap="jet")
    ax2.set_title("Grad-CAM Heatmap")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(original, cmap="gray")
    ax3.imshow(resized_cam, cmap="jet", alpha=0.45)
    ax3.set_title("Overlay")
    ax3.axis("off")

    fig.suptitle(
        f"{title}\n"
        f"Slice probability={probability:.4f} | "
        f"Subject probability={subject_probability:.4f} | "
        f"Threshold={threshold:.2f} | "
        f"Prediction={predicted_class_name}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(
    records: list[RepresentativeSlice],
    path: Path,
) -> None:
    rows = [asdict(record) for record in records]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_gradcam(
    subject_result_path: Path,
    slice_predictions_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    image_size: int,
    positive_class_index: int,
    device_name: str,
    seed: int,
) -> dict[str, Any]:
    set_reproducibility(seed)

    subject_result = load_json(subject_result_path)
    rows = load_slice_predictions(slice_predictions_path)

    subject_probability = float(
        subject_result["subject_probability_positive"]
    )
    threshold = float(subject_result["locked_threshold"])
    predicted_class_name = str(subject_result["predicted_class_name"])
    primary_volume = Path(subject_result["primary_volume"]).expanduser().resolve()

    if not primary_volume.exists():
        raise FileNotFoundError(
            f"Primary MRI volume does not exist: {primary_volume}"
        )

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    logging.info("Device: %s", device)
    logging.info("Loading checkpoint: %s", checkpoint_path)

    model, num_outputs = load_model(checkpoint_path, device)
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    image = nib.load(str(primary_volume))
    volume = reduce_to_3d(
        np.asarray(image.dataobj, dtype=np.float32)
    )
    volume = normalize_volume(volume)

    selected = select_representative_rows(
        rows=rows,
        subject_probability=subject_probability,
    )

    transform = build_transform(image_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    representative_records: list[RepresentativeSlice] = []
    result_items: list[dict[str, Any]] = []

    try:
        for strategy, row in selected.items():
            axis = int(row["axis"])
            slice_index = int(row["slice_index"])
            original_slice = slice_from_volume(
                volume,
                axis,
                slice_index,
            )

            input_tensor = transform(
                slice_to_pil(original_slice)
            ).unsqueeze(0).to(device)

            cam, recomputed_probability = gradcam.generate(
                input_tensor=input_tensor,
                num_outputs=num_outputs,
                positive_class_index=positive_class_index,
            )

            output_file = f"gradcam_{strategy}.png"
            output_path = output_dir / output_file

            save_gradcam_figure(
                original=original_slice,
                cam=cam,
                output_path=output_path,
                title=(
                    f"{strategy.replace('_', ' ').title()} "
                    f"(axis={axis}, slice={slice_index})"
                ),
                probability=recomputed_probability,
                subject_probability=subject_probability,
                threshold=threshold,
                predicted_class_name=predicted_class_name,
            )

            record = RepresentativeSlice(
                strategy=strategy,
                rank=int(row["rank"]),
                slice_index=slice_index,
                axis=axis,
                probability_positive=float(row["probability_positive"]),
                subject_probability_positive=subject_probability,
                absolute_distance_to_subject_probability=abs(
                    float(row["probability_positive"]) - subject_probability
                ),
                predicted_class_name=str(row["predicted_class_name"]),
                output_file=output_file,
            )
            representative_records.append(record)

            result_items.append(
                {
                    **asdict(record),
                    "recomputed_probability_positive": round(
                        recomputed_probability,
                        8,
                    ),
                    "probability_difference_from_step24b": round(
                        recomputed_probability
                        - float(row["probability_positive"]),
                        8,
                    ),
                }
            )
            logging.info(
                "%s: axis=%d slice=%d probability=%.6f",
                strategy,
                axis,
                slice_index,
                recomputed_probability,
            )
    finally:
        gradcam.close()

    write_csv(
        representative_records,
        output_dir / "representative_slices.csv",
    )

    summary = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "case_id": subject_result["case_id"],
        "subject_result": str(subject_result_path.resolve()),
        "slice_predictions": str(slice_predictions_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "primary_volume": str(primary_volume),
        "device": str(device),
        "target_layer": "model.features[-1]",
        "subject_probability_positive": subject_probability,
        "locked_threshold": threshold,
        "predicted_class_name": predicted_class_name,
        "representative_slices": result_items,
        "research_use_only": True,
        "clinical_diagnosis": False,
        "interpretation_warning": (
            "Grad-CAM indicates spatial regions associated with model output. "
            "It does not establish causal or clinical evidence."
        ),
    }

    (output_dir / "gradcam_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate representative Grad-CAM images from STEP 24B outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--subject-result",
        required=True,
        type=Path,
        help="Path to STEP 24B subject_result.json.",
    )
    parser.add_argument(
        "--slice-predictions",
        required=True,
        type=Path,
        help="Path to STEP 24B slice_predictions.csv.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to trained EfficientNet-B0 checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <inference-dir>/gradcam.",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    subject_result_path = args.subject_result.expanduser().resolve()
    slice_predictions_path = args.slice_predictions.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else subject_result_path.parent / "gradcam"
    )

    try:
        summary = run_gradcam(
            subject_result_path=subject_result_path,
            slice_predictions_path=slice_predictions_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            image_size=args.image_size,
            positive_class_index=args.positive_class_index,
            device_name=args.device,
            seed=args.seed,
        )
    except Exception as exc:
        logging.exception("Grad-CAM generation failed: %s", exc)
        return 1

    print("\n" + "=" * 78)
    print(f"CASE ID      : {summary['case_id']}")
    print(f"PREDICTION   : {summary['predicted_class_name']}")
    print(f"PROBABILITY  : {summary['subject_probability_positive']:.6f}")
    print(f"IMAGES       : {len(summary['representative_slices'])}")
    print(f"OUTPUT       : {output_dir}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
