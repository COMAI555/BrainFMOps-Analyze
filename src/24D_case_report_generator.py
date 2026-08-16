#!/usr/bin/env python3
"""
24D_case_report_generator.py

BrainFMOps-Analyze — STEP 24D
Automatic HTML Case Report Generator

Purpose
-------
Combine outputs from:
- STEP 24A: Case Readiness Checker
- STEP 24B: Subject-Level Inference
- STEP 24C: Representative Grad-CAM
- STEP 24C.5: MRI Volume Selector

into one self-contained HTML report.

Inputs
------
- readiness_report.json
- subject_result.json
- gradcam_summary.json
- volume_selection_report.json (optional)
- representative_slices.csv (optional)
- Grad-CAM PNG images (optional but recommended)

Output
------
- <CASE_ID>_report.html
- report_manifest.json
- report_summary.txt

Important
---------
This report is for research use only.
It is not a clinical diagnosis or a medical-device output.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import logging
import mimetypes
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCRIPT_VERSION = "1.0.0"


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Optional[Path], required: bool = True) -> dict[str, Any]:
    if path is None:
        return {}

    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required JSON file does not exist: {path}")
        return {}

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_csv(path: Optional[Path]) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def safe_text(value: Any) -> str:
    if value is None:
        return "N/A"
    return html.escape(str(value))


def safe_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def safe_percent(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "N/A"


def status_class(status: str) -> str:
    normalized = str(status).upper()
    if normalized == "READY":
        return "status-ready"
    if normalized == "READY_WITH_WARNINGS":
        return "status-warning"
    if normalized == "NOT_READY":
        return "status-error"
    return "status-neutral"


def image_to_data_uri(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_gradcam_images(
    gradcam_summary: dict[str, Any],
    gradcam_dir: Path,
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []

    for item in gradcam_summary.get("representative_slices", []):
        filename = item.get("output_file")
        if not filename:
            continue

        path = gradcam_dir / filename
        if not path.exists():
            logging.warning("Grad-CAM image missing: %s", path)
            continue

        images.append(
            {
                "strategy": item.get("strategy", "unknown"),
                "slice_index": item.get("slice_index"),
                "axis": item.get("axis"),
                "probability_positive": item.get(
                    "recomputed_probability_positive",
                    item.get("probability_positive"),
                ),
                "path": str(path.resolve()),
                "data_uri": image_to_data_uri(path),
            }
        )

    return images


def make_key_value_rows(items: list[tuple[str, Any]]) -> str:
    rows = []
    for label, value in items:
        rows.append(
            f"""
            <tr>
              <th>{safe_text(label)}</th>
              <td>{safe_text(value)}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def make_list(items: list[Any], empty_message: str = "None") -> str:
    if not items:
        return f"<p class='muted'>{safe_text(empty_message)}</p>"

    return "<ul>" + "".join(
        f"<li>{safe_text(item)}</li>" for item in items
    ) + "</ul>"


def make_representative_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "<p class='muted'>Representative slice table was not available.</p>"

    body = []
    for row in rows:
        body.append(
            f"""
            <tr>
              <td>{safe_text(row.get('strategy'))}</td>
              <td>{safe_text(row.get('axis'))}</td>
              <td>{safe_text(row.get('slice_index'))}</td>
              <td>{safe_float(row.get('probability_positive'), 6)}</td>
              <td>{safe_float(row.get('subject_probability_positive'), 6)}</td>
              <td>{safe_float(row.get('absolute_distance_to_subject_probability'), 6)}</td>
            </tr>
            """
        )

    return f"""
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Axis</th>
            <th>Slice</th>
            <th>Slice Probability</th>
            <th>Subject Probability</th>
            <th>Absolute Distance</th>
          </tr>
        </thead>
        <tbody>
          {''.join(body)}
        </tbody>
      </table>
    </div>
    """


def build_html(
    readiness: dict[str, Any],
    subject: dict[str, Any],
    gradcam: dict[str, Any],
    volume_selection: dict[str, Any],
    representative_rows: list[dict[str, str]],
    gradcam_images: list[dict[str, Any]],
) -> str:
    case_id = str(subject.get("case_id") or readiness.get("case_id") or "UNKNOWN_CASE")
    readiness_status = str(readiness.get("status", "UNKNOWN"))
    predicted_class = str(subject.get("predicted_class_name", "N/A"))
    probability = subject.get("subject_probability_positive")
    threshold = subject.get("locked_threshold")
    decision_margin = subject.get("decision_margin")

    stats = readiness.get("statistics", {})
    folder_validation = readiness.get("folder_validation", {})
    quality_summary = readiness.get("quality_summary", {})
    slice_summary = subject.get("slice_probability_summary", {})

    image_cards = []
    for item in gradcam_images:
        strategy_label = str(item["strategy"]).replace("_", " ").title()
        image_cards.append(
            f"""
            <figure class="image-card">
              <img src="{item['data_uri']}" alt="{safe_text(strategy_label)}">
              <figcaption>
                <strong>{safe_text(strategy_label)}</strong><br>
                Axis {safe_text(item.get('axis'))},
                Slice {safe_text(item.get('slice_index'))},
                P(positive)={safe_float(item.get('probability_positive'), 6)}
              </figcaption>
            </figure>
            """
        )

    volume_section = ""
    if volume_selection:
        volume_section = f"""
        <section>
          <h2>4. MRI Volume Selection</h2>
          <div class="card">
            <table>
              {make_key_value_rows([
                  ("Original shape", volume_selection.get("original_shape")),
                  ("Candidate volume count", volume_selection.get("candidate_volume_count")),
                  ("Selected volume index", volume_selection.get("selected_volume_index")),
                  ("Selected score", safe_float(volume_selection.get("selected_score"), 6)),
                  ("Selected output", volume_selection.get("selected_volume_file")),
                  ("Selection method", volume_selection.get("selection_method")),
              ])}
            </table>
          </div>
          <div class="notice">
            The selector provides a deterministic engineering choice. It does not infer clinical sequence semantics.
          </div>
        </section>
        """

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_text(case_id)} — BrainFMOps-Analyze Report</title>
<style>
  :root {{
    --bg: #f4f6f8;
    --card: #ffffff;
    --text: #1f2933;
    --muted: #6b7280;
    --border: #d8dee6;
    --accent: #1f4e79;
    --ready: #16784a;
    --warning: #a15c00;
    --error: #b42318;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.55;
  }}
  .container {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px;
  }}
  .header {{
    background: linear-gradient(135deg, #183b56, #275d8c);
    color: white;
    padding: 28px;
    border-radius: 14px;
    margin-bottom: 22px;
  }}
  .header h1 {{ margin: 0 0 8px; font-size: 30px; }}
  .header p {{ margin: 4px 0; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
  }}
  .metric {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
  }}
  .metric .label {{
    color: var(--muted);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .metric .value {{
    margin-top: 6px;
    font-size: 24px;
    font-weight: 700;
  }}
  section {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px;
    margin-top: 18px;
  }}
  h2 {{
    margin-top: 0;
    color: var(--accent);
    border-bottom: 2px solid #e6ebf1;
    padding-bottom: 10px;
  }}
  h3 {{ color: #334e68; }}
  table {{
    width: 100%;
    border-collapse: collapse;
  }}
  th, td {{
    text-align: left;
    vertical-align: top;
    border-bottom: 1px solid var(--border);
    padding: 10px 12px;
  }}
  th {{
    width: 34%;
    background: #f8fafc;
  }}
  .table-wrap {{ overflow-x: auto; }}
  .status {{
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    font-weight: 700;
  }}
  .status-ready {{ color: var(--ready); background: #eaf8f1; }}
  .status-warning {{ color: var(--warning); background: #fff4df; }}
  .status-error {{ color: var(--error); background: #fdecec; }}
  .status-neutral {{ color: #374151; background: #edf0f3; }}
  .notice {{
    margin-top: 14px;
    padding: 14px 16px;
    background: #fff7e6;
    border-left: 5px solid #d68a00;
    border-radius: 8px;
  }}
  .critical {{
    padding: 16px;
    background: #fff0f0;
    border-left: 5px solid var(--error);
    border-radius: 8px;
    font-weight: 700;
  }}
  .image-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
    gap: 18px;
  }}
  .image-card {{
    margin: 0;
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    background: white;
  }}
  .image-card img {{
    width: 100%;
    display: block;
  }}
  figcaption {{
    padding: 12px 14px;
    color: #374151;
  }}
  .muted {{ color: var(--muted); }}
  .footer {{
    margin-top: 24px;
    color: var(--muted);
    text-align: center;
    font-size: 13px;
  }}
  @media print {{
    body {{ background: white; }}
    .container {{ max-width: none; padding: 0; }}
    section, .metric, .header {{
      break-inside: avoid;
      box-shadow: none;
    }}
  }}
</style>
</head>
<body>
<div class="container">

  <header class="header">
    <h1>BrainFMOps-Analyze Case Report</h1>
    <p><strong>Case ID:</strong> {safe_text(case_id)}</p>
    <p><strong>Generated:</strong> {safe_text(utc_now_iso())}</p>
    <p><strong>Pipeline:</strong> Quality-aware and reproducible subject-level brain MRI classification</p>
  </header>

  <div class="grid">
    <div class="metric">
      <div class="label">Readiness</div>
      <div class="value">
        <span class="status {status_class(readiness_status)}">
          {safe_text(readiness_status)}
        </span>
      </div>
    </div>
    <div class="metric">
      <div class="label">Prediction</div>
      <div class="value">{safe_text(predicted_class)}</div>
    </div>
    <div class="metric">
      <div class="label">Positive Probability</div>
      <div class="value">{safe_float(probability, 6)}</div>
    </div>
    <div class="metric">
      <div class="label">Locked Threshold</div>
      <div class="value">{safe_float(threshold, 4)}</div>
    </div>
  </div>

  <section>
    <h2>1. Executive Summary</h2>
    <p>
      The MRI case passed the technical readiness gate with status
      <strong>{safe_text(readiness_status)}</strong>.
      The subject-level model produced a positive-class probability of
      <strong>{safe_float(probability, 6)}</strong>, compared with the locked
      decision threshold of <strong>{safe_float(threshold, 4)}</strong>.
      The resulting research classification was
      <strong>{safe_text(predicted_class)}</strong>.
    </p>
    <div class="critical">
      Research-use only. This output is not a clinical diagnosis, not a medical-device decision,
      and must not be used as a substitute for qualified medical interpretation.
    </div>
  </section>

  <section>
    <h2>2. Case Readiness Assessment</h2>
    <div class="grid">
      <div class="card">
        <h3>Folder and Volume Summary</h3>
        <table>
          {make_key_value_rows([
              ("Case directory", readiness.get("case_dir")),
              ("Total volume candidates", stats.get("total_volume_candidates")),
              ("Readable volumes", stats.get("readable_volumes")),
              ("Rejected volumes", stats.get("rejected_volumes")),
              ("Readable ratio", safe_percent(stats.get("readable_ratio"))),
              ("Primary volume", stats.get("primary_volume")),
          ])}
        </table>
      </div>
      <div class="card">
        <h3>OASIS Structure</h3>
        <table>
          {make_key_value_rows([
              ("PROCESSED folder", folder_validation.get("has_processed_folder")),
              ("RAW folder", folder_validation.get("has_raw_folder")),
              ("FSL_SEG folder", folder_validation.get("has_fsl_seg_folder")),
              ("TXT metadata files", len(folder_validation.get("metadata_txt_files", []))),
              ("XML metadata files", len(folder_validation.get("metadata_xml_files", []))),
          ])}
        </table>
      </div>
    </div>

    <h3>Decision Reasons</h3>
    {make_list(readiness.get("decision_reasons", []), "No readiness decision reasons were recorded.")}

    <h3>Warnings</h3>
    {make_list(readiness.get("warnings", []), "No readiness warnings were recorded.")}

    <h3>Errors</h3>
    {make_list(readiness.get("errors", []), "No readiness errors were recorded.")}
  </section>

  <section>
    <h2>3. Subject-Level Inference</h2>
    <table>
      {make_key_value_rows([
          ("Model checkpoint", subject.get("model_checkpoint")),
          ("Device", subject.get("device")),
          ("Volume shape", subject.get("volume_shape")),
          ("Selected axis", subject.get("selected_axis")),
          ("Candidate slice count", subject.get("candidate_slice_count")),
          ("Selected slice count", subject.get("num_selected_slices")),
          ("Aggregation method", subject.get("config", {}).get("aggregation")),
          ("Subject probability", safe_float(probability, 8)),
          ("Locked threshold", safe_float(threshold, 4)),
          ("Decision margin", safe_float(decision_margin, 8)),
          ("Predicted class", predicted_class),
          ("Runtime seconds", safe_float(subject.get("runtime_seconds"), 4)),
      ])}
    </table>

    <h3>Slice Probability Distribution</h3>
    <table>
      {make_key_value_rows([
          ("Mean", safe_float(slice_summary.get("mean"), 8)),
          ("Standard deviation", safe_float(slice_summary.get("std"), 8)),
          ("Minimum", safe_float(slice_summary.get("min"), 8)),
          ("Median", safe_float(slice_summary.get("median"), 8)),
          ("Maximum", safe_float(slice_summary.get("max"), 8)),
      ])}
    </table>
  </section>

  {volume_section}

  <section>
    <h2>5. Representative Grad-CAM Explanations</h2>
    <p>
      Grad-CAM highlights spatial regions associated with the model output.
      These maps are explanatory visualizations, not evidence of causality,
      pathology localization, or clinical validity.
    </p>
    <div class="image-grid">
      {''.join(image_cards) if image_cards else "<p class='muted'>No Grad-CAM images were available.</p>"}
    </div>

    <h3>Representative Slice Metadata</h3>
    {make_representative_table(representative_rows)}
  </section>

  <section>
    <h2>6. Reproducibility Record</h2>
    <table>
      {make_key_value_rows([
          ("Readiness script version", readiness.get("script_version")),
          ("Inference script version", subject.get("script_version")),
          ("Grad-CAM script version", gradcam.get("script_version")),
          ("Volume selector script version", volume_selection.get("script_version") if volume_selection else "Not provided"),
          ("Random seed", subject.get("config", {}).get("seed")),
          ("Image size", subject.get("config", {}).get("image_size")),
          ("Positive class index", subject.get("config", {}).get("positive_class_index")),
          ("Negative class name", subject.get("config", {}).get("negative_class_name")),
          ("Positive class name", subject.get("config", {}).get("positive_class_name")),
          ("Research-use only", subject.get("research_use_only")),
      ])}
    </table>
  </section>

  <section>
    <h2>7. Interpretation Boundaries</h2>
    <ul>
      <li>This pipeline performs research-oriented technical classification.</li>
      <li>The probability value is model-specific and is not a clinical risk estimate.</li>
      <li>The locked threshold was applied exactly as configured in the research workflow.</li>
      <li>Grad-CAM does not prove causal reasoning or disease localization.</li>
      <li>Clinical deployment requires independent validation, governance, regulatory review, and qualified human oversight.</li>
    </ul>
  </section>

  <div class="footer">
    BrainFMOps-Analyze STEP 24D · Script version {safe_text(SCRIPT_VERSION)}
  </div>

</div>
</body>
</html>
"""
    return html_doc


def write_manifest(
    output_path: Path,
    readiness_path: Path,
    subject_path: Path,
    gradcam_path: Path,
    volume_selection_path: Optional[Path],
    representative_csv_path: Optional[Path],
    gradcam_images: list[dict[str, Any]],
    case_id: str,
) -> None:
    manifest = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "case_id": case_id,
        "html_report": str(output_path.resolve()),
        "source_files": {
            "readiness_report": str(readiness_path.resolve()),
            "subject_result": str(subject_path.resolve()),
            "gradcam_summary": str(gradcam_path.resolve()),
            "volume_selection_report": (
                str(volume_selection_path.resolve())
                if volume_selection_path is not None
                and volume_selection_path.exists()
                else None
            ),
            "representative_slices_csv": (
                str(representative_csv_path.resolve())
                if representative_csv_path is not None
                and representative_csv_path.exists()
                else None
            ),
        },
        "embedded_gradcam_images": [
            {
                "strategy": item["strategy"],
                "path": item["path"],
                "slice_index": item["slice_index"],
                "axis": item["axis"],
            }
            for item in gradcam_images
        ],
        "self_contained_html": True,
        "research_use_only": True,
        "clinical_diagnosis": False,
    }

    manifest_path = output_path.parent / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_summary(
    output_path: Path,
    readiness: dict[str, Any],
    subject: dict[str, Any],
    image_count: int,
) -> None:
    case_id = subject.get("case_id") or readiness.get("case_id") or "UNKNOWN_CASE"

    lines = [
        "=" * 78,
        "BrainFMOps-Analyze — Automatic HTML Case Report",
        "=" * 78,
        f"Case ID              : {case_id}",
        f"Readiness status     : {readiness.get('status', 'UNKNOWN')}",
        f"Prediction           : {subject.get('predicted_class_name', 'N/A')}",
        f"Positive probability : {safe_float(subject.get('subject_probability_positive'), 6)}",
        f"Locked threshold     : {safe_float(subject.get('locked_threshold'), 4)}",
        f"Grad-CAM images      : {image_count}",
        f"HTML report          : {output_path.resolve()}",
        "",
        "Research-use notice",
        "-" * 78,
        "This report is not a clinical diagnosis or medical-device output.",
        "=" * 78,
    ]

    summary_path = output_path.parent / "report_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a self-contained HTML report from BrainFMOps-Analyze "
            "STEP 24A–24C.5 outputs."
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
        "--subject-result",
        required=True,
        type=Path,
        help="Path to STEP 24B subject_result.json.",
    )
    parser.add_argument(
        "--gradcam-summary",
        required=True,
        type=Path,
        help="Path to STEP 24C gradcam_summary.json.",
    )
    parser.add_argument(
        "--volume-selection-report",
        type=Path,
        default=None,
        help="Optional STEP 24C.5 volume_selection_report.json.",
    )
    parser.add_argument(
        "--representative-slices",
        type=Path,
        default=None,
        help="Optional STEP 24C representative_slices.csv.",
    )
    parser.add_argument(
        "--gradcam-dir",
        type=Path,
        default=None,
        help="Directory containing Grad-CAM images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <inference-dir>/<CASE_ID>_report.html.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)

    readiness_path = args.readiness_report.expanduser().resolve()
    subject_path = args.subject_result.expanduser().resolve()
    gradcam_path = args.gradcam_summary.expanduser().resolve()

    volume_selection_path = (
        args.volume_selection_report.expanduser().resolve()
        if args.volume_selection_report is not None
        else None
    )
    representative_csv_path = (
        args.representative_slices.expanduser().resolve()
        if args.representative_slices is not None
        else None
    )

    try:
        readiness = load_json(readiness_path, required=True)
        subject = load_json(subject_path, required=True)
        gradcam = load_json(gradcam_path, required=True)
        volume_selection = load_json(
            volume_selection_path,
            required=False,
        )
        representative_rows = load_csv(representative_csv_path)

        case_id = str(
            subject.get("case_id")
            or readiness.get("case_id")
            or "UNKNOWN_CASE"
        )

        gradcam_dir = (
            args.gradcam_dir.expanduser().resolve()
            if args.gradcam_dir is not None
            else gradcam_path.parent
        )
        gradcam_images = resolve_gradcam_images(
            gradcam_summary=gradcam,
            gradcam_dir=gradcam_dir,
        )

        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else subject_path.parent / f"{case_id}_report.html"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_html = build_html(
            readiness=readiness,
            subject=subject,
            gradcam=gradcam,
            volume_selection=volume_selection,
            representative_rows=representative_rows,
            gradcam_images=gradcam_images,
        )
        output_path.write_text(report_html, encoding="utf-8")

        write_manifest(
            output_path=output_path,
            readiness_path=readiness_path,
            subject_path=subject_path,
            gradcam_path=gradcam_path,
            volume_selection_path=volume_selection_path,
            representative_csv_path=representative_csv_path,
            gradcam_images=gradcam_images,
            case_id=case_id,
        )
        write_summary(
            output_path=output_path,
            readiness=readiness,
            subject=subject,
            image_count=len(gradcam_images),
        )

    except Exception as exc:
        logging.exception("HTML report generation failed: %s", exc)
        return 1

    print("\n" + "=" * 78)
    print(f"CASE ID       : {case_id}")
    print(f"REPORT        : {output_path}")
    print(f"GRADCAM IMAGES: {len(gradcam_images)}")
    print(f"SELF-CONTAINED: YES")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
