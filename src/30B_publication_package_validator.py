
#!/usr/bin/env python3
"""
30B_publication_package_validator.py

BrainFMOps-Analyze — STEP 30B
Publication Package Validator

Purpose
-------
Validate whether the BrainFMOps-Analyze research package contains the
artifacts required for manuscript preparation.

Validation domains
------------------
1. Dataset completeness
2. Full-cohort evaluation
3. Performance reports
4. Statistical validation
5. Explainability outputs
6. Reproducibility metadata
7. Provenance and leakage control
8. Publication figures
9. Smart publication package

Outputs
-------
30B_Publication_Readiness/
├── publication_readiness_report.json
├── publication_readiness_checklist.csv
├── publication_readiness_summary.txt
├── publication_readiness_report.html
└── missing_artifacts.csv

Important interpretation
------------------------
The validator distinguishes among:
- ENGINEERING_READY
- MANUSCRIPT_READY_WITH_LIMITATIONS
- NOT_READY_FOR_INDEPENDENT_PERFORMANCE_CLAIM
- NOT_READY

A high engineering completeness score does not override incomplete provenance.
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


SCRIPT_VERSION = "1.0.0"


@dataclass
class CheckResult:
    domain: str
    item: str
    status: str
    score_awarded: float
    score_maximum: float
    evidence: str
    severity: str
    recommendation: str


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Could not read JSON %s: %s", path, exc)
        return {}


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return sum(1 for _ in csv.DictReader(file_obj))


def find_first(root: Path, filename: str) -> Optional[Path]:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def find_all(root: Path, filename: str) -> list[Path]:
    return list(root.rglob(filename))


def count_matching(root: Path, pattern: str) -> int:
    return sum(1 for path in root.rglob(pattern) if path.is_file())


def add_check(
    checks: list[CheckResult],
    domain: str,
    item: str,
    passed: bool,
    score: float,
    evidence: str,
    severity: str,
    recommendation: str,
    partial: bool = False,
) -> None:
    if passed:
        status = "PASS"
        awarded = score
    elif partial:
        status = "PARTIAL"
        awarded = score * 0.5
    else:
        status = "FAIL"
        awarded = 0.0

    checks.append(
        CheckResult(
            domain=domain,
            item=item,
            status=status,
            score_awarded=awarded,
            score_maximum=score,
            evidence=evidence,
            severity=severity,
            recommendation=recommendation,
        )
    )


def build_checks(root: Path) -> tuple[list[CheckResult], dict[str, Any]]:
    checks: list[CheckResult] = []
    evidence: dict[str, Any] = {}

    evaluation_csv = root / "evaluation_summary.csv"
    evaluation_rows = count_csv_rows(evaluation_csv)
    evidence["evaluation_rows"] = evaluation_rows

    add_check(
        checks,
        "Dataset",
        "Full evaluation summary exists",
        evaluation_csv.exists(),
        6,
        str(evaluation_csv),
        "critical",
        "Generate STEP 25A full evaluation summary.",
    )

    add_check(
        checks,
        "Dataset",
        "Full cohort contains at least 100 subjects",
        evaluation_rows >= 100,
        6,
        f"{evaluation_rows} records",
        "critical",
        "Run the complete evaluation cohort; pilot results are insufficient.",
    )

    manifest_path = find_first(root, "dataset_manifest.json")
    manifest = load_json(manifest_path)
    manifest_subjects = manifest.get("dataset_statistics", {}).get(
        "total_subjects"
    )
    evidence["dataset_manifest"] = str(manifest_path) if manifest_path else None
    evidence["manifest_subjects"] = manifest_subjects

    add_check(
        checks,
        "Dataset",
        "Dataset manifest exists",
        bool(manifest),
        5,
        str(manifest_path) if manifest_path else "Not found",
        "major",
        "Run STEP 28A Dataset Manifest Generator.",
    )

    add_check(
        checks,
        "Dataset",
        "Manifest subject count matches evaluation",
        (
            bool(manifest)
            and manifest_subjects is not None
            and int(manifest_subjects) == evaluation_rows
        ),
        3,
        f"manifest={manifest_subjects}, evaluation={evaluation_rows}",
        "major",
        "Regenerate the manifest from the current evaluation CSV.",
        partial=bool(manifest) and manifest_subjects is not None,
    )

    performance_path = find_first(root, "performance_report.json")
    performance = load_json(performance_path)
    classification = performance.get("classification_evaluation", {})
    classification_available = bool(classification.get("available"))
    labeled_count = classification.get("labeled_case_count", 0)
    metrics = classification.get("metrics", {})

    evidence["performance_report"] = (
        str(performance_path) if performance_path else None
    )
    evidence["classification_available"] = classification_available
    evidence["labeled_count"] = labeled_count
    evidence["performance_metrics"] = metrics

    add_check(
        checks,
        "Performance",
        "Full performance report exists",
        bool(performance),
        6,
        str(performance_path) if performance_path else "Not found",
        "critical",
        "Run STEP 25D Full Evaluation Report Builder.",
    )

    add_check(
        checks,
        "Performance",
        "Classification metrics are available",
        classification_available,
        5,
        f"labeled cases={labeled_count}",
        "critical",
        "Provide valid labels and rerun STEP 25B.",
    )

    add_check(
        checks,
        "Performance",
        "At least 100 labeled subjects",
        int(labeled_count or 0) >= 100,
        4,
        f"{labeled_count} labeled cases",
        "major",
        "Increase the labeled independent evaluation cohort.",
        partial=int(labeled_count or 0) > 0,
    )

    required_metric_names = (
        "accuracy",
        "precision",
        "sensitivity_recall",
        "specificity",
        "f1_score",
        "roc_auc",
        "pr_auc",
    )
    available_metric_count = sum(
        metrics.get(name) is not None
        for name in required_metric_names
    )

    add_check(
        checks,
        "Performance",
        "Core classification metrics are complete",
        available_metric_count == len(required_metric_names),
        5,
        f"{available_metric_count}/{len(required_metric_names)} metrics",
        "major",
        "Regenerate the full performance report.",
        partial=available_metric_count > 0,
    )

    statistical_path = find_first(root, "statistical_validation_report.json")
    statistical = load_json(statistical_path)
    statistical_metrics = statistical.get("metrics", {})
    bootstrap = statistical.get("stratified_bootstrap", {})
    evidence["statistical_report"] = (
        str(statistical_path) if statistical_path else None
    )

    add_check(
        checks,
        "Statistics",
        "Statistical validation report exists",
        bool(statistical),
        6,
        str(statistical_path) if statistical_path else "Not found",
        "critical",
        "Run STEP 25D to generate full STEP 25C outputs.",
    )

    core_stat_names = (
        "matthews_correlation_coefficient",
        "cohens_kappa",
        "brier_score",
        "log_loss",
    )
    core_stat_count = sum(
        statistical_metrics.get(name) is not None
        for name in core_stat_names
    )

    add_check(
        checks,
        "Statistics",
        "Advanced statistics are available",
        core_stat_count == len(core_stat_names),
        4,
        f"{core_stat_count}/{len(core_stat_names)} statistics",
        "major",
        "Regenerate STEP 25C from the full labeled cohort.",
        partial=core_stat_count > 0,
    )

    bootstrap_required = (
        "accuracy",
        "sensitivity_recall",
        "specificity",
        "f1_score",
        "roc_auc",
    )
    bootstrap_count = sum(
        bool(bootstrap.get(name))
        for name in bootstrap_required
    )

    add_check(
        checks,
        "Statistics",
        "Bootstrap confidence intervals are available",
        bootstrap_count == len(bootstrap_required),
        4,
        f"{bootstrap_count}/{len(bootstrap_required)} intervals",
        "major",
        "Run bootstrap validation with at least 2,000 iterations.",
        partial=bootstrap_count > 0,
    )

    html_reports = count_matching(root, "*_report.html")
    gradcam_summaries = count_matching(root, "gradcam_summary.json")
    evidence["html_reports"] = html_reports
    evidence["gradcam_summaries"] = gradcam_summaries

    add_check(
        checks,
        "Explainability",
        "HTML case reports generated for full cohort",
        evaluation_rows > 0 and html_reports >= evaluation_rows,
        5,
        f"{html_reports}/{evaluation_rows} HTML reports",
        "major",
        "Regenerate missing STEP 24D case reports.",
        partial=html_reports > 0,
    )

    add_check(
        checks,
        "Explainability",
        "Grad-CAM summaries generated for full cohort",
        evaluation_rows > 0 and gradcam_summaries >= evaluation_rows,
        5,
        f"{gradcam_summaries}/{evaluation_rows} Grad-CAM summaries",
        "major",
        "Regenerate missing STEP 24C outputs.",
        partial=gradcam_summaries > 0,
    )

    figure_dir = root / "30A_Paper_Figures"
    expected_figures = [
        "Fig_1_BrainFMOps_Analyze_Pipeline.png",
        "Fig_2_Dataset_and_Readiness_Profile.png",
        "Fig_3_Subject_Level_Performance.png",
        "Fig_4_Statistical_Validation.png",
        "Fig_5_Explainability_Representative_Cases.png",
        "Fig_6_Provenance_and_Reproducibility.png",
    ]
    figure_exists = [
        (figure_dir / filename).exists()
        for filename in expected_figures
    ]
    figure_count = sum(figure_exists)
    evidence["publication_figure_count"] = figure_count

    add_check(
        checks,
        "Figures",
        "Six paper figures exist",
        figure_count == len(expected_figures),
        7,
        f"{figure_count}/{len(expected_figures)} figures",
        "major",
        "Rerun STEP 30A after completing full reports.",
        partial=figure_count > 0,
    )

    figure_summary_path = figure_dir / "paper_figure_summary.txt"
    figure_manifest_path = figure_dir / "paper_figure_manifest.json"
    figure_manifest = load_json(figure_manifest_path)
    figure_warnings = figure_manifest.get("warnings", [])
    evidence["figure_warnings"] = figure_warnings

    add_check(
        checks,
        "Figures",
        "Paper figures have no unresolved generation warnings",
        len(figure_warnings) == 0,
        3,
        f"{len(figure_warnings)} warning(s)",
        "major",
        "Resolve warnings and rerun STEP 30A.",
        partial=figure_count == len(expected_figures),
    )

    provenance_path = find_first(root, "subject_overlap_report.json")
    provenance = load_json(provenance_path)
    provenance_decision = provenance.get("decision", "NOT_PROVIDED")
    evidence["provenance_report"] = (
        str(provenance_path) if provenance_path else None
    )
    evidence["provenance_decision"] = provenance_decision

    add_check(
        checks,
        "Provenance",
        "Subject overlap audit exists",
        bool(provenance),
        5,
        str(provenance_path) if provenance_path else "Not found",
        "critical",
        "Run STEP 27A Subject Overlap Audit.",
    )

    add_check(
        checks,
        "Provenance",
        "Training/validation/test/evaluation provenance is complete",
        provenance_decision in {"LEAKAGE_FREE", "LEAKAGE_FOUND"},
        4,
        provenance_decision,
        "critical",
        "Locate the original split manifests used to train the checkpoint.",
        partial=provenance_decision == "INCOMPLETE_PROVENANCE",
    )

    add_check(
        checks,
        "Provenance",
        "Evaluation is leakage-free",
        provenance_decision == "LEAKAGE_FREE",
        7,
        provenance_decision,
        "critical",
        (
            "Do not claim independent test performance until original split "
            "manifests prove no overlap."
        ),
    )

    input_hashes = manifest.get("input_hashes", {})
    has_hashes = bool(
        input_hashes.get("evaluation_csv_sha256")
        and input_hashes.get("labels_csv_sha256")
    )
    manifest_uuid = manifest.get("manifest_uuid")

    add_check(
        checks,
        "Reproducibility",
        "Dataset hashes are recorded",
        has_hashes,
        4,
        str(input_hashes),
        "major",
        "Regenerate STEP 28A with evaluation and labels inputs.",
    )

    add_check(
        checks,
        "Reproducibility",
        "Dataset manifest UUID is recorded",
        bool(manifest_uuid),
        3,
        str(manifest_uuid or "Not found"),
        "major",
        "Regenerate STEP 28A.",
    )

    batch_config_path = root / "batch_configuration.json"
    add_check(
        checks,
        "Reproducibility",
        "Batch configuration is preserved",
        batch_config_path.exists(),
        3,
        str(batch_config_path),
        "major",
        "Preserve STEP 25A batch_configuration.json.",
    )

    smart_package = root / "29A1_Smart_Publication_Package"
    smart_manifest_path = smart_package / "smart_publication_manifest.json"
    smart_manifest = load_json(smart_manifest_path)

    add_check(
        checks,
        "Packaging",
        "Smart publication package exists",
        bool(smart_manifest),
        4,
        str(smart_manifest_path),
        "major",
        "Run STEP 29A.1 Smart Publication Collector.",
    )

    missing_expected = smart_manifest.get(
        "missing_expected_artifact_count",
        None,
    )
    add_check(
        checks,
        "Packaging",
        "Smart package has no missing expected artifacts",
        missing_expected == 0,
        3,
        f"missing={missing_expected}",
        "minor",
        "Inspect missing_expected_artifacts.csv and regenerate missing outputs.",
        partial=isinstance(missing_expected, int),
    )

    return checks, evidence


def summarize_domains(checks: list[CheckResult]) -> dict[str, Any]:
    domains: dict[str, dict[str, float]] = {}

    for check in checks:
        domain = domains.setdefault(
            check.domain,
            {"awarded": 0.0, "maximum": 0.0},
        )
        domain["awarded"] += check.score_awarded
        domain["maximum"] += check.score_maximum

    for domain in domains.values():
        domain["percentage"] = (
            100.0 * domain["awarded"] / domain["maximum"]
            if domain["maximum"] else 0.0
        )

    return domains


def determine_readiness(
    checks: list[CheckResult],
    score_percentage: float,
    provenance_decision: str,
) -> dict[str, Any]:
    critical_failures = [
        check
        for check in checks
        if check.status == "FAIL" and check.severity == "critical"
    ]

    engineering_failures = [
        check
        for check in critical_failures
        if check.domain != "Provenance"
    ]

    engineering_ready = len(engineering_failures) == 0
    independent_claim_ready = provenance_decision == "LEAKAGE_FREE"

    if not engineering_ready:
        status = "NOT_READY"
    elif not independent_claim_ready:
        status = "MANUSCRIPT_READY_WITH_LIMITATIONS"
    elif score_percentage >= 85:
        status = "READY_FOR_MANUSCRIPT"
    else:
        status = "ENGINEERING_READY"

    return {
        "status": status,
        "engineering_ready": engineering_ready,
        "independent_performance_claim_ready": independent_claim_ready,
        "critical_failure_count": len(critical_failures),
        "critical_failures": [
            f"{check.domain}: {check.item}"
            for check in critical_failures
        ],
        "interpretation": (
            "The engineering package is complete, but independent test "
            "performance cannot be claimed because provenance is incomplete."
            if engineering_ready and not independent_claim_ready
            else (
                "The package is ready for manuscript preparation."
                if engineering_ready and independent_claim_ready
                else "Critical research artifacts are still missing."
            )
        ),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def build_html(report: dict[str, Any]) -> str:
    check_rows = []
    for check in report["checks"]:
        check_rows.append(
            f"""
            <tr>
              <td>{check['domain']}</td>
              <td>{check['item']}</td>
              <td>{check['status']}</td>
              <td>{check['score_awarded']:.1f}/{check['score_maximum']:.1f}</td>
              <td>{check['evidence']}</td>
              <td>{check['recommendation']}</td>
            </tr>
            """
        )

    domain_rows = []
    for name, values in report["domain_scores"].items():
        domain_rows.append(
            f"""
            <tr>
              <td>{name}</td>
              <td>{values['awarded']:.1f}</td>
              <td>{values['maximum']:.1f}</td>
              <td>{values['percentage']:.1f}%</td>
            </tr>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BrainFMOps Publication Readiness</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
h1, h2 {{ margin-bottom: 8px; }}
.summary {{ border: 1px solid #bbb; padding: 16px; margin-bottom: 24px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #eee; }}
.notice {{ border-left: 5px solid #777; padding: 12px; background: #f5f5f5; }}
</style>
</head>
<body>
<h1>BrainFMOps-Analyze Publication Readiness Report</h1>
<div class="summary">
<p><strong>Status:</strong> {report['readiness']['status']}</p>
<p><strong>Overall score:</strong> {report['overall_score_percentage']:.1f}%</p>
<p><strong>Engineering ready:</strong> {report['readiness']['engineering_ready']}</p>
<p><strong>Independent performance claim ready:</strong>
{report['readiness']['independent_performance_claim_ready']}</p>
<p>{report['readiness']['interpretation']}</p>
</div>

<h2>Domain Scores</h2>
<table>
<tr><th>Domain</th><th>Awarded</th><th>Maximum</th><th>Percentage</th></tr>
{''.join(domain_rows)}
</table>

<h2>Detailed Checklist</h2>
<table>
<tr>
<th>Domain</th><th>Item</th><th>Status</th><th>Score</th>
<th>Evidence</th><th>Recommendation</th>
</tr>
{''.join(check_rows)}
</table>

<div class="notice">
<strong>Methodological boundary:</strong>
A high package score does not establish leakage-free or clinically valid
performance. Independent test claims require complete split provenance and
a leakage-free decision from STEP 27A.
</div>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate BrainFMOps research artifacts for manuscript readiness."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    root = args.evaluation_root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Evaluation root not found: {root}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "30B_Publication_Readiness"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        checks, evidence = build_checks(root)

        total_awarded = sum(check.score_awarded for check in checks)
        total_maximum = sum(check.score_maximum for check in checks)
        overall_percentage = (
            100.0 * total_awarded / total_maximum
            if total_maximum else 0.0
        )

        domain_scores = summarize_domains(checks)
        provenance_decision = evidence.get(
            "provenance_decision",
            "NOT_PROVIDED",
        )
        readiness = determine_readiness(
            checks,
            overall_percentage,
            provenance_decision,
        )

        report = {
            "schema_version": "1.0",
            "script_version": SCRIPT_VERSION,
            "generated_at_utc": utc_now_iso(),
            "evaluation_root": str(root),
            "overall_score_awarded": total_awarded,
            "overall_score_maximum": total_maximum,
            "overall_score_percentage": overall_percentage,
            "domain_scores": domain_scores,
            "readiness": readiness,
            "evidence": evidence,
            "checks": [asdict(check) for check in checks],
            "research_use_only": True,
            "clinical_diagnosis": False,
            "methodological_boundary": (
                "Engineering completeness does not establish independent "
                "clinical performance. Complete subject-level provenance is "
                "required before making leakage-free test claims."
            ),
        }

        (output_dir / "publication_readiness_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        write_csv(
            [asdict(check) for check in checks],
            output_dir / "publication_readiness_checklist.csv",
        )

        missing_rows = [
            {
                "domain": check.domain,
                "item": check.item,
                "severity": check.severity,
                "evidence": check.evidence,
                "recommendation": check.recommendation,
            }
            for check in checks
            if check.status != "PASS"
        ]
        write_csv(
            missing_rows,
            output_dir / "missing_artifacts.csv",
        )

        lines = [
            "=" * 88,
            "BrainFMOps-Analyze — Publication Readiness Summary",
            "=" * 88,
            f"Overall score                     : {overall_percentage:.2f}%",
            f"Readiness status                  : {readiness['status']}",
            f"Engineering ready                 : {readiness['engineering_ready']}",
            (
                "Independent performance claim ready: "
                f"{readiness['independent_performance_claim_ready']}"
            ),
            f"Critical failures                 : {readiness['critical_failure_count']}",
            "",
            "Domain Scores",
            "-" * 88,
        ]

        for name, values in domain_scores.items():
            lines.append(
                f"{name:26s}: {values['awarded']:.1f}/"
                f"{values['maximum']:.1f} "
                f"({values['percentage']:.1f}%)"
            )

        lines.extend(
            [
                "",
                "Interpretation",
                "-" * 88,
                readiness["interpretation"],
                "",
                "Non-passing checks",
                "-" * 88,
            ]
        )

        non_passing = [
            check for check in checks if check.status != "PASS"
        ]
        if non_passing:
            for check in non_passing:
                lines.append(
                    f"- [{check.status}] {check.domain}: {check.item}"
                )
        else:
            lines.append("- None")

        lines.extend(
            [
                "",
                "Methodological boundary",
                "-" * 88,
                report["methodological_boundary"],
                "=" * 88,
            ]
        )

        (output_dir / "publication_readiness_summary.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        (output_dir / "publication_readiness_report.html").write_text(
            build_html(report),
            encoding="utf-8",
        )

    except Exception as exc:
        logging.exception("Publication validation failed: %s", exc)
        return 1

    print("\n" + "=" * 88)
    print("PUBLICATION PACKAGE VALIDATION COMPLETED")
    print(f"OVERALL SCORE        : {overall_percentage:.2f}%")
    print(f"STATUS               : {readiness['status']}")
    print(f"ENGINEERING READY    : {readiness['engineering_ready']}")
    print(
        "INDEPENDENT CLAIM   : "
        f"{readiness['independent_performance_claim_ready']}"
    )
    print(f"CRITICAL FAILURES    : {readiness['critical_failure_count']}")
    print(f"OUTPUT               : {output_dir}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    sys.exit(main())
