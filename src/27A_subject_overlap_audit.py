#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, logging, re, sys
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_VERSION = "1.0.0"
DEFAULT_ID_COLUMNS = ("case_id","subject_id","subject","id","patient_id","participant_id","oasis_id")

@dataclass
class SplitInfo:
    name: str
    source: str
    source_type: str
    subject_count: int
    duplicate_count: int
    subjects: list[str]
    warnings: list[str]

def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def normalize_subject_id(value: str) -> str:
    value = Path(value.strip()).name
    match = re.search(r"(OAS\d+_\d+_MR\d+)", value, flags=re.I)
    if match:
        return match.group(1).upper()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-").upper()

def deduplicate(values):
    normalized = [normalize_subject_id(v) for v in values if str(v).strip()]
    counts = Counter(normalized)
    return sorted(counts), sum(c - 1 for c in counts.values() if c > 1)

def detect_id_column(fieldnames, requested):
    if requested:
        if requested not in fieldnames:
            raise ValueError(f"ID column '{requested}' not found. Available: {fieldnames}")
        return requested
    lower_map = {x.lower(): x for x in fieldnames}
    for candidate in DEFAULT_ID_COLUMNS:
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError(f"Cannot detect subject ID column. Available: {fieldnames}")

def load_csv(path: Path, id_column: Optional[str]):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        column = detect_id_column(reader.fieldnames, id_column)
        values = [row[column].strip() for row in reader if row.get(column,"").strip()]
    return values, ([] if values else ["No subject IDs found."])

def load_txt(path: Path):
    values = [x.strip() for x in path.read_text(encoding="utf-8-sig").splitlines()
              if x.strip() and not x.strip().startswith("#")]
    return values, ([] if values else ["No subject IDs found."])

def extract_json_values(obj):
    values = []
    if isinstance(obj, str):
        values.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            values.extend(extract_json_values(item))
    elif isinstance(obj, dict):
        for key, item in obj.items():
            if key.lower() in {"case_id","subject_id","subject","id","patient_id",
                               "participant_id","subjects","train_subjects",
                               "val_subjects","test_subjects"}:
                values.extend(extract_json_values(item))
    return values

def load_json_source(path: Path):
    values = extract_json_values(json.loads(path.read_text(encoding="utf-8")))
    return values, ([] if values else ["No recognizable subject IDs found."])

def load_directory(path: Path):
    values = []
    for item in path.rglob("*"):
        if item.is_dir():
            sid = normalize_subject_id(item.name)
            if re.fullmatch(r"OAS\d+_\d+_MR\d+", sid):
                values.append(sid)
    return values, ([] if values else ["No OASIS-style subject folders found."])

def load_split(name, source, id_column):
    if source is None:
        return None
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"{name} source not found: {source}")
    if source.is_dir():
        values, warnings, source_type = *load_directory(source), "directory"
    elif source.suffix.lower() == ".csv":
        values, warnings, source_type = *load_csv(source, id_column), "csv"
    elif source.suffix.lower() in {".txt",".list"}:
        values, warnings, source_type = *load_txt(source), "txt"
    elif source.suffix.lower() == ".json":
        values, warnings, source_type = *load_json_source(source), "json"
    else:
        raise ValueError(f"Unsupported source type: {source}")
    subjects, duplicates = deduplicate(values)
    if duplicates:
        warnings.append(f"Removed {duplicates} duplicate subject occurrence(s).")
    return SplitInfo(name, str(source), source_type, len(subjects), duplicates, subjects, warnings)

def overlaps(split_map):
    names = list(split_map)
    pair_rows = []
    matrix = {a:{} for a in names}
    for a in names:
        sa = set(split_map[a].subjects)
        for b in names:
            matrix[a][b] = len(sa & set(split_map[b].subjects))
    for i,a in enumerate(names):
        for b in names[i+1:]:
            sa, sb = set(split_map[a].subjects), set(split_map[b].subjects)
            ov = sorted(sa & sb)
            pair_rows.append({
                "split_a":a,"split_b":b,"count_a":len(sa),"count_b":len(sb),
                "overlap_count":len(ov),
                "overlap_rate_a":len(ov)/len(sa) if sa else 0.0,
                "overlap_rate_b":len(ov)/len(sb) if sb else 0.0,
                "overlap_subjects":" | ".join(ov),
            })
    return pair_rows, matrix

def membership_rows(split_map):
    all_subjects = sorted(set().union(*(set(x.subjects) for x in split_map.values())))
    rows = []
    for sid in all_subjects:
        row = {"subject_id":sid}
        memberships = []
        for name, info in split_map.items():
            present = sid in set(info.subjects)
            row[name] = int(present)
            if present: memberships.append(name)
        row["membership_count"] = len(memberships)
        row["memberships"] = " | ".join(memberships)
        rows.append(row)
    return rows

def decision(split_map, pair_rows):
    required = {"training","validation","test","evaluation"}
    missing = sorted(required - set(split_map))
    if missing:
        return "INCOMPLETE_PROVENANCE", [], [
            "Missing provenance source(s): " + ", ".join(missing),
            "Do not claim leakage-free evaluation until all relevant manifests are supplied."
        ]
    critical = {
        frozenset(("training","validation")),
        frozenset(("training","test")),
        frozenset(("training","evaluation")),
        frozenset(("validation","test")),
        frozenset(("validation","evaluation")),
        frozenset(("test","evaluation")),
    }
    reasons = []
    for row in pair_rows:
        if frozenset((row["split_a"],row["split_b"])) in critical and row["overlap_count"] > 0:
            reasons.append(f'{row["split_a"]} ∩ {row["split_b"]} = {row["overlap_count"]} subject(s)')
    if reasons:
        return "LEAKAGE_FOUND", reasons, []
    return "LEAKAGE_FREE", ["No subject overlap detected among supplied manifests."], []

def write_csv(rows, path):
    if not rows: return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def plot_matrix(matrix, path, dpi):
    names = list(matrix)
    values = np.array([[matrix[a][b] for b in names] for a in names], dtype=int)
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(values); fig.colorbar(im, ax=ax, label="Shared subjects")
    ax.set_xticks(range(len(names)), labels=names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), labels=names)
    ax.set_title("Subject Overlap Matrix")
    midpoint = values.max()/2 if values.size else 0
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j,i,str(values[i,j]),ha="center",va="center",
                    color="white" if values[i,j] > midpoint else "black")
    fig.tight_layout(); fig.savefig(path,dpi=dpi,bbox_inches="tight"); plt.close(fig)

def plot_distribution(split_map, path, dpi):
    names = list(split_map); counts = [split_map[n].subject_count for n in names]
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(len(names)); ax.bar(x, counts)
    ax.set_xticks(x, labels=names, rotation=25, ha="right")
    ax.set_ylabel("Unique subjects"); ax.set_title("Subject Distribution by Split")
    for xi,c in zip(x,counts): ax.text(xi,c,str(c),ha="center",va="bottom")
    fig.tight_layout(); fig.savefig(path,dpi=dpi,bbox_inches="tight"); plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-source", type=Path)
    p.add_argument("--validation-source", type=Path)
    p.add_argument("--test-source", type=Path)
    p.add_argument("--evaluation-source", type=Path, required=True)
    p.add_argument("--training-id-column")
    p.add_argument("--validation-id-column")
    p.add_argument("--test-id-column")
    p.add_argument("--evaluation-id-column")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    eval_source = args.evaluation_source.expanduser().resolve()
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              (eval_source if eval_source.is_dir() else eval_source.parent) / "27A_Subject_Overlap_Audit")
    output.mkdir(parents=True, exist_ok=True)

    split_map = {}
    try:
        specs = [
            ("training",args.training_source,args.training_id_column),
            ("validation",args.validation_source,args.validation_id_column),
            ("test",args.test_source,args.test_id_column),
            ("evaluation",args.evaluation_source,args.evaluation_id_column),
        ]
        for name,source,column in specs:
            info = load_split(name,source,column)
            if info:
                split_map[name]=info
                logging.info("%s: %d unique subjects",name,info.subject_count)

        pair_rows, matrix = overlaps(split_map)
        members = membership_rows(split_map)
        final_decision, reasons, warnings = decision(split_map,pair_rows)
        for info in split_map.values():
            warnings.extend(f"{info.name}: {w}" for w in info.warnings)

        write_csv(pair_rows, output/"subject_overlap_pairs.csv")
        write_csv(members, output/"subject_membership_matrix.csv")
        plot_matrix(matrix, output/"Fig_overlap_matrix.png", args.dpi)
        plot_distribution(split_map, output/"Fig_subject_distribution.png", args.dpi)

        report = {
            "schema_version":"1.0","script_version":SCRIPT_VERSION,
            "generated_at_utc":utc_now_iso(),"decision":final_decision,
            "split_information":{n:asdict(i) for n,i in split_map.items()},
            "pairwise_overlaps":pair_rows,"overlap_matrix":matrix,
            "decision_reasons":reasons,"warnings":warnings,
            "interpretation_boundary":"Conclusion is valid only for supplied manifests."
        }
        (output/"subject_overlap_report.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
        )

        lines = [
            "="*84,
            "BrainFMOps-Analyze — Subject Overlap and Provenance Audit",
            "="*84,
            f"FINAL DECISION : {final_decision}",
            "",
            "Subject Counts",
            "-"*84,
        ]
        for n,i in split_map.items():
            lines.append(f"{n:20s}: {i.subject_count}")
        lines += ["","Reasons","-"*84] + [f"- {x}" for x in reasons]
        lines += ["","Warnings","-"*84] + ([f"- {x}" for x in warnings] or ["- None"])
        (output/"provenance_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")

    except Exception as exc:
        logging.exception("Subject overlap audit failed: %s", exc)
        return 1

    print("\n"+"="*84)
    print(f"DECISION   : {final_decision}")
    for n,i in split_map.items():
        print(f"{n.upper():11s}: {i.subject_count}")
    print(f"OUTPUT     : {output}")
    print("="*84)
    if final_decision == "INCOMPLETE_PROVENANCE":
        print("WARNING: Do not claim leakage-free evaluation yet.")
    elif final_decision == "LEAKAGE_FOUND":
        print("WARNING: Overlap detected; full evaluation is not an independent test.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
