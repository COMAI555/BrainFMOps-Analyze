# BrainFMOps-Analyze

BrainFMOps-Analyze is a research prototype for quality-aware, reproducible,
subject-level brain MRI classification. The repository contains dataset-readiness
checks, leakage-aware splitting, EfficientNet-B0 training and inference,
subject-level aggregation, Grad-CAM generation, statistical evaluation, and
publication-support notebooks.

## Research boundary

This software is for research and methodological audit only. It is not a medical
device, clinical decision-support system, or diagnostic tool. The archived
case study does not establish clinical or comparative classifier performance.

## Public data policy

The original OASIS MRI files and clinical spreadsheet are not redistributed.
Users must obtain OASIS data from the official source and comply with its
data-use terms. Local data, checkpoints, derived images, and private outputs
belong in ignored folders such as `data/`, `workspace/`, `checkpoints/`, and
`outputs/`.

The public evaluation table contains release-only identifiers and the minimum
fields required to verify the reported evaluation arithmetic:

```text
public_data/evaluation_predictions_212_sanitized.csv
```

## Repository structure

```text
configs/       Public experiment configuration
docs/          Public audit and provenance metadata
notebooks/     Evaluation and publication-support notebooks
public_data/   Sanitized evaluation outputs only
src/           Pipeline source code
validation/    Public evaluation and repository-audit checks
```

## Environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Verify the released evaluation table

From the repository root:

```bash
python validation/verify_public_evaluation.py
```

Expected result:

```text
Subjects: 212
TP=73, TN=36, FP=88, FN=15
Accuracy=0.514151
Precision=0.453416
Recall=0.829545
Specificity=0.290323
Brier score=0.243408
Verification: PASS
```

## Pre-publication audit

Run before every public release:

```bash
python validation/audit_repository.py
```

The audit checks for local absolute paths, possible secrets, blocked MRI/model
files, stored notebook outputs, and subject identifiers in public data files.

## Notebook execution

Start Jupyter from the repository root so relative paths resolve consistently:

```bash
jupyter lab
```

Notebooks that require original OASIS clinical data or MRI files will stop with
a missing-file error until the user supplies those files locally. This is
intentional; the repository never fabricates labels or redistributes source data.

## Reproducibility status

The recovered 212-subject evaluation table permits independent verification of
the confusion matrix, threshold-dependent metrics, ROC-AUC, Brier score, and
fixed-bin calibration calculations. It does not restore the original optimizer
state, complete development manifest, or full training provenance. The archived
experiment is therefore only partially reproducible.

## Citation and licence

Author metadata, `CITATION.cff`, and the final software licence will be added
after the author order, affiliations, and copyright holder are confirmed.
