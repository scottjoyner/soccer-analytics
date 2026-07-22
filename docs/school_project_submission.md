# School Project Submission Guide

This guide packages the repo into a Friday-ready submission for a project about fine-tuning a soccer prediction model and analyzing scoring quality.

## One-command run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
python -m soccer_edge.school_project \
  --source examples/school_project_training.csv \
  --output-dir data/processed/school_project
```

## What to submit

Submit these files/folders:

```text
docs/school_project_arxiv_paper.md
data/processed/school_project/PROJECT_RUN_SUMMARY.md
data/processed/school_project/metrics.json
data/processed/school_project/metrics.csv
data/processed/school_project/hyperparameter_results.csv
data/processed/school_project/predictions.csv
data/processed/school_project/confidence_bins.csv
data/processed/school_project/calibration_review/
data/processed/school_project/MODEL_CARD.md
data/processed/school_project/DATA_CARD.md
data/processed/school_project/artifact_index.json
```

Optional, if the grader wants the trained model artifact:

```text
data/processed/school_project/final_model.joblib
data/processed/school_project/tuned_prediction_model/
```

## What the project demonstrates

- A supervised soccer match outcome prediction problem.
- Task-specific model fine-tuning through a logistic-regression regularization sweep.
- A held-out train/test split.
- Scoring with accuracy, log loss, and multiclass Brier score.
- Confidence-bin calibration analysis.
- Dataset version hashing.
- Model/data cards.
- A paper-style writeup with method, metrics, limitations, and reproducibility details.

## Before submitting

1. Run the command above.
2. Open `data/processed/school_project/PROJECT_RUN_SUMMARY.md`.
3. Copy the actual metric values into Section 7 of `docs/school_project_arxiv_paper.md`, or attach both files so the grader can see the generated results directly.
4. Do not manually improve metrics. Re-run after changing features instead.
5. If asked whether this is real-world production data, say: the included fixture is a classroom-safe reproducible demonstration dataset; the code also supports verified open and licensed soccer data sources.

## Fast explanation for presentation

> I built a soccer outcome prediction pipeline. I fine-tuned a multinomial logistic-regression model by sweeping regularization strength, selected the best model on a held-out split, and evaluated it with accuracy, log loss, Brier score, and calibration bins. I also exported predictions, metrics, model metadata, dataset hashes, a model card, a data card, and an arXiv-style paper.

## Commands for inspection

```bash
cat data/processed/school_project/PROJECT_RUN_SUMMARY.md
cat data/processed/school_project/metrics.json
head data/processed/school_project/predictions.csv
cat data/processed/school_project/hyperparameter_results.csv
cat data/processed/school_project/confidence_bins.csv
```
