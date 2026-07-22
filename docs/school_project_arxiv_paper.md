# Fine-Tuning and Calibration Analysis of a Soccer Match Outcome Prediction Model

**Scott Joyner**  
Course Project Submission  
Repository: `soccer-analytics`

## Abstract

This project develops a reproducible soccer analytics pipeline for training, tuning, and evaluating a match outcome prediction model. The system converts soccer match-level features into a supervised learning table, performs task-specific fine-tuning through a regularization sweep, exports held-out predictions, and analyzes scoring quality using accuracy, log loss, multiclass Brier score, and confidence-bin calibration. The final implementation also records dataset version hashes, model metadata, model and data cards, and a paper-ready run summary. The project is designed to support real open soccer data such as StatsBomb Open Data, Metrica Sports Sample Data, OpenFootball JSON results, and football-data.co.uk CSV results, while also including a small classroom-safe demonstration fixture so that the workflow can be reproduced without restricted video or paid data access.

## 1. Introduction

Soccer outcome prediction is a challenging supervised learning problem because match results are affected by tactical state, player form, team strength, shot quality, fatigue, and stochastic events. A project-grade system should therefore do more than produce a class label: it should document the data source, train a model, score the predictions on held-out examples, and report whether predicted probabilities are well calibrated.

The goal of this project is to build an end-to-end soccer prediction pipeline that can be run locally and submitted as a complete analysis artifact. The repository includes modules for open-data ingestion, local rights-gated video processing, player and team feature generation, model training, calibration review, and promotion checks. For this school submission, the final workflow focuses on a lightweight prediction model that can be tuned and evaluated without optional GPU, computer vision, or deep-learning dependencies.

## 2. Data

The project supports several data tiers:

1. **Open event and result data.** StatsBomb Open Data can provide events, lineups, and selected 360 freeze-frame context; OpenFootball and football-data.co.uk can provide match result context; Metrica can provide anonymized event/tracking data.
2. **Licensed local footage.** The repository contains rights-gated commands for processing only local footage with recorded ownership or license references.
3. **Classroom demonstration fixture.** The file `examples/school_project_training.csv` provides a reproducible tabular dataset for the final project workflow. It contains match-level features such as recent shots, expected-goal proxies, player-form proxies, pressure indexes, rest days, and a three-class result label: home win (`H`), draw (`D`), or away win (`A`).

The final submission workflow versions the input dataset using SHA-256 hashes and writes a data card. This prevents the model results from becoming detached from the exact dataset used in the run.

## 3. Feature Representation

Each match is represented by a fixed-length vector:

\[
x_i = [s_h, s_a, xg_h, xg_a, f_h, f_a, p_h, p_a, r_h, r_a]
\]

where `s` denotes recent shot volume, `xg` denotes expected-goal proxy features, `f` denotes player-form proxy features, `p` denotes pressure index features, and `r` denotes rest days. The label is a three-way categorical outcome:

\[
y_i \in \{H, D, A\}.
\]

The broader repository can extend this representation with player-level event aggregates, rolling player form, YOLO-derived player/ball detections, and pitch-state tensors.

## 4. Model Fine-Tuning Method

The final project workflow uses a standardized multinomial logistic-regression classifier. Although lightweight, this is a strong baseline for a school project because it produces interpretable feature weights and calibrated class probabilities.

The model is fine-tuned by sweeping the inverse regularization strength:

\[
C \in \{0.01, 0.1, 1.0, 10.0\}.
\]

For each candidate value, the workflow trains on a stratified training split and scores on a held-out test split. The selected model maximizes held-out accuracy, with log loss as a tie-breaker. The final bundle stores the fitted model, metadata, selected features, and performance metrics.

## 5. Evaluation Metrics

The workflow reports the following metrics:

**Accuracy.** The fraction of held-out matches for which the predicted class equals the observed label.

\[
\text{Accuracy}=\frac{1}{n}\sum_{i=1}^n \mathbb{1}(\hat{y}_i=y_i)
\]

**Log loss.** A probability-sensitive scoring rule that penalizes confident wrong predictions.

\[
\text{LogLoss}=-\frac{1}{n}\sum_{i=1}^n \log p_{i,y_i}
\]

**Multiclass Brier score.** A squared-error probability score over all classes.

\[
\text{Brier}=\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K (p_{i,k}-\mathbb{1}[y_i=k])^2
\]

**Confidence-bin calibration.** Predictions are grouped into confidence bins. For each bin, the workflow compares average confidence to empirical accuracy.

## 6. Reproducible Experiment

Run the project workflow from the repository root:

```bash
python -m soccer_edge.school_project \
  --source examples/school_project_training.csv \
  --output-dir data/processed/school_project
```

The command writes:

- `data/processed/school_project/final_model.joblib`
- `data/processed/school_project/predictions.csv`
- `data/processed/school_project/metrics.json`
- `data/processed/school_project/metrics.csv`
- `data/processed/school_project/hyperparameter_results.csv`
- `data/processed/school_project/calibration_review/metrics.json`
- `data/processed/school_project/calibration_review/calibration.json`
- `data/processed/school_project/confidence_bins.csv`
- `data/processed/school_project/MODEL_CARD.md`
- `data/processed/school_project/DATA_CARD.md`
- `data/processed/school_project/PROJECT_RUN_SUMMARY.md`
- `data/processed/school_project/artifact_index.json`

## 7. Results

The exact values for the final run are written to `data/processed/school_project/PROJECT_RUN_SUMMARY.md` and `metrics.json`. These should be copied into the table below after running the command locally.

| Metric | Value from run artifact |
| --- | --- |
| Selected regularization `C` | See `metrics.json` |
| Held-out accuracy | See `metrics.json` |
| Held-out log loss | See `metrics.json` |
| Held-out multiclass Brier score | See `metrics.json` |
| Training rows | See `metrics.json` |
| Test rows | See `metrics.json` |

Do not manually inflate these values. If results are unsatisfactory, change the features, re-run the workflow, and report the new artifacts.

## 8. Analysis and Discussion

This workflow is intentionally designed around probability quality rather than only class accuracy. A model that is correct 60% of the time but systematically overconfident may be less useful than a slightly less accurate model with better-calibrated probabilities. Log loss and Brier score help expose this behavior.

The hyperparameter sweep demonstrates how regularization affects model performance. Small `C` values apply stronger regularization and may underfit; large `C` values allow larger weights and may overfit. The selected model is the candidate with the best held-out tradeoff under the project scoring rule.

The model is also traceable: dataset hashes, model metadata, data cards, model cards, and prediction rows are exported alongside the model. This makes the result reproducible and auditable.

## 9. Limitations

The bundled classroom fixture is intentionally small and should not be interpreted as a production-quality soccer forecasting dataset. It is useful for demonstrating the modeling pipeline, hyperparameter tuning, probability scoring, and analysis workflow. A stronger version of this project would train on larger open datasets, include true player minutes from lineups, add opponent-adjusted player form, and compare several model families.

The project also avoids unauthorized public video scraping. Computer vision and fine-tuning commands in the broader repository are rights-gated and should be used only with owned, licensed, or compatible-license footage.

## 10. Conclusion

This project provides a reproducible end-to-end soccer prediction workflow that trains and tunes a model, exports held-out predictions, evaluates probability quality, and packages the outputs into a submission-ready analysis. The implementation is structured so that the same workflow can be extended from the classroom fixture to verified open or licensed soccer datasets.

## References

1. StatsBomb Open Data. Public soccer event and lineup data for research and analysis.
2. Metrica Sports Sample Data. Public sample tracking and event data.
3. OpenFootball football.json. Open match result and fixture JSON data.
4. football-data.co.uk. Historical football results and odds-style CSV data.
5. SoccerNet. Benchmark datasets for soccer video understanding and game-state reconstruction.
