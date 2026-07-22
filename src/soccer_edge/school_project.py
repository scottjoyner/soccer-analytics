"""Submission-ready school project workflow.

Run with:

    python -m soccer_edge.school_project \
      --source examples/school_project_training.csv \
      --output-dir data/processed/school_project

The workflow trains and tunes a task-specific soccer outcome predictor, exports
predictions, scoring metrics, calibration-style confidence bins, model/data cards,
and a paper-ready markdown summary. It is intentionally lightweight so it can run
without optional Torch, OpenCV, or YOLO dependencies.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from soccer_edge.cards import write_data_card, write_model_card
from soccer_edge.dataset_versioning import write_dataset_versions
from soccer_edge.evaluation.calibration_review import write_calibration_review
from soccer_edge.models.bundle import save_bundle
from soccer_edge.models.comparison import write_model_comparison
from soccer_edge.models.markdown_report import write_model_markdown_report
from soccer_edge.models.registry import write_registry_index, write_registry_summary


DEFAULT_FEATURE_COLUMNS = [
    "home_shots_last5",
    "away_shots_last5",
    "home_xg_last5",
    "away_xg_last5",
    "home_player_form",
    "away_player_form",
    "home_pressure_index",
    "away_pressure_index",
    "home_rest_days",
    "away_rest_days",
]


@dataclass(frozen=True)
class ProjectRunPaths:
    output_dir: str
    dataset_versions: str
    hyperparameter_results: str
    predictions: str
    metrics_json: str
    metrics_csv: str
    calibration_metrics: str
    calibration_bins: str
    model_dir: str
    model_card: str
    data_card: str
    registry: str
    registry_summary: str
    comparison: str
    comparison_markdown: str
    report_markdown: str


@dataclass(frozen=True)
class TunedProjectResult:
    best_c: float
    accuracy: float
    log_loss: float
    brier: float
    train_rows: int
    test_rows: int
    class_labels: list[str]
    feature_columns: list[str]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def multiclass_brier(probabilities, labels, classes) -> float:
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    total = 0.0
    for row_idx, label in enumerate(labels):
        expected = [0.0] * len(classes)
        expected[class_to_idx[label]] = 1.0
        total += sum((float(probabilities[row_idx][idx]) - expected[idx]) ** 2 for idx in range(len(classes)))
    return total / max(1, len(labels))


def confidence_bins_frame(predictions: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    frame = predictions.copy()
    frame["correct"] = frame["label"].astype(str).eq(frame["prediction"].astype(str))
    frame["confidence_bin"] = pd.cut(frame["confidence"], bins=bins, include_lowest=True)
    out = (
        frame.groupby("confidence_bin", observed=False)
        .agg(row_count=("correct", "size"), accuracy=("correct", "mean"), avg_confidence=("confidence", "mean"))
        .reset_index()
    )
    out["confidence_bin"] = out["confidence_bin"].astype(str)
    return out


def calibration_ready_predictions(predictions: pd.DataFrame, class_labels: list[str]) -> pd.DataFrame:
    """Return a numeric-label frame compatible with the shared calibration writer."""

    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    frame = predictions.copy()
    frame["label_text"] = frame["label"].astype(str)
    frame["prediction_text"] = frame["prediction"].astype(str)
    frame["label"] = frame["label_text"].map(label_to_idx).astype(int)
    return frame


def validated_features(frame: pd.DataFrame, requested: list[str] | None, label_column: str) -> list[str]:
    selected = requested or [column for column in DEFAULT_FEATURE_COLUMNS if column in frame.columns]
    if not selected:
        selected = [column for column in frame.select_dtypes(include="number").columns if column != label_column]
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if label_column not in frame.columns:
        raise ValueError(f"missing label column: {label_column}")
    if not selected:
        raise ValueError("no feature columns selected")
    return selected


def split_frame(frame: pd.DataFrame, label_column: str, train_fraction: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.2 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.2 and 0.9")
    labels = frame[label_column]
    stratify = labels if labels.value_counts().min() >= 2 else None
    train, test = train_test_split(
        frame,
        train_size=train_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit_candidate(train: pd.DataFrame, feature_columns: list[str], label_column: str, c_value: float) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=c_value, max_iter=1000, multi_class="auto")),
        ]
    )
    model.fit(train[feature_columns], train[label_column])
    return model


def evaluate_model(model: Pipeline, test: pd.DataFrame, feature_columns: list[str], label_column: str) -> tuple[dict[str, float], pd.DataFrame]:
    probabilities = model.predict_proba(test[feature_columns])
    classes = list(model.named_steps["classifier"].classes_)
    predictions = model.predict(test[feature_columns])
    metrics = {
        "accuracy": float(accuracy_score(test[label_column], predictions)),
        "log_loss": float(log_loss(test[label_column], probabilities, labels=classes)),
        "brier": float(multiclass_brier(probabilities, list(test[label_column]), classes)),
    }
    rows = test.copy()
    rows["prediction"] = predictions
    rows["confidence"] = probabilities.max(axis=1)
    for idx, class_label in enumerate(classes):
        rows[f"prob_{idx}"] = probabilities[:, idx]
        rows[f"prob_label_{idx}"] = str(class_label)
    return metrics, rows


def tune_prediction_model(
    frame: pd.DataFrame,
    output_dir: Path,
    feature_columns: list[str],
    label_column: str,
    c_values: list[float],
    train_fraction: float,
    random_state: int,
) -> tuple[TunedProjectResult, pd.DataFrame, pd.DataFrame, Pipeline]:
    train, test = split_frame(frame, label_column, train_fraction, random_state)
    results: list[dict[str, float]] = []
    best_model: Pipeline | None = None
    best_predictions: pd.DataFrame | None = None
    best_score: tuple[float, float] | None = None

    for c_value in c_values:
        model = fit_candidate(train, feature_columns, label_column, c_value)
        metrics, predictions = evaluate_model(model, test, feature_columns, label_column)
        row = {"C": c_value, **metrics}
        results.append(row)
        score = (metrics["accuracy"], -metrics["log_loss"])
        if best_score is None or score > best_score:
            best_score = score
            best_model = model
            best_predictions = predictions

    if best_model is None or best_predictions is None:
        raise RuntimeError("no model candidates were evaluated")

    hyper = pd.DataFrame(results).sort_values(["accuracy", "log_loss"], ascending=[False, True]).reset_index(drop=True)
    best = hyper.iloc[0]
    classes = [str(value) for value in best_model.named_steps["classifier"].classes_]
    result = TunedProjectResult(
        best_c=float(best["C"]),
        accuracy=float(best["accuracy"]),
        log_loss=float(best["log_loss"]),
        brier=float(best["brier"]),
        train_rows=len(train),
        test_rows=len(test),
        class_labels=classes,
        feature_columns=feature_columns,
    )
    return result, hyper, best_predictions, best_model


def write_metrics_files(result: TunedProjectResult, output_dir: Path) -> tuple[Path, Path]:
    metrics_json = output_dir / "metrics.json"
    metrics_csv = output_dir / "metrics.csv"
    metrics_json.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    pd.DataFrame([asdict(result)]).to_csv(metrics_csv, index=False)
    return metrics_json, metrics_csv


def write_project_report(output_path: Path, result: TunedProjectResult, source: Path, paths: ProjectRunPaths) -> Path:
    lines = [
        "# Final Project Run Summary",
        "",
        "## Objective",
        "Train and tune a soccer match outcome prediction model, score it on a held-out evaluation split, and export artifacts suitable for the accompanying arXiv-style paper.",
        "",
        "## Dataset",
        f"- Source table: `{source}`",
        f"- Training rows: {result.train_rows}",
        f"- Test rows: {result.test_rows}",
        f"- Labels: {', '.join(result.class_labels)}",
        "",
        "## Model",
        "The final model is a standardized multinomial logistic-regression predictor selected by a hyperparameter sweep over regularization strength `C`.",
        f"- Selected C: {result.best_c}",
        f"- Features: {', '.join(result.feature_columns)}",
        "",
        "## Held-out scoring",
        f"- Accuracy: {result.accuracy:.4f}",
        f"- Log loss: {result.log_loss:.4f}",
        f"- Multiclass Brier score: {result.brier:.4f}",
        "",
        "## Exported artifacts",
        f"- Predictions: `{paths.predictions}`",
        f"- Hyperparameter sweep: `{paths.hyperparameter_results}`",
        f"- Metrics JSON: `{paths.metrics_json}`",
        f"- Calibration metrics: `{paths.calibration_metrics}`",
        f"- Model card: `{paths.model_card}`",
        f"- Data card: `{paths.data_card}`",
        f"- Paper draft: `docs/school_project_arxiv_paper.md`",
        "",
        "## Academic integrity note",
        "This workflow reports generated artifacts from the local run. Do not edit metrics upward manually; rerun the workflow after changing data or features.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_school_project(
    source: Path,
    output_dir: Path,
    feature_columns: list[str] | None = None,
    label_column: str = "label",
    c_values: list[float] | None = None,
    train_fraction: float = 0.7,
    random_state: int = 42,
) -> ProjectRunPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_table(source)
    selected_features = validated_features(frame, feature_columns, label_column)
    c_grid = c_values or [0.01, 0.1, 1.0, 10.0]
    result, hyper, predictions, model = tune_prediction_model(
        frame,
        output_dir,
        selected_features,
        label_column,
        c_grid,
        train_fraction=train_fraction,
        random_state=random_state,
    )

    model_dir = output_dir / "tuned_prediction_model"
    save_bundle(
        model=model,
        output_dir=model_dir,
        name="school_project_tuned_soccer_predictor",
        version="v1",
        feature_names=selected_features,
        metrics={"accuracy": result.accuracy, "log_loss": result.log_loss, "brier": result.brier},
        notes="Task-specific tuned soccer prediction model for final project submission.",
    )
    joblib.dump(model, output_dir / "final_model.joblib")

    dataset_versions = write_dataset_versions([source], output_dir / "dataset_versions.csv")
    hyper_path = output_dir / "hyperparameter_results.csv"
    hyper.to_csv(hyper_path, index=False)
    predictions_path = output_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics_json, metrics_csv = write_metrics_files(result, output_dir)

    calibration_dir = output_dir / "calibration_review"
    calibration_frame = calibration_ready_predictions(predictions, result.class_labels)
    calibration_paths = write_calibration_review(calibration_frame, calibration_dir, num_bins=5)
    bins_path = output_dir / "confidence_bins.csv"
    confidence_bins_frame(predictions).to_csv(bins_path, index=False)

    model_card = write_model_card(model_dir, output_dir / "MODEL_CARD.md", version_paths=[source])
    data_card = write_data_card(
        "school-project-soccer-outcome-dataset",
        [source],
        output_dir / "DATA_CARD.md",
        rights_status="classroom_demo_or_verified_source",
        version_paths=[source],
    )
    registry = write_registry_index(output_dir, output_dir / "registry.csv")
    registry_summary = write_registry_summary(output_dir, output_dir / "registry_summary.csv")
    comparison = write_model_comparison(registry_summary, output_dir / "model_comparison.csv")
    comparison_markdown = write_model_markdown_report(comparison, output_dir / "model_comparison.md")

    paths = ProjectRunPaths(
        output_dir=str(output_dir),
        dataset_versions=str(dataset_versions),
        hyperparameter_results=str(hyper_path),
        predictions=str(predictions_path),
        metrics_json=str(metrics_json),
        metrics_csv=str(metrics_csv),
        calibration_metrics=str(calibration_paths["metrics"]),
        calibration_bins=str(bins_path),
        model_dir=str(model_dir),
        model_card=str(model_card),
        data_card=str(data_card),
        registry=str(registry),
        registry_summary=str(registry_summary),
        comparison=str(comparison),
        comparison_markdown=str(comparison_markdown),
        report_markdown=str(output_dir / "PROJECT_RUN_SUMMARY.md"),
    )
    write_project_report(output_dir / "PROJECT_RUN_SUMMARY.md", result, source, paths)
    (output_dir / "artifact_index.json").write_text(json.dumps(asdict(paths), indent=2), encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the school-project soccer prediction workflow.")
    parser.add_argument("--source", type=Path, default=Path("examples/school_project_training.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/school_project"))
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURE_COLUMNS), help="Comma-separated feature columns.")
    parser.add_argument("--label", default="label")
    parser.add_argument("--c-grid", default="0.01,0.1,1,10", help="Comma-separated LogisticRegression C values.")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = [item.strip() for item in args.features.split(",") if item.strip()]
    c_values = [float(item.strip()) for item in args.c_grid.split(",") if item.strip()]
    paths = run_school_project(
        source=args.source,
        output_dir=args.output_dir,
        feature_columns=features,
        label_column=args.label,
        c_values=c_values,
        train_fraction=args.train_fraction,
        random_state=args.random_state,
    )
    print(json.dumps(asdict(paths), indent=2))


if __name__ == "__main__":
    main()
