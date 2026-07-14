"""Match-outcome prediction from CV detections + match results.

This is the footage-agnostic training half of the pipeline. It turns YOLO detection
tables (player/ball per frame, optionally pitch-projected) into per-match feature rows,
joins them to match results (home/away score), derives winner/score labels, and finetunes
a winner classifier plus home/away score regressors.

In production the `match_id` associations and final scores come from approved event data
(e.g. StatsBomb). Public video URLs are never inputs; only local licensed detections are.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from soccer_edge.models.bundle import save_bundle
from soccer_edge.models.prediction_export import export_bundle_predictions
from soccer_edge.models.simple_classifier import fit_simple_classifier

PLAYER_CLASSES = {"player", "person"}
BALL_CLASSES = {"ball", "sports ball"}

FEATURE_COLUMNS = [
    "n_player",
    "n_ball",
    "avg_det_per_frame",
    "ball_center_x",
    "ball_center_y",
]


def derive_winner_label(home_score: float, away_score: float) -> int:
    """0 = home win, 1 = draw, 2 = away win."""

    if home_score > away_score:
        return 0
    if away_score > home_score:
        return 2
    return 1


def match_result_labels(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    for column in ("home_score", "away_score"):
        if column not in frame.columns:
            raise ValueError(f"results table missing required column: {column}")
    frame["winner"] = [
        derive_winner_label(home, away) for home, away in zip(frame["home_score"], frame["away_score"])
    ]
    return frame


def aggregate_video_features(detections: pd.DataFrame, video_width: float = 1920.0, video_height: float = 1080.0) -> dict:
    if detections is None or len(detections) == 0:
        return {
            "n_frames": 0,
            "n_player": 0,
            "n_ball": 0,
            "avg_det_per_frame": 0.0,
            "ball_center_x": 0.0,
            "ball_center_y": 0.0,
        }
    classes = detections["class_name"].astype(str).str.lower()
    n_player = int(classes.isin(PLAYER_CLASSES).sum())
    n_ball = int(classes.isin(BALL_CLASSES).sum())
    n_frames = int(detections["frame_idx"].nunique())
    ball = detections[classes.isin(BALL_CLASSES)]
    if len(ball) > 0:
        ball_center_x = float(((ball["x1"] + ball["x2"]) / 2.0).mean())
        ball_center_y = float(((ball["y1"] + ball["y2"]) / 2.0).mean())
    else:
        ball_center_x, ball_center_y = 0.0, 0.0
    return {
        "n_frames": n_frames,
        "n_player": n_player,
        "n_ball": n_ball,
        "avg_det_per_frame": len(detections) / max(n_frames, 1),
        "ball_center_x": ball_center_x / max(video_width, 1.0),
        "ball_center_y": ball_center_y / max(video_height, 1.0),
    }


def build_prediction_dataset(
    detections: pd.DataFrame,
    results: pd.DataFrame,
    match_id: str | None = None,
) -> pd.DataFrame:
    features = aggregate_video_features(detections)
    if match_id is not None:
        features["match_id"] = match_id
        row = results[results["match_id"] == match_id]
        if len(row) == 0:
            raise ValueError(f"no match result for match_id={match_id!r}")
        result = row.iloc[0]
        features["home_score"] = int(result["home_score"])
        features["away_score"] = int(result["away_score"])
        features["winner"] = derive_winner_label(result["home_score"], result["away_score"])
    return pd.DataFrame([features])


def _fit_score_regressor(frame: pd.DataFrame, feature_columns: list[str], target: str, output_dir: Path) -> dict[str, Path]:
    model = RandomForestRegressor(n_estimators=50, random_state=0)
    model.fit(frame[feature_columns], frame[target])
    predictions = model.predict(frame[feature_columns])
    metrics = {"mse": float(mean_squared_error(frame[target], predictions))}
    return save_bundle(
        model=model,
        output_dir=output_dir,
        name="score_regressor",
        version="v0",
        feature_names=feature_columns,
        metrics=metrics,
        notes=f"target={target}",
    )


def build_prediction_dataset_multi(
    results: pd.DataFrame,
    detections_by_match: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a multi-match labeled dataset from per-match detection tables."""

    labeled = match_result_labels(results)
    rows: list[dict] = []
    for match_id in labeled["match_id"]:
        detections = detections_by_match.get(match_id)
        if detections is None:
            continue
        row = aggregate_video_features(detections)
        row["match_id"] = match_id
        result = labeled[labeled["match_id"] == match_id].iloc[0]
        row["home_score"] = int(result["home_score"])
        row["away_score"] = int(result["away_score"])
        row["winner"] = int(result["winner"])
        rows.append(row)
    if not rows:
        raise ValueError("no matches in results matched the supplied detection tables")
    return pd.DataFrame(rows)


def train_match_predictor(
    dataset: pd.DataFrame,
    output_dir: Path,
    feature_columns: Iterable[str] | None = None,
) -> dict[str, object]:
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    output_dir = Path(output_dir)
    missing = [column for column in feature_columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if "winner" not in dataset.columns:
        raise ValueError("dataset missing 'winner' label column")

    winner_paths = fit_simple_classifier(dataset, feature_columns, "winner", output_dir / "winner")
    home_paths = _fit_score_regressor(dataset, feature_columns, "home_score", output_dir / "home_score")
    away_paths = _fit_score_regressor(dataset, feature_columns, "away_score", output_dir / "away_score")

    dataset_path = output_dir / "dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    predictions_path = output_dir / "predictions.csv"
    export_bundle_predictions(output_dir / "winner", dataset_path, predictions_path, feature_columns=feature_columns)

    return {
        "dataset": dataset_path,
        "winner_model": winner_paths["model"],
        "winner_metadata": winner_paths["metadata"],
        "home_score_model": home_paths["model"],
        "away_score_model": away_paths["model"],
        "predictions": predictions_path,
    }


# Occupancy grid defaults for the CNN winner/score path.
GRID_CHANNELS = 3
GRID_HEIGHT = 8
GRID_WIDTH = 8


def grid_column_names(channels: int = GRID_CHANNELS, height: int = GRID_HEIGHT, width: int = GRID_WIDTH) -> list[str]:
    return [f"g{idx}" for idx in range(channels * height * width)]


def frame_occupancy_grid(
    detections_frame: pd.DataFrame,
    channels: int = GRID_CHANNELS,
    height_bins: int = GRID_HEIGHT,
    width_bins: int = GRID_WIDTH,
    video_width: float = 1920.0,
    video_height: float = 1080.0,
) -> np.ndarray:
    """Flattened occupancy grid for one frame from player/ball detections.

    Channel 0 = home players, 1 = away players, 2 = ball. Without pitch calibration
    and team assignment, all players are placed in both player channels; calibration
    + team ID is a later enhancement.
    """

    grid = np.zeros((channels, height_bins, width_bins), dtype=np.float32)
    if detections_frame is None or len(detections_frame) == 0:
        return grid.flatten()

    classes = detections_frame["class_name"].astype(str).str.lower()
    players = detections_frame[classes.isin(PLAYER_CLASSES)]
    balls = detections_frame[classes.isin(BALL_CLASSES)]
    width_bins_f = max(float(width_bins), 1.0)
    height_bins_f = max(float(height_bins), 1.0)

    def bin_index(cx: float, cy: float) -> tuple[int, int]:
        col = min(int(cx / max(video_width, 1.0) * width_bins_f), width_bins - 1)
        row = min(int(cy / max(video_height, 1.0) * height_bins_f), height_bins - 1)
        return row, col

    for _, detection in players.iterrows():
        cx = (float(detection["x1"]) + float(detection["x2"])) / 2.0
        cy = (float(detection["y1"]) + float(detection["y2"])) / 2.0
        row, col = bin_index(cx, cy)
        grid[0, row, col] += 1.0
        if channels > 1:
            grid[1, row, col] += 1.0
    for _, detection in balls.iterrows():
        cx = (float(detection["x1"]) + float(detection["x2"])) / 2.0
        cy = (float(detection["y1"]) + float(detection["y2"])) / 2.0
        row, col = bin_index(cx, cy)
        grid[min(channels - 1, 2), row, col] = 1.0
    return grid.flatten()


def _grid_row(detections: pd.DataFrame, frame_idx: int, channels: int, height_bins: int, width_bins: int, video_width: float, video_height: float) -> dict:
    grid = frame_occupancy_grid(
        detections[detections["frame_idx"] == frame_idx],
        channels=channels,
        height_bins=height_bins,
        width_bins=width_bins,
        video_width=video_width,
        video_height=video_height,
    )
    row = {name: float(value) for name, value in zip(grid_column_names(channels, height_bins, width_bins), grid)}
    row["frame_idx"] = frame_idx
    return row


def build_match_grid_table(
    detections: pd.DataFrame,
    results: pd.DataFrame,
    match_id: str,
    channels: int = GRID_CHANNELS,
    height_bins: int = GRID_HEIGHT,
    width_bins: int = GRID_WIDTH,
    video_width: float = 1920.0,
    video_height: float = 1080.0,
) -> pd.DataFrame:
    labeled = match_result_labels(results)
    row = labeled[labeled["match_id"] == match_id]
    if len(row) == 0:
        raise ValueError(f"no match result for match_id={match_id!r}")
    result = row.iloc[0]
    frames = sorted(detections["frame_idx"].unique().tolist()) if len(detections) > 0 else []
    grid_rows = [
        _grid_row(detections, frame_idx, channels, height_bins, width_bins, video_width, video_height)
        for frame_idx in frames
    ]
    table = pd.DataFrame(grid_rows)
    table["match_id"] = match_id
    table["home_score"] = int(result["home_score"])
    table["away_score"] = int(result["away_score"])
    table["winner"] = int(result["winner"])
    return table


def build_match_grid_table_multi(
    results: pd.DataFrame,
    detections_by_match: dict[str, pd.DataFrame],
    channels: int = GRID_CHANNELS,
    height_bins: int = GRID_HEIGHT,
    width_bins: int = GRID_WIDTH,
    video_width: float = 1920.0,
    video_height: float = 1080.0,
) -> pd.DataFrame:
    labeled = match_result_labels(results)
    tables: list[pd.DataFrame] = []
    for match_id in labeled["match_id"]:
        detections = detections_by_match.get(match_id)
        if detections is None:
            continue
        tables.append(
            build_match_grid_table(
                detections,
                labeled,
                match_id,
                channels=channels,
                height_bins=height_bins,
                width_bins=width_bins,
                video_width=video_width,
                video_height=video_height,
            )
        )
    if not tables:
        raise ValueError("no matches in results matched the supplied detection tables")
    return pd.concat(tables, ignore_index=True)


def train_match_predictor_cnn(
    grid_table: pd.DataFrame,
    output_dir: Path,
    channels: int = GRID_CHANNELS,
    height_bins: int = GRID_HEIGHT,
    width_bins: int = GRID_WIDTH,
    sequence_length: int = 4,
    device: str | None = None,
    epochs: int = 1,
    batch_size: int = 4,
    hidden_size: int = 128,
) -> dict[str, object]:
    from soccer_edge.models.cnn_runner import train_cnn_from_npz
    from soccer_edge.models.tensor_samples import build_npz_from_table

    output_dir = Path(output_dir)
    spatial_columns = grid_column_names(channels, height_bins, width_bins)
    npz_path = output_dir / "grid_samples.npz"
    build_npz_from_table(
        source=_write_grid_csv(grid_table, output_dir),
        output_path=npz_path,
        spatial_columns=spatial_columns,
        label_column="winner",
        sequence_length=sequence_length,
        channels=channels,
        height=height_bins,
        width=width_bins,
        group_column="match_id",
        order_column="frame_idx",
    )
    model_paths = train_cnn_from_npz(
        npz_path,
        output_dir / "winner_cnn",
        output_classes=3,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        device=device,
    )
    return {"npz": npz_path, "model": model_paths["model"], "metadata": model_paths["metadata"]}


def _write_grid_csv(grid_table: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "grid_table.csv"
    grid_table.to_csv(path, index=False)
    return path
