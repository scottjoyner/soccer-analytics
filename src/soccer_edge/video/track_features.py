"""Track-style features derived from per-frame YOLO detections.

The detector runs without a tracker or pitch calibration, so we reconstruct
possession/proximity/movement signals directly from per-frame bounding boxes.
All distances are normalized by video width so features are resolution
independent.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

PLAYER_CLASSES = {"player", "person"}
BALL_CLASSES = {"ball", "sports ball"}
NEAR_BALL_FRACTION = 0.06  # proximity radius as a fraction of video width


def _center(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cx"] = (out["x1"] + out["x2"]) / 2.0
    out["cy"] = (out["y1"] + out["y2"]) / 2.0
    out["area"] = (out["x2"] - out["x1"]) * (out["y1"] - out["y2"]).abs()
    return out


def build_match_track_features(
    detections: pd.DataFrame,
    video_width: float = 1920.0,
    video_height: float = 1080.0,
) -> dict:
    """Aggregate per-frame detections into a single match-level feature row."""
    keys = [
        "n_frames",
        "ball_rate",
        "person_frame_rate",
        "ball_det_count",
        "person_det_count",
        "player_ball_min_dist",
        "players_near_ball_mean",
        "contested_rate",
        "ball_movement",
        "player_box_area_mean",
    ]
    if detections is None or len(detections) == 0:
        return {k: 0.0 for k in keys}

    df = _center(detections)
    classes = df["class_name"].astype(str).str.lower()
    players = df[classes.isin(PLAYER_CLASSES)]
    balls = df[classes.isin(BALL_CLASSES)]

    n_frames = int(df["frame_idx"].nunique())
    ball_frames = sorted(balls["frame_idx"].unique().tolist())
    person_frames = int(players["frame_idx"].nunique())

    ball_rate = len(ball_frames) / max(n_frames, 1)
    person_frame_rate = person_frames / max(n_frames, 1)

    player_ball_min_dist = 0.0
    players_near_ball_mean = 0.0
    contested_rate = 0.0
    ball_movement = 0.0
    near_radius = NEAR_BALL_FRACTION * video_width

    if len(ball_frames) > 0:
        per_frame_ball = balls.groupby("frame_idx")[["cx", "cy"]].mean()
        per_frame_players = {
            fid: players[players["frame_idx"] == fid][["cx", "cy"]].values
            for fid in ball_frames
        }
        min_dists = []
        near_counts = []
        contested = 0
        prev = None
        for fid in ball_frames:
            bc = per_frame_ball.loc[fid].values
            pcs = per_frame_players.get(fid, [])
            if len(pcs) > 0:
                d = (((pcs - bc) ** 2).sum(axis=1) ** 0.5).min()
                min_dists.append(d / max(video_width, 1.0))
                near = int((((pcs - bc) ** 2).sum(axis=1) ** 0.5).min() < near_radius)
                near_counts.append(near)
                if near >= 1:
                    contested += 1
            if prev is not None:
                ball_movement += float((((bc - prev) ** 2).sum() ** 0.5) / max(video_width, 1.0))
            prev = bc
        player_ball_min_dist = float(pd.Series(min_dists).mean()) if min_dists else 0.0
        players_near_ball_mean = float(pd.Series(near_counts).mean()) if near_counts else 0.0
        contested_rate = contested / len(ball_frames)

    player_box_area_mean = float(players["area"].mean() / max(video_width * video_height, 1.0)) if len(players) else 0.0

    return {
        "n_frames": n_frames,
        "ball_rate": ball_rate,
        "person_frame_rate": person_frame_rate,
        "ball_det_count": int(len(balls)),
        "person_det_count": int(len(players)),
        "player_ball_min_dist": player_ball_min_dist,
        "players_near_ball_mean": players_near_ball_mean,
        "contested_rate": contested_rate,
        "ball_movement": ball_movement / max(len(ball_frames) - 1, 1),
        "player_box_area_mean": player_box_area_mean,
    }


def build_track_dataset(
    results: pd.DataFrame,
    detections_by_match: dict[str, pd.DataFrame],
    video_width: float = 1920.0,
    video_height: float = 1080.0,
) -> pd.DataFrame:
    """Build a multi-match labeled dataset from per-match track features."""
    from soccer_edge.pipeline.match_predictor import match_result_labels

    labeled = match_result_labels(results)
    rows: list[dict] = []
    for match_id in labeled["match_id"]:
        detections = detections_by_match.get(match_id)
        if detections is None:
            continue
        row = build_match_track_features(detections, video_width=video_width, video_height=video_height)
        row["match_id"] = match_id
        result = labeled[labeled["match_id"] == match_id].iloc[0]
        row["home_score"] = int(result["home_score"])
        row["away_score"] = int(result["away_score"])
        row["winner"] = int(result["winner"])
        rows.append(row)
    if not rows:
        raise ValueError("no matches in results matched the supplied detection tables")
    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "ball_rate",
    "person_frame_rate",
    "player_ball_min_dist",
    "players_near_ball_mean",
    "contested_rate",
    "ball_movement",
    "player_box_area_mean",
]


def evaluate_match_predictor(
    dataset: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
) -> dict:
    """Leakage-safe train/test split + calibrated winner/score models.

    Returns in-sample and out-of-sample metrics (accuracy, Brier score for the
    winner, MSE for the score regressors) plus written predictions.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import accuracy_score, brier_score_loss, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    missing = [c for c in feature_columns if c not in dataset.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if "winner" not in dataset.columns:
        raise ValueError("dataset missing 'winner' label column")

    X = dataset[feature_columns].fillna(0.0).values
    y_win = dataset["winner"].values
    y_home = dataset["home_score"].values
    y_away = dataset["away_score"].values

    (X_tr, X_te, yw_tr, yw_te, yh_tr, yh_te, ya_tr, ya_te) = train_test_split(
        X, y_win, y_home, y_away, test_size=test_size, random_state=seed, stratify=y_win
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("cal", CalibratedClassifierCV(LogisticRegression(max_iter=2000), cv=3)),
    ])
    clf.fit(X_tr, yw_tr)
    yw_tr_pred = clf.predict(X_tr)
    yw_te_pred = clf.predict(X_te)
    yw_te_proba = clf.predict_proba(X_te)

    def fit_reg(y_tr_target, y_te_target):
        reg = RandomForestRegressor(n_estimators=50, random_state=seed)
        reg.fit(X_tr, y_tr_target)
        return reg.predict(X_tr), reg.predict(X_te)

    home_tr_pred, home_te_pred = fit_reg(yh_tr, yh_te)
    away_tr_pred, away_te_pred = fit_reg(ya_tr, ya_te)

    return {
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "winner_accuracy_train": float(accuracy_score(yw_tr, yw_tr_pred)),
        "winner_accuracy_test": float(accuracy_score(yw_te, yw_te_pred)),
        "winner_brier_test": float(brier_score_loss(yw_te, yw_te_proba)),
        "home_score_mse_train": float(mean_squared_error(yh_tr, home_tr_pred)),
        "home_score_mse_test": float(mean_squared_error(yh_te, home_te_pred)),
        "away_score_mse_train": float(mean_squared_error(ya_tr, away_tr_pred)),
        "away_score_mse_test": float(mean_squared_error(ya_te, away_te_pred)),
        "test_indices": list(range(len(X_te))),
    }
