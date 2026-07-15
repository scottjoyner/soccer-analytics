import pandas as pd
from pathlib import Path

from soccer_edge.pipeline.match_predictor import (
    aggregate_video_features,
    build_prediction_dataset_multi,
    derive_winner_label,
    match_result_labels,
    train_match_predictor,
)


def test_run_capture_to_match_predictor_orchestrates(monkeypatch, tmp_path) -> None:
    import pandas as pd
    from soccer_edge.pipeline import match_predictor as mp

    # Skip the real YOLO model; write a synthetic detections table as run_yolo_detection would.
    def fake_detect(input_path, output_dir, model_path, **kwargs):
        from soccer_edge.video.detector import Detection
        from soccer_edge.video.state_tables import write_video_state_tables

        detections = [
            Detection(frame_idx=0, class_name="player", confidence=0.9, x1=0, y1=0, x2=10, y2=10),
            Detection(frame_idx=1, class_name="ball", confidence=0.8, x1=5, y1=5, x2=15, y2=15),
        ]
        return write_video_state_tables(output_dir=output_dir, detections=detections)

    # The trainer needs labeled data from >=2 classes; a single captured match
    # provides one. Stub it so the orchestration (detect->merge->dataset) is
    # exercised without requiring a multi-class training set in this unit test.
    def fake_train(dataset, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "winner_model.pkl").write_bytes(b"stub")
        pd.DataFrame([{"match_id": "m1", "pred_winner": 0}]).to_csv(out / "predictions.csv", index=False)
        return {"winner_model": out / "winner_model.pkl", "predictions": out / "predictions.csv"}

    monkeypatch.setattr(mp, "train_match_predictor", fake_train)

    monkeypatch.setattr(mp, "run_yolo_detection", fake_detect)

    results = pd.DataFrame(
        [
            {"match_id": "m1", "home_score": 2, "away_score": 1},
            {"match_id": "m2", "home_score": 0, "away_score": 3},
            {"match_id": "m3", "home_score": 1, "away_score": 1},
        ]
    )
    out = tmp_path / "out"
    paths = mp.run_capture_to_match_predictor(
        video_path=tmp_path / "clip.mp4",
        results=results,
        output_dir=out,
        model_path="yolov8n.pt",
        match_id="m1",
        event_source=None,
    )
    assert paths["dataset"].exists()
    assert paths["winner_model"].exists()
    assert paths["predictions"].exists()
    dataset = pd.read_csv(paths["dataset"])
    assert dataset.iloc[0]["winner"] == 0  # home win


def test_merge_match_features_merges_detection_and_event() -> None:
    from soccer_edge.pipeline.match_predictor import merge_match_features

    detections = pd.DataFrame(
        [
            {"frame_idx": 0, "class_name": "player", "x1": 0, "y1": 0, "x2": 10, "y2": 10},
            {"frame_idx": 1, "class_name": "person", "x1": 0, "y1": 0, "x2": 10, "y2": 10},
            {"frame_idx": 2, "class_name": "ball", "x1": 5, "y1": 5, "x2": 15, "y2": 15},
        ]
    )
    results = pd.DataFrame(
        [{"match_id": "m1", "home_score": 2, "away_score": 1}]
    )
    event = pd.DataFrame(
        [{"match_id": "m1", "home_xg": 1.4, "away_xg": 0.6, "home_xt": 0.3, "away_xt": 0.2}]
    )
    merged = merge_match_features({"m1": detections}, results, event_features=event)
    assert len(merged) == 1
    row = merged.iloc[0]
    # detection features: 2 players (player+person), 1 ball
    assert row["n_player"] == 2
    assert row["n_ball"] == 1
    assert row["winner"] == 0  # home win
    # event features merged by match_id
    assert row["home_xg"] == 1.4
    assert row["away_xt"] == 0.2


def test_merge_match_features_skips_unlabeled_matches() -> None:
    from soccer_edge.pipeline.match_predictor import merge_match_features

    detections = pd.DataFrame(
        [{"frame_idx": 0, "class_name": "player", "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    )
    results = pd.DataFrame([{"match_id": "m1", "home_score": 1, "away_score": 0}])
    # detection for m2 has no result -> must be skipped
    merged = merge_match_features({"m1": detections, "m2": detections}, results)
    assert list(merged["match_id"]) == ["m1"]


def test_derive_winner_label() -> None:
    assert derive_winner_label(2, 1) == 0
    assert derive_winner_label(0, 2) == 2
    assert derive_winner_label(1, 1) == 1


def test_match_result_labels_adds_winner() -> None:
    results = pd.DataFrame(
        [
            {"match_id": "m1", "home_score": 2, "away_score": 1},
            {"match_id": "m2", "home_score": 0, "away_score": 3},
            {"match_id": "m3", "home_score": 1, "away_score": 1},
        ]
    )
    labeled = match_result_labels(results)
    assert labeled["winner"].tolist() == [0, 2, 1]


def test_aggregate_video_features_handles_empty() -> None:
    features = aggregate_video_features(pd.DataFrame())
    assert features["n_player"] == 0
    assert features["n_ball"] == 0


def test_aggregate_video_features_counts_classes() -> None:
    detections = pd.DataFrame(
        [
            {"frame_idx": 0, "class_name": "player", "x1": 0, "y1": 0, "x2": 10, "y2": 20},
            {"frame_idx": 0, "class_name": "person", "x1": 0, "y1": 0, "x2": 10, "y2": 20},
            {"frame_idx": 1, "class_name": "ball", "x1": 5, "y1": 5, "x2": 10, "y2": 10},
        ]
    )
    features = aggregate_video_features(detections, video_width=100.0, video_height=100.0)
    assert features["n_player"] == 2
    assert features["n_ball"] == 1
    assert features["n_frames"] == 2
    assert 0.0 < features["ball_center_x"] < 1.0


def test_train_match_predictor_writes_bundles(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {"match_id": "m1", "home_score": 2, "away_score": 1},
            {"match_id": "m2", "home_score": 0, "away_score": 3},
            {"match_id": "m3", "home_score": 1, "away_score": 1},
        ]
    )
    detections = {
        "m1": pd.DataFrame([{"frame_idx": 0, "class_name": "player", "x1": 0, "y1": 0, "x2": 10, "y2": 20}]),
        "m2": pd.DataFrame([{"frame_idx": 0, "class_name": "player", "x1": 0, "y1": 0, "x2": 10, "y2": 20}]),
        "m3": pd.DataFrame([{"frame_idx": 0, "class_name": "ball", "x1": 5, "y1": 5, "x2": 10, "y2": 10}]),
    }
    labeled = match_result_labels(results)
    dataset = build_prediction_dataset_multi(labeled, detections)
    paths = train_match_predictor(dataset, tmp_path)
    assert paths["winner_model"].exists()
    assert paths["home_score_model"].exists()
    assert paths["away_score_model"].exists()
    assert paths["predictions"].exists()
    preds = pd.read_csv(paths["predictions"])
    assert {"prob_0", "prob_1", "prob_2"}.issubset(preds.columns)
