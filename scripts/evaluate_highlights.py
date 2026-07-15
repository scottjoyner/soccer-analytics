"""Out-of-sample evaluation of the match-predictor on the 98 highlights.

Builds two feature sets from the per-match detections:
  v1: aggregate_video_features (n_player, n_ball, avg_det_per_frame, ball_center)
  v2: build_match_track_features (possession/proximity/movement signals)
Runs a leakage-safe train/test split + calibrated winner classifier and score
regressors, reporting in-sample vs out-of-sample metrics.
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

from soccer_edge.pipeline.match_predictor import build_prediction_dataset_multi
from soccer_edge.video.track_features import build_track_dataset, evaluate_match_predictor

REPO = Path("/home/scott/git/soccer-analytics")
DET_ROOT = REPO / "data/processed/highlights/detections"
RESULTS = REPO / "data/processed/highlights/match_results.csv"
OUT = REPO / "data/processed/highlights/training/eval"
OUT.mkdir(parents=True, exist_ok=True)


def load():
    results = pd.read_csv(RESULTS)
    by_match = {}
    for p in sorted(DET_ROOT.glob("*/*detections.parquet")):
        by_match[p.parent.name] = pd.read_parquet(p)
    return results, by_match


def main():
    results, by_match = load()

    v1 = build_prediction_dataset_multi(results, by_match)
    v1.to_csv(OUT / "detection_features_v1.csv", index=False)
    v2 = build_track_dataset(results, by_match)
    v2.to_csv(OUT / "detection_features_v2.csv", index=False)

    v1_metrics = evaluate_match_predictor(v1, feature_columns=[
        "n_player", "n_ball", "avg_det_per_frame", "ball_center_x", "ball_center_y"])
    v2_metrics = evaluate_match_predictor(v2)

    out = {"v1_count_features": v1_metrics, "v2_track_features": v2_metrics}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== V1 (count features) ===")
    for k, val in v1_metrics.items():
        print(f"  {k}: {val}")
    print("=== V2 (track features) ===")
    for k, val in v2_metrics.items():
        print(f"  {k}: {val}")
    print("wrote", OUT / "metrics.json")


if __name__ == "__main__":
    main()
