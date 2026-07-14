"""Build the cleaned highlights training dataset from per-match detections and fine-tune
the match-predictor (winner + score) and its CNN variant.

Reads:
  data/processed/highlights/match_results.csv
  data/processed/highlights/detections/<match_id>/detections.parquet
Writes:
  data/processed/highlights/training/detection_features.csv   (tabular training set)
  data/processed/highlights/training/grid_table.csv           (per-frame occupancy grids)
  data/processed/highlights/training/grid_samples.npz         (CNN tensor samples)
  data/processed/highlights/training/model/                   (tabular bundles + predictions)
  data/processed/highlights/training/cnn_model/               (CNN bundle)
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from soccer_edge.pipeline.match_predictor import (
    build_match_grid_table_multi,
    build_prediction_dataset_multi,
    train_match_predictor,
    train_match_predictor_cnn,
)

REPO = Path("/home/scott/git/soccer-analytics")
DET_ROOT = REPO / "data/processed/highlights/detections"
RESULTS = REPO / "data/processed/highlights/match_results.csv"
OUT = REPO / "data/processed/highlights/training"
OUT.mkdir(parents=True, exist_ok=True)


def load_detections() -> dict[str, pd.DataFrame]:
    by_match: dict[str, pd.DataFrame] = {}
    total_rows = 0
    total_player = 0
    total_ball = 0
    for path in sorted(DET_ROOT.glob("*/*detections.parquet")):
        mid = path.parent.name
        df = pd.read_parquet(path)
        by_match[mid] = df
        total_rows += len(df)
        classes = df["class_name"].astype(str).str.lower()
        total_player += int(classes.isin({"player", "person"}).sum())
        total_ball += int(classes.isin({"ball", "sports ball"}).sum())
    return by_match, dict(total_rows=total_rows, total_player=total_player, total_ball=total_ball)


def main() -> None:
    results = pd.read_csv(RESULTS)
    by_match, stats = load_detections()
    print(f"detection tables: {len(by_match)} matches, rows={stats['total_rows']}, "
          f"player_det={stats['total_player']}, ball_det={stats['total_ball']}")

    dataset = build_prediction_dataset_multi(results, by_match)
    print(f"tabular training set: {len(dataset)} matches, features={list(dataset.columns)}")
    dataset.to_csv(OUT / "detection_features.csv", index=False)

    grid = build_match_grid_table_multi(results, by_match)
    print(f"CNN grid table: {len(grid)} frame-rows, columns={len(grid.columns)}")
    grid.to_csv(OUT / "grid_table.csv", index=False)

    print("=== fine-tune tabular match-predictor (winner + score) ===")
    tabular = train_match_predictor(dataset, OUT / "model")
    print("tabular paths:", {k: str(v) for k, v in tabular.items()})

    print("=== fine-tune CNN match-predictor (grid -> winner) ===")
    cnn = train_match_predictor_cnn(grid, OUT / "cnn_model", sequence_length=4, epochs=3, batch_size=8, hidden_size=128)
    print("cnn paths:", {k: str(v) for k, v in cnn.items()})

    summary = (
        f"# Highlights Training Summary\n\n"
        f"- Matches processed: {len(by_match)}/98\n"
        f"- Total detection rows: {stats['total_rows']}\n"
        f"- Player detections: {stats['total_player']}, Ball detections: {stats['total_ball']}\n"
        f"- Tabular training set: {len(dataset)} matches (features: {', '.join(dataset.columns)})\n"
        f"- CNN grid samples: {len(grid)} frame-rows\n\n"
        f"Models fine-tuned:\n"
        f"- tabular winner+score -> {OUT/'model'}\n"
        f"- CNN grid winner -> {OUT/'cnn_model'}\n"
    )
    (OUT / "training_summary.md").write_text(summary, encoding="utf-8")
    print("wrote", OUT / "training_summary.md")


if __name__ == "__main__":
    main()
