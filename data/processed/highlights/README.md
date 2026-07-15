# Highlights Dataset (FIFA World Cup 2026)

Aggregated from permitted local footage for a research project.

- Source dir: /media/scott/SSD_4TB/fileserver/highlights (98 local MP4 highlight files)
- Rights status: licensed (user-confirmed permitted local files)
- rights_reference: permitted-local:/media/scott/SSD_4TB/fileserver/highlights
- Lineage: results parsed from filenames (no audiovisual content scraped/downloaded);
  CV detection blocked on originals by AV1 codec (no decoder in media reader);
  transcoded-to-H.264 clips process via detect-yolo (validated on 1 clip).

## Files
- match_results.csv: 98 matches (home/away, scores, penalties, winner, stage, source path)
- team_aggregates.csv: 50 teams (matches, W/D/L, GF, GA, GD)
- match_training_features.csv: per-match joined team stats + labels (winner, scores)
  for match-predictor / simple classifier fine-tuning
- detections/<match_id>/detections.parquet (+ tracks/ball_states/player_states): per-match
  YOLO detection tables for all 98 matches (AV1 originals transcoded to H.264, then
  detect-yolo). Totals: 88,127 detection rows (86,007 player, 1,541 ball).
- training/: cleaned training dataset + fine-tuned models
  - detection_features.csv: tabular set (n_player, n_ball, avg_det_per_frame, ball_center_x/y) + labels
  - grid_table.csv / grid_samples.npz: per-frame occupancy grids for the CNN winner model
  - model/: tabular winner classifier + home/away score regressors (predictions.csv, dataset.csv)
  - cnn_model/: CNN grid->winner bundle
  - cnn_eval/: out-of-sample CNN winner evaluation (68/30 match hold-out) — metrics.json
    reports sequence/match accuracy (51.6% / 50.0%, at the majority-class baseline)
    and winner Brier (0.617); trained only on the train split
  - training_summary.md: processing + fine-tune summary
- highlights_training_data.zip: bundle of the above, **including the fine-tuned model .joblib binaries**

## Evaluation
Out-of-sample results are scripted (leakage-safe, stratified 68/30 match hold-out):

```bash
python scripts/evaluate_highlights.py   # tabular winner + score, calibrated
python scripts/evaluate_cnn.py          # CNN winner (sequence + match accuracy, Brier)
```

Neither the tabular nor the CNN model beats the majority-class baseline: highlight
reels are highlight-selected and too short for aggregate detection counts to carry
final-score signal. See `paper/arxiv_draft.tex` for the full discussion.

Generated: 2026-07-14T21:08:46.550884Z
