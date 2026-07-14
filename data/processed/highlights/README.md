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
  - training_summary.md: processing + fine-tune summary
- highlights_training_data.zip: bundle of the above (model .joblib binaries excluded)

Generated: 2026-07-14T21:08:46.550884Z
