# Highlights Training Summary

- Matches processed: 98/98
- Total detection rows: 88127
- Player detections: 86007, Ball detections: 1541
- Tabular training set: 98 matches (features: n_frames, n_player, n_ball, avg_det_per_frame, ball_center_x, ball_center_y, match_id, home_score, away_score, winner)
- CNN grid samples: 7878 frame-rows

Models fine-tuned:
- tabular winner+score -> /home/scott/git/soccer-analytics/data/processed/highlights/training/model
- CNN grid winner -> /home/scott/git/soccer-analytics/data/processed/highlights/training/cnn_model

## Out-of-sample evaluation (leakage-safe 68/30 stratified split, calibrated classifier)

Two feature sets compared (scripts/evaluate_highlights.py):

| Feature set / metric | Count | Track |
|----------------------|-------|-------|
| Winner accuracy (test) | 53.3% | 50.0% |
| Majority-class baseline | 49.0% | 49.0% |
| Winner Brier (test) | 0.623 | 0.614 |
| Home-score MSE (test) | 1.770 | 1.864 |
| Away-score MSE (test) | 2.345 | 2.066 |
| Train / test rows | 68 / 30 | 68 / 30 |

Track features (training/eval/detection_features_v2.csv): ball_rate,
person_frame_rate, player_ball_min_dist, players_near_ball_mean,
contested_rate, ball_movement, player_box_area_mean, n_frames, n_player,
n_ball.

Conclusion: neither feature set beats the baseline; calibrated Brier scores
(~0.61) confirm near-random probability quality. Highlight-clip CV aggregates
carry no outcome signal. See paper/arxiv_draft.tex.
