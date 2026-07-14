# Highlights Training Summary

- Matches processed: 98/98
- Total detection rows: 88127
- Player detections: 86007, Ball detections: 1541
- Tabular training set: 98 matches (features: n_frames, n_player, n_ball, avg_det_per_frame, ball_center_x, ball_center_y, match_id, home_score, away_score, winner)
- CNN grid samples: 7878 frame-rows

Models fine-tuned:
- tabular winner+score -> /home/scott/git/soccer-analytics/data/processed/highlights/training/model
- CNN grid winner -> /home/scott/git/soccer-analytics/data/processed/highlights/training/cnn_model
