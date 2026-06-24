# Low-Level Design

## 1. Scope

This document defines the first implementation pass for the soccer analytics research system. The initial build focuses on data ingestion, licensed-video processing, feature engineering, probability modeling, calibration, and offline evaluation.

This first version does not implement live execution against any external financial or prediction platform.

## 2. Package layout

```text
src/soccer_edge/
  __init__.py
  cli.py
  config.py
  ingest/
    __init__.py
    statsbomb_loader.py
    metrica_loader.py
    soccernet_loader.py
    video_discovery.py
    fixtures_loader.py
  video/
    __init__.py
    batch_runner.py
    detector.py
    tracker.py
    homography.py
    possession.py
    pitch_features.py
    clip_registry.py
  features/
    __init__.py
    prematch_features.py
    inplay_features.py
    cv_features.py
    labels.py
  models/
    __init__.py
    train_prematch.py
    train_inplay.py
    calibrate.py
    evaluate.py
    registry.py
  evaluation/
    __init__.py
    simulator.py
    replay.py
    scoring.py
  store/
    __init__.py
    parquet.py
    db.py
    neo4j_graph.py
```

## 3. Configuration

Use `pydantic-settings` for environment-backed configuration.

```text
SOCCER_EDGE_ENV=dev
SOCCER_EDGE_DATA_DIR=data
SOCCER_EDGE_ENABLE_EXTERNAL_EXECUTION=false
SOCCER_EDGE_YOUTUBE_API_KEY=
SOCCER_EDGE_NEO4J_URI=bolt://localhost:7687
SOCCER_EDGE_NEO4J_USER=neo4j
SOCCER_EDGE_NEO4J_PASSWORD=
```

All external execution must remain disabled in this bootstrap.

## 4. Data schemas

### Match

```text
match_id
competition_id
season_id
kickoff_utc
home_team_id
away_team_id
venue
neutral_site
```

### Event

```text
event_id
match_id
period
timestamp_seconds
team_id
player_id
event_type
x_m
y_m
outcome
```

### Detection

```text
video_id
frame_idx
timestamp_seconds
class_name
confidence
x1
y1
x2
y2
```

### Track state

```text
video_id
frame_idx
timestamp_seconds
track_id
class_name
confidence
x_m
y_m
bbox_x1
bbox_y1
bbox_x2
bbox_y2
```

### Prediction

```text
prediction_id
match_id
model_name
model_version
timestamp_utc
target
home_win_probability
draw_probability
away_win_probability
next_goal_home_probability
next_goal_away_probability
no_goal_probability
```

## 5. Ingest modules

### StatsBomb loader

Input folder:

```text
data/raw/statsbomb/
```

Outputs:

```text
data/processed/statsbomb/competitions.parquet
data/processed/statsbomb/matches.parquet
data/processed/statsbomb/events.parquet
data/processed/statsbomb/lineups.parquet
data/processed/statsbomb/shots.parquet
data/processed/statsbomb/freeze_frames.parquet
```

Validation:

- all event rows have match IDs;
- timestamps are non-negative;
- coordinates are normalized into meters;
- output schemas are stable.

### Metrica loader

Input folder:

```text
data/raw/metrica/
```

Outputs:

```text
data/processed/metrica/tracking_home.parquet
data/processed/metrica/tracking_away.parquet
data/processed/metrica/events.parquet
```

Coordinate policy:

- normalize to a 105m x 68m pitch;
- preserve original coordinates;
- add `x_m` and `y_m` fields.

### SoccerNet loader

Input folder:

```text
data/raw/soccernet/
```

Outputs:

```text
data/processed/soccernet/annotations.parquet
data/processed/soccernet/tracks.parquet
```

### Video discovery

Purpose: collect search metadata for manual review and rights-clearing only.

The module must never download, cache, mirror, or store audiovisual content.

Output:

```text
data/interim/video_discovery/video_candidates.parquet
```

Columns:

```text
video_id
url
title
channel
published_at
query
snippet
rights_status
notes
created_at
```

Allowed `rights_status` values:

```text
pending
licensed
rejected
owned
compatible_license
```

## 6. Video pipeline

Licensed local input folder:

```text
data/raw/video_licensed/
```

Outputs:

```text
data/features/video/<video_id>/detections.parquet
data/features/video/<video_id>/tracks.parquet
data/features/video/<video_id>/ball_state.parquet
data/features/video/<video_id>/player_state.parquet
data/features/video/<video_id>/cv_features.parquet
```

### Detector

Detect classes:

- player;
- goalkeeper;
- referee;
- ball;
- staff or other if supported by the model.

### Tracker

Requirements:

- stable track IDs;
- gap metadata;
- confidence propagation;
- ball interpolation metadata.

### Homography

Manual calibration file:

```json
{
  "video_id": "example_match_clip",
  "pitch_length_m": 105.0,
  "pitch_width_m": 68.0,
  "pixel_points": [[100, 900], [350, 250], [1500, 250], [1800, 900]],
  "pitch_points_m": [[0, 68], [0, 0], [105, 0], [105, 68]],
  "confidence": 0.8,
  "notes": "manual initial calibration"
}
```

### Possession model

Inputs:

- ball pitch position;
- player pitch positions;
- team assignment;
- ball velocity;
- recent possession history;
- detection confidence.

Rules:

- use meters rather than pixels;
- avoid forced possession when confidence is low;
- expose uncertainty;
- require short-window persistence.

## 7. Feature engineering

### CV features

```text
nearest_player_distance_to_ball
second_nearest_player_distance_to_ball
players_within_3m_ball
players_within_5m_ball
team_in_possession
possession_chain_seconds
ball_speed_mps
ball_acceleration_mps2
team_compactness_home
team_compactness_away
home_defensive_line_height
away_defensive_line_height
pressure_score
box_entry
final_third_entry
progressive_carry
distance_to_goal
defenders_between_ball_and_goal
```

### Prematch features

```text
elo_home
elo_away
elo_diff
form_points_last_5_home
form_points_last_5_away
goals_for_last_5_home
goals_for_last_5_away
goals_against_last_5_home
goals_against_last_5_away
rest_days_home
rest_days_away
neutral_site
```

### In-play features

Rolling windows:

```text
last_60s
last_300s
last_600s
```

Feature families:

- score state;
- cards;
- shots;
- expected goals;
- field tilt;
- possession;
- pressure;
- spatial control.

## 8. Label generation and leakage prevention

Rules:

- features must be generated only from information available at prediction time;
- future outcomes are used only as labels;
- train, validation, and test splits must be time-based;
- no same-match future events can leak into earlier feature rows.

Targets:

```text
final_home_win
final_draw
final_away_win
next_goal_team
goal_within_5min
goal_within_10min
no_goal_next_10min
```

## 9. Model training

Commands:

```bash
soccer-edge train prematch
soccer-edge train inplay
soccer-edge calibrate
```

Outputs:

```text
data/models/<model_name>/model.pkl
data/models/<model_name>/feature_list.json
data/models/<model_name>/metrics.json
```

Metrics:

- log loss;
- Brier score;
- calibration error;
- reliability curve data;
- offline decision-quality score.

## 10. Evaluation

Evaluation loop:

1. Load historical feature rows.
2. Generate prediction at timestamp T.
3. Compare to known outcome.
4. Score calibration and accuracy.
5. Aggregate results by time, league, team, and confidence bucket.

## 11. CLI design

```bash
soccer-edge --help
soccer-edge ingest statsbomb --path data/raw/statsbomb
soccer-edge ingest metrica --path data/raw/metrica
soccer-edge ingest soccernet --path data/raw/soccernet
soccer-edge discover video --query "soccer goal highlights"
soccer-edge video process --input data/raw/video_licensed
soccer-edge features build
soccer-edge train prematch
soccer-edge train inplay
soccer-edge calibrate
soccer-edge evaluate
```

## 12. Test plan

Initial tests:

- config loads with safe defaults;
- coordinate normalization maps values into pitch bounds;
- video discovery does not create download paths;
- possession assignment returns no possession under low confidence;
- label generation prevents future leakage;
- CLI smoke test succeeds.

## 13. Implementation order

1. Project skeleton and CLI.
2. Config and storage utilities.
3. Dataset loaders.
4. Synthetic sample data and tests.
5. CV processing interfaces.
6. Feature builders.
7. Baseline model training.
8. Calibration.
9. Offline evaluation.
10. Optional Neo4j export.
11. Dashboard/API.