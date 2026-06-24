# Data Pipeline Design

## Goal

Create a repeatable pipeline for turning soccer match data and licensed local video clips into model-ready training data.

The pipeline must support:

- World Cup match metadata;
- full-match local video files where permitted;
- goal montage or highlight clips where permitted;
- open event and tracking datasets;
- frame-level player and ball state extraction;
- supervised labels for match and in-play prediction tasks.

## Key rule

The system should never assume that a public video URL is a permitted training asset.

A public URL is only discovery metadata. A video file becomes usable only after it is placed in a licensed local source folder and represented in a manifest with an allowed rights status.

## Data layers

```text
Discovery metadata
  -> candidate URLs, titles, channels, notes, rights status

Licensed local source files
  -> local videos owned, licensed, or otherwise approved for processing

Video manifest
  -> match ID, clip ID, teams, timestamp coverage, rights status, local path

CV outputs
  -> detections, tracks, ball state, player state, pitch coordinates

Feature tables
  -> pressure, possession, compactness, distance to goal, field tilt

Training tables
  -> features joined to labels and match outcomes
```

## Recommended folders

```text
data/raw/video_licensed/
  world_cup/
    full_matches/
    goal_montages/
    highlights/
  club/
    full_matches/
    goal_montages/

manifests/
  video_manifest.csv
  world_cup_matches.csv
```

## Video manifest columns

```text
video_id
match_id
competition
season
home_team
away_team
clip_type
source_url
local_path
period
start_match_second
end_match_second
rights_status
notes
```

Allowed `clip_type` values:

```text
full_match
goal_montage
highlight
training_clip
unknown
```

Allowed `rights_status` values:

```text
owned
licensed
compatible_license
pending
rejected
```

Only these statuses are processable:

```text
owned
licensed
compatible_license
```

## World Cup ingestion strategy

For World Cup work, separate three datasets:

1. `world_cup_matches.csv`
   - match ID;
   - competition;
   - date;
   - teams;
   - group or knockout stage;
   - venue if known;
   - final score when complete.

2. `video_manifest.csv`
   - local source files that are approved for processing;
   - clip timing and match alignment;
   - rights status and notes.

3. `events_or_labels.csv`
   - goals;
   - cards;
   - substitutions;
   - shots;
   - important timestamps;
   - derived labels.

Goal montage videos can be useful for learning shot/goal buildup patterns, but they are biased. They overrepresent scoring sequences and underrepresent normal possession. Treat them as event-focused training examples, not full-match outcome datasets.

## Clip alignment

Each local video should be aligned to match time:

```text
video timestamp -> match period -> match second
```

Minimum useful alignment fields:

```text
period
start_match_second
end_match_second
```

For goal montages, each goal segment should eventually become a separate row:

```text
clip_id
parent_video_id
match_id
scoring_team
period
goal_match_second
segment_start_second
segment_end_second
```

## CV extraction outputs

For each processable video, write:

```text
data/features/video/<video_id>/detections.parquet
data/features/video/<video_id>/tracks.parquet
data/features/video/<video_id>/ball_state.parquet
data/features/video/<video_id>/player_state.parquet
data/features/video/<video_id>/cv_features.parquet
```

## Critical feature families

- player distance to ball;
- nearest and second-nearest player to ball;
- pressure around ball;
- ball speed and acceleration;
- possession confidence;
- possession chain length;
- team compactness;
- defensive line height;
- attacking line height;
- box entries;
- final-third entries;
- distance to goal;
- defenders between ball and goal.

## Bias controls

Goal montage data is not representative of full matches. The model must track source type and use source-aware validation.

Recommended controls:

- keep `clip_type` in the feature table;
- train separate goal-window models and match-outcome models;
- never evaluate match-outcome models only on goal montages;
- include full-match data for probability modeling;
- use goal montages mainly for event recognition, shot buildup, pressure, and goal-window labels.

## First implementation tasks

1. Add a manifest parser.
2. Validate local paths are under `data/raw/video_licensed`.
3. Validate rights status before processing.
4. Add manifest tests.
5. Add World Cup match metadata schema.
6. Add CV batch runner that consumes manifest rows.
7. Add clip alignment helpers.
8. Add goal-segment schema.
9. Add feature extraction stubs.
10. Add one synthetic manifest fixture for tests.