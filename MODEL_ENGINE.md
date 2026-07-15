# Predictive Model Engine

## Goal

Build a model engine that predicts match state outcomes from current game-state inputs.

The first deep-learning targets are:

- final home/draw/away probability;
- next event window probability;
- score/no-score window probability;
- possession-danger state;
- pressure and field-position advantage.

## Input representation

The model should combine two streams:

1. **Spatial grid input** for CNN-style learning.
2. **Temporal sequence input** for recurrent or transformer-style learning.

## Spatial grid

Each frame or second can be represented as a pitch grid:

```text
channels x height_bins x width_bins
```

Initial channels:

```text
0 = home/player occupancy
1 = away/player occupancy
2 = ball occupancy
```

Future channels:

```text
3 = home velocity magnitude
4 = away velocity magnitude
5 = ball velocity magnitude
6 = home pressure field
7 = away pressure field
8 = distance-to-goal field
9 = possession mask
```

The current implementation starts with `build_occupancy_grid(...)`.

## Temporal state

The temporal stream should use rolling windows of game state, for example:

```text
last 10 seconds
last 30 seconds
last 60 seconds
last 300 seconds
```

Each timestep can include tabular features:

```text
match_second
score_diff
home_red_cards
away_red_cards
ball_x_m
ball_y_m
ball_speed_mps
nearest_player_distance_to_ball
players_within_3m_ball
players_within_5m_ball
possession_team_code
possession_chain_seconds
pressure_score
field_tilt
home_compactness
away_compactness
home_defensive_line_height
away_defensive_line_height
distance_to_goal
defenders_between_ball_and_goal
```

## Model families

### CNN

Use `FieldStateCNN` for a single current field-state grid. This is useful when the model should learn spatial patterns such as pressure around the ball, defensive line shape, and box-entry danger.

### Temporal model

Use `TemporalModelSpec` for recurrent-style sequence models over tabular state. The initial committed spec is framework-neutral so CI does not require Torch.

### Hybrid model

The intended next model is a hybrid:

```text
per-frame CNN encoder -> sequence model -> probability head
```

The CNN converts each pitch grid into an embedding. The sequence model consumes embeddings plus tabular features and predicts calibrated probabilities.

## Labels

Labels must be leakage-safe.

For a feature row at time `t`, a future-window label may only use outcomes in:

```text
(t, t + window]
```

An event at exactly `t` is already known at prediction time and must not count as a future label.

## Training flow

1. Build frame/player/ball state tables.
2. Convert each frame or second into spatial grids.
3. Build rolling temporal sequences.
4. Generate leakage-safe labels.
5. Split by match date, not random row split.
6. Train baseline tabular model.
7. Train CNN model.
8. Train temporal model.
9. Train hybrid model.
10. Calibrate output probabilities.
11. Evaluate log loss, Brier score, calibration error, and performance by confidence bucket.
12. Report out-of-sample metrics behind a leakage-safe hold-out (stratified match
    split), never in-sample only. For the highlight-clip study this is scripted in
    `scripts/evaluate_highlights.py` (tabular) and `scripts/evaluate_cnn.py` (CNN),
    with `scripts/batch_cnn_eval.sh` sweeping seeds on a batch machine.

## Fine-tuning strategy

Start with generic soccer datasets and then fine-tune on the target distribution:

1. Pretrain on open event/tracking data where labels are reliable.
2. Fine-tune on licensed World Cup-style match video state.
3. Fine-tune a separate goal-window model using goal montage segments.
4. Keep match-outcome training separate from goal-montage training to avoid selection bias.
5. Calibrate on held-out full-match data.

## Immediate implementation tasks

1. Convert track states to pitch-space player state rows.
2. Build per-second occupancy grids.
3. Build rolling sequence samples.
4. Add baseline dataset class for Torch when optional ML deps are installed.
5. Add CNN training loop.
6. Add temporal training loop.
7. Add calibration wrapper for model outputs.
