# TODO Roadmap

## Phase 0 — Repo bootstrap

- [x] Add README.
- [x] Add high-level design.
- [x] Add low-level design.
- [x] Add implementation roadmap.
- [x] Add Codex implementation prompt.
- [x] Add Python package scaffold.
- [x] Add safe environment template.
- [x] Add git ignore rules.

## Phase 1 — Project foundation

- [ ] Finalize package name and CLI entry point.
- [ ] Add configuration loader.
- [ ] Add structured logging.
- [ ] Add linting and test tooling.
- [ ] Add CLI smoke tests.
- [ ] Add CI workflow.

## Phase 2 — Storage layer

- [ ] Implement Parquet read/write helpers.
- [ ] Implement DuckDB query helper.
- [ ] Define schemas for matches, events, detections, tracks, features, predictions, and evaluations.
- [ ] Add schema validation tests.
- [ ] Add optional Neo4j export path.

## Phase 3 — Open data ingestion

- [ ] Implement StatsBomb loader.
- [ ] Implement Metrica loader.
- [ ] Implement SoccerNet loader.
- [ ] Implement fixture/result loader.
- [ ] Normalize coordinates to 105m x 68m where possible.
- [ ] Preserve original coordinates alongside normalized coordinates.
- [ ] Add data lineage columns to each output table.

## Phase 4 — Video discovery metadata

- [ ] Implement video discovery metadata schema.
- [ ] Store only title, URL, channel, date, search query, notes, and rights status.
- [ ] Add tests proving no audiovisual download helper exists.
- [ ] Add rights statuses: pending, licensed, rejected, owned, compatible_license.
- [ ] Allow video processing only from local licensed folders.

## Phase 5 — Licensed video processing

- [ ] Implement batch runner for local licensed videos.
- [ ] Implement YOLO detector wrapper.
- [ ] Implement tracker wrapper.
- [ ] Add ball interpolation with confidence metadata.
- [ ] Add manual homography calibration loader.
- [ ] Convert pixel detections to pitch-space positions.
- [ ] Write detections, tracks, ball state, and player state to Parquet.

## Phase 6 — CV feature generation

- [ ] Implement nearest player to ball.
- [ ] Implement players within 3m and 5m of ball.
- [ ] Implement ball speed and acceleration.
- [ ] Implement possession confidence.
- [ ] Implement possession chain duration.
- [ ] Implement pressure score.
- [ ] Implement team compactness.
- [ ] Implement defensive and attacking line height.
- [ ] Implement final-third and box entries.
- [ ] Implement distance to goal and defenders between ball and goal.

## Phase 7 — Supervised feature tables

- [ ] Build prematch feature table.
- [ ] Build in-play rolling-window feature table.
- [ ] Build label generation with leakage prevention.
- [ ] Add time-based train/validation/test split.
- [ ] Add synthetic fixtures to test label leakage.

## Phase 8 — Modeling

- [ ] Train baseline model.
- [ ] Train Elo/logistic baseline.
- [ ] Train LightGBM or XGBoost tabular model.
- [ ] Add calibration wrapper.
- [ ] Report log loss, Brier score, and calibration curves.
- [ ] Save feature list, model artifact, metrics, and version metadata.

## Phase 9 — Offline evaluation

- [ ] Implement historical replay evaluator.
- [ ] Score predictions by confidence bucket.
- [ ] Score predictions by league, team, and time window.
- [ ] Add report writer.
- [ ] Add notebook for calibration review.

## Phase 10 — Interface and operations

- [ ] Add FastAPI service for predictions.
- [ ] Add local dashboard.
- [ ] Add model registry metadata.
- [ ] Add experiment tracking.
- [ ] Add runtime documentation.
- [ ] Add Docker profile for optional services.

## Phase 11 — Hardening

- [ ] Add robust tests for every feature builder.
- [ ] Add data validation reports.
- [ ] Add drift monitoring.
- [ ] Add duplicate match/video detection.
- [ ] Add reproducible seed control.
- [ ] Add model card template.
- [ ] Add safety checklist for future integrations.