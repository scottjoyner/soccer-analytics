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
- [x] Add data pipeline design.

## Phase 1 — Project foundation

- [x] Finalize package name and CLI entry point.
- [x] Add configuration loader.
- [ ] Add structured logging.
- [x] Add linting and test tooling.
- [x] Add CLI smoke tests.
- [x] Add CI workflow.

## Phase 2 — Storage layer

- [x] Implement table read/write helpers.
- [x] Implement DuckDB query helper.
- [x] Define schemas for matches, events, frames, ball states, player states, features, and labels.
- [x] Add basic schema tests.
- [ ] Add optional Neo4j export path.

## Phase 3 — Open data ingestion

- [x] Implement first local StatsBomb JSON loader.
- [ ] Implement Metrica loader.
- [ ] Implement SoccerNet loader.
- [x] Implement generic match catalog loader.
- [x] Add coordinate normalization utility for 105m x 68m pitch.
- [ ] Preserve original coordinates alongside normalized coordinates.
- [ ] Add data lineage columns to each output table.

## Phase 4 — Video discovery metadata

- [x] Implement video discovery metadata schema.
- [x] Store only title, URL, channel, date, search query, notes, and rights status.
- [x] Add tests proving no audiovisual download helper exists.
- [x] Add default rights status: pending.
- [x] Allow video processing only from local licensed folders.

## Phase 5 — Licensed video processing

- [x] Implement batch planner for local licensed videos.
- [x] Add example video manifest.
- [x] Add detector interface and null detector.
- [x] Add initial tracker interface.
- [x] Add ball interpolation with confidence metadata.
- [x] Add manual pitch calibration metadata.
- [x] Add homography transform implementation.
- [ ] Convert pixel detections to pitch-space positions.
- [ ] Write detections, tracks, ball state, and player state to table files.

## Phase 6 — CV feature generation

- [x] Implement nearest player to ball.
- [x] Implement players within radius of ball.
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
- [x] Add leakage-safe label helper.
- [ ] Add time-based train/validation/test split.
- [x] Add synthetic tests for label leakage.

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

## Immediate next commits

1. Convert detections and tracks into pitch-space state rows.
2. Add ball speed and acceleration features.
3. Add possession confidence and chain-duration features.
4. Write CV outputs to table files from the video batch planner.
5. Add first baseline training table builder.