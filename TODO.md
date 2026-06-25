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
- [x] Add predictive model engine design.

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
- [x] Convert pixel detections to pitch-space positions.
- [x] Write detections, tracks, ball state, and player state to table files.

## Phase 6 — CV feature generation

- [x] Implement nearest player to ball.
- [x] Implement players within radius of ball.
- [x] Add occupancy-grid tensor builder for CNN inputs.
- [x] Implement ball speed and acceleration.
- [x] Implement possession confidence.
- [x] Implement possession chain duration.
- [x] Implement pressure score.
- [x] Implement team compactness.
- [x] Implement defensive and attacking line height.
- [x] Implement final-third and box entries.
- [x] Implement distance to goal and players between ball and goal.

## Phase 7 — Supervised feature tables

- [ ] Build prematch feature table.
- [ ] Build in-play rolling-window feature table.
- [x] Add leakage-safe label helper.
- [ ] Add time-based train/validation/test split.
- [x] Add synthetic tests for label leakage.

## Phase 8 — Modeling

- [x] Add optional ML requirements file.
- [x] Add game-state feature and training config objects.
- [x] Add CNN model shell for field-state grids.
- [x] Add temporal model spec for recurrent-style sequence models.
- [x] Add hybrid CNN plus temporal model implementation.
- [x] Add Torch dataset for rolling game-state samples.
- [x] Add CNN training loop.
- [x] Add temporal training loop.
- [x] Add calibration wrapper.
- [x] Report log loss, Brier score, accuracy, and calibration curves.
- [x] Save feature list, model artifact, metrics, and version metadata.

## Phase 9 — Offline evaluation

- [x] Implement historical replay evaluator.
- [x] Score predictions by confidence bucket.
- [ ] Score predictions by league, team, and time window.
- [ ] Add report writer.
- [ ] Add notebook for calibration review.

## Phase 10 — CLI workflows

- [x] Add feature build command.
- [x] Add model bundle save command.
- [x] Add model evaluate command.

## Immediate next commits

1. Add league/team/time-window grouped replay scoring.
2. Add report writer for metrics, calibration, and replay outputs.
3. Add end-to-end sample data fixture that exercises feature build -> bundle save -> evaluate.
4. Add Neo4j export path for match, frame, feature, and model-run metadata.
5. Add structured logging.
