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
- [ ] Write detections, tracks, ball state, and player state to table files.

## Phase 6 — CV feature generation

- [x] Implement nearest player to ball.
- [x] Implement players within radius of ball.
- [x] Add occupancy-grid tensor builder for CNN inputs.
- [x] Implement ball speed and acceleration.
- [x] Implement possession confidence.
- [x] Implement possession chain duration.
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

- [x] Add optional ML requirements file.
- [x] Add game-state feature and training config objects.
- [x] Add CNN model shell for field-state grids.
- [x] Add temporal model spec for recurrent-style sequence models.
- [ ] Add hybrid CNN plus temporal model implementation.
- [x] Add Torch dataset for rolling game-state samples.
- [x] Add CNN training loop.
- [ ] Add temporal training loop.
- [ ] Add calibration wrapper.
- [ ] Report log loss, Brier score, and calibration curves.
- [ ] Save feature list, model artifact, metrics, and version metadata.

## Phase 9 — Offline evaluation

- [ ] Implement historical replay evaluator.
- [ ] Score predictions by confidence bucket.
- [ ] Score predictions by league, team, and time window.
- [ ] Add report writer.
- [ ] Add notebook for calibration review.

## Immediate next commits

1. Write detections, tracks, ball state, and player state to table files.
2. Add pressure score and compactness features.
3. Add hybrid CNN plus temporal model implementation.
4. Add temporal training loop.
5. Add calibration wrapper and metrics reporting.