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
- [x] Add AGENTS.md for agent operation.

## Phase 1 — Project foundation

- [x] Finalize package name and CLI entry point.
- [x] Add configuration loader.
- [x] Add structured logging.
- [x] Add linting and test tooling.
- [x] Add CI workflow.
- [x] Add optional ML CI workflow job.
- [x] Add model/data card validation checks in CI.
- [x] Add CI smoke test for example video-review commands.

## Phase 2 — Storage layer

- [x] Implement table read/write helpers.
- [x] Implement DuckDB query helper.
- [x] Define schemas for matches, events, frames, ball states, player states, features, and labels.
- [x] Add basic schema tests.
- [x] Add graph export payload path for match, frame, feature, and model-run metadata.
- [x] Add Neo4j graph export payloads for dataset versions, annotation audits, and model evaluation summaries.
- [x] Add graph export file writers for dataset/version/evaluation payload batches.

## Phase 3 — Open data ingestion

- [x] Implement first local StatsBomb JSON loader.
- [x] Implement Metrica loader.
- [x] Implement SoccerNet loader.
- [x] Implement generic match catalog loader.
- [x] Add coordinate normalization utility for 105m x 68m pitch.
- [x] Preserve original coordinates alongside normalized coordinates.
- [x] Add data lineage columns to output tables.
- [x] Wire local StatsBomb/Metrica/SoccerNet loaded frames into stored processed tables.
- [x] Add processed StatsBomb fixture coverage.
- [x] Add richer example fixtures for processed StatsBomb/Metrica/SoccerNet paths.
- [x] Add deeper training data sourcing strategy.
- [x] Add training source catalog helper.
- [x] Add rights-aware raw data source catalog.
- [x] Add raw player data sourcing guide.

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
- [x] Replace placeholder video process command with first local pipeline stub.
- [x] Add optional media reader dependency gate.
- [x] Add frame sampling and adapter execution loop.
- [x] Add media inference adapter implementation.
- [x] Add optional local object model bridge.
- [x] Add local footage catalog manifest command.
- [x] Add video-frame export that creates `image_path` rows directly from local footage.
- [x] Add frame/detection join helper to attach `image_path` to detection rows by `frame_idx`.
- [x] Add calibration-aware pixel-to-pitch conversion into the media processing loop.
- [x] Load homography calibration files from JSON/YAML in the media process command.
- [x] Add calibration visual QA plots for pitch-space projection.
- [x] Add QA summary markdown for calibration error statistics.
- [x] Add annotation export format for local object model fine-tuning.
- [x] Add annotation train/val splitter for exported frames and labels.
- [x] Add annotation label audit summaries by class, frame, and split.
- [x] Add annotation dataset config writer for local object-model training.
- [x] Add active learning sampling for low-confidence frames.
- [x] Add object crop export for low-confidence review rows.
- [x] Add automatic correction-review UI export for class/bbox changes.
- [x] Add automatic correction merge helpers for reviewed low-confidence crops.
- [x] Add crop-review HTML contact sheet generation.
- [x] Add dataset versioning/hashing for frame manifests, annotation tables, and data cards.
- [x] Add dataset version IDs into data-card and model-card metadata.
- [x] Add dataset-card/model-card cross-links to graph payload IDs.
- [x] Add automatic data-card population from training source catalog plus manifest stats.
- [x] Add richer examples with tiny local image fixtures.

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
- [x] Add player-match event stats for model features.
- [x] Add rolling player form features for model inputs.

## Phase 7 — Supervised feature tables

- [x] Build prematch feature table.
- [x] Build in-play rolling-window feature table.
- [x] Carry source metadata through rolling feature tables.
- [x] Add leakage-safe label helper.
- [x] Add time-based train/validation/test split.
- [x] Add synthetic tests for label leakage.
- [x] Add tensor sample builder from rolling grid feature tables.
- [x] Add tensor builder support for multi-match grouped sequences.
- [x] Add grouped tensor samples with time ordering column support.

## Phase 8 — Modeling

- [x] Add optional ML requirements file.
- [x] Add game-state feature and training config objects.
- [x] Add CNN model shell for field-state grids.
- [x] Add temporal model spec for recurrent-style sequence models.
- [x] Add hybrid CNN plus temporal model implementation.
- [x] Add Torch dataset for rolling game-state samples.
- [x] Add CNN training loop.
- [x] Add temporal training loop.
- [x] Add simple sklearn training helper and CLI command.
- [x] Add full CNN training CLI command for tensor datasets.
- [x] Add prediction export command from saved model bundles.
- [x] Add prediction export tests for CNN bundles when Torch is available.
- [x] Add CNN probability calibration review path.
- [x] Add calibration wrapper.
- [x] Report log loss, Brier score, accuracy, and calibration curves.
- [x] Save feature list, model artifact, metrics, and version metadata.
- [x] Add model registry index for saved bundles.
- [x] Add richer model registry summaries by metrics and created date.
- [x] Add run summary writer for comparison, markdown, and calibration artifacts.
- [x] Add model/data cards for promoted fine-tuned bundles.
- [x] Add full object-model training command behind optional dependency gates.
- [x] Add object-model evaluation ingest for precision/recall by class.
- [x] Add object-model evaluation visual confusion matrix output.

## Phase 9 — Offline evaluation

- [x] Implement historical replay evaluator.
- [x] Score predictions by confidence bucket.
- [x] Score predictions by league, team, and time window.
- [x] Add report writer.
- [x] Add calibration review script.
- [x] Add model comparison report from registry summary and evaluation metrics.
- [x] Add model comparison markdown report writer.
- [x] Add promotion gate command that validates cards, versions, audits, and object metrics together.

## Phase 10 — CLI workflows

- [x] Add feature build command.
- [x] Add processed ingest command.
- [x] Add prematch feature command.
- [x] Add in-play feature command.
- [x] Add player stats and player form feature commands.
- [x] Add tensor sample command.
- [x] Add simple training command.
- [x] Add CNN training command.
- [x] Add local training chain command.
- [x] Add local fine-tuning pipeline command.
- [x] Add dry-run plan validation that checks command paths and missing inputs.
- [x] Add local fine-tune dry-run mode that writes a runnable shell plan without executing optional dependencies.
- [x] Add object model training command.
- [x] Add model bundle save command.
- [x] Add model prediction command.
- [x] Add CNN prediction command.
- [x] Add model registry command.
- [x] Add model registry summary command.
- [x] Add model compare command.
- [x] Add model compare markdown command.
- [x] Add model run summary command.
- [x] Add model/data card commands.
- [x] Add automatic data-card command.
- [x] Add model/data card validation command.
- [x] Add model evaluate command.
- [x] Add model calibration review command.
- [x] Add CNN calibration review command.
- [x] Add object-model evaluation command.
- [x] Add object confusion matrix command.
- [x] Add source catalog and raw-source catalog commands.
- [x] Add graph payload export commands.
- [x] Add local footage catalog command.
- [x] Add local object model video process command.
- [x] Add frame export, frame image join, correction review/merge, annotation export/split/audit, dataset versions, low-confidence sampling, crop export, contact sheet, calibration QA/summary, and annotation-config commands.
- [x] Add CLI command to run the full tiny example pipeline end to end.
- [x] Add end-to-end sample workflow fixture.
- [x] Add examples directory with tiny CSV fixtures and NPZ generation command.
- [x] Add richer examples for complete processed video and pitch-calibrated outputs.
- [x] Add local pipeline command docs.
- [x] Add docs for NPZ tensor dataset format.

## Immediate next commits

1. Add player per-90 and minutes-adjusted feature normalization.
2. Add lineup-aware expected-starter features and team roster aggregation.
3. Add raw-source manifest validation against allowed/blocked modality rules.
4. Add graph payload file writers for player-match and player-form features.
5. Add source-specific ingestion adapters for OpenFootball and football-data.co.uk CSVs.
