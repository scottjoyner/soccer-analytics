# Soccer-Edge Fine-Tuning and Data-Source Report

**Project:** Soccer-Edge rights-safe local soccer analytics pipeline  
**Repository:** `https://github.com/scottjoyner/soccer-analytics`  
**Report scope:** open-data ingestion, licensed local-footage processing, object-detection fine-tuning preparation, match-outcome modeling, evaluation, and promotion controls.  
**Prepared for:** internal team presentation  

## Executive summary

Soccer-Edge is a local-first research pipeline that combines open soccer event/tracking data with rights-approved local footage. It produces lineage-tagged Parquet tables, frame-level detection datasets, spatial features, tensor samples, model bundles, calibration reports, data cards, model cards, and promotion-gate reports.

The work completed so far demonstrates a reproducible end-to-end system rather than a production-ready prediction product. The system can:

- ingest StatsBomb, Metrica, SoccerNet, OpenFootball, and football-data.co.uk data;
- process approved local footage with a YOLO abstraction;
- export player/ball detections and annotation-review artifacts;
- build match-level computer-vision features;
- train a tabular winner/score model;
- train a CNN from occupancy-grid tensors;
- evaluate models on held-out matches;
- refuse promotion when a model does not meet baseline and calibration thresholds.

The 98-match highlight case study produced 88,127 detection rows and 7,878 grid frame rows. The key result is deliberately negative: highlight-derived features did not demonstrate reliable match-outcome lift. The open-event model, by contrast, showed materially stronger calibrated performance in the recorded study. This identifies the next product priority: full-match footage and/or richer open event/tracking supervision, not another model trained on the same highlight-only aggregates.

## 1. System objective

The intended workflow is:

```text
approved sources
  -> normalized tables
  -> detections / tracking / pitch state
  -> feature tables
  -> train/validation/test split by match
  -> calibrated models
  -> evaluation and data/model cards
  -> promotion gate
```

The platform is research/offline only. External execution and real-money actions are disabled.

## 2. Data-source inventory and citations

### 2.1 StatsBomb Open Data

**Used for:** event data, match metadata, lineups, shots, passes, carries, pressures, and event-derived feature tables such as xG/xT-style features.

**Repository:** [StatsBomb Open Data](https://github.com/statsbomb/open-data)  
**Specification:** [StatsBomb Open Data Specification](https://github.com/statsbomb/open-data/blob/master/doc/StatsBomb%20Open%20Data%20Specification%20v1.1.pdf)

**Citation text:**

> StatsBomb. *StatsBomb Open Data*. GitHub repository. https://github.com/statsbomb/open-data

**Attribution/use note:** cite StatsBomb and follow the repository terms and attribution requirements. The source is non-audiovisual event data; it is suitable for reproducible event-feature research without acquiring broadcast video.

### 2.2 Metrica Sports Sample Data

**Used for:** sample event and tracking data, synchronization experiments, pitch-coordinate normalization, possession/pressure feature validation, and tracking-table schema tests.

**Repository:** [Metrica Sports Sample Data](https://github.com/metrica-sports/sample-data)

**Citation text:**

> Metrica Sports. *Metrica Sports Sample Tracking and Event Data*. GitHub repository. https://github.com/metrica-sports/sample-data

**Attribution/use note:** acknowledge Metrica Sports and respect the repository’s stated use conditions. The sample data is used here as a tracking/event benchmark, not as a substitute for rights-cleared broadcast footage.

### 2.3 SoccerNet

**Used for:** benchmark annotations and task-specific validation where the selected subset and access terms allow it. SoccerNet is especially relevant to action spotting, broadcast-video understanding, and related benchmark tasks.

**Dataset page:** [SoccerNet Data](https://www.soccer-net.org/data)  
**FAQ and access notes:** [SoccerNet FAQ](https://www.soccer-net.org/faq)  
**Repository:** [SoccerNet-v3](https://github.com/SoccerNet/SoccerNet-v3)  
**Foundational paper:** [SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Giancola_SoccerNet_A_Scalable_CVPR_2018_paper.pdf)

**Citation text:**

> Giancola, S., Amato, G., Choe, M., et al. *SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos*. CVPR Workshops, 2018. https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Giancola_SoccerNet_A_Scalable_CVPR_2018_paper.pdf

**Attribution/use note:** SoccerNet’s access requirements vary by data modality and task. In particular, its data page states that access to broadcast video requires completing the applicable NDA. Only approved subsets should enter this pipeline.

### 2.4 OpenFootball

**Used for:** open match-result archives and structured competition/season results.

**Project site:** [Open Football Data / football.db](https://openfootball.github.io/)  
**Repository organization:** [OpenFootball on GitHub](https://github.com/openfootball)

**Citation text:**

> OpenFootball. *Open Football Data / football.db*. https://openfootball.github.io/

**Attribution/use note:** OpenFootball describes its data as free and open public-domain football data. Preserve source attribution and the original dataset version/path in lineage fields.

### 2.5 football-data.co.uk

**Used for:** historical results and, where permitted for the specific use, historical odds/result CSVs.

**Site:** [Football-Data.co.uk](https://www.football-data.co.uk/)  
**Data page:** [Football-Data historical data](https://www.football-data.co.uk/data.php)

**Citation text:**

> Football-Data.co.uk. *Football Results, Statistics & Soccer Betting Odds Data*. https://www.football-data.co.uk/data.php

**Attribution/use note:** check the current site terms and intended commercial/non-commercial use before distributing derived results or odds data.

### 2.6 Licensed local footage

**Used for:** frame export, YOLO detection, detection-review preparation, local object-model fine-tuning, occupancy grids, and video-derived match features.

The 98-match highlights study used permitted local files supplied to the project. The repository records the rights posture in the data card:

- rights status: `licensed`;
- rights reference: permitted local source path;
- allowed use: offline research, model training, calibration, and evaluation inside the repository;
- restricted use: redistribute restricted footage or process media without recorded rights.

The source videos are not cited as public URLs or redistributable assets. Public video URLs remain discovery metadata only. They are not processing inputs.

## 3. Rights and provenance controls

Rights compliance is implemented as a gate, not just a statement in the report.

Processable footage must have:

- an approved `rights_status`: `owned`, `licensed`, or `compatible_license`;
- a nonempty `rights_reference`;
- a manifest row;
- a matching `video_id`;
- an existing local regular file;
- a resolved path that matches the approved input;
- no blocked remote modality such as YouTube, Twitch, HTTP, HTTPS, RTMP, or RTSP.

Derived tables preserve lineage fields such as:

- `lineage_source_name`;
- `lineage_source_path`;
- `lineage_dataset_version`;
- `lineage_ingested_at_utc`.

This makes it possible to trace a feature or model output back to its source table and source rights context.

## 4. What was fine-tuned and built

### 4.1 Open-data ingestion

The source adapters were implemented and exercised for:

- StatsBomb JSON competitions, matches, events, and lineups;
- Metrica events and tracking CSVs;
- SoccerNet JSON and CSV examples;
- OpenFootball result CSVs;
- football-data result CSVs.

The normalized outputs are Parquet tables, with lineage metadata carried into derived tables.

### 4.2 Computer-vision abstraction

The video path uses an Ultralytics YOLO abstraction. The detector emits frame-level bounding boxes with:

- frame index;
- class name;
- confidence;
- bounding-box coordinates.

The intended local object taxonomy is:

- player;
- ball;
- referee;
- goalkeeper.

COCO-style detector names are normalized where appropriate, for example `person` to `player` and `sports ball` to `ball`.

The pipeline also includes:

- frame export;
- image-path joining;
- low-confidence sampling;
- crop export;
- HTML contact-sheet review;
- correction merge;
- normalized annotation export;
- grouped train/validation splitting;
- annotation audits;
- YOLO dataset layout generation;
- dataset hashing and versioning.

### 4.3 Local object-model fine-tuning path

The intended fine-tuning loop is:

```text
approved local footage
  -> exported frames
  -> model detections
  -> image-path join
  -> low-confidence queue
  -> crop/contact-sheet review
  -> corrected annotations
  -> grouped train/validation split
  -> YOLO data.yaml
  -> optional object-model training
  -> object evaluation and confusion matrix
```

The train/validation split is grouped by physical image so the same frame cannot leak into both splits.

### 4.4 Match-outcome tabular model

The highlight study builds per-match features from detection output:

- `n_player`;
- `n_ball`;
- `avg_det_per_frame`;
- `ball_center_x`;
- `ball_center_y`.

The model path uses:

- a calibrated winner classifier;
- home-score regression;
- away-score regression;
- match-level train/test splitting;
- model bundle metadata;
- prediction export;
- evaluation metrics;
- promotion checks.

### 4.5 CNN occupancy-grid model

Detections are rasterized into occupancy grids with initial channels for:

- home/player occupancy;
- away/player occupancy;
- ball occupancy.

The CNN path produces:

- NPZ tensor samples;
- grouped temporal windows;
- a CNN winner model bundle;
- sequence-level and match-level evaluation;
- winner Brier score;
- repeated-CV and seed-sweep support.

### 4.6 Realtime state path

The repository also contains a realtime research path with:

- rolling state;
- class-aware nearest-centroid matching;
- low-confidence review queue;
- windowed possession/territory/pressure aggregates;
- smoothed expected-winner probability;
- expected-winner, momentum, and comeback triggers.

This path is explicitly a lightweight research abstraction. It does not provide player re-identification, a Kalman filter, full pitch calibration, or real-money execution.

## 5. Fine-tuning case study: 98 permitted highlight clips

The processed study contains:

- 98 matches;
- 50 distinct teams;
- 88,127 total detection rows;
- 86,007 player detections;
- 1,541 ball detections;
- 7,878 occupancy-grid frame rows;
- a 98-match match-training table;
- tabular winner and score models;
- a CNN occupancy-grid winner model;
- data-card and training-summary artifacts.

The source match results were parsed from the supplied local filenames into a structured results table. No audiovisual content was scraped or downloaded by the pipeline.

### Example: HL001

The team-facing case study is:

- Argentina 3–1 Switzerland;
- match ID: `HL001`;
- 150 analyzed frames;
- 1,567 raw detections;
- 1,514 `person` detections;
- 45 `sports ball` detections;
- 8 detections from other classes.

The existing HL001 bundle also exposes an important quality limitation: the processed `tracks.parquet`, `player_states.parquet`, and `ball_states.parquet` tables are empty. Therefore the report does not present possession, pressure, or player-level tracking claims for this bundle as completed measurements.

## 6. Evaluation results

### 6.1 Highlight-derived features

The recorded 68/30 held-out evaluation reported:

| Metric | Count features | Track features |
|---|---:|---:|
| Winner accuracy | 53.3% | 50.0% |
| Majority baseline | 49.0% | 49.0% |
| Winner Brier | 0.623 | 0.614 |
| Home-score MSE | 1.770 | 1.864 |
| Away-score MSE | 2.345 | 2.066 |
| Train/test matches | 68/30 | 68/30 |

The CNN repeated-CV summary reported approximately:

- 49.3% ± 1.8% sequence accuracy;
- 48.9% ± 1.3% match accuracy;
- 0.630 ± 0.011 winner Brier.

Interpretation: short highlight clips and aggregate detection counts do not provide reliable final-score signal. A small accuracy difference above baseline is not enough to justify promotion when probability calibration remains weak and the dataset is highlight-selected.

### 6.2 Open-event model

The project paper records a separate StatsBomb Premier League 2023/24 event-data evaluation:

- 380 matches;
- repeated 5-fold × 10 cross-validation;
- calibrated winner classifier;
- 63.3% ± 2.9% accuracy;
- 0.164 Brier score;
- home/away score MSE of 0.96 / 0.87;
- 41.3% majority baseline.

This is the strategically important comparison. The same general evaluation discipline produces materially stronger results when supplied with rich event supervision rather than highlight-only detection aggregates.

## 7. What the work proves

The current work proves that the platform can:

1. enforce a rights-safe media boundary;
2. normalize heterogeneous soccer data sources;
3. preserve source lineage through derived tables;
4. generate reproducible frame/detection/annotation artifacts;
5. construct model-ready CV and event features;
6. train tabular and CNN model paths;
7. evaluate at match level without row leakage;
8. produce calibration and promotion metadata;
9. refuse to promote a model that does not demonstrate sufficient evidence.

## 8. What it does not yet prove

It does not yet prove:

- reliable player identity across frames;
- complete possession inference from broadcast footage;
- calibrated pitch-space tactical metrics for arbitrary camera views;
- predictive lift from short highlight videos;
- production-grade live winner or score prediction;
- profitability or market edge;
- suitability for real-money execution.

## 9. Recommended next fine-tuning phase

The next phase should prioritize data quality and temporal alignment over model complexity:

1. Obtain written rights for representative full-match local footage.
2. Create match manifests with period and match-second alignment.
3. Validate detector outputs on a manually reviewed frame sample.
4. Correct player/ball labels and train a local object model.
5. Add stable tracking and pitch calibration.
6. Join CV state to event timestamps.
7. Train in-play features using future-window labels.
8. Evaluate by match, not by frame.
9. Compare against event-only and market/Elo baselines.
10. Promote only if calibration and out-of-sample lift pass the gate.

The strongest demo message for the team is:

> We have built a rights-safe, traceable pipeline that turns approved soccer data and local footage into inspectable training artifacts. The first honest result is that highlight-only CV aggregates are not enough for outcome prediction. The next value step is full-match, time-aligned state data.

## 10. Reproducibility artifacts

Primary local artifacts:

- `README.md` — product and pipeline overview;
- `docs/training_data_sourcing.md` — source policy and fine-tuning workflow;
- `data/processed/highlights/DATA_CARD.md` — rights and dataset card;
- `data/processed/highlights/training/training_summary.md` — case-study summary;
- `data/processed/highlights/match_results.csv` — match labels;
- `data/processed/highlights/training/detection_features.csv` — match-level CV features;
- `data/processed/highlights/detections/` — per-match detection/state tables;
- `paper/arxiv_draft.tex` — technical paper draft and bibliography;
- `demo/HL001-argentina-switzerland.html` — team-facing concrete case study.

## References

1. StatsBomb. *StatsBomb Open Data*. https://github.com/statsbomb/open-data
2. StatsBomb. *StatsBomb Open Data Specification v1.1*. https://github.com/statsbomb/open-data/blob/master/doc/StatsBomb%20Open%20Data%20Specification%20v1.1.pdf
3. Metrica Sports. *Metrica Sports Sample Tracking and Event Data*. https://github.com/metrica-sports/sample-data
4. Giancola, S., Amato, G., Choe, M., et al. *SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos*. CVPR Workshops, 2018. https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Giancola_SoccerNet_A_Scalable_CVPR_2018_paper.pdf
5. SoccerNet. *Data*. https://www.soccer-net.org/data
6. SoccerNet. *FAQ*. https://www.soccer-net.org/faq
7. OpenFootball. *Open Football Data / football.db*. https://openfootball.github.io/
8. Football-Data.co.uk. *Football Results, Statistics & Soccer Betting Odds Data*. https://www.football-data.co.uk/data.php
9. Redmon, J., Divvala, S., Girshick, R., Farhadi, A. *You Only Look Once: Unified, Real-Time Object Detection*. CVPR, 2016.
10. Ultralytics. *Ultralytics YOLO Documentation*. https://docs.ultralytics.com/
11. Pedregosa, F., et al. *Scikit-learn: Machine Learning in Python*. JMLR, 2011. https://jmlr.org/papers/v12/pedregosa11a.html
12. Paszke, A., et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS, 2019. https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library
