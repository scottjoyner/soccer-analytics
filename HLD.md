# High-Level Design

## 1. Mission

Build a robust soccer analytics and probability engine that can ingest open/licensed soccer data, extract computer-vision features from licensed match video, train calibrated probability models, and evaluate prediction-market opportunities through paper trading and backtesting.

The system should support both pre-match and in-play modeling. It should be safe by default: research mode first, paper trading second, real execution only after explicit future gates.

## 2. Non-goals

This system will not:

- guarantee profitable betting;
- use unauthorized downloaded video;
- scrape protected services;
- use insider/private information;
- place real trades by default;
- bypass market or platform restrictions;
- ignore legal, geographic, or exchange-specific requirements.

## 3. Primary use cases

### 3.1 Data research

Ingest open and licensed soccer datasets, normalize them into common schemas, and build repeatable feature sets for modeling.

### 3.2 Video-derived feature extraction

Run object detection, tracking, pitch calibration, possession estimation, and pressure modeling on licensed local match clips.

### 3.3 Probability modeling

Train pre-match and in-play probability models that output calibrated probabilities for:

- home/team A win;
- draw;
- away/team B win;
- next goal;
- goal within a time window;
- no goal within a time window.

### 3.4 Market edge detection

Compare calibrated probabilities with prediction-market prices and flag paper-trade opportunities where edge exceeds fees, spread, slippage, and uncertainty buffers.

### 3.5 Paper trading and backtesting

Replay market states and model states, simulate orders, and measure calibration, ROI, drawdown, hit rate, edge decay, and liquidity constraints.

## 4. Architecture

```text
                    +---------------------+
                    |  Data Sources       |
                    |---------------------|
                    | Open events         |
                    | Tracking data       |
                    | Licensed videos     |
                    | Fixtures/results    |
                    | Market snapshots    |
                    | YouTube metadata    |
                    +----------+----------+
                               |
                               v
+------------------+   +-------+--------+   +------------------+
| Ingest Layer     |-->| Normalization  |-->| Parquet/DuckDB   |
+------------------+   +----------------+   +--------+---------+
                                                        |
                                                        v
+------------------+   +----------------+   +------------------+
| Video/CV Layer   |-->| Feature Layer  |-->| Training Tables  |
+------------------+   +----------------+   +--------+---------+
                                                        |
                                                        v
                                               +--------+---------+
                                               | Model Layer      |
                                               | Calibration      |
                                               +--------+---------+
                                                        |
                                                        v
                                               +--------+---------+
                                               | Backtest + Paper |
                                               | Trading Layer    |
                                               +------------------+
```

## 5. Major components

### 5.1 Ingest layer

Responsibilities:

- read source-specific formats;
- preserve raw data lineage;
- normalize IDs, timestamps, team names, player names, coordinates, and match state;
- write clean Parquet tables.

Initial ingest modules:

- `statsbomb_loader.py`
- `metrica_loader.py`
- `soccernet_loader.py`
- `fixtures_loader.py`
- `market_loader.py`
- `youtube_discovery.py`

### 5.2 Video/CV layer

Responsibilities:

- process licensed local video files;
- detect players, goalkeepers, referees, and ball;
- track objects across frames;
- convert pixels to pitch coordinates;
- infer possession and pressure;
- produce per-frame and per-second feature tables.

Core outputs:

- `detections.parquet`
- `tracks.parquet`
- `ball_state.parquet`
- `player_state.parquet`
- `team_state.parquet`
- `cv_features.parquet`

### 5.3 Feature layer

Responsibilities:

- generate leakage-safe supervised learning features;
- create rolling windows for in-play modeling;
- join event, tracking, video, fixture, and market state;
- create labels only from future outcomes after feature timestamps.

Feature groups:

- pre-match team strength;
- form and schedule;
- market-implied probability;
- score state and match clock;
- event momentum;
- CV pressure and spatial control;
- possession and progression;
- liquidity and market movement.

### 5.4 Model layer

Initial model stack:

- baseline market-implied model;
- Elo/logistic baseline;
- LightGBM/XGBoost tabular model;
- calibrated probability wrapper;
- later sequence model for in-play temporal state.

Evaluation metrics:

- log loss;
- Brier score;
- calibration error;
- reliability curves;
- top-edge bucket performance;
- edge decay over time;
- out-of-sample ROI in paper backtests.

### 5.5 Backtest layer

Responsibilities:

- replay historical match states and market snapshots;
- prevent future leakage;
- model order fills with spread/slippage;
- enforce risk constraints;
- produce run reports.

### 5.6 Trading layer

Initial implementation must be paper-only.

Future real execution, if added, must require all of:

- `ENABLE_REAL_TRADING=true`;
- explicit market configuration;
- max stake per trade;
- max daily loss;
- kill switch;
- jurisdiction/compliance review;
- separate manual approval step.

## 6. Data storage design

### 6.1 Local files

Use Parquet as the primary interchange/storage format.

```text
data/raw/          # untouched source data, ignored by git
data/interim/      # normalized intermediate outputs
data/processed/    # clean tables
data/features/     # model-ready features
data/models/       # trained artifacts
data/predictions/  # model outputs and paper trades
```

### 6.2 DuckDB

Use DuckDB for local analytical queries over Parquet.

### 6.3 Optional Neo4j

Neo4j can be used to represent graph relationships:

```text
(:Competition)-[:HAS_MATCH]->(:Match)
(:Team)-[:PLAYED_IN]->(:Match)
(:Player)-[:APPEARED_IN]->(:Match)
(:Match)-[:HAS_FRAME]->(:Frame)
(:Frame)-[:HAS_PLAYER_STATE]->(:PlayerState)
(:Frame)-[:HAS_BALL_STATE]->(:BallState)
(:Prediction)-[:FOR_MATCH]->(:Match)
(:TradeSignal)-[:BASED_ON]->(:Prediction)
```

## 7. Model lifecycle

1. Ingest data.
2. Normalize and validate schemas.
3. Build feature tables.
4. Train baseline model.
5. Train stronger model.
6. Calibrate probabilities.
7. Evaluate out of sample.
8. Backtest against market data.
9. Paper trade.
10. Promote only if calibration and risk metrics pass.

## 8. Risk model

Each trade signal must include:

- model probability;
- market implied probability;
- edge;
- uncertainty buffer;
- liquidity estimate;
- max stake fraction;
- reason codes;
- paper/live mode flag;
- timestamp and model version.

## 9. Initial milestone definition

Milestone 1 is complete when:

- one open dataset can be ingested;
- one licensed local video can be processed;
- frame-level player/ball features are written to Parquet;
- a baseline in-play probability model trains;
- calibration metrics are produced;
- a paper backtest runs without real execution.

## 10. Future milestones

- robust pitch calibration UI;
- stronger ball tracking and occlusion handling;
- SoccerNet game-state reconstruction benchmarking;
- live fixture and market snapshot collection;
- full market replay;
- dashboard/API;
- Neo4j export;
- sequence model for temporal in-play state;
- automated model registry and experiment tracking.