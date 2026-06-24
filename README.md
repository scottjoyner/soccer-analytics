# Soccer Analytics

A research-first soccer probability engine for building calibrated match outcome models from open event data, licensed/local video-derived computer-vision features, and prediction-market data.

> Status: initial architecture scaffold. The default mode is research and paper trading only.

## Purpose

This repository is designed to become a robust system for predicting soccer match probabilities, especially:

- pre-match home/draw/away probabilities;
- in-play win/draw/loss probabilities;
- next-goal and goal-within-window probabilities;
- market edge detection versus prediction-market prices;
- paper trading and backtesting before any live execution.

The project is not intended to guarantee betting profits. Every model output should be treated as a probabilistic research signal that must be calibrated, backtested, and risk-limited.

## Core principle

The edge is not simply predicting the winner. The edge is:

```text
calibrated_model_probability - market_implied_probability - fees - spread - slippage - uncertainty_buffer
```

No real-money execution should exist until the research system has survived out-of-sample testing, paper trading, and explicit risk review.

## Data policy and guardrails

This repo separates three categories of data:

1. **Open/licensed data**: event, lineup, tracking, and competition data from sources that explicitly permit local research usage.
2. **Licensed/local video**: videos stored locally only when the user owns them, has permission to process them, or they are distributed under terms that allow processing.
3. **Discovery metadata**: YouTube or web search metadata used only to find candidate clips for review and rights-clearing.

Important restrictions:

- Do not download, cache, mirror, or store YouTube audiovisual content through this repo.
- YouTube integration must only store metadata such as title, URL, channel, query, and rights status.
- Computer-vision processing must only run against local files in `data/raw/video_licensed/` or another explicitly licensed local path.
- Default trading mode is paper trading.
- Real trading must remain disabled unless a future implementation adds explicit environment gates, bankroll limits, kill switches, and jurisdiction/compliance checks.

## System overview

The system has four major layers:

1. **Ingest**
   - StatsBomb/Open event data
   - Metrica sample tracking/event data
   - SoccerNet tracking/game-state data
   - fixture and market snapshots
   - YouTube discovery metadata only

2. **Video/CV feature factory**
   - object detection for players, referees, goalkeepers, and ball;
   - multi-object tracking;
   - camera movement compensation;
   - pitch-space homography;
   - possession and pressure estimation;
   - per-frame and rolling-window features.

3. **Modeling**
   - baseline Elo/logistic models;
   - LightGBM/XGBoost tabular models;
   - calibrated in-play models;
   - leakage-safe train/test splits;
   - probability calibration and explainability.

4. **Backtesting and paper trading**
   - market replay;
   - slippage/spread assumptions;
   - edge thresholds;
   - bankroll and risk limits;
   - paper-order logs.

## Proposed repository layout

```text
soccer-analytics/
  README.md
  HLD.md
  LLD.md
  TODO.md
  CODEX_PROMPT.md
  pyproject.toml
  docker-compose.yml
  .env.example
  src/soccer_edge/
    ingest/
    video/
    features/
    models/
    backtest/
    trading/
    store/
    cli.py
  tests/
  data/
    raw/
    interim/
    processed/
    features/
    models/
    predictions/
```

The `data/` folders are intentionally ignored by git except for `.gitkeep` files.

## Quickstart

```bash
# clone
git clone https://github.com/scottjoyner/soccer-analytics.git
cd soccer-analytics

# create environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# inspect commands
soccer-edge --help
```

## Initial commands to implement

```bash
soccer-edge ingest statsbomb --path data/raw/statsbomb
soccer-edge ingest metrica --path data/raw/metrica
soccer-edge ingest soccernet --path data/raw/soccernet
soccer-edge discover youtube --query "World Cup goal highlights"
soccer-edge video process --input data/raw/video_licensed --output data/features/video
soccer-edge features build
soccer-edge train prematch
soccer-edge train inplay
soccer-edge calibrate
soccer-edge backtest
soccer-edge paper-trade
```

## First milestone

Milestone 1 should prove the end-to-end research loop without real trading:

- load at least one open event/tracking dataset;
- process one licensed local soccer video;
- extract per-frame player/ball states;
- generate CV features such as nearest player to ball, pressure score, possession chain, compactness, and distance to goal;
- train a baseline in-play probability model;
- calibrate probabilities;
- run a paper backtest.

## Safety and compliance defaults

- No real-money orders by default.
- No unauthorized video downloading.
- No scraping protected services.
- No insider information.
- No bypassing market/platform/geographic restrictions.
- No claims that the model guarantees profit.

## Documentation

- `HLD.md` describes the high-level architecture.
- `LLD.md` describes implementation-level module design.
- `TODO.md` tracks phased work.
- `CODEX_PROMPT.md` contains prompts for Codex/agent implementation.