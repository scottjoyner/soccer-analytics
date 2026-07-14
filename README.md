# Soccer-Edge

A **rights-safe, local-first soccer analytics pipeline** that turns open event
data and *permitted local footage* into structured tables, detection datasets,
and match-outcome models — with a hard, tested rights gate at every footage
boundary.

> Scope: offline research only. No real-money actions, no market execution, and
> **no scraping or downloading of audiovisual content** (YouTube or otherwise).
> Public video URLs are discovery metadata only.

---

## What it does

1. **Ingests open data** — StatsBomb Open Data, Metrica Sports sample data, and
   approved SoccerNet subsets — into normalized, lineage-tagged Parquet tables.
2. **Detects players and the ball** in permitted local video via a YOLOv8
   abstraction (`detect-yolo`), emitting per-frame boxes, tracks, and
   pitch-space states.
3. **Aggregates player and team statistics** across matches, with optional
   team/opponent splits (`features player-aggregate`).
4. **Fine-tunes match-outcome models** — a tabular winner classifier + home/away
   score regressors, and a convolutional occupancy-grid winner model
   (`train match-predictor`, `train match-predictor-cnn`).
5. **Enforces rights by construction** — any approved footage row must carry a
   recorded `rights_reference`, and the footage file is opened only when its
   path matches the approved manifest entry.

---

## Data policy & rights gate

This repository treats data-rights as a system property, not documentation:

- Footage `rights_status` is one of `owned`, `licensed`, `compatible_license`,
  `pending`, or `blocked`.
- Any **approved** status (`owned`/`licensed`/`compatible_license`) **requires a
  non-empty `rights_reference`** (an explicit written-rights pointer).
- A single assertion, `assert_processable`, is reused across the command
  surface: `discover video`, `catalog-local`, `video plan`, `detect-yolo`,
  `process`, `process-local-model`, `export-frames`, `train player-ball`, and
  `train local-finetune`. When a manifest row is named, footage is processed
  only if its `rights_reference` is recorded **and** its `local_path` matches the
  input.
- Public video URLs are stored as discovery metadata only; they are never
  processing inputs.

---

## Install

```bash
git clone https://github.com/scottjoyner/soccer-analytics.git
cd soccer-analytics
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -r requirements-ml.txt   # optional: YOLO + PyTorch ML stack
soccer-edge --help
```

---

## Repository layout

```text
soccer-analytics/
  README.md            # this file
  AGENTS.md            # operating rules (safety + data policy)
  paper/               # arXiv-style draft paper
  src/soccer_edge/
    ingest/            # open-data loaders + processed tables
    video/             # detection, tracking, manifest, rights gate
    features/          # player/team aggregation, tensors
    models/            # classifiers, regressors, CNN, calibration
    pipeline/          # match_predictor, object_finetune
    player_stats.py    # per-match + cross-match player aggregation
    cli.py             # the soccer-edge command surface
  scripts/             # reproducible batch + training scripts
  data/processed/highlights/   # processed highlights dataset (see its README)
  tests/
```

`data/` is git-ignored except `.gitkeep`; the processed highlights dataset and
its zip bundle are force-added so the research artifact is reproducible.

---

## Pipeline usage

### Open event data

```bash
soccer-edge ingest write-processed --source examples/statsbomb --source-type statsbomb --output-dir data/processed/ingest
soccer-edge features player-stats  --events data/processed/ingest/statsbomb_events.parquet --output data/processed/player_match_stats.csv
soccer-edge features player-aggregate --player-stats data/processed/player_match_stats.csv --output data/processed/player_aggregates.csv
soccer-edge features player-aggregate --player-stats data/processed/player_match_stats.csv --split-by opponent --output data/processed/player_aggregates_by_opponent.csv
```

### Permitted local footage

```bash
# 1. record rights and catalog local files
soccer-edge video catalog-local --root /path/to/licensed --output manifests/local_video_manifest.csv \
  --rights-status licensed --rights-reference "license-file:///perms/permit.pdf"

# 2. plan what is processable (skips rows without a recorded rights_reference)
soccer-edge video plan --manifest manifests/local_video_manifest.csv --licensed-root /path/to/licensed

# 3. detect players/ball (gate refuses unproven footage)
soccer-edge video detect-yolo --input /path/to/licensed/clip.mp4 --model-path yolov8n.pt \
  --output-dir data/processed/video_yolo --manifest manifests/local_video_manifest.csv --video-id clip
```

### Match-outcome models

```bash
soccer-edge train match-predictor --detections det1.parquet det2.parquet \
  --results match_results.csv --output-dir data/processed/match_predictor
soccer-edge train match-predictor-cnn --detections det1.parquet det2.parquet \
  --results match_results.csv --output-dir data/processed/match_predictor_cnn
```

---

## Highlights case study (reproducible)

`data/processed/highlights/` contains a processed dataset built from **98
permitted FIFA World Cup 2026 highlight clips** provided to the authors as local
files. Processing is fully scripted and re-runnable:

```bash
python scripts/process_highlights_batch.py     # transcode AV1->H.264 + detect-yolo per match
python scripts/build_highlights_training.py    # aggregate + fine-tune both models
```

The original clips are AV1-encoded; the local media reader has no AV1 decoder,
so each clip is transcoded to H.264 with FFmpeg (a rights-preserving local
transformation — originals stay the system of record) before detection. Match
results were parsed from the filenames; no audiovisual content was scraped.

| Metric | Value |
| --- | --- |
| Matches processed | 98 (50 distinct teams) |
| Detection rows | 88,127 (86,007 player / 1,541 ball) |
| CNN grid frame-rows | 7,878 |
| Winner accuracy (in-sample) | 48.0% (majority baseline 49.0%) |
| Home-score regressor MSE | 0.372 (in-sample) |
| Away-score regressor MSE | 0.204 (in-sample) |

**Honest limitation.** Short highlight reels are highlight-selected and contain
far fewer frames than a full match, so aggregate detection counts carry almost
no signal about the final score. The tabular model does **not** beat the
majority-class baseline for winner prediction. The pipeline therefore
demonstrates end-to-end reproducibility and governance, and pinpoints where
predictive signal must come from: full-match footage or rich open event data
(e.g., xT, pressure). All reported figures are in-sample; no generalization is
claimed. See `paper/arxiv_draft.tex`.

The shipped `highlights_training_data.zip` includes the parsed aggregates,
per-match detection tables, the cleaned training dataset, predictions, and the
**fine-tuned model binaries**.

---

## Reproducibility & artifacts

- `data/processed/highlights/README.md` — dataset documentation + lineage.
- `data/processed/highlights/DATA_CARD.md` — generated data card.
- `data/processed/highlights/highlights_training_data.zip` — bundled artifact.
- `paper/arxiv_draft.tex` — arXiv-style draft describing the system and study.

---

## Safety & compliance

- No real-money execution; offline research only.
- No scraping or downloading of protected audiovisual content.
- Researchers remain responsible for confirming their own rights to any local
  footage they process.
- Model outputs are probabilistic research signals, not guarantees.

## License

Code is provided for research use. Footage and event data remain under their
respective rights holders; this repository distributes only metadata, parsed
aggregates, and derived detection tables for permitted/licensed inputs.
