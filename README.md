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

1. **Ingests open data** — StatsBomb Open Data, Metrica Sports sample data,
   approved SoccerNet subsets, OpenFootball match results, and football-data.co.uk
   results — into normalized, lineage-tagged Parquet tables.
2. **Detects players and the ball** in permitted local video via a YOLOv8
   abstraction (`detect-yolo`), emitting per-frame boxes, tracks, and
   pitch-space states.
3. **Aggregates player and team statistics** across matches, with optional
   team/opponent splits (`features player-aggregate`).
4. **Fine-tunes match-outcome models** — a tabular winner classifier + home/away
   score regressors, and a convolutional occupancy-grid winner model
   (`train match-predictor`, `train match-predictor-cnn`).
5. **Captures permitted local footage for intake** — records screen, webcam, or a
   screenshot (`capture screen|webcam|image`), optionally running live YOLO
   detection while recording (`--detect`), and registers each capture as a
   rights-gated manifest row ready for the pipeline.
6. **Gates model promotion on measured lift** — the promotion gate refuses to
   promote a model bundle unless it beats the majority-class baseline and clears a
   Brier threshold, closing the loop from evaluation to a promoted artifact.
7. **Enforces rights by construction** — any approved footage row must carry a
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
  `process`, `process-local-model`, `export-frames`, `train player-ball`,
  `train local-finetune`, and the `capture` commands. The rights gate is
  mandatory for every footage-processing command: it requires both
  `--manifest` and `--video-id`, and footage is opened **only** if its
  `rights_reference` is recorded, its `local_path` exists, **and** that path
  matches the input. Passing no manifest is no longer allowed (a silent no-op
  was removed). `run_yolo_detection` enforces the same gate internally, so
  un-gated footage cannot be processed via the library either.
- A **modality blocklist** (`configs/modality_rules.json`) rejects any manifest
  row whose `source_url`/`clip_type` references a blocked modality (`youtube`,
  `youtu.be`, `twitch`, `stream`, or an `http(s)/rtmp/rtsp` scheme), so remote
  URLs can never become processing inputs. Local captures use a `capture://`
  scheme that stays within the gate.
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

> On an **externally-managed** Python (PEP 668) where `pip install` is blocked,
> or where `python3 -m venv` fails because `python3-venv` is not installed, use
> one of:
> - `pip install --break-system-packages -e .` plus the ML requirements (fine for a
>   dedicated research box); or
> - `pip install --user virtualenv && python3 -m virtualenv --system-site-packages .venv`
>   to reuse an already-installed `torch`/`numpy`/`pandas` without re-downloading.
> The CNN/object-model stack needs `torch`, `ultralytics`, and `opencv-python`.

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

# 3. detect players/ball (gate requires an approved --manifest/--video-id; refuses otherwise)
soccer-edge video detect-yolo --input /path/to/licensed/clip.mp4 --model-path yolov8n.pt \
  --output-dir data/processed/video_yolo --manifest manifests/local_video_manifest.csv --video-id clip
```

### Capture intake (screen / webcam / image)

Record permitted local footage straight into the pipeline. Every capture requires
an approved `--rights-status` (`owned`/`licensed`/`compatible_license`) and a
recorded `--rights-reference`; the command refuses otherwise. This is for content
you own or are licensed for — capturing third-party streams (YouTube, Twitch, …)
is prohibited by the rights gate.

```bash
# save footage + append a rights-gated manifest row
soccer-edge capture screen --duration 30 --fps 20 \
  --rights-status owned --rights-reference "personal-recording://self"
soccer-edge capture webcam --duration 30 --device 0 \
  --rights-status owned --rights-reference "personal-recording://self"

# real-time mode: run YOLO detection on each frame as it is recorded,
# writing a detections table (and an annotated video with --annotate)
soccer-edge capture screen --detect --object-model-path models/yolov8n.pt --annotate \
  --duration 30 --rights-status owned --rights-reference "personal-recording://self"
```

Captures land in `data/raw/video_licensed/captures/`. Robustness: a capture that
records zero frames (e.g. a closed webcam) raises instead of writing a dangling
manifest row; `--duration` must be positive (a duration of `0` is no longer
silently coerced to 10s); the video writer is released on error; and the container
codec is chosen from the `--output` suffix when the default is used.

The command prints the next
step — feeding the saved file into `train local-finetune` (or, for `--detect`, the
detections CSV into `video prepare-object-dataset`).

Requires the optional `mss` package for screen capture (`pip install mss`, included
in `requirements-ml.txt`).

#### Capture to match predictor

A single command wires a captured clip (or any approved local footage) end to end
into the match-outcome model: rights-gated YOLO detection, per-match feature
aggregation, merge with open-event features, and training. Detections are produced
by `run_yolo_detection`, which enforces the rights gate, and the per-match features
are joined to `match_results.csv` by `match_id` (optionally alongside open-event
xG/xT features from `--event-source`). This is the join point that lets captured
footage train alongside the other disparate data sources.

```bash
soccer-edge capture to-match-predictor \
  --input data/raw/video_licensed/captures/screen_001.mp4 \
  --model-path models/yolov8n.pt --results match_results.csv \
  --output-dir data/processed/capture_predictor --match-id m1 \
  --manifest manifests/video_manifest.csv --video-id screen_001 \
  --event-source examples/statsbomb --stride 5
```

The rights gate is mandatory for this command: a `--manifest`/`--video-id` pair
pointing at an approved, rights-referenced row under the licensed root is required
(bypassing it is intentionally not exposed on the CLI). Public URLs are discovery
metadata only and can never be opened as inputs.

### Match-outcome models

```bash
soccer-edge train match-predictor --detections det1.parquet det2.parquet \
  --results match_results.csv --output-dir data/processed/match_predictor
soccer-edge train match-predictor-cnn --detections det1.parquet det2.parquet \
  --results match_results.csv --output-dir data/processed/match_predictor_cnn
```

All match-outcome models are keyed by `match_id` and detection-agnostic: the
per-match CV features (`n_player`, `n_ball`, ball-center, …) are class-aliased so
that COCO names (`person`, `sports ball`) map to `player`/`ball`, and they merge
with open-event features (from StatsBomb/Metrica/OpenFootball/football-data) via
`merge_match_features`. Matches that have detections but no result row are skipped
because they cannot be labeled.

### Evaluation → promotion gate

Model promotion is gated on measured, out-of-sample lift. Convert an eval
`metrics.json` into a predictive-metrics table, run the gate, and promote only if
it passes:

```bash
soccer-edge model eval-to-metrics --metrics-json <OUT>/metrics.json \
  --output data/processed/predictive_metrics.csv --model-name cnn-v1
soccer-edge model promotion-gate \
  --model-card-path MODEL_CARD.md --data-card-path DATA_CARD.md \
  --dataset-versions versions.csv --audit-dir audit --object-metrics obj_metrics.csv \
  --predictive-metrics data/processed/predictive_metrics.csv \
  --majority-baseline-rate 0.50 --min-accuracy-lift 0.02 --max-brier 0.50
soccer-edge model promote --bundle-dir <candidate> --promoted-root models/promoted \
  --predictive-metrics data/processed/predictive_metrics.csv \
  --majority-baseline-rate 0.50 --min-accuracy-lift 0.02 --max-brier 0.50 \
  --dataset-versions versions.csv --audit-dir audit --object-metrics obj_metrics.csv
```

`model promote` exits non-zero and writes nothing when the gate fails, so the
highlight-clip models (which show no lift) cannot be promoted until a rights-clean
source yields genuine lift. The gate derives the majority baseline from the
`baseline_accuracy` recorded by `eval-to-metrics` / `model evaluate` when
`--majority-baseline-rate` is omitted; a no-lift model therefore cannot slip
through by forgetting the flag.

---

## Highlights case study (reproducible)

`data/processed/highlights/` contains a processed dataset built from **98
permitted FIFA World Cup 2026 highlight clips** provided to the authors as local
files. Processing is fully scripted and re-runnable:

```bash
python scripts/process_highlights_batch.py     # transcode AV1->H.264 + detect-yolo per match
python scripts/build_highlights_training.py    # aggregate + fine-tune both models
python scripts/evaluate_highlights.py          # out-of-sample tabular + calibrated eval
python scripts/evaluate_cnn.py                 # out-of-sample CNN winner eval (68/30 hold-out)
SEEDS="0 1 2 3 4" ./scripts/batch_cnn_eval.sh <OUT_ROOT> 10   # off-box seed sweep
```

For mean±sd across folds (matching the tabular StatsBomb reporting), use the
repeated-CV variant directly:

```bash
python scripts/evaluate_cnn.py --folds 5 --repeats 1 --epochs 10 --seed 0 \
  --output-dir data/processed/highlights/training/cnn_eval_cv
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
| Winner accuracy (in-sample fine-tune) | 48.0% (majority baseline 49.0%) |
| Home-score regressor MSE (in-sample) | 0.372 |
| Away-score regressor MSE (in-sample) | 0.204 |
| Winner accuracy (out-of-sample, tabular) | 53.3% count / 50.0% track (base 49.0%) |
| Winner accuracy (out-of-sample, CNN) | 51.6% sequence / 50.0% match (base 51.6% / 50.0%) |

**Honest limitation.** Short highlight reels are highlight-selected and contain
far fewer frames than a full match, so aggregate detection counts carry almost
no signal about the final score. Both the tabular and CNN winner classifiers
fail to beat the majority-class baseline out-of-sample (CNN: 51.6% sequence /
50.0% match accuracy at baseline; winner Brier 0.617), confirming the
probabilities are no better than chance. The pipeline therefore demonstrates
end-to-end reproducibility and governance, and pinpoints where predictive signal
must come from: full-match footage or rich open event data (e.g., xT,
pressure). All reported figures are out-of-sample only; no generalization is
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
