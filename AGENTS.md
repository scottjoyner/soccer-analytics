# AGENTS.md

## Mission

Build and operate the local soccer analytics pipeline for research, feature generation, model training, and model review. The agent should prioritize a rights-safe, reproducible workflow that turns approved local footage and open soccer data into structured tables, tensor samples, model bundles, predictions, and calibration reports.

## Hard safety and data rules

1. Do not scrape, download, cache, or import audiovisual content from YouTube or other platforms unless explicit written rights are present and recorded in the manifest.
2. Raw footage collection means collecting from approved local locations only: local disk, NAS, mounted drives, Tailscale-accessible folders, or user-provided files.
3. Public video URLs may be stored only as discovery metadata. They are not processing inputs.
4. Only process media rows where `rights_status` is one of: `owned`, `licensed`, or `compatible_license`.
5. Preserve lineage fields on all processed tables and derived feature tables.
6. Never enable external execution or real-money actions. This repository is research/offline evaluation only.

## Current pipeline commands

Install locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Optional ML and local object model stack:

```bash
pip install -r requirements-ml.txt
```

Process open/local data:

```bash
soccer-edge ingest write-processed --source examples/statsbomb --source-type statsbomb --output-dir data/processed/examples/ingest
soccer-edge ingest write-processed --source examples/metrica --source-type metrica --output-dir data/processed/examples/ingest
soccer-edge ingest write-processed --source examples/soccernet --source-type soccernet --output-dir data/processed/examples/ingest
```

Build model inputs:

```bash
soccer-edge features inplay --source data/processed/game_state.csv --output data/processed/inplay_features.parquet --columns speed,pressure --window-seconds 60
soccer-edge features tensor-samples --source examples/tiny_grid_features.csv --output data/processed/examples/tiny_tensor_samples.npz --columns g0,g1,g2,g3 --label label --channels 1 --height 2 --width 2 --sequence-length 2 --group match_id --order timestamp_seconds
```

Train and review:

```bash
soccer-edge train simple --source examples/tiny_training.csv --columns speed_last,pressure_last --label label --output-dir data/processed/examples/simple_model
soccer-edge model predict --bundle-dir data/processed/examples/simple_model --source examples/tiny_training.csv --output data/processed/examples/predictions.csv
soccer-edge model registry --root-dir data/processed/examples --output data/processed/examples/registry.csv
soccer-edge model registry-summary --root-dir data/processed/examples --output data/processed/examples/registry_summary.csv
soccer-edge model compare --registry data/processed/examples/registry_summary.csv --output data/processed/examples/comparison.csv
soccer-edge model compare-markdown --comparison data/processed/examples/comparison.csv --output data/processed/examples/comparison.md
soccer-edge model run-summary --registry data/processed/examples/registry_summary.csv --predictions data/processed/examples/predictions.csv --output-dir data/processed/examples/run_summary
soccer-edge model calibration-review --predictions data/processed/examples/predictions.csv --output-dir data/processed/examples/calibration_review
```

Run the tiny local smoke pipeline:

```bash
soccer-edge examples tiny --repo-root . --output-dir data/processed/examples/tiny_pipeline
```

## Raw footage collection workflow

The agent may organize local footage, but must not fetch audiovisual files from the public internet.

1. Search approved local roots provided by the user, such as `/mnt`, `/media`, `/data`, or Tailscale-mounted shares.
2. Copy or reference only files that the user owns or has licensed for processing.
3. Create or update a video manifest with columns: `video_id`, `match_id`, `clip_type`, `local_path`, `rights_status`, `notes`.
4. Catalog approved footage and plan processable rows:

```bash
soccer-edge video catalog-local --root data/raw/video_licensed --output manifests/local_video_manifest.csv --rights-status owned
soccer-edge video plan --manifest manifests/local_video_manifest.csv --licensed-root data/raw/video_licensed
```

5. Run the scaffold process command or the optional local object-model command:

```bash
soccer-edge video process --input data/raw/video_licensed/clip.mp4 --output-dir data/processed/video_pipeline --frame-count 100
soccer-edge video process-local-model --input data/raw/video_licensed/clip.mp4 --model-path models/local-object-model.pt --output-dir data/processed/video_model --stride 5 --max-samples 100
```

6. When a local object model is configured, use the media reader and media inference adapter to convert per-frame outputs into table-ready rows.

## Fine-tuning pipeline target

The agent should prepare data for model fine-tuning in this order:

1. Ingest local/open event data into processed Parquet tables.
2. Process approved local footage into detections/tracks/state tables.
3. Convert pixel-space detections to pitch-space state when calibration is available.
4. Build rolling feature tables and preserve source metadata.
5. Build tensor samples grouped by `match_id` and ordered by `timestamp_seconds` or `frame_idx`.
6. Train baseline tabular models and CNN tensor models.
7. Export predictions, calibration reports, registry summaries, and markdown comparison reports.
8. Promote only model bundles with reproducible metadata, feature names, metrics, and lineage.

## Suggested next implementation tasks

1. Add calibration-aware pixel-to-pitch conversion into the media processing loop.
2. Add an end-to-end fine-tuning command that runs ingest -> features -> tensor samples -> train -> predict -> review.
3. Add active learning sampling for low-confidence frames.
4. Add annotation export format for object model fine-tuning.
5. Add richer examples for complete processed video and pitch-calibrated outputs.

## Quality gates

Before committing new work, run:

```bash
pytest
ruff check src tests
```

For optional ML changes, also run:

```bash
pip install -r requirements-ml.txt
pytest tests/test_cnn_runner.py tests/test_cnn_outputs.py tests/test_cnn_review.py tests/test_cnn_training.py tests/test_torch_optional.py tests/test_temporal_training.py
```
