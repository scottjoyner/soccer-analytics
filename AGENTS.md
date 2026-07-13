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

## Training data sourcing

Use `docs/training_data_sourcing.md` as the primary source policy. The recommended source tiers are:

1. Open event/tracking data: StatsBomb Open Data, Metrica Sports Sample Data, and approved SoccerNet subsets.
2. Rights-approved local footage: user-owned, licensed, or compatible-license files on local disk, NAS, mounted storage, or Tailscale shares.
3. Self-generated annotations: local frame exports, detections, low-confidence review queues, corrected labels, crop sheets, and local annotation configs.

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

Run the frame/detection review path:

```bash
soccer-edge video export-frames --input data/raw/video_licensed/clip.mp4 --output-dir data/processed/frames --manifest-output data/processed/frame_manifest.csv --stride 5 --max-frames 100
soccer-edge video process-local-model --input data/raw/video_licensed/clip.mp4 --model-path models/local-object-model.pt --output-dir data/processed/video_model --stride 5 --max-samples 100 --calibration configs/pitch_calibration.json
soccer-edge video attach-frame-images --detections data/processed/video_model/detections.parquet --frame-manifest data/processed/frame_manifest.csv --output data/processed/detections_with_images.csv
soccer-edge video sample-low-confidence --source data/processed/detections_with_images.csv --output data/processed/low_confidence.csv --threshold 0.5 --limit 100
soccer-edge video export-crops --source data/processed/low_confidence.csv --output-dir data/processed/crops --manifest-output data/processed/crop_manifest.csv --image-path-column image_path
soccer-edge video contact-sheet --source data/processed/crop_manifest.csv --output data/processed/crop_review.html
```

Run calibration and annotation prep:

```bash
soccer-edge video calibration-qa --calibration configs/pitch_calibration.json --csv-output data/processed/calibration_qa.csv --svg-output data/processed/calibration_qa.svg
soccer-edge video calibration-summary --source data/processed/calibration_qa.csv --output data/processed/calibration_qa.md
soccer-edge video export-annotations --source data/processed/detections_with_images.csv --output-dir data/processed/annotations --classes player,ball --image-width 1920 --image-height 1080
soccer-edge video split-annotations --source data/processed/detections_with_images.csv --train-output data/processed/annotations/train.csv --val-output data/processed/annotations/val.csv --train-fraction 0.8
soccer-edge video annotation-config --root data/processed/annotations --train-images images/train --val-images images/val --classes player,ball --output data/processed/annotations/data.yaml
```

Train and review:

```bash
soccer-edge train simple --source examples/tiny_training.csv --columns speed_last,pressure_last --label label --output-dir data/processed/examples/simple_model
soccer-edge model predict --bundle-dir data/processed/examples/simple_model --source examples/tiny_training.csv --output data/processed/examples/predictions.csv
soccer-edge model run-summary --registry data/processed/examples/registry_summary.csv --predictions data/processed/examples/predictions.csv --output-dir data/processed/examples/run_summary
soccer-edge model model-card --bundle-dir data/processed/examples/simple_model --output data/processed/examples/MODEL_CARD.md
soccer-edge model data-card --dataset-name local-example --sources examples/tiny_training.csv,examples/tiny_grid_features.csv --output data/processed/examples/DATA_CARD.md --rights-status owned
soccer-edge model validate-cards --model-card-path data/processed/examples/MODEL_CARD.md --data-card-path data/processed/examples/DATA_CARD.md
```

## Fine-tuning pipeline target

The agent should prepare data for model fine-tuning in this order:

1. Ingest local/open event data into processed Parquet tables.
2. Catalog approved local footage and export frame manifests.
3. Process approved local footage into detections/tracks/state tables.
4. Join detection rows to frame image paths by `frame_idx`.
5. Convert pixel-space detections to pitch-space state when calibration is available.
6. Build rolling feature tables and preserve source metadata.
7. Export normalized annotations, train/validation splits, low-confidence review queues, object crops, and contact sheets.
8. Train baseline tabular models, CNN tensor models, and optional local object models.
9. Export predictions, calibration reports, registry summaries, cards, and markdown comparison reports.
10. Promote only model bundles with reproducible metadata, feature names, metrics, cards, and lineage.

## Suggested next implementation tasks

1. Add annotation label audit summaries by class, frame, and split.
2. Add dataset versioning/hashing for frame manifests, annotation tables, and data cards.
3. Add automatic data-card population from training source catalog plus manifest stats.
4. Add object-model evaluation ingest for precision/recall by class.
5. Add a full local fine-tuning pipeline command that chains frame export -> detection -> join -> review -> split -> config -> train.

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
