# AGENTS.md

## Mission

Build and operate the local soccer analytics pipeline for research, feature generation, model training, and model review. The agent should prioritize a rights-safe, reproducible workflow that turns approved local footage and open soccer data into structured tables, tensor samples, model bundles, predictions, calibration reports, data cards, graph payloads, and review artifacts.

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

> On an externally-managed Python (PEP 668) where `pip install` is blocked, or
> where `python3 -m venv` fails because `python3-venv` is missing, install with
> `pip install --break-system-packages` (dedicated research box) or build a venv
> with `python3 -m virtualenv --system-site-packages .venv` to reuse an installed
> `torch`/`numpy`/`pandas`. The ML stack needs `torch`, `ultralytics`, `opencv-python`.

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
soccer-edge video merge-corrections --base data/processed/detections_with_images.csv --corrections data/processed/reviewed_corrections.csv --output data/processed/corrected_detections.csv --keys crop_path
soccer-edge video split-annotations --source data/processed/corrected_detections.csv --train-output data/processed/annotations/train.csv --val-output data/processed/annotations/val.csv --train-fraction 0.8
soccer-edge video audit-annotations --source data/processed/corrected_detections.csv --output-dir data/processed/annotation_audit
soccer-edge video dataset-versions --paths data/processed/frame_manifest.csv,data/processed/corrected_detections.csv,data/processed/annotations/train.csv,data/processed/annotations/val.csv --output data/processed/dataset_versions.csv
```

Run calibration and annotation prep:

```bash
soccer-edge video calibration-qa --calibration configs/pitch_calibration.json --csv-output data/processed/calibration_qa.csv --svg-output data/processed/calibration_qa.svg
soccer-edge video calibration-summary --source data/processed/calibration_qa.csv --output data/processed/calibration_qa.md
soccer-edge video export-annotations --source data/processed/corrected_detections.csv --output-dir data/processed/annotations --classes player,ball --image-width 1920 --image-height 1080
soccer-edge video prepare-object-dataset --source data/processed/corrected_detections.csv --output-dir data/processed/annotations/yolo --classes player,ball --train-fraction 0.8
```

The `prepare-object-dataset` command builds the ultralytics-compatible YOLO layout
(`images/{train,val}` + `labels/{train,val}` + `data.yaml`) from detection rows that carry
an `image_path` column, so `train object-model` can consume it directly. The full
`train local-finetune` pipeline runs this step automatically (its `annotation_config` points
at `annotations/yolo/data.yaml`). The `annotation-config` command alone only writes a
`data.yaml` and assumes you have already arranged the `images/`+`labels/` directories.

Capture local footage for intake:

```bash
soccer-edge capture screen --duration 30 --fps 20 --rights-status owned --rights-reference "license-abc"
soccer-edge capture webcam --duration 30 --device 0 --rights-status owned --rights-reference "license-abc"
soccer-edge capture image --rights-status owned --rights-reference "license-abc"
```

The `capture` commands record screen/webcam/video or a screenshot to
`data/raw/video_licensed/captures/` and append a rights-gated manifest row (source scheme
`capture://screen|webcam|image`). Every capture requires `rights_status` in
{owned, licensed, compatible_license} and a recorded `rights_reference`; the command refuses
to capture otherwise. This is for content you own or are licensed/compatible-license for —
capturing third-party streams (YouTube, Twitch, etc.) is prohibited (see Hard safety and
data rules). After capture, feed the saved file straight into the rights-gated pipeline:

```bash
soccer-edge train local-finetune --input <saved_file> --manifest manifests/video_manifest.csv --video-id <video_id> --object-model-path models/yolov8n.pt
```

Generate dataset metadata and evaluate object detections:

```bash
soccer-edge model source-catalog --output data/processed/training_sources.csv
soccer-edge model auto-data-card --dataset-name local-finetune-dataset --manifests data/processed/frame_manifest.csv,data/processed/corrected_detections.csv,data/processed/crop_manifest.csv --output data/processed/DATA_CARD.md --version-paths data/processed/frame_manifest.csv,data/processed/corrected_detections.csv,data/processed/annotations/train.csv,data/processed/annotations/val.csv
soccer-edge model object-eval --source data/processed/object_eval_rows.csv --output data/processed/object_eval.csv
soccer-edge model object-confusion --source data/processed/object_eval_rows.csv --table-output data/processed/object_confusion.csv --svg-output data/processed/object_confusion.svg
soccer-edge model data-card --dataset-name local-dataset --sources data/processed/corrected_detections.csv --output data/processed/DATA_CARD.md --version-paths data/processed/corrected_detections.csv
```

Run the full local fine-tuning path when optional media/object-model dependencies are installed:

```bash
soccer-edge train local-finetune \
  --input data/raw/video_licensed/clip.mp4 \
  --object-model-path models/local-object-model.pt \
  --output-dir data/processed/local_finetune \
  --classes player,ball \
  --calibration-path configs/pitch_calibration.json \
  --stride 5 \
  --max-frames 100
```

Dry-run the full path without optional media/model dependencies:

```bash
soccer-edge train local-finetune \
  --input data/raw/video_licensed/clip.mp4 \
  --object-model-path models/local-object-model.pt \
  --output-dir data/processed/local_finetune \
  --classes player,ball \
  --calibration-path configs/pitch_calibration.json \
  --dry-run-plan data/processed/local_finetune/plan.sh
```

Out-of-sample evaluation of the highlight-clip models (leakage-safe, stratified
68/30 match hold-out; the CNN winner classifier is trained only on the train
split):

```bash
python scripts/evaluate_highlights.py            # tabular winner + score, calibrated
python scripts/evaluate_cnn.py                   # CNN winner, sequence + match accuracy, Brier
SEEDS="0 1 2 3 4" ./scripts/batch_cnn_eval.sh <OUT_ROOT> 10   # seed-sweep on a batch machine
# repeated-CV variant reporting mean +/- sd across folds (matches StatsBomb reporting):
python scripts/evaluate_cnn.py --folds 5 --repeats 1 --epochs 10 --output-dir <OUT_ROOT>_cv
```

The CNN evaluation caps PyTorch threads (`OMP_NUM_THREADS`, default 8) to avoid
CPU/memory oversubscription crashes; heavy sweeps should run on the batch box,
not the interactive machine.

## Fine-tuning pipeline target

The agent should prepare data for model fine-tuning in this order:

1. Ingest local/open event data into processed Parquet tables.
2. Catalog approved local footage and export frame manifests.
3. Process approved local footage into detections/tracks/state tables.
4. Join detection rows to frame image paths by `frame_idx`.
5. Convert pixel-space detections to pitch-space state when calibration is available.
6. Review low-confidence crops and merge corrected rows.
7. Split annotation rows without leaking frame groups across train/validation outputs.
8. Build audits, dataset versions, source catalogs, automatic data cards, and graph export payloads.
9. Export normalized annotations, low-confidence review queues, object crops, and contact sheets.
10. Train baseline tabular models, CNN tensor models, and optional local object models.
11. Export predictions, object evaluation metrics/confusion matrices, calibration reports, registry summaries, cards, and markdown comparison reports.
12. Promote only model bundles with reproducible metadata, feature names, metrics, cards, versions, and lineage.

## Suggested next implementation tasks

Completed in recent work:

1. Add out-of-sample CNN highlight-clip winner evaluation (`scripts/evaluate_cnn.py`)
   with a stratified 68/30 match hold-out and a seed-sweep batch wrapper
   (`scripts/batch_cnn_eval.sh`) for off-box compute.
2. Add leakage-safe, calibrated out-of-sample tabular evaluation
   (`scripts/evaluate_highlights.py`).
3. Add a repeated-CV variant of the CNN out-of-sample eval (`--folds`/`--repeats`)
   reporting mean±sd, like the StatsBomb results.

Forward-looking:

1. Extend the highlight study with full-match footage or richer open event data
   once rights-clean sources are available.
2. Promote a model bundle through the promotion-gate command once a rights-clean
   source yields lift over the majority-class baseline. Note: the highlight-clip
   CNN still shows no lift (51.6% sequence / 50.0% match accuracy at baseline;
   winner Brier ~0.62 across seeds), so the promotion gate remains untriggered.

   The `model promotion-gate` command now enforces this automatically: pass
   `--predictive-metrics` (a table with `accuracy`/`brier` columns),
   `--majority-baseline-rate`, `--min-accuracy-lift` (default 0.02), and
   `--max-brier`. The `beats_majority_baseline` and `brier_within_threshold`
   checks fail the gate when a bundle shows no lift or is uncalibrated.

   The eval scripts emit a `metrics.json` whose schema differs per script. Normalize
   it into the promotion-gate predictive metrics table with:

   ```bash
   soccer-edge model eval-to-metrics --metrics-json <OUT_ROOT>/metrics.json --output data/processed/predictive_metrics.csv --model-name cnn-v1
   ```

   Then run the gate:

   ```bash
   soccer-edge model promotion-gate \
     --model-card-path <model_card> --data-card-path <data_card> \
     --dataset-versions <versions.csv> --audit-dir <audit_dir> --object-metrics <obj_metrics.csv> \
     --predictive-metrics data/processed/predictive_metrics.csv \
     --majority-baseline-rate 0.50 --min-accuracy-lift 0.02 --max-brier 0.50
   ```

   A passing gate promotes the bundle into `models/promoted/<name>/<version>/`
   (copied bundle + cards + metrics + gate report + `promotion.json`):

   ```bash
   soccer-edge model promote \
     --bundle-dir <candidate_bundle> --promoted-root models/promoted \
     --model-card-path <model_card> --data-card-path <data_card> \
     --dataset-versions <versions.csv> --audit-dir <audit_dir> --object-metrics <obj_metrics.csv> \
     --predictive-metrics data/processed/predictive_metrics.csv \
     --majority-baseline-rate 0.50 --min-accuracy-lift 0.02 --max-brier 0.50
   soccer-edge model promoted-list --promoted-root models/promoted --output data/processed/promoted_models.csv
   ```

   `model promote` exits non-zero (and writes nothing) when the gate fails, so the
   highlight-clip CNN cannot be promoted until a rights-clean source yields real lift.

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
