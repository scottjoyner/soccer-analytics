# Training Data Sourcing Strategy

This project should build training data from a layered, rights-safe source strategy. The goal is to support soccer event modeling, tracking/state modeling, local object detection, calibration QA, and future fine-tuning without scraping or downloading unauthorized audiovisual content.

## Source tiers

### Tier 1 — Open event and tracking data

Use these sources first because they can bootstrap features, labels, schema validation, and baseline models without requiring local video rights.

| Source | Best use | Pipeline path | Rights posture |
| --- | --- | --- | --- |
| StatsBomb Open Data | Event data, lineups, selected 360 freeze-frame context | `soccer-edge ingest write-processed --source <statsbomb-root> --source-type statsbomb` | Public research/open-data use; attribution required by source terms. |
| Metrica Sports Sample Data | Tracking and event synchronization, pitch coordinate normalization, possession/pressure features | `soccer-edge ingest write-processed --source <metrica-root> --source-type metrica` | Sample data for public analysis; acknowledge source. |
| SoccerNet annotations | Benchmark labels for action spotting, calibration, tracking-style tasks, depending on subset access | `soccer-edge ingest write-processed --source <soccernet-root> --source-type soccernet` | Respect SoccerNet access terms and task-specific data restrictions. |

### Tier 2 — Licensed or owned local footage

Use this for object detection, frame extraction, crop review, pitch calibration, and model fine-tuning.

Approved locations:

- User-owned local disk folders.
- NAS or mounted storage.
- Tailscale-accessible shares controlled by the user.
- Partner-provided footage where written processing rights are recorded.

Never use a public URL as a media input unless written rights are recorded in the local manifest.

Required manifest columns:

```text
video_id,match_id,clip_type,local_path,rights_status,notes
```

Allowed `rights_status` values for processing:

```text
owned,licensed,compatible_license
```

### Tier 3 — Self-generated annotations

Use local object model outputs and review tools to create higher-quality training labels.

Recommended loop:

```bash
soccer-edge video export-frames --input data/raw/video_licensed/clip.mp4 --output-dir data/processed/frames --manifest-output data/processed/frame_manifest.csv --stride 5 --max-frames 100
soccer-edge video process-local-model --input data/raw/video_licensed/clip.mp4 --model-path models/local-object-model.pt --output-dir data/processed/video_model --stride 5 --max-samples 100 --calibration configs/pitch_calibration.json
soccer-edge video attach-frame-images --detections data/processed/video_model/detections.parquet --frame-manifest data/processed/frame_manifest.csv --output data/processed/detections_with_images.csv
soccer-edge video sample-low-confidence --source data/processed/detections_with_images.csv --output data/processed/low_confidence.csv --threshold 0.5 --limit 100
soccer-edge video export-crops --source data/processed/low_confidence.csv --output-dir data/processed/crops --manifest-output data/processed/crop_manifest.csv --image-path-column image_path
soccer-edge video contact-sheet --source data/processed/crop_manifest.csv --output data/processed/crop_review.html
```

Human review should promote corrected rows back into the annotation set. The corrected annotations can then be exported, split, audited, and versioned:

```bash
soccer-edge video export-annotations --source data/processed/corrected_detections.csv --output-dir data/processed/annotations --classes player,ball --image-width 1920 --image-height 1080
soccer-edge video split-annotations --source data/processed/corrected_detections.csv --train-output data/processed/annotations/train.csv --val-output data/processed/annotations/val.csv --train-fraction 0.8
soccer-edge video audit-annotations --source data/processed/corrected_detections.csv --output-dir data/processed/annotation_audit
soccer-edge video dataset-versions --paths data/processed/frame_manifest.csv,data/processed/corrected_detections.csv,data/processed/annotations/train.csv,data/processed/annotations/val.csv --output data/processed/dataset_versions.csv
soccer-edge video annotation-config --root data/processed/annotations --train-images images/train --val-images images/val --classes player,ball --output data/processed/annotations/data.yaml
```

### Tier 4 — Evaluation and promotion metadata

Before promoting a fine-tuned dataset or model, generate a source catalog, automatic data card, and class-level object metrics.

```bash
soccer-edge model source-catalog --output data/processed/training_sources.csv
soccer-edge model auto-data-card --dataset-name local-finetune-dataset --manifests data/processed/frame_manifest.csv,data/processed/corrected_detections.csv,data/processed/crop_manifest.csv --output data/processed/DATA_CARD.md
soccer-edge model object-eval --source data/processed/object_eval_rows.csv --output data/processed/object_eval.csv
```

## Training-data priorities

1. **Baseline event model**: use StatsBomb and Metrica event tables to train simple tabular models and validate scoring/calibration plumbing.
2. **State and tracking model**: use Metrica tracking data to validate pitch-space state features and rolling-window table generation.
3. **Broadcast/video model**: use SoccerNet only after confirming access terms for the specific task/subset.
4. **Local object model**: use owned/licensed local footage, exported frames, detection review queues, and corrected annotations.
5. **Fine-tuning bundle**: promote only datasets with data cards, source manifests, rights status, feature lineage, model cards, dataset versions, audits, and calibration reports.

## Full local fine-tuning command

When optional media and local object-model dependencies are installed, the command below chains frame export, local object-model detections, image-path joins, low-confidence review, crops, annotations, train/validation split, annotation config, automatic data card, and optional object-model training.

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

## Minimum dataset card checklist

Every training dataset should have:

- Dataset name and version.
- Source roots or manifests.
- Rights status.
- Allowed use.
- Restricted use.
- Label definitions.
- Class list.
- Train/validation split method.
- Known bias or quality issues.
- Link to calibration QA artifacts when pitch-space labels are used.
- Dataset asset hashes.
- Annotation audit summaries.

## Label taxonomy

Initial local object classes:

```text
player
ball
referee
goalkeeper
```

Initial event/state labels:

```text
shot
pass
carry
pressure
turnover
goal
set_piece
```

## Quality gates before training

Run these before model training:

```bash
soccer-edge model validate-cards --data-card-path data/processed/DATA_CARD.md
soccer-edge video calibration-qa --calibration configs/pitch_calibration.json --csv-output data/processed/calibration_qa.csv --svg-output data/processed/calibration_qa.svg
soccer-edge video calibration-summary --source data/processed/calibration_qa.csv --output data/processed/calibration_qa.md
soccer-edge video audit-annotations --source data/processed/corrected_detections.csv --output-dir data/processed/annotation_audit
soccer-edge video dataset-versions --paths data/processed/frame_manifest.csv,data/processed/corrected_detections.csv,data/processed/annotations/train.csv,data/processed/annotations/val.csv --output data/processed/dataset_versions.csv
pytest
ruff check src tests
```

## What the agent should not do

- Do not download match videos from public platforms for training.
- Do not silently process footage with `pending`, `unknown`, or missing rights status.
- Do not mix train and validation labels from the same frame group unless explicitly testing leakage.
- Do not promote a model without data cards, model cards, dataset versions, audits, and calibration/evaluation artifacts.
