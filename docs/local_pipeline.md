# Local Pipeline

Use these commands after installing the package in editable mode.

```bash
soccer-edge ingest raw-sources --output data/processed/raw_data_sources.csv
soccer-edge ingest write-processed --source data/raw/metrica --source-type metrica --output-dir data/processed/ingest
soccer-edge features inplay --source data/processed/game_state.csv --output data/processed/inplay_features.parquet --columns speed,pressure --window-seconds 60
soccer-edge features player-stats --events data/processed/ingest/events.parquet --output data/processed/player_match_stats.csv
soccer-edge features player-form --player-stats data/processed/player_match_stats.csv --output data/processed/player_form.csv --window 5 --order-column match_id
soccer-edge features player-aggregate --player-stats data/processed/player_match_stats.csv --output data/processed/player_aggregates.csv
soccer-edge features player-aggregate --player-stats data/processed/player_match_stats.csv --split-by opponent --output data/processed/player_aggregates_by_opponent.csv
soccer-edge train simple --source data/processed/inplay_features.parquet --columns speed_last,pressure_last --label label --output-dir data/processed/simple_model
soccer-edge model predict --bundle-dir data/processed/simple_model --source data/processed/inplay_features.parquet --output data/processed/predictions.csv
soccer-edge model registry --root-dir data/processed --output data/processed/model_registry.csv
soccer-edge model registry-summary --root-dir data/processed --output data/processed/model_registry_summary.csv
soccer-edge model compare --registry data/processed/model_registry_summary.csv --output data/processed/model_comparison.csv
soccer-edge model compare-markdown --comparison data/processed/model_comparison.csv --output data/processed/model_comparison.md
soccer-edge model run-summary --registry data/processed/model_registry_summary.csv --predictions data/processed/predictions.csv --output-dir data/processed/run_summary
soccer-edge model model-card --bundle-dir data/processed/simple_model --output data/processed/MODEL_CARD.md --version-paths data/processed/inplay_features.parquet --graph-ids ModelRun::simple
soccer-edge model data-card --dataset-name local-dataset --sources data/raw/video_licensed,data/processed/inplay_features.parquet --output data/processed/DATA_CARD.md --rights-status owned --version-paths data/processed/inplay_features.parquet --graph-ids DatasetVersion::local-dataset
soccer-edge model validate-cards --model-card-path data/processed/MODEL_CARD.md --data-card-path data/processed/DATA_CARD.md
soccer-edge model calibration-review --predictions data/processed/predictions.csv --output-dir data/processed/calibration_review
```

Optional CNN training uses an NPZ file with `spatial` and `labels` arrays. You can build the NPZ from flattened grid columns, train, export CNN outputs, and run calibration review.

```bash
soccer-edge features tensor-samples --source data/processed/grid_features.csv --output data/processed/tensor_samples.npz --columns g0,g1,g2,g3 --label label --channels 1 --height 2 --width 2 --group match_id --order timestamp_seconds
soccer-edge train cnn --source data/processed/tensor_samples.npz --output-dir data/processed/cnn_model --output-classes 3 --epochs 5 --batch-size 8
soccer-edge model predict-cnn --bundle-dir data/processed/cnn_model --source data/processed/tensor_samples.npz --output data/processed/cnn_predictions.csv
soccer-edge model calibration-review-cnn --bundle-dir data/processed/cnn_model --source data/processed/tensor_samples.npz --output-dir data/processed/cnn_calibration_review
```

Local media processing remains restricted to files you have rights to process. The media reader gate uses optional OpenCV support, and the media inference adapter converts local model outputs into table-ready rows. Every command that opens raw footage (`export-frames`, `process`, `process-local-model`, `detect-yolo`, `train player-ball`, and `train local-finetune` run mode) enforces a rights gate: pass `--manifest` and `--video-id` pointing at an approved `catalog-local` row whose `rights_reference` is recorded and whose `local_path` matches the input. Omitting these flags keeps the legacy trust-local-file behavior. The gate additionally enforces a modality blocklist (`configs/modality_rules.json`) that rejects public/remote sources (`youtube`, `twitch`, `stream`, `http(s)/rtmp/rtsp`). The in-repo `configs/pitch_calibration.json` and `models/local-object-model.pt` (a YOLOv8n base detector) are provided so the fine-tuning path runs without external downloads.

```bash
soccer-edge video catalog-local --root data/raw/video_licensed --output manifests/local_video_manifest.csv --rights-status owned --rights-reference <written-rights-reference>
soccer-edge video plan --manifest manifests/local_video_manifest.csv --licensed-root data/raw/video_licensed
soccer-edge video export-frames --input data/raw/video_licensed/clip.mp4 --output-dir data/processed/frames --manifest-output data/processed/frame_manifest.csv --stride 5 --max-frames 100 --manifest manifests/local_video_manifest.csv --video-id clip --licensed-root data/raw/video_licensed
soccer-edge video calibration-qa --calibration configs/pitch_calibration.json --csv-output data/processed/calibration_qa.csv --svg-output data/processed/calibration_qa.svg
soccer-edge video calibration-summary --source data/processed/calibration_qa.csv --output data/processed/calibration_qa.md
soccer-edge video process-local-model --input data/raw/video_licensed/clip.mp4 --model-path models/local-object-model.pt --output-dir data/processed/video_model --stride 5 --max-samples 100 --calibration configs/pitch_calibration.json --manifest manifests/local_video_manifest.csv --video-id clip --licensed-root data/raw/video_licensed
soccer-edge video detect-yolo --input data/raw/video_licensed/clip.mp4 --model-path models/yolov8n.pt --output-dir data/processed/video_yolo --stride 5 --max-frames 100 --manifest manifests/local_video_manifest.csv --video-id clip --licensed-root data/raw/video_licensed
soccer-edge video attach-frame-images --detections data/processed/video_model/detections.parquet --frame-manifest data/processed/frame_manifest.csv --output data/processed/detections_with_images.csv
soccer-edge video sample-low-confidence --source data/processed/detections_with_images.csv --output data/processed/low_confidence.csv --threshold 0.5 --limit 100
soccer-edge video export-crops --source data/processed/low_confidence.csv --output-dir data/processed/crops --manifest-output data/processed/crop_manifest.csv --image-path-column image_path
soccer-edge video contact-sheet --source data/processed/crop_manifest.csv --output data/processed/crop_review.html
soccer-edge video correction-review --source data/processed/crop_manifest.csv --html-output data/processed/correction_review.html --template-output data/processed/reviewed_corrections.csv --keys crop_path
soccer-edge video merge-corrections --base data/processed/detections_with_images.csv --corrections data/processed/reviewed_corrections.csv --output data/processed/corrected_detections.csv --keys crop_path
soccer-edge video export-annotations --source data/processed/corrected_detections.csv --output-dir data/processed/annotations --classes player,ball --image-width 1920 --image-height 1080
soccer-edge video split-annotations --source data/processed/corrected_detections.csv --train-output data/processed/annotations/train.csv --val-output data/processed/annotations/val.csv --train-fraction 0.8
soccer-edge video audit-annotations --source data/processed/corrected_detections.csv --output-dir data/processed/annotation_audit
soccer-edge video dataset-versions --paths data/processed/frame_manifest.csv,data/processed/corrected_detections.csv,data/processed/annotations/train.csv,data/processed/annotations/val.csv --output data/processed/dataset_versions.csv
```

Object-model evaluation, graph export, and promotion checks:

```bash
soccer-edge model source-catalog --output data/processed/training_sources.csv
soccer-edge model object-eval --source data/processed/object_eval_rows.csv --output data/processed/object_eval.csv
soccer-edge model object-confusion --source data/processed/object_eval_rows.csv --table-output data/processed/object_confusion.csv --svg-output data/processed/object_confusion.svg
soccer-edge model graph-payloads --source data/processed/dataset_versions.csv --output data/processed/dataset_version_payloads.jsonl --kind dataset-version
soccer-edge model graph-payloads --source data/processed/player_match_stats.csv --output data/processed/player_match_payloads.jsonl --kind player-match
soccer-edge model graph-payloads --source data/processed/player_form.csv --output data/processed/player_form_payloads.jsonl --kind player-form
soccer-edge model graph-audit-payloads --audit-dir data/processed/annotation_audit --output data/processed/annotation_audit_payloads.jsonl
soccer-edge model graph-payloads --source data/processed/object_eval.csv --output data/processed/object_eval_payloads.jsonl --kind object-evaluation
soccer-edge model auto-data-card --dataset-name local-finetune-dataset --manifests data/processed/frame_manifest.csv,data/processed/corrected_detections.csv,data/processed/crop_manifest.csv --output data/processed/DATA_CARD.md --version-paths data/processed/frame_manifest.csv,data/processed/corrected_detections.csv,data/processed/annotations/train.csv,data/processed/annotations/val.csv
soccer-edge model promotion-gate --model-card-path data/processed/MODEL_CARD.md --data-card-path data/processed/DATA_CARD.md --dataset-versions data/processed/dataset_versions.csv --audit-dir data/processed/annotation_audit --object-metrics data/processed/object_eval.csv --output data/processed/promotion_gate.md --min-f1 0.1
```

Optional local object-model training uses the annotation config and runs through the optional ML dependency stack.

```bash
soccer-edge video annotation-config --root data/processed/annotations --train-images images/train --val-images images/val --classes player,ball --output data/processed/annotations/data.yaml
soccer-edge train object-model --data-config data/processed/annotations/data.yaml --base-model models/local-object-model.pt --output-dir data/processed/object_training --run-name local_object_model --epochs 50 --image-size 640
```

Full local fine-tuning path, when optional media/object-model dependencies are installed:

```bash
soccer-edge train local-finetune \
  --input data/raw/video_licensed/clip.mp4 \
  --object-model-path models/local-object-model.pt \
  --output-dir data/processed/local_finetune \
  --classes player,ball \
  --calibration-path configs/pitch_calibration.json \
  --stride 5 \
  --max-frames 100 \
  --manifest manifests/local_video_manifest.csv \
  --video-id clip \
  --licensed-root data/raw/video_licensed
```

Fine-tune a player/ball detector directly on approved local footage:

```bash
soccer-edge train player-ball \
  --input data/raw/video_licensed/clip.mp4 \
  --base-model models/yolov8n.pt \
  --output-dir data/processed/player_ball_finetune \
  --stride 5 \
  --max-frames 100 \
  --manifest manifests/local_video_manifest.csv \
  --video-id clip \
  --licensed-root data/raw/video_licensed
```

Dry-run the full fine-tuning path without invoking optional media/model dependencies:

```bash
soccer-edge train local-finetune \
  --input data/raw/video_licensed/clip.mp4 \
  --object-model-path models/local-object-model.pt \
  --output-dir data/processed/local_finetune \
  --classes player,ball \
  --calibration-path configs/pitch_calibration.json \
  --dry-run-plan data/processed/local_finetune/plan.sh \
  --validate-plan-inputs
```

Run the local chain from approved footage plus existing tabular/grid feature files:

```bash
soccer-edge train local-chain \
  --footage-root data/raw/video_licensed \
  --tabular-source examples/tiny_training.csv \
  --grid-source examples/tiny_grid_features.csv \
  --output-dir data/processed/local_training_chain \
  --tabular-columns speed_last,pressure_last \
  --grid-columns g0,g1,g2,g3 \
  --order timestamp_seconds \
  --rights-reference <written-rights-reference>
```

Run the tiny example pipeline:

```bash
soccer-edge examples tiny --repo-root . --output-dir data/processed/examples/tiny_pipeline
```
