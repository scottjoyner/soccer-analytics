# Local Pipeline

Use these commands after installing the package in editable mode.

```bash
soccer-edge ingest write-processed --source data/raw/metrica --source-type metrica --output-dir data/processed/ingest
soccer-edge features inplay --source data/processed/game_state.csv --output data/processed/inplay_features.parquet --columns speed,pressure --window-seconds 60
soccer-edge train simple --source data/processed/inplay_features.parquet --columns speed_last,pressure_last --label label --output-dir data/processed/simple_model
soccer-edge model predict --bundle-dir data/processed/simple_model --source data/processed/inplay_features.parquet --output data/processed/predictions.csv
soccer-edge model registry --root-dir data/processed --output data/processed/model_registry.csv
soccer-edge model registry-summary --root-dir data/processed --output data/processed/model_registry_summary.csv
soccer-edge model compare --registry data/processed/model_registry_summary.csv --output data/processed/model_comparison.csv
soccer-edge model compare-markdown --comparison data/processed/model_comparison.csv --output data/processed/model_comparison.md
soccer-edge model run-summary --registry data/processed/model_registry_summary.csv --predictions data/processed/predictions.csv --output-dir data/processed/run_summary
soccer-edge model model-card --bundle-dir data/processed/simple_model --output data/processed/MODEL_CARD.md
soccer-edge model data-card --dataset-name local-dataset --sources data/raw/video_licensed,data/processed/inplay_features.parquet --output data/processed/DATA_CARD.md --rights-status owned
soccer-edge model calibration-review --predictions data/processed/predictions.csv --output-dir data/processed/calibration_review
```

Optional CNN training uses an NPZ file with `spatial` and `labels` arrays. You can build the NPZ from flattened grid columns, train, export CNN outputs, and run calibration review.

```bash
soccer-edge features tensor-samples --source data/processed/grid_features.csv --output data/processed/tensor_samples.npz --columns g0,g1,g2,g3 --label label --channels 1 --height 2 --width 2 --group match_id --order timestamp_seconds
soccer-edge train cnn --source data/processed/tensor_samples.npz --output-dir data/processed/cnn_model --output-classes 3 --epochs 5 --batch-size 8
soccer-edge model predict-cnn --bundle-dir data/processed/cnn_model --source data/processed/tensor_samples.npz --output data/processed/cnn_predictions.csv
soccer-edge model calibration-review-cnn --bundle-dir data/processed/cnn_model --source data/processed/tensor_samples.npz --output-dir data/processed/cnn_calibration_review
```

Local media processing remains restricted to files you have rights to process. The media reader gate uses optional OpenCV support, and the media inference adapter converts local model outputs into table-ready rows.

```bash
soccer-edge video catalog-local --root data/raw/video_licensed --output manifests/local_video_manifest.csv --rights-status owned
soccer-edge video plan --manifest manifests/local_video_manifest.csv --licensed-root data/raw/video_licensed
soccer-edge video process-local-model --input data/raw/video_licensed/clip.mp4 --model-path models/local-object-model.pt --output-dir data/processed/video_model --stride 5 --max-samples 100
soccer-edge video export-annotations --source data/processed/video_model/detections.parquet --output-dir data/processed/annotations --classes player,ball --image-width 1920 --image-height 1080
soccer-edge video sample-low-confidence --source data/processed/video_model/detections.parquet --output data/processed/low_confidence.csv --threshold 0.5 --limit 100
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
  --order timestamp_seconds
```

Run the tiny example pipeline:

```bash
soccer-edge examples tiny --repo-root . --output-dir data/processed/examples/tiny_pipeline
```
