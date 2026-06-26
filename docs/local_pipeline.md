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
soccer-edge model calibration-review --predictions data/processed/predictions.csv --output-dir data/processed/calibration_review
```

Optional CNN training uses an NPZ file with `spatial` and `labels` arrays. You can build the NPZ from flattened grid columns, train, export CNN outputs, and run calibration review.

```bash
soccer-edge features tensor-samples --source data/processed/grid_features.csv --output data/processed/tensor_samples.npz --columns g0,g1,g2,g3 --label label --channels 1 --height 2 --width 2 --group match_id
soccer-edge train cnn --source data/processed/tensor_samples.npz --output-dir data/processed/cnn_model --output-classes 3 --epochs 5 --batch-size 8
soccer-edge model predict-cnn --bundle-dir data/processed/cnn_model --source data/processed/tensor_samples.npz --output data/processed/cnn_predictions.csv
soccer-edge model calibration-review-cnn --bundle-dir data/processed/cnn_model --source data/processed/tensor_samples.npz --output-dir data/processed/cnn_calibration_review
```

Local media processing remains restricted to files you have rights to process. The media reader gate uses optional OpenCV support, and the media inference adapter converts local model outputs into table-ready rows.
