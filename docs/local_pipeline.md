# Local Pipeline

Use these commands after installing the package in editable mode.

```bash
soccer-edge ingest write-processed --source data/raw/metrica --source-type metrica --output-dir data/processed/ingest
soccer-edge features inplay --source data/processed/game_state.csv --output data/processed/inplay_features.parquet --columns speed,pressure --window-seconds 60
soccer-edge train simple --source data/processed/inplay_features.parquet --columns speed_last,pressure_last --label label --output-dir data/processed/simple_model
soccer-edge model predict --bundle-dir data/processed/simple_model --source data/processed/inplay_features.parquet --output data/processed/predictions.csv
soccer-edge model registry --root-dir data/processed --output data/processed/model_registry.csv
soccer-edge model registry-summary --root-dir data/processed --output data/processed/model_registry_summary.csv
soccer-edge model calibration-review --predictions data/processed/predictions.csv --output-dir data/processed/calibration_review
```

Optional CNN training uses an NPZ file with `spatial` and `labels` arrays.

```bash
soccer-edge train cnn --source data/processed/tensor_samples.npz --output-dir data/processed/cnn_model --output-classes 3 --epochs 5 --batch-size 8
```

Local media processing remains restricted to files you have rights to process. The current command writes the state-table scaffold and is ready for a future sampler and adapter.
