# Local Pipeline

Use these commands after installing the package in editable mode.

```bash
soccer-edge ingest write-processed --source data/raw/metrica --source-type metrica --output-dir data/processed/ingest
soccer-edge features inplay --source data/processed/game_state.csv --output data/processed/inplay_features.parquet --columns speed,pressure --window-seconds 60
soccer-edge train simple --source data/processed/inplay_features.parquet --columns speed_last,pressure_last --label label --output-dir data/processed/simple_model
soccer-edge model registry --root-dir data/processed --output data/processed/model_registry.csv
soccer-edge model calibration-review --predictions data/processed/predictions.csv --output-dir data/processed/calibration_review
```

Local media processing remains restricted to files you have rights to process. The current command writes the state-table scaffold and is ready for a future sampler and adapter.
