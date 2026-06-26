# Examples

These tiny fixtures are for smoke-testing local commands.

Build tensor samples from flattened grid columns:

```bash
soccer-edge features tensor-samples \
  --source examples/tiny_grid_features.csv \
  --output data/processed/examples/tiny_tensor_samples.npz \
  --columns g0,g1,g2,g3 \
  --label label \
  --channels 1 \
  --height 2 \
  --width 2 \
  --sequence-length 2 \
  --group match_id
```

Train a simple model and export predictions:

```bash
soccer-edge train simple \
  --source examples/tiny_training.csv \
  --columns speed_last,pressure_last \
  --label label \
  --output-dir data/processed/examples/simple_model

soccer-edge model predict \
  --bundle-dir data/processed/examples/simple_model \
  --source examples/tiny_training.csv \
  --output data/processed/examples/predictions.csv
```
