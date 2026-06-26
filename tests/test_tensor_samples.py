import numpy as np
import pandas as pd
import pytest

from soccer_edge.models.tensor_samples import build_npz_from_table, build_npz_tensor_samples


def test_build_npz_tensor_samples(tmp_path) -> None:
    columns = [f"g{i}" for i in range(12)]
    frame = pd.DataFrame([{**{column: float(i) for i, column in enumerate(columns)}, "label": 1} for _ in range(2)])
    path = build_npz_tensor_samples(
        frame,
        spatial_columns=columns,
        label_column="label",
        output_path=tmp_path / "samples.npz",
        sequence_length=2,
        channels=3,
        height=2,
        width=2,
    )
    data = np.load(path)
    assert data["spatial"].shape == (1, 2, 3, 2, 2)
    assert data["labels"].tolist() == [1]


def test_build_npz_from_table(tmp_path) -> None:
    columns = [f"g{i}" for i in range(4)]
    source = tmp_path / "grid.csv"
    pd.DataFrame([{**{column: 0.0 for column in columns}, "label": 0}]).to_csv(source, index=False)
    path = build_npz_from_table(source, tmp_path / "out.npz", columns, "label", channels=1, height=2, width=2)
    assert path.exists()


def test_build_npz_tensor_samples_validates_shape(tmp_path) -> None:
    with pytest.raises(ValueError):
        build_npz_tensor_samples(pd.DataFrame([{"label": 0}]), [], "label", tmp_path / "bad.npz", channels=1, height=2, width=2)
