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


def test_build_npz_tensor_samples_grouped(tmp_path) -> None:
    columns = [f"g{i}" for i in range(4)]
    rows = []
    for match_id in ["m1", "m2"]:
        for label in [0, 1]:
            rows.append({**{column: 0.0 for column in columns}, "label": label, "match_id": match_id})
    frame = pd.DataFrame(rows)
    path = build_npz_tensor_samples(
        frame,
        spatial_columns=columns,
        label_column="label",
        output_path=tmp_path / "grouped.npz",
        sequence_length=2,
        channels=1,
        height=2,
        width=2,
        group_column="match_id",
    )
    data = np.load(path)
    assert data["spatial"].shape == (2, 2, 1, 2, 2)
    assert data["labels"].tolist() == [1, 1]


def test_build_npz_tensor_samples_orders_within_group(tmp_path) -> None:
    columns = ["g0"]
    frame = pd.DataFrame(
        [
            {"match_id": "m1", "timestamp_seconds": 2.0, "g0": 2.0, "label": 2},
            {"match_id": "m1", "timestamp_seconds": 1.0, "g0": 1.0, "label": 1},
        ]
    )
    path = build_npz_tensor_samples(
        frame,
        spatial_columns=columns,
        label_column="label",
        output_path=tmp_path / "ordered.npz",
        sequence_length=2,
        channels=1,
        height=1,
        width=1,
        group_column="match_id",
        order_column="timestamp_seconds",
    )
    data = np.load(path)
    assert data["spatial"].reshape(2).tolist() == [1.0, 2.0]
    assert data["labels"].tolist() == [2]


def test_build_npz_from_table(tmp_path) -> None:
    columns = [f"g{i}" for i in range(4)]
    source = tmp_path / "grid.csv"
    pd.DataFrame([{**{column: 0.0 for column in columns}, "label": 0}]).to_csv(source, index=False)
    path = build_npz_from_table(source, tmp_path / "out.npz", columns, "label", channels=1, height=2, width=2)
    assert path.exists()


def test_build_npz_tensor_samples_validates_shape(tmp_path) -> None:
    with pytest.raises(ValueError):
        build_npz_tensor_samples(pd.DataFrame([{"label": 0}]), [], "label", tmp_path / "bad.npz", channels=1, height=2, width=2)
