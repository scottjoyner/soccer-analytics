import pandas as pd

from soccer_edge.dataset_versioning import asset_version, dataset_version_id, dataset_versions, file_sha256, write_dataset_versions


def test_file_sha256(tmp_path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("abc", encoding="utf-8")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_asset_version_for_table(tmp_path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame([{"x": 1}]).to_csv(path, index=False)
    version = asset_version(path)
    assert version.row_count == 1
    assert version.column_count == 1


def test_dataset_version_id_is_stable(tmp_path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame([{"x": 1}]).to_csv(a, index=False)
    pd.DataFrame([{"y": 2}]).to_csv(b, index=False)
    assert dataset_version_id([a, b]) == dataset_version_id([b, a])
    assert dataset_version_id([a, b]).startswith("ds_")


def test_write_dataset_versions(tmp_path) -> None:
    path = tmp_path / "data.csv"
    output = tmp_path / "versions.csv"
    pd.DataFrame([{"x": 1}]).to_csv(path, index=False)
    result = write_dataset_versions([path], output)
    assert result.exists()
    assert len(dataset_versions([path])) == 1
