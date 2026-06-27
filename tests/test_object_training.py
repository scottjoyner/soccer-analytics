from soccer_edge.object_training import ObjectTrainingConfig, object_training_kwargs, write_object_training_config


def test_object_training_kwargs(tmp_path) -> None:
    config = ObjectTrainingConfig(
        data_config=tmp_path / "data.yaml",
        base_model=tmp_path / "base.pt",
        output_dir=tmp_path / "runs",
        run_name="demo",
        epochs=3,
        image_size=320,
    )
    kwargs = object_training_kwargs(config)
    assert kwargs["data"].endswith("data.yaml")
    assert kwargs["epochs"] == 3
    assert kwargs["imgsz"] == 320
    assert kwargs["name"] == "demo"


def test_write_object_training_config(tmp_path) -> None:
    config = ObjectTrainingConfig(data_config=tmp_path / "data.yaml", base_model=tmp_path / "base.pt", output_dir=tmp_path / "runs")
    path = write_object_training_config(config)
    assert path.exists()
    assert "data_config" in path.read_text(encoding="utf-8")
