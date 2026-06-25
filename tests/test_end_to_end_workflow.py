import pandas as pd

from soccer_edge.evaluation.replay import replay_predictions
from soccer_edge.models.bundle import load_bundle, save_bundle
from soccer_edge.run_reports import write_json, write_table
from soccer_edge.schemas import BallStateRecord
from soccer_edge.video.state_tables import write_video_state_tables


def test_end_to_end_fixture_workflow(tmp_path) -> None:
    state_dir = tmp_path / "state"
    paths = write_video_state_tables(
        state_dir,
        ball_states=[BallStateRecord("video", 1, 1.0, 50.0, 34.0, 0.9)],
    )
    assert paths["ball_states"].exists()

    bundle_dir = tmp_path / "bundle"
    save_bundle(
        model={"kind": "demo"},
        output_dir=bundle_dir,
        name="demo",
        version="v0",
        feature_names=["x"],
        metrics={"accuracy": 1.0},
    )
    model, metadata = load_bundle(bundle_dir)
    assert model["kind"] == "demo"
    assert metadata.metrics["accuracy"] == 1.0

    predictions = pd.DataFrame(
        [{"match_id": "m1", "timestamp_seconds": 1.0, "label": 0, "prob_0": 0.8, "prob_1": 0.1, "prob_2": 0.1}]
    )
    result = replay_predictions(predictions)
    assert result.metrics.accuracy == 1.0

    report_dir = tmp_path / "report"
    write_json(result.metrics, report_dir / "metrics.json")
    write_table(predictions, report_dir / "predictions.csv")
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "predictions.csv").exists()
