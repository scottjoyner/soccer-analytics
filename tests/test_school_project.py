import pandas as pd

from soccer_edge.school_project import DEFAULT_FEATURE_COLUMNS, calibration_ready_predictions, run_school_project


def make_project_frame() -> pd.DataFrame:
    rows = []
    labels = ["H", "A", "D"] * 6
    for idx, label in enumerate(labels):
        home_adv = 1 if label == "H" else 0
        away_adv = 1 if label == "A" else 0
        drawish = 1 if label == "D" else 0
        rows.append(
            {
                "match_id": f"m{idx:03d}",
                "home_shots_last5": 14 + home_adv * 2 + drawish,
                "away_shots_last5": 14 + away_adv * 2 + drawish,
                "home_xg_last5": 1.5 + home_adv * 0.5,
                "away_xg_last5": 1.5 + away_adv * 0.5,
                "home_player_form": 0.55 + home_adv * 0.2,
                "away_player_form": 0.55 + away_adv * 0.2,
                "home_pressure_index": 0.50 + home_adv * 0.2,
                "away_pressure_index": 0.50 + away_adv * 0.2,
                "home_rest_days": 5 + home_adv,
                "away_rest_days": 5 + away_adv,
                "label": label,
            }
        )
    return pd.DataFrame(rows)


def test_calibration_ready_predictions() -> None:
    frame = pd.DataFrame([{"label": "H", "prediction": "H", "prob_0": 0.1, "prob_1": 0.8, "prob_2": 0.1}])
    ready = calibration_ready_predictions(frame, ["A", "H", "D"])
    assert ready.iloc[0]["label"] == 1
    assert ready.iloc[0]["label_text"] == "H"


def test_run_school_project(tmp_path) -> None:
    source = tmp_path / "school.csv"
    output_dir = tmp_path / "out"
    make_project_frame().to_csv(source, index=False)
    paths = run_school_project(source, output_dir, feature_columns=DEFAULT_FEATURE_COLUMNS, train_fraction=0.7, random_state=7)
    assert (output_dir / "final_model.joblib").exists()
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "calibration_review" / "metrics.json").exists()
    assert (output_dir / "PROJECT_RUN_SUMMARY.md").exists()
    assert paths.model_card.endswith("MODEL_CARD.md")
