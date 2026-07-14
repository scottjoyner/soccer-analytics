import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app

runner = CliRunner()


def test_raw_sources_and_player_stats_cli(tmp_path) -> None:
    raw_sources = tmp_path / "raw_sources.csv"
    events = tmp_path / "events.csv"
    player_stats = tmp_path / "player_stats.csv"
    player_form = tmp_path / "player_form.csv"
    result = runner.invoke(app, ["ingest", "raw-sources", "--output", str(raw_sources)])
    assert result.exit_code == 0
    assert raw_sources.exists()

    pd.DataFrame([{"match_id": 1, "player_name": "A", "team_name": "Home", "event_type": "Shot", "shot_outcome": "Goal"}]).to_csv(events, index=False)
    result = runner.invoke(app, ["features", "player-stats", "--events", str(events), "--output", str(player_stats)])
    assert result.exit_code == 0
    assert player_stats.exists()
    result = runner.invoke(app, ["features", "player-form", "--player-stats", str(player_stats), "--output", str(player_form), "--window", "3"])
    assert result.exit_code == 0
    assert player_form.exists()


def test_review_graph_and_promotion_cli(tmp_path) -> None:
    crops = tmp_path / "crops.csv"
    review_html = tmp_path / "review.html"
    corrections = tmp_path / "corrections.csv"
    versions = tmp_path / "versions.csv"
    graph = tmp_path / "versions.jsonl"
    audit_dir = tmp_path / "audit"
    metrics = tmp_path / "object_metrics.csv"
    model_card = tmp_path / "MODEL_CARD.md"
    data_card = tmp_path / "DATA_CARD.md"
    promotion = tmp_path / "promotion.md"
    pd.DataFrame([{"crop_path": "a.jpg", "class_name": "player", "x1": 0, "y1": 0, "x2": 1, "y2": 1}]).to_csv(crops, index=False)
    result = runner.invoke(app, ["video", "correction-review", "--source", str(crops), "--html-output", str(review_html), "--template-output", str(corrections)])
    assert result.exit_code == 0
    assert review_html.exists()
    assert corrections.exists()

    result = runner.invoke(app, ["video", "dataset-versions", "--paths", str(crops), "--output", str(versions)])
    assert result.exit_code == 0
    result = runner.invoke(app, ["model", "graph-payloads", "--source", str(versions), "--output", str(graph), "--kind", "dataset-version"])
    assert result.exit_code == 0
    assert graph.exists()

    audit_dir.mkdir()
    pd.DataFrame([{"class_name": "player", "row_count": 1}]).to_csv(audit_dir / "by_class.csv", index=False)
    pd.DataFrame([{"class_name": "player", "f1": 1.0}]).to_csv(metrics, index=False)
    model_card.write_text("# Model Card\n\n## Intended use\n\n## Features\n\n## Metrics\n\n## Limitations\n", encoding="utf-8")
    data_card.write_text("# Data Card\n\n## Sources\n\n## Lineage\n\n## Allowed use\n\n## Restrictions\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "model",
            "promotion-gate",
            "--model-card-path",
            str(model_card),
            "--data-card-path",
            str(data_card),
            "--dataset-versions",
            str(versions),
            "--audit-dir",
            str(audit_dir),
            "--object-metrics",
            str(metrics),
            "--output",
            str(promotion),
        ],
    )
    assert result.exit_code == 0
    assert promotion.exists()
