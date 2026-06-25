"""Command-line interface for soccer analytics research workflows."""

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from soccer_edge.app_logging import configure_logging, get_logger
from soccer_edge.config import get_settings
from soccer_edge.evaluation.calibration_review import write_calibration_review
from soccer_edge.evaluation.replay import replay_predictions
from soccer_edge.features.table_builders import build_inplay_rolling_table, build_prematch_table
from soccer_edge.ingest.metrica_loader import ingest_metrica as run_metrica_ingest
from soccer_edge.ingest.soccernet_loader import ingest_soccernet as run_soccernet_ingest
from soccer_edge.ingest.statsbomb_loader import ingest_statsbomb as run_statsbomb_ingest
from soccer_edge.ingest.video_discovery import build_candidate
from soccer_edge.models.bundle import save_bundle
from soccer_edge.video.batch_runner import build_processing_plan
from soccer_edge.video.state_tables import write_video_state_tables

app = typer.Typer(help="Soccer analytics research CLI.")
ingest_app = typer.Typer(help="Ingest open soccer datasets.")
discover_app = typer.Typer(help="Discover candidate video metadata.")
video_app = typer.Typer(help="Process licensed local soccer videos.")
features_app = typer.Typer(help="Build model feature tables.")
train_app = typer.Typer(help="Train probability models.")
model_app = typer.Typer(help="Save, score, and inspect model outputs.")

app.add_typer(ingest_app, name="ingest")
app.add_typer(discover_app, name="discover")
app.add_typer(video_app, name="video")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(model_app, name="model")

console = Console()
logger = get_logger("soccer_edge.cli")


@app.callback()
def main(json_logs: bool = typer.Option(False, help="Emit structured JSON logs.")) -> None:
    """Configure global CLI options."""

    configure_logging(json_logs=json_logs)
    logger.debug("cli configured")


@app.command()
def doctor() -> None:
    """Print environment defaults."""

    settings = get_settings()
    console.print("Soccer analytics environment", style="bold")
    console.print(f"env={settings.env}")
    console.print(f"data_dir={settings.data_dir}")
    console.print(f"external_execution_enabled={settings.external_execution_enabled}")


@ingest_app.command("statsbomb")
def ingest_statsbomb(path: Path = typer.Option(..., exists=False)) -> None:
    """Prepare StatsBomb ingest."""

    console.print(run_statsbomb_ingest(path))


@ingest_app.command("metrica")
def ingest_metrica(path: Path = typer.Option(..., exists=False)) -> None:
    """Prepare Metrica ingest."""

    console.print(run_metrica_ingest(path))


@ingest_app.command("soccernet")
def ingest_soccernet(path: Path = typer.Option(..., exists=False)) -> None:
    """Prepare SoccerNet ingest."""

    console.print(run_soccernet_ingest(path))


@discover_app.command("video")
def discover_video(
    query: str = typer.Option(...),
    url: str = typer.Option("https://example.com/manual-review"),
    title: str = typer.Option("Manual review candidate"),
) -> None:
    """Store candidate video metadata only."""

    candidate = build_candidate(url=url, title=title, query=query)
    console.print(candidate)


@video_app.command("plan")
def plan_video_processing(
    manifest: Path = typer.Option(..., exists=True),
    licensed_root: Path = typer.Option(Path("data/raw/video_licensed")),
) -> None:
    """Plan processable local video rows from a manifest."""

    plan = build_processing_plan(manifest, licensed_root)
    console.print(f"processable={len(plan.processable)} skipped={len(plan.skipped)}")


@video_app.command("process")
def process_video(input_path: Path = typer.Option(..., "--input", exists=False)) -> None:
    """Process local licensed video files."""

    console.print(f"Licensed video processing placeholder: {input_path}")


@features_app.command("build")
def build_features(output_dir: Path = typer.Option(Path("data/processed/state_tables"))) -> None:
    """Create empty state-table files as the first feature-build target."""

    paths = write_video_state_tables(output_dir)
    console.print({name: str(path) for name, path in paths.items()})


@features_app.command("prematch")
def build_prematch_features(
    matches: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/prematch_features.parquet")),
) -> None:
    """Build a prematch feature table from match rows."""

    frame = pd.read_parquet(matches) if matches.suffix == ".parquet" else pd.read_csv(matches)
    table = build_prematch_table(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    console.print(f"wrote={output} rows={len(table)}")


@features_app.command("inplay")
def build_inplay_features(
    source: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/inplay_features.parquet")),
    columns: str = typer.Option(..., help="Comma-separated feature columns."),
    window_seconds: float = typer.Option(60.0),
) -> None:
    """Build rolling in-play feature rows."""

    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    feature_columns = [column.strip() for column in columns.split(",") if column.strip()]
    table = build_inplay_rolling_table(frame, feature_columns=feature_columns, window_seconds=window_seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    console.print(f"wrote={output} rows={len(table)}")


@train_app.command("prematch")
def train_prematch() -> None:
    """Train prematch model placeholder."""

    console.print("Prematch training placeholder")


@train_app.command("inplay")
def train_inplay() -> None:
    """Train in-play model placeholder."""

    console.print("In-play training placeholder")


@model_app.command("save-demo")
def save_demo_model(output_dir: Path = typer.Option(Path("data/processed/model_demo"))) -> None:
    """Save a small demo model bundle with metadata."""

    paths = save_bundle(
        model={"kind": "demo"},
        output_dir=output_dir,
        name="demo",
        version="v0",
        feature_names=["demo_feature"],
        metrics={"accuracy": 0.0},
        notes="Demo bundle for pipeline validation.",
    )
    console.print({name: str(path) for name, path in paths.items()})


@model_app.command("evaluate")
def evaluate_model(predictions: Path = typer.Option(..., exists=True)) -> None:
    """Evaluate a CSV or parquet file with label/probability columns."""

    frame = pd.read_parquet(predictions) if predictions.suffix == ".parquet" else pd.read_csv(predictions)
    result = replay_predictions(frame)
    console.print(result)


@model_app.command("calibration-review")
def calibration_review(
    predictions: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/calibration_review")),
    num_bins: int = typer.Option(10),
) -> None:
    """Write metrics and calibration artifacts from probability rows."""

    frame = pd.read_parquet(predictions) if predictions.suffix == ".parquet" else pd.read_csv(predictions)
    paths = write_calibration_review(frame, output_dir=output_dir, num_bins=num_bins)
    console.print({name: str(path) for name, path in paths.items()})


@app.command()
def calibrate() -> None:
    """Calibrate model probabilities."""

    console.print("Use: soccer-edge model calibration-review --predictions <csv-or-parquet>")


@app.command()
def evaluate() -> None:
    """Evaluate models offline."""

    console.print("Use: soccer-edge model evaluate --predictions <csv-or-parquet>")
