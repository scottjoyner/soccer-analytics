"""Command-line interface for soccer analytics research workflows."""

from pathlib import Path

import typer
from rich.console import Console

from soccer_edge.config import get_settings
from soccer_edge.ingest.metrica_loader import ingest_metrica as run_metrica_ingest
from soccer_edge.ingest.soccernet_loader import ingest_soccernet as run_soccernet_ingest
from soccer_edge.ingest.statsbomb_loader import ingest_statsbomb as run_statsbomb_ingest
from soccer_edge.ingest.video_discovery import build_candidate

app = typer.Typer(help="Soccer analytics research CLI.")
ingest_app = typer.Typer(help="Ingest open soccer datasets.")
discover_app = typer.Typer(help="Discover candidate video metadata.")
video_app = typer.Typer(help="Process licensed local soccer videos.")
features_app = typer.Typer(help="Build model feature tables.")
train_app = typer.Typer(help="Train probability models.")

app.add_typer(ingest_app, name="ingest")
app.add_typer(discover_app, name="discover")
app.add_typer(video_app, name="video")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")

console = Console()


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


@video_app.command("process")
def process_video(input_path: Path = typer.Option(..., "--input", exists=False)) -> None:
    """Process local licensed video files."""

    console.print(f"Licensed video processing placeholder: {input_path}")


@features_app.command("build")
def build_features() -> None:
    """Build feature tables."""

    console.print("Feature build placeholder")


@train_app.command("prematch")
def train_prematch() -> None:
    """Train prematch model placeholder."""

    console.print("Prematch training placeholder")


@train_app.command("inplay")
def train_inplay() -> None:
    """Train in-play model placeholder."""

    console.print("In-play training placeholder")


@app.command()
def calibrate() -> None:
    """Calibrate model probabilities."""

    console.print("Calibration placeholder")


@app.command()
def evaluate() -> None:
    """Evaluate models offline."""

    console.print("Offline evaluation placeholder")
