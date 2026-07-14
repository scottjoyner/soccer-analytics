from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TrainingSource:
    name: str
    tier: str
    modality: str
    best_use: str
    rights_posture: str
    ingest_command: str
    notes: str = ""


def default_training_sources() -> list[TrainingSource]:
    return [
        TrainingSource(
            name="StatsBomb Open Data",
            tier="open-data",
            modality="events,lineups,360-context",
            best_use="event modeling, lineup context, selected freeze-frame context",
            rights_posture="public research/open-data use with attribution",
            ingest_command="soccer-edge ingest write-processed --source <statsbomb-root> --source-type statsbomb --output-dir data/processed/ingest",
        ),
        TrainingSource(
            name="Metrica Sports Sample Data",
            tier="open-data",
            modality="tracking,events",
            best_use="tracking/state features, pitch normalization, rolling-window examples",
            rights_posture="sample data for public analysis; acknowledge source",
            ingest_command="soccer-edge ingest write-processed --source <metrica-root> --source-type metrica --output-dir data/processed/ingest",
        ),
        TrainingSource(
            name="SoccerNet",
            tier="restricted-benchmark",
            modality="video annotations, calibration, tracking-style benchmark subsets",
            best_use="benchmark evaluation and task-specific video understanding research after access approval",
            rights_posture="respect SoccerNet access terms and task-specific restrictions",
            ingest_command="soccer-edge ingest write-processed --source <soccernet-root> --source-type soccernet --output-dir data/processed/ingest",
        ),
        TrainingSource(
            name="Owned or licensed local footage",
            tier="local-rights-approved",
            modality="video,frames,crops,annotations",
            best_use="object detection, crop review, pitch calibration, local fine-tuning",
            rights_posture="process only with owned, licensed, or compatible_license manifest rows",
            ingest_command="soccer-edge video catalog-local --root <footage-root> --output manifests/local_video_manifest.csv --rights-status owned --rights-reference <written-rights-reference>",
        ),
    ]


def training_sources_frame(sources: list[TrainingSource] | None = None) -> pd.DataFrame:
    selected = sources if sources is not None else default_training_sources()
    return pd.DataFrame([asdict(source) for source in selected])


def write_training_sources(output: Path, sources: list[TrainingSource] | None = None) -> Path:
    frame = training_sources_frame(sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, index=False)
    return output
