from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RawDataSource:
    name: str
    url: str
    modality: str
    player_stats_level: str
    ingestion_status: str
    rights_posture: str
    notes: str


def default_raw_data_sources() -> list[RawDataSource]:
    return [
        RawDataSource(
            name="StatsBomb Open Data",
            url="https://github.com/statsbomb/open-data",
            modality="events,lineups,selected-360",
            player_stats_level="player event aggregates, shots, passes, goals, pressures",
            ingestion_status="implemented: statsbomb loader and player event aggregation",
            rights_posture="free public research use with StatsBomb attribution requirements",
            notes="Best first source for real named player features from open event data.",
        ),
        RawDataSource(
            name="Metrica Sports Sample Data",
            url="https://github.com/metrica-sports/sample-data",
            modality="tracking,events",
            player_stats_level="anonymized player tracking/event aggregates",
            ingestion_status="implemented: metrica loader; player names anonymized",
            rights_posture="sample public analysis data; acknowledge source for public use",
            notes="Good for tracking/state features, not named player scouting features.",
        ),
        RawDataSource(
            name="OpenFootball football.json",
            url="https://github.com/openfootball/football.json",
            modality="fixtures,results,teams,leagues",
            player_stats_level="team and match context only",
            ingestion_status="cataloged only",
            rights_posture="public-domain-style open football JSON; verify repository terms before redistribution",
            notes="Useful for schedule/result context and league normalization.",
        ),
        RawDataSource(
            name="football-data.co.uk",
            url="https://www.football-data.co.uk/",
            modality="match results, odds-style CSVs, league tables",
            player_stats_level="team/match context only",
            ingestion_status="cataloged only",
            rights_posture="free football results/odds CSVs; respect site terms and attribution expectations",
            notes="Useful for historical match-result baselines, not player-level modeling.",
        ),
        RawDataSource(
            name="SoccerNet / SoccerNet-GSR",
            url="https://www.soccer-net.org/",
            modality="benchmark video annotations, action spotting, calibration, game-state reconstruction",
            player_stats_level="role/team/jersey-position labels for approved subsets",
            ingestion_status="implemented: generic local soccernet loader; access terms required",
            rights_posture="restricted benchmark access; follow SoccerNet task-specific terms",
            notes="Use only after approved access; do not treat public URLs as media inputs.",
        ),
        RawDataSource(
            name="SoccerTrack v2",
            url="https://arxiv.org/abs/2508.01802",
            modality="multi-view full-pitch video, pitch positions, jersey identities, ball actions",
            player_stats_level="jersey/player tracking and action labels in research dataset",
            ingestion_status="candidate research source; verify release files and terms before ingestion",
            rights_posture="public research dataset claim in paper; verify dataset license before processing",
            notes="Potentially strong source for full-pitch state reconstruction and player movement features.",
        ),
        RawDataSource(
            name="Google Research Football simulated tracking",
            url="https://arxiv.org/abs/2503.19809",
            modality="simulated tracking/events",
            player_stats_level="synthetic agent/player movement and state features",
            ingestion_status="candidate simulation source",
            rights_posture="research/simulation workflow; verify repository/license before storing generated data",
            notes="Useful for pretraining and stress-testing model architecture before real tracking data.",
        ),
    ]


def raw_data_sources_frame(sources: list[RawDataSource] | None = None) -> pd.DataFrame:
    selected = sources if sources is not None else default_raw_data_sources()
    return pd.DataFrame([asdict(source) for source in selected])


def write_raw_data_sources(output: Path, sources: list[RawDataSource] | None = None) -> Path:
    frame = raw_data_sources_frame(sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, index=False)
    return output
