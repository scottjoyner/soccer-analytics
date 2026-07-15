"""Derive match-outcome features from StatsBomb Open Data event JSON.

This module reads the canonical StatsBomb directory layout
(competitions.json, matches/<comp>/<season>.json, events/<match_id>.json) and
produces one row per match with home/away event-derived features plus the
match labels (home_score, away_score, winner) taken from the match metadata.

These open-event features are the supervision signal the highlight-clip CV
features lack: possession, pressure, shots, and expected goals all correlate
with final score and outcome.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EVENT_TYPES = [
    "Pass",
    "Pressure",
    "Carry",
    "Foul Committed",
    "Ball Recovery",
    "Interception",
    "Duel",
    "Clearance",
    "Block",
    "Miscontrol",
    "Dispossessed",
    "Shot",
]

SHOT_ON_TARGET = {"Goal", "Saved"}


def _read_json(path: Path) -> object:
    import json

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_match_metadata(source_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(source_dir.glob("matches/*/*.json")):
        for m in _read_json(path):
            if not isinstance(m, dict) or "match_id" not in m:
                continue
            out[str(m["match_id"])] = m
    return out


def _shot_xg(event: dict) -> float:
    shot = event.get("shot")
    if not isinstance(shot, dict):
        return 0.0
    try:
        return float(shot.get("statsbomb_xg") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _shot_on_target(event: dict) -> bool:
    shot = event.get("shot")
    if not isinstance(shot, dict):
        return False
    outcome = shot.get("outcome") or {}
    name = (outcome.get("name") or "") if isinstance(outcome, dict) else ""
    return name in SHOT_ON_TARGET


def _empty_counts() -> dict[str, int]:
    return {f"n_{t.lower().replace(' ', '_')}": 0 for t in EVENT_TYPES}


def build_match_event_features(source_dir: Path, competition_ids: list[int] | None = None) -> pd.DataFrame:
    """Return one row per match with home/away event features and labels."""
    source_dir = Path(source_dir)
    metadata = _load_match_metadata(source_dir)
    rows: list[dict[str, object]] = []

    for path in sorted(source_dir.glob("events/*.json")):
        match_id = path.stem
        meta = metadata.get(match_id)
        if meta is None:
            continue
        if competition_ids is not None and meta.get("competition", {}).get("competition_id") not in competition_ids:
            continue
        home_name = (meta.get("home_team") or {}).get("home_team_name")
        away_name = (meta.get("away_team") or {}).get("away_team_name")
        home_score = meta.get("home_score")
        away_score = meta.get("away_score")
        if home_score is None or away_score is None or home_name is None or away_name is None:
            continue

        home = _empty_counts()
        away = _empty_counts()
        home_xg = 0.0
        away_xg = 0.0
        home_on_target = 0
        away_on_target = 0

        for event in _read_json(path):
            if not isinstance(event, dict):
                continue
            team_name = (event.get("team") or {}).get("name")
            if team_name not in (home_name, away_name):
                continue
            type_name = (event.get("type") or {}).get("name")
            if type_name not in EVENT_TYPES:
                continue
            bucket = home if team_name == home_name else away
            bucket[f"n_{type_name.lower().replace(' ', '_')}"] += 1
            if type_name == "Shot":
                if team_name == home_name:
                    home_xg += _shot_xg(event)
                    home_on_target += int(_shot_on_target(event))
                else:
                    away_xg += _shot_xg(event)
                    away_on_target += int(_shot_on_target(event))

        row: dict[str, object] = {
            "match_id": match_id,
            "competition_id": meta.get("competition", {}).get("competition_id"),
            "season_id": meta.get("season", {}).get("season_id"),
            "home_team": home_name,
            "away_team": away_name,
            "home_score": int(home_score),
            "away_score": int(away_score),
            "home_xg": round(home_xg, 4),
            "away_xg": round(away_xg, 4),
            "home_shots_on_target": home_on_target,
            "away_shots_on_target": away_on_target,
            "winner": 0 if home_score > away_score else (2 if away_score > home_score else 1),
        }
        for key, val in home.items():
            row[f"home_{key}"] = val
        for key, val in away.items():
            row[f"away_{key}"] = val
        rows.append(row)

    return pd.DataFrame(rows)


def default_event_features() -> list[str]:
    """Home/away event-feature columns used for modeling (excludes labels/ids)."""
    cols: list[str] = []
    for side in ("home", "away"):
        cols.extend(
            [
                f"{side}_xg",
                f"{side}_shots_on_target",
                f"{side}_n_shot",
                f"{side}_n_pass",
                f"{side}_n_pressure",
                f"{side}_n_carry",
                f"{side}_n_foul_committed",
                f"{side}_n_ball_recovery",
                f"{side}_n_interception",
                f"{side}_n_clearance",
                f"{side}_n_block",
            ]
        )
    return cols
