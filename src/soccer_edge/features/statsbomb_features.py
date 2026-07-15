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

import numpy as np
import pandas as pd

from soccer_edge.features.expected_threat import (
    N_CELLS,
    loc_to_cell,
    mirror_location,
    solve_xt,
    team_xt,
)

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

# On-ball actions that advance possession and therefore accumulate xT.
XPTRANSITION_TYPES = ["Pass", "Carry", "Dribble", "Shot"]

# A pressure counts as "regained" if the same team wins the ball within this
# many subsequent events.
PRESSURE_REGAIN_WINDOW = 3

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


def _action_end_location(event: dict, type_name: str) -> list[float] | None:
    """Return the destination [x, y] for an xT-transition action, if known."""
    nested = {
        "Pass": event.get("pass"),
        "Carry": event.get("carry"),
        "Dribble": event.get("dribble"),
        "Shot": event.get("shot"),
    }.get(type_name)
    if isinstance(nested, dict):
        end = nested.get("end_location")
        if isinstance(end, (list, tuple)) and len(end) >= 2:
            return [float(end[0]), float(end[1])]
    return None


def _empty_counts() -> dict[str, int]:
    return {f"n_{t.lower().replace(' ', '_')}": 0 for t in EVENT_TYPES}


def build_match_event_features(
    source_dir: Path,
    competition_ids: list[int] | None = None,
    xt_fit_matches: list[str] | None = None,
) -> pd.DataFrame:
    """Return one row per match with home/away event features and labels.

    Includes count features, expected goals, expected threat (xT), and pressure
    regains. The xT surface is fit from the event data itself. By default it is
    fit league-wide; pass ``xt_fit_matches`` to fit it only on a subset of match
    ids (e.g. a cross-validation training fold) so test matches are scored with a
    surface that never saw them.
    """
    source_dir = Path(source_dir)
    metadata = _load_match_metadata(source_dir)

    trans_counts = np.zeros((N_CELLS, N_CELLS), dtype=float)
    shot_goals = np.zeros(N_CELLS, dtype=float)
    shot_totals = np.zeros(N_CELLS, dtype=float)
    match_transitions: dict[str, dict[str, list[tuple[int, int]]]] = {}
    match_pressure_regains: dict[str, dict[str, int]] = {}
    match_shot_cells: dict[str, dict[int, list[int]]] = {}
    rows_meta: list[dict[str, object]] = []

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

        events = _read_json(path)
        if not isinstance(events, list):
            continue

        # Pass 1: decide per-team orientation from shot locations.
        shot_x_by_team: dict[str, list[float]] = {home_name: [], away_name: []}
        for event in events:
            if not isinstance(event, dict):
                continue
            if (event.get("type") or {}).get("name") != "Shot":
                continue
            team_name = (event.get("team") or {}).get("name")
            loc = event.get("location")
            if team_name in shot_x_by_team and isinstance(loc, (list, tuple)) and len(loc) >= 1:
                shot_x_by_team[team_name].append(float(loc[0]))
        mirror_team = {
            home_name: np.mean(shot_x_by_team[home_name]) < 60 if shot_x_by_team[home_name] else False,
            away_name: np.mean(shot_x_by_team[away_name]) < 60 if shot_x_by_team[away_name] else False,
        }

        home = _empty_counts()
        away = _empty_counts()
        home_xg = away_xg = 0.0
        home_on_target = away_on_target = 0
        transitions: dict[str, list[tuple[int, int]]] = {home_name: [], away_name: []}
        pressure_regains = {home_name: 0, away_name: 0}
        match_shot_cells[match_id] = {}
        last_pressure_idx: dict[str, int] = {home_name: -1, away_name: -1}
        counted_pressures: set[tuple[str, int]] = set()

        for idx, event in enumerate(events):
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

            loc = event.get("location")
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                sx, sy = float(loc[0]), float(loc[1])
            else:
                sx = sy = None

            if type_name == "Shot":
                is_goal = (
                    _shot_on_target(event)
                    and (event.get("shot") or {}).get("outcome", {}).get("name") == "Goal"
                )
                if team_name == home_name:
                    home_xg += _shot_xg(event)
                    home_on_target += int(_shot_on_target(event))
                else:
                    away_xg += _shot_xg(event)
                    away_on_target += int(_shot_on_target(event))
                if sx is not None:
                    cell = loc_to_cell(*(mirror_location(sx, sy) if mirror_team[team_name] else (sx, sy)))
                    if cell >= 0:
                        shot_totals[cell] += 1
                        if is_goal:
                            shot_goals[cell] += 1
                        bucket = match_shot_cells[match_id].setdefault(cell, [0, 0])
                        bucket[0] += int(is_goal)
                        bucket[1] += 1

            if type_name in XPTRANSITION_TYPES and sx is not None:
                start = loc_to_cell(*(mirror_location(sx, sy) if mirror_team[team_name] else (sx, sy)))
                end_loc = _action_end_location(event, type_name)
                if end_loc is not None:
                    ex, ey = end_loc
                    if mirror_team[team_name]:
                        ex, ey = mirror_location(ex, ey)
                    end = loc_to_cell(ex, ey)
                else:
                    end = start
                if start >= 0:
                    transitions[team_name].append((start, end))
                    trans_counts[start, end] += 1

            if type_name == "Pressure":
                last_pressure_idx[team_name] = idx
            elif type_name == "Ball Recovery":
                lp = last_pressure_idx[team_name]
                if lp >= 0 and (idx - lp) <= PRESSURE_REGAIN_WINDOW and (team_name, lp) not in counted_pressures:
                    pressure_regains[team_name] += 1
                    counted_pressures.add((team_name, lp))

        match_transitions[match_id] = transitions
        match_pressure_regains[match_id] = pressure_regains
        rows_meta.append(
            {
                "match_id": match_id,
                "meta": meta,
                "home_name": home_name,
                "away_name": away_name,
                "home": home,
                "away": away,
                "home_xg": round(home_xg, 4),
                "away_xg": round(away_xg, 4),
                "home_on_target": home_on_target,
                "away_on_target": away_on_target,
            }
        )

    xt = solve_xt(trans_counts, shot_goals, shot_totals) if xt_fit_matches is None else fit_xt_surface(
        match_transitions, match_shot_cells, xt_fit_matches
    )

    rows: list[dict[str, object]] = []
    for r in rows_meta:
        meta = r["meta"]
        home_score = int(meta["home_score"])
        away_score = int(meta["away_score"])
        home_name = r["home_name"]
        away_name = r["away_name"]
        row: dict[str, object] = {
            "match_id": r["match_id"],
            "competition_id": meta.get("competition", {}).get("competition_id"),
            "season_id": meta.get("season", {}).get("season_id"),
            "home_team": home_name,
            "away_team": away_name,
            "home_score": home_score,
            "away_score": away_score,
            "home_xg": r["home_xg"],
            "away_xg": r["away_xg"],
            "home_shots_on_target": r["home_on_target"],
            "away_shots_on_target": r["away_on_target"],
            "home_xt": round(team_xt(xt, match_transitions[r["match_id"]][home_name]), 4),
            "away_xt": round(team_xt(xt, match_transitions[r["match_id"]][away_name]), 4),
            "home_pressure_regains": match_pressure_regains[r["match_id"]][home_name],
            "away_pressure_regains": match_pressure_regains[r["match_id"]][away_name],
            "winner": 0 if home_score > away_score else (2 if away_score > home_score else 1),
        }
        for key, val in r["home"].items():
            row[f"home_{key}"] = val
        for key, val in r["away"].items():
            row[f"away_{key}"] = val
        rows.append(row)

    return pd.DataFrame(rows)


def fit_xt_surface(
    match_transitions: dict[str, dict[str, list[tuple[int, int]]]],
    match_shot_cells: dict[str, dict[int, list[int]]],
    match_ids: list[str] | None = None,
) -> np.ndarray:
    """Fit an xT surface from a subset of matches' transitions and shot cells.

    Pass ``match_ids`` to fit only on a cross-validation training fold so the
    surface never sees the test matches it is later applied to.
    """
    trans_counts = np.zeros((N_CELLS, N_CELLS), dtype=float)
    shot_goals = np.zeros(N_CELLS, dtype=float)
    shot_totals = np.zeros(N_CELLS, dtype=float)
    ids = match_ids if match_ids is not None else list(match_transitions.keys())
    for mid in ids:
        for team in match_transitions.get(mid, {}).values():
            for start, end in team:
                if 0 <= start < N_CELLS and 0 <= end < N_CELLS:
                    trans_counts[start, end] += 1
        for cell, (goals, totals) in match_shot_cells.get(mid, {}).items():
            shot_goals[cell] += goals
            shot_totals[cell] += totals
    return solve_xt(trans_counts, shot_goals, shot_totals)


def build_match_event_features_fold(
    source_dir: Path,
    train_match_ids: list[str],
    competition_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Build match features with the xT surface fit only on ``train_match_ids``.

    Every returned row (including held-out test matches) uses this training-fold
    surface, eliminating the negligible league-wide xT leakage.
    """
    return build_match_event_features(source_dir, competition_ids=competition_ids, xt_fit_matches=train_match_ids)


def default_event_features() -> list[str]:
    """Home/away event-feature columns used for modeling (excludes labels/ids)."""
    cols: list[str] = []
    for side in ("home", "away"):
        cols.extend(
            [
                f"{side}_xg",
                f"{side}_xt",
                f"{side}_shots_on_target",
                f"{side}_pressure_regains",
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
