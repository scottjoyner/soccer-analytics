"""Data-driven expected threat (xT) for StatsBomb event data.

Implements the canonical xT surface (Karun Singh, 2018): a Markov fixed point
over a possession-transition matrix where the reward is the empirical
probability of scoring (a goal) from each pitch cell. The grid is fit from the
event data itself, so it stays rights-clean and reproducible.

StatsBomb stores both teams' event coordinates in the same absolute frame, so a
team attacking the left-hand goal has actions heading toward x=0. To build a
single xT surface we mirror the coordinates of any team whose shots are, on
average, closer to x=0 than x=120, so every attack points toward x=120.
"""
from __future__ import annotations

import numpy as np

COLS = 12
ROWS = 8
N_CELLS = COLS * ROWS
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


def loc_to_cell(x: float, y: float) -> int:
    """Map a StatsBomb pitch location to a flat cell index."""
    if x is None or y is None:
        return -1
    col = int(min(COLS - 1, max(0, x / PITCH_LENGTH * COLS)))
    row = int(min(ROWS - 1, max(0, y / PITCH_WIDTH * ROWS)))
    return row * COLS + col


def cell_coords(index: int) -> tuple[int, int]:
    return divmod(index, COLS)


def mirror_location(x: float, y: float) -> tuple[float, float]:
    """Mirror a location so the attack points toward x=120."""
    return PITCH_LENGTH - x, PITCH_WIDTH - y


def solve_xt(
    transition_counts: np.ndarray,
    shot_goals: np.ndarray,
    shot_totals: np.ndarray,
    iterations: int = 50,
    smoothing: float = 1e-3,
) -> np.ndarray:
    """Solve the xT fixed point from transition counts and shot/goal counts.

    xT = P @ (G + (1 - G) * xT), iterated to convergence.
    """
    counts = transition_counts.astype(float).copy()
    counts += smoothing
    p = counts / counts.sum(axis=1, keepdims=True)
    n = counts.shape[0]
    g = np.zeros(n)
    nonzero = shot_totals > 0
    g[nonzero] = shot_goals[nonzero] / shot_totals[nonzero]

    xt = g.copy()
    for _ in range(iterations):
        xt = p @ (g + (1.0 - g) * xt)
    return xt


def team_xt(xt: np.ndarray, transitions: list[tuple[int, int]]) -> float:
    """Sum the threat gained over a team's possession actions."""
    total = 0.0
    for start, end in transitions:
        if start < 0 or end < 0:
            continue
        total += float(xt[end] - xt[start])
    return total
