import numpy as np

from soccer_edge.features.expected_threat import (
    N_CELLS,
    loc_to_cell,
    mirror_location,
    solve_xt,
    team_xt,
)


def test_loc_to_cell_bounds() -> None:
    assert loc_to_cell(0, 0) == 0
    assert loc_to_cell(120, 80) == N_CELLS - 1
    assert loc_to_cell(-5, -5) == 0
    assert loc_to_cell(999, 999) == N_CELLS - 1


def test_mirror_location() -> None:
    x, y = mirror_location(10.0, 20.0)
    assert x == 110.0 and y == 60.0


def test_solve_xt_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    t = rng.integers(0, 10, size=(N_CELLS, N_CELLS)).astype(float)
    goals = rng.integers(0, 3, size=N_CELLS).astype(float)
    totals = rng.integers(0, 6, size=N_CELLS).astype(float)
    xt = solve_xt(t, goals, totals)
    assert xt.shape == (N_CELLS,)
    assert np.all(np.isfinite(xt))
    assert float(xt.min()) >= 0.0
    assert float(xt.max()) <= 1.0


def test_solve_xt_goal_cell_higher() -> None:
    # Two cells: cell 0 leads to a goal (G=0.5), cell 1 is an isolated dead end.
    t = np.zeros((2, 2))
    t[0, 0] = 10.0
    t[1, 1] = 10.0
    goals = np.array([5.0, 0.0])
    totals = np.array([10.0, 0.0])
    xt = solve_xt(t, goals, totals, smoothing=0.0)
    assert xt[0] > xt[1]


def test_team_xt_sums_transitions() -> None:
    xt = np.array([0.0, 0.1, 0.4, 0.9])
    transitions = [(0, 1), (1, 2), (2, 3)]
    assert team_xt(xt, transitions) == 0.1 + 0.3 + 0.5
    assert team_xt(xt, [(-1, 2)]) == 0.0
