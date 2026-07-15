"""Out-of-sample evaluation of a match-outcome model on StatsBomb Open Data.

Builds home/away event-derived features (shots, xG, passes, pressures, ...) from
local StatsBomb event JSON and runs a repeated stratified cross-validated,
calibrated winner classifier + score regressors. This demonstrates that the
same pipeline yields real predictive lift when fed open event data (the signal
source the highlight-clip features lack).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


from soccer_edge.evaluation.match_eval import repeated_cv_match_predictor
from soccer_edge.features.statsbomb_features import (
    build_match_event_features,
    build_match_event_features_fold,
    default_event_features,
)

CANDIDATE_DIRS = [
    Path("/tmp/opencode/sb"),
    Path("/tmp/opencode/statsbomb-open-data"),
    Path("examples/statsbomb"),
]
OUT = Path("data/processed/event_eval")
OUT.mkdir(parents=True, exist_ok=True)

# Basic feature set (no xT / pressure regains) for ablation.
BASIC_FEATURES = [
    col
    for col in default_event_features()
    if not col.endswith("_xt") and not col.endswith("_pressure_regains")
]


def _find_source() -> Path | None:
    for p in CANDIDATE_DIRS:
        if (p / "competitions.json").exists() and list(p.glob("events/*.json")):
            return p
    return None


def _report(label: str, m: dict) -> None:
    print(f"\n=== {label} (5-fold x10 repeats, n={m['n_matches']}) ===")
    print(f"  winner accuracy : {m['winner_accuracy_mean']:.3f} +/- {m['winner_accuracy_std']:.3f}")
    print(f"  majority baseline: {m['majority_baseline_accuracy_mean']:.3f} +/- {m['majority_baseline_accuracy_std']:.3f}")
    print(f"  winner Brier     : {m['winner_brier_mean']:.3f} +/- {m['winner_brier_std']:.3f}")
    print(f"  home score MSE   : {m['home_score_mse_mean']:.3f} +/- {m['home_score_mse_std']:.3f}")
    print(f"  away score MSE   : {m['away_score_mse_mean']:.3f} +/- {m['away_score_mse_std']:.3f}")


def main() -> int:
    source = _find_source()
    if source is None:
        print("No StatsBomb event data found in candidate dirs", file=sys.stderr)
        return 1

    print(f"source: {source}")
    frame = build_match_event_features(source)
    frame.to_csv(OUT / "event_features.csv", index=False)

    richer = default_event_features()
    print(f"matches: {len(frame)}  basic features: {len(BASIC_FEATURES)}  richer: {len(richer)}")

    def _per_fold_xt(train_idx, test_idx, frm):
        train_ids = frm.iloc[train_idx]["match_id"].astype(str).tolist()
        return build_match_event_features_fold(source, train_ids)

    basic_metrics = repeated_cv_match_predictor(frame, BASIC_FEATURES, n_splits=5, n_repeats=10)
    richer_metrics = repeated_cv_match_predictor(
        frame, richer, n_splits=5, n_repeats=10, feature_refit_fn=_per_fold_xt
    )
    metrics = {"basic": basic_metrics, "richer": richer_metrics}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _report("StatsBomb basic event features (xG + counts)", basic_metrics)
    _report("StatsBomb richer event features (+xT, pressure regains)", richer_metrics)
    print(f"wrote {OUT / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
