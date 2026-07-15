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
    default_event_features,
)

CANDIDATE_DIRS = [
    Path("/tmp/opencode/sb"),
    Path("/tmp/opencode/statsbomb-open-data"),
    Path("examples/statsbomb"),
]
OUT = Path("data/processed/event_eval")
OUT.mkdir(parents=True, exist_ok=True)


def _find_source() -> Path | None:
    for p in CANDIDATE_DIRS:
        if (p / "competitions.json").exists() and list(p.glob("events/*.json")):
            return p
    return None


def main() -> int:
    source = _find_source()
    if source is None:
        print("No StatsBomb event data found in candidate dirs", file=sys.stderr)
        return 1

    print(f"source: {source}")
    frame = build_match_event_features(source)
    features = default_event_features()
    frame.to_csv(OUT / "event_features.csv", index=False)
    print(f"matches: {len(frame)}  features: {len(features)}")

    metrics = repeated_cv_match_predictor(frame, features, n_splits=5, n_repeats=10)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== StatsBomb event-feature model (5-fold x10 repeats) ===")
    print(f"  winner accuracy : {metrics['winner_accuracy_mean']:.3f} +/- {metrics['winner_accuracy_std']:.3f}")
    print(f"  majority baseline: {metrics['majority_baseline_accuracy_mean']:.3f} +/- {metrics['majority_baseline_accuracy_std']:.3f}")
    print(f"  winner Brier     : {metrics['winner_brier_mean']:.3f} +/- {metrics['winner_brier_std']:.3f}")
    print(f"  home score MSE   : {metrics['home_score_mse_mean']:.3f} +/- {metrics['home_score_mse_std']:.3f}")
    print(f"  away score MSE   : {metrics['away_score_mse_mean']:.3f} +/- {metrics['away_score_mse_std']:.3f}")
    print(f"wrote {OUT / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
