"""Out-of-sample evaluation of the highlight-clip CNN winner classifier.

Loads the 98 processed highlight detections and labels, splits matches into a
stratified train/test hold-out, trains the FieldStateCNN on train, and reports
held-out sequence- and match-level accuracy plus a multiclass Brier score.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from soccer_edge.evaluation.cnn_eval import (
    evaluate_cnn_out_of_sample,
    evaluate_cnn_repeated_cv,
)

REPO = Path("/home/scott/git/soccer-analytics")
DET_ROOT = REPO / "data/processed/highlights/detections"
RESULTS = REPO / "data/processed/highlights/match_results.csv"
OUT = REPO / "data/processed/highlights/training/cnn_eval"


def _limit_threads() -> None:
    """Cap PyTorch threads to avoid CPU/memory oversubscription crashes.

    With 24 logical cores the default thread pools can multiply resident memory
    (BLAS workspace per thread) and starve the box when large grids are resident.
    """
    try:
        import torch  # noqa: WPS433 (late import, optional dependency)
    except Exception:  # pragma: no cover - torch optional
        return
    n = int(os.environ.get("OMP_NUM_THREADS", min(8, os.cpu_count() or 1)))
    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, n // 2))


def load(max_matches: int | None = None):
    results = pd.read_csv(RESULTS)
    by_match: dict[str, pd.DataFrame] = {}
    for p in sorted(DET_ROOT.glob("*/*detections.parquet")):
        by_match[p.parent.name] = pd.read_parquet(p)
        if max_matches is not None and len(by_match) >= max_matches:
            break
    return results, by_match


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-matches", type=int, default=None, help="cap number of matches loaded (safety)")
    parser.add_argument("--epochs", type=int, default=5, help="CNN training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="CNN batch size")
    parser.add_argument("--output-dir", type=str, default=str(OUT), help="output directory for metrics + model")
    parser.add_argument("--seed", type=int, default=0, help="train/test split random seed")
    parser.add_argument("--folds", type=int, default=1, help="CV folds; >1 enables repeated-CV")
    parser.add_argument("--repeats", type=int, default=1, help="repeats for repeated stratified CV")
    args = parser.parse_args()

    _limit_threads()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, by_match = load(args.max_matches)
    if not by_match:
        print("No highlight detections found", file=sys.stderr)
        return 1
    print(f"loaded {len(by_match)} matches")

    if args.folds > 1:
        metrics = evaluate_cnn_repeated_cv(
            results,
            by_match,
            out_dir,
            n_splits=args.folds,
            repeats=args.repeats,
            epochs=args.epochs,
            batch_size=args.batch_size,
            random_state=args.seed,
        )
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print("\n=== Highlight-clip CNN (repeated-CV, held-out matches) ===")
        print(f"  folds / repeats    : {metrics['n_folds']} / {metrics['repeats']}")
        print(
            f"  sequence accuracy  : {metrics['sequence_accuracy_mean']:.3f} "
            f"+/- {metrics['sequence_accuracy_std']:.3f} "
            f"(base {metrics['sequence_baseline_accuracy_mean']:.3f} "
            f"+/- {metrics['sequence_baseline_accuracy_std']:.3f})"
        )
        print(
            f"  match accuracy     : {metrics['match_accuracy_mean']:.3f} "
            f"+/- {metrics['match_accuracy_std']:.3f} "
            f"(base {metrics['match_baseline_accuracy_mean']:.3f} "
            f"+/- {metrics['match_baseline_accuracy_std']:.3f})"
        )
        print(
            f"  winner Brier       : {metrics['winner_brier_mean']:.3f} "
            f"+/- {metrics['winner_brier_std']:.3f}"
        )
        print(f"wrote {out_dir / 'metrics.json'}")
        return 0

    metrics = evaluate_cnn_out_of_sample(
        results,
        by_match,
        out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== Highlight-clip CNN (out-of-sample, held-out matches) ===")
    print(f"  train/test matches : {metrics['n_train_matches']} / {metrics['n_test_matches']}")
    print(f"  train/test seqs    : {metrics['n_train_sequences']} / {metrics['n_test_sequences']}")
    print(f"  sequence accuracy  : {metrics['sequence_accuracy']:.3f} (base {metrics['sequence_baseline_accuracy']:.3f})")
    print(f"  match accuracy     : {metrics['match_accuracy']:.3f} (base {metrics['match_baseline_accuracy']:.3f})")
    print(f"  winner Brier       : {metrics['winner_brier']:.3f}")
    print(f"wrote {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
