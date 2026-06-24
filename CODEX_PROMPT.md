# Codex Implementation Prompt

Use this prompt to start the implementation work in this repository.

## Prompt

You are implementing `scottjoyner/soccer-analytics`, a Python 3.12 research system for soccer probability modeling.

The goal is to build a clean, testable package that can:

1. ingest open soccer event/tracking datasets;
2. process licensed local soccer videos;
3. extract player, ball, possession, pressure, and pitch-control features;
4. train calibrated match-outcome and in-play probability models;
5. evaluate model quality offline with leakage-safe splits;
6. keep all external execution disabled in the initial implementation.

Read these docs first:

- `README.md`
- `HLD.md`
- `LLD.md`
- `TODO.md`

## Non-negotiable guardrails

- Do not add code that downloads audiovisual content from YouTube or similar platforms.
- Discovery modules may store metadata only: title, URL, channel, publish date, query, notes, and rights status.
- Video processing must only operate on local licensed files.
- Do not implement real-money execution.
- Do not store secrets in the repo.
- Do not hardcode local user paths.
- Add tests for leakage prevention and safe defaults.

## First implementation batch

Implement the project foundation:

1. `pyproject.toml` dependencies and CLI entry point.
2. `src/soccer_edge/config.py` using `pydantic-settings`.
3. `src/soccer_edge/cli.py` using Typer.
4. package `__init__.py` files.
5. storage helpers in `src/soccer_edge/store/parquet.py`.
6. placeholder ingest commands for StatsBomb, Metrica, SoccerNet, and video discovery.
7. placeholder video command for licensed local processing.
8. placeholder model/evaluation commands.
9. tests proving:
   - package imports;
   - config defaults are safe;
   - CLI loads;
   - video discovery has no download helper;
   - external execution defaults to disabled.

## Expected CLI shape

```bash
soccer-edge --help
soccer-edge ingest statsbomb --path data/raw/statsbomb
soccer-edge ingest metrica --path data/raw/metrica
soccer-edge ingest soccernet --path data/raw/soccernet
soccer-edge discover video --query "soccer goal highlights"
soccer-edge video process --input data/raw/video_licensed
soccer-edge features build
soccer-edge train prematch
soccer-edge train inplay
soccer-edge calibrate
soccer-edge evaluate
```

## Design expectations

- Use type hints throughout.
- Prefer small modules over large scripts.
- Keep functions deterministic where possible.
- Make all paths configurable.
- Use Parquet for tabular outputs.
- Use DuckDB for local analytical querying.
- Keep Neo4j optional.
- Use clear error messages.
- Do not silently skip bad data.
- Use time-based splits for model evaluation.

## Suggested dependencies

Core:

```text
pandas
polars
numpy
pyarrow
duckdb
pydantic
pydantic-settings
typer
rich
scikit-learn
joblib
```

Optional CV/modeling extras:

```text
opencv-python
ultralytics
supervision
lightgbm
xgboost
neo4j
```

Dev:

```text
pytest
ruff
mypy
```

## Implementation order

1. Make the package installable.
2. Make `soccer-edge --help` work.
3. Implement safe config defaults.
4. Add storage helper functions.
5. Add ingest command stubs.
6. Add video command stubs.
7. Add feature/model/evaluation command stubs.
8. Add tests.
9. Run `ruff`, `mypy`, and `pytest`.
10. Commit the working scaffold.

## Follow-up prompt after first implementation

After the first implementation batch, use this follow-up:

```text
Review the current soccer-analytics repo against README.md, HLD.md, LLD.md, and TODO.md. Identify missing modules, unsafe assumptions, failing tests, weak abstractions, and any place where the implementation might allow unauthorized video processing or external execution. Then patch the repo so the CLI, config, tests, and storage helpers are clean and ready for the first real data loader.
```

## Second implementation batch

After the foundation is stable, implement:

1. StatsBomb Open Data loader.
2. Metrica sample loader.
3. common coordinate normalization utilities.
4. schema validation.
5. synthetic sample tests.
6. baseline feature table builder.
7. simple logistic baseline model.
8. calibration report writer.

## Definition of done for bootstrap

Bootstrap is complete when:

- `pip install -e .[dev]` works;
- `soccer-edge --help` works;
- tests pass;
- safe defaults are enforced;
- docs are aligned with the package structure;
- no module can download audiovisual content;
- no module can execute external transactions.