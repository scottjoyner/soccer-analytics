# Raw and Player Data Sourcing

This project should treat raw data sourcing as a manifest-first process. The repository can catalog public/open sources and can ingest local copies after the user has verified source terms, but it should not scrape restricted sites or download unauthorized match video.

## Preferred source order

1. **StatsBomb Open Data** — best first source for named player event features. Use events, lineups, matches, and selected 360 freeze-frame files.
2. **Metrica Sports Sample Data** — useful for tracking/state features and synchronized events, but player and team names are anonymized.
3. **OpenFootball football.json** — useful for league, team, fixture, and result context, not detailed player modeling.
4. **football-data.co.uk** — useful for historical match-result baselines and CSV examples, not player-level features.
5. **SoccerNet / SoccerNet-GSR** — strong benchmark source for action spotting, calibration, and game-state reconstruction when access terms are approved.
6. **SoccerTrack v2 and simulated tracking datasets** — research candidates for full-pitch state reconstruction and pretraining after terms are verified.

## Source catalog command

```bash
soccer-edge ingest raw-sources --output data/processed/raw_data_sources.csv
```

The catalog records each source's URL, modality, player-stat depth, ingestion status, rights posture, and usage notes.

## Player event stats

Build player-match stats from processed event rows:

```bash
soccer-edge features player-stats \
  --events data/processed/ingest/events.parquet \
  --output data/processed/player_match_stats.csv
```

Build leakage-safe rolling player form features:

```bash
soccer-edge features player-form \
  --player-stats data/processed/player_match_stats.csv \
  --output data/processed/player_form.csv \
  --window 5 \
  --order-column match_id
```

## Initial player metrics

The player stats builder starts with model-safe event aggregates:

- Total events.
- Shots.
- Goals.
- Passes.
- Completed passes.
- Pass completion rate.
- Carries.
- Dribbles.
- Pressures.
- Interceptions.
- Tackles.
- Fouls committed.
- Max observed match minute.
- Rolling form features using prior matches only.

## Model use

Use player features as inputs only when they are available before the prediction timestamp. For prematch models, use prior-match rolling form. For in-play models, aggregate only events observed before the current game clock. Do not leak full-match player totals into prematch or early in-play predictions.

## Future expansion

- Add lineup-aware expected starters and bench indicators.
- Add minutes-adjusted per-90 player features.
- Add team-level roster aggregation from player form.
- Add opponent-adjusted form and competition-level normalization.
- Add graph payloads for player-season and player-match features.
