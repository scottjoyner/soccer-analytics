import pytest

from soccer_edge.models.run_metadata import RunMetadata
from soccer_edge.schemas import FeatureRecord, FrameRecord, MatchRecord
from soccer_edge.store.graph_export import (
    feature_payload,
    frame_payload,
    match_payload,
    player_form_payload,
    player_match_payload,
    run_payload,
)


def test_match_payload() -> None:
    row = MatchRecord("m1", "cup", "2026", "2026-06-11", "A", "B")
    payload = match_payload(row)
    assert "MERGE" in payload["statement"]
    assert payload["props"]["match_id"] == "m1"


def test_frame_and_feature_payloads() -> None:
    frame = FrameRecord("video", "m1", 1, 1.0)
    feature = FeatureRecord("m1", 1.0, "x", 0.5)
    assert frame_payload(frame)["props"]["frame_idx"] == 1
    assert feature_payload(feature)["props"]["feature_name"] == "x"


def test_run_payload() -> None:
    row = RunMetadata("demo", "v1", "now", ["x"], {"accuracy": 1.0})
    assert run_payload(row)["props"]["name"] == "demo"


def test_payload_requires_key() -> None:
    with pytest.raises(ValueError):
        frame_payload({"x": 1})


def test_player_match_payload() -> None:
    row = {"match_id": "m1", "player_name": "A", "team_name": "T", "goals": 1}
    payload = player_match_payload(row)
    assert "PlayerMatch" in payload["statement"]
    assert payload["props"]["player_match_id"] == "m1::A"
    assert payload["props"]["goals"] == 1


def test_player_form_payload() -> None:
    row = {"player_name": "A", "match_id": "m1", "goals_form_5": 0.4}
    payload = player_form_payload(row)
    assert "PlayerForm" in payload["statement"]
    assert payload["props"]["player_form_id"] == "A::m1"
    assert payload["props"]["goals_form_5"] == 0.4
