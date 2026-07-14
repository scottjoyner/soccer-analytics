from pathlib import Path

import pytest

from soccer_edge.local_finetune_plan import local_finetune_shell_plan, shell_quote, validate_local_finetune_inputs, write_local_finetune_shell_plan


def test_shell_quote() -> None:
    assert shell_quote("abc/def") == "abc/def"
    assert shell_quote("a b") == "'a b'"


def test_local_finetune_shell_plan() -> None:
    plan = local_finetune_shell_plan(Path("clip.mp4"), Path("model.pt"), Path("out"), calibration_path=Path("calibration.json"))
    assert plan.startswith("#!/usr/bin/env bash")
    assert "soccer-edge video export-frames" in plan
    assert "--calibration calibration.json" in plan
    assert "soccer-edge model auto-data-card" in plan


def test_validate_local_finetune_inputs(tmp_path) -> None:
    clip = tmp_path / "clip.mp4"
    model = tmp_path / "model.pt"
    clip.write_bytes(b"demo")
    result = validate_local_finetune_inputs(clip, model)
    assert not result.ok
    assert str(model) in result.missing_paths


def test_write_local_finetune_shell_plan(tmp_path) -> None:
    path = write_local_finetune_shell_plan(tmp_path / "plan.sh", Path("clip.mp4"), Path("model.pt"), Path("out"))
    assert path.exists()
    assert "local-finetune-dataset" in path.read_text(encoding="utf-8")


def test_write_local_finetune_shell_plan_validation_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        write_local_finetune_shell_plan(tmp_path / "plan.sh", Path("missing.mp4"), Path("missing.pt"), Path("out"), validate_inputs=True)
