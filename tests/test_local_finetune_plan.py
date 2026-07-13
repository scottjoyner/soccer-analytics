from pathlib import Path

from soccer_edge.local_finetune_plan import local_finetune_shell_plan, shell_quote, write_local_finetune_shell_plan


def test_shell_quote() -> None:
    assert shell_quote("abc/def") == "abc/def"
    assert shell_quote("a b") == "'a b'"


def test_local_finetune_shell_plan() -> None:
    plan = local_finetune_shell_plan(Path("clip.mp4"), Path("model.pt"), Path("out"), calibration_path=Path("calibration.json"))
    assert plan.startswith("#!/usr/bin/env bash")
    assert "soccer-edge video export-frames" in plan
    assert "--calibration calibration.json" in plan
    assert "soccer-edge model auto-data-card" in plan


def test_write_local_finetune_shell_plan(tmp_path) -> None:
    path = write_local_finetune_shell_plan(tmp_path / "plan.sh", Path("clip.mp4"), Path("model.pt"), Path("out"))
    assert path.exists()
    assert "local-finetune-dataset" in path.read_text(encoding="utf-8")
