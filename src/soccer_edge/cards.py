from pathlib import Path

from soccer_edge.models.run_metadata import read_run_metadata


def write_model_card(
    bundle_dir: Path,
    output_path: Path,
    intended_use: str = "Offline soccer analytics research.",
    limitations: str = "Not for real-money execution or guaranteed outcome prediction.",
) -> Path:
    metadata = read_run_metadata(bundle_dir / "metadata.json")
    lines = [
        f"# Model Card: {metadata.name}",
        "",
        f"Version: {metadata.version}",
        f"Created: {metadata.created_at_utc}",
        "",
        "## Intended use",
        intended_use,
        "",
        "## Features",
        "\n".join(f"- {feature}" for feature in metadata.feature_names),
        "",
        "## Metrics",
        "\n".join(f"- {name}: {value}" for name, value in metadata.metrics.items()),
        "",
        "## Limitations",
        limitations,
        "",
        "## Notes",
        metadata.notes or "No additional notes.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_data_card(
    dataset_name: str,
    source_paths: list[Path],
    output_path: Path,
    rights_status: str = "owned",
    lineage_note: str = "Generated from approved local or open sources.",
) -> Path:
    lines = [
        f"# Data Card: {dataset_name}",
        "",
        f"Rights status: {rights_status}",
        "",
        "## Sources",
        "\n".join(f"- {path}" for path in source_paths),
        "",
        "## Lineage",
        lineage_note,
        "",
        "## Allowed use",
        "Offline research, model training, calibration, and evaluation within this repository.",
        "",
        "## Restrictions",
        "Do not redistribute restricted footage or process media without recorded rights status.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
