from pathlib import Path

import pandas as pd


DROP_ACTIONS = {"drop", "delete", "remove", "reject"}
KEEP_ACTIONS = {"keep", "accept", "update", "correct"}


def correction_target_columns(corrections: pd.DataFrame, prefix: str = "corrected_") -> dict[str, str]:
    return {column: column.removeprefix(prefix) for column in corrections.columns if column.startswith(prefix)}


def merge_reviewed_corrections(
    base: pd.DataFrame,
    corrections: pd.DataFrame,
    key_columns: list[str],
    action_column: str = "review_action",
    corrected_prefix: str = "corrected_",
) -> pd.DataFrame:
    missing_base = [column for column in key_columns if column not in base.columns]
    missing_corrections = [column for column in key_columns if column not in corrections.columns]
    if missing_base:
        raise ValueError(f"base missing key columns: {missing_base}")
    if missing_corrections:
        raise ValueError(f"corrections missing key columns: {missing_corrections}")

    merged = base.copy()
    target_map = correction_target_columns(corrections, corrected_prefix)
    indexed_corrections = corrections.set_index(key_columns, drop=False)
    keep_mask = []
    for idx, row in merged.iterrows():
        key = tuple(row[column] for column in key_columns)
        lookup_key = key[0] if len(key) == 1 else key
        if lookup_key not in indexed_corrections.index:
            keep_mask.append(True)
            continue
        correction_row = indexed_corrections.loc[lookup_key]
        if isinstance(correction_row, pd.DataFrame):
            correction_row = correction_row.iloc[-1]
        action = str(correction_row.get(action_column, "update")).lower()
        if action in DROP_ACTIONS:
            keep_mask.append(False)
            continue
        keep_mask.append(True)
        if action in KEEP_ACTIONS or target_map:
            for source_column, target_column in target_map.items():
                value = correction_row[source_column]
                if pd.notna(value):
                    merged.at[idx, target_column] = value
            merged.at[idx, "review_status"] = action
    return merged.loc[keep_mask].reset_index(drop=True)


def merge_reviewed_corrections_from_tables(
    base_path: Path,
    corrections_path: Path,
    output_path: Path,
    key_columns: list[str],
    action_column: str = "review_action",
) -> Path:
    base = pd.read_parquet(base_path) if base_path.suffix == ".parquet" else pd.read_csv(base_path)
    corrections = pd.read_parquet(corrections_path) if corrections_path.suffix == ".parquet" else pd.read_csv(corrections_path)
    merged = merge_reviewed_corrections(base, corrections, key_columns=key_columns, action_column=action_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        merged.to_parquet(output_path, index=False)
    else:
        merged.to_csv(output_path, index=False)
    return output_path
