from pathlib import Path

import duckdb
import pandas as pd


def query_tables(sql: str, data_dir: Path = Path("data")) -> pd.DataFrame:
    connection = duckdb.connect(database=":memory:")
    connection.sql(f"SET home_directory='{data_dir}'")
    return connection.sql(sql).df()
