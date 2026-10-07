"""Stage 4 SQL / DuckDB layer: processed Parquet -> DuckDB database -> SQL result tables.

DuckDB is the single data source for the later stages (decision log D-020):

    loans_clean     the processed Parquet from Stage 2, loaded as a table
    sample_split    ID -> "development" / "holdout", the D-013 split made once
    model_dataset   view with ID, sample, Status and only the 16 D-017 candidate features
    model_scope_dataset  model_dataset without credit_type = EQUI: the main model's
                         population (D-026)
    leakage_demo_dataset  view that adds back the D-017-excluded fields, for the
                          Stage 5 leakage demonstration only (D-024)

Steps (each is a separate function so it can be tested on its own):

    connect                    open (or create) the DuckDB database file
    load_loans                 load data/processed/loans_clean.parquet into DuckDB
    assign_sample_split        stratified 70/30 split on Status, seed 42 (D-013)
    create_model_dataset_view  sql/model_dataset.sql and sql/model_scope_dataset.sql
    create_leakage_demo_view   sql/leakage_demo_dataset.sql
    run_query                  run one sql/<name>.sql file and return its result
    run_analyses               all aggregate SQL analyses, keyed by output name
    save_sql_tables            write them to artifacts/sql_*.csv (aggregates only)

The analyses themselves are plain SQL files in sql/, readable on their own.

Run from the project root (after `python -m src.data_processing`):

    python -m src.db
"""

from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

# SQL files that produce an aggregate result table, and the parameters each one needs.
# Each result is saved as artifacts/sql_<name>.csv.
ANALYSIS_QUERIES = {
    "dq_reconciliation": {},
    "dq_missingness": {},
    "portfolio_by_segment": {"min_segment_size": config.SQL_MIN_SEGMENT_SIZE},
    "risk_deciles": {"n_bins": config.SQL_N_BINS, "equi": config.EQUI_LEVEL},
    "risk_segment_crosses": {
        "equi": config.EQUI_LEVEL, "min_segment_size": config.SQL_MIN_SEGMENT_SIZE,
    },
    "split_summary": {},
    "model_input_ranges": {
        "q_lower": config.INPUT_RANGE_QUANTILES[0], "q_upper": config.INPUT_RANGE_QUANTILES[1],
    },
}

# ---------------------------------------------------------------------------
# Database set-up
# ---------------------------------------------------------------------------


def connect(path: Path | str = config.DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB database file, creating it if needed. Use ":memory:" for tests."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def load_loans(con: duckdb.DuckDBPyConnection, parquet_path: Path = config.CLEAN_DATA_PATH) -> int:
    """(Re)create the loans table from the processed Parquet file. Returns the row count."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {parquet_path}. "
            "Run `python -m src.data_processing` first."
        )
    con.execute(
        f"CREATE OR REPLACE TABLE {config.LOANS_TABLE} AS SELECT * FROM read_parquet($path)",
        {"path": str(parquet_path)},
    )
    return con.execute(f"SELECT count(*) FROM {config.LOANS_TABLE}").fetchone()[0]


def split_ids(ids: pd.Series, target: pd.Series) -> pd.DataFrame:
    """Assign each ID to the development or hold-out sample (D-013).

    Stratified on the target with a fixed seed, so the default rate is the same in
    both samples and the split is identical every time it is rebuilt. The IDs are
    sorted first, so the result does not depend on the order the rows were read in.
    """
    order = ids.sort_values().index
    dev_ids, holdout_ids = train_test_split(
        ids.loc[order],
        test_size=config.TEST_SIZE,
        stratify=target.loc[order],
        random_state=config.RANDOM_SEED,
    )
    split = pd.concat([
        pd.DataFrame({config.ID_COL: dev_ids, "sample": config.SAMPLE_DEVELOPMENT}),
        pd.DataFrame({config.ID_COL: holdout_ids, "sample": config.SAMPLE_HOLDOUT}),
    ])
    return split.sort_values(config.ID_COL, ignore_index=True)


def assign_sample_split(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Make the D-013 split from the loans table and store it as the split table."""
    loans = con.execute(
        f"SELECT {config.ID_COL}, {config.TARGET_COL} FROM {config.LOANS_TABLE}"
    ).df()
    split = split_ids(loans[config.ID_COL], loans[config.TARGET_COL])
    con.register("split_frame", split)
    con.execute(f"CREATE OR REPLACE TABLE {config.SPLIT_TABLE} AS SELECT * FROM split_frame")
    con.unregister("split_frame")
    return split


# ---------------------------------------------------------------------------
# SQL files
# ---------------------------------------------------------------------------


def read_sql(name: str, sql_dir: Path = config.SQL_DIR) -> str:
    """Return the text of sql/<name>.sql."""
    return (sql_dir / f"{name}.sql").read_text(encoding="utf-8")


def run_query(con: duckdb.DuckDBPyConnection, name: str, **params) -> pd.DataFrame:
    """Run sql/<name>.sql with DuckDB named parameters ($n_bins, ...) and return the result."""
    return con.execute(read_sql(name), params or None).df()


def create_model_dataset_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create the model_dataset view (sql/model_dataset.sql, D-017 feature scope) and,
    on top of it, the model_scope_dataset view (sql/model_scope_dataset.sql, D-026)."""
    con.execute(read_sql("model_dataset"))
    con.execute(read_sql("model_scope_dataset"))


def create_leakage_demo_view(con: duckdb.DuckDBPyConnection) -> None:
    """Create the leakage_demo_dataset view (sql/leakage_demo_dataset.sql, D-024)."""
    con.execute(read_sql("leakage_demo_dataset"))


def run_analyses(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Run every aggregate SQL analysis and return the results, keyed by name."""
    return {name: run_query(con, name, **params) for name, params in ANALYSIS_QUERIES.items()}


def save_sql_tables(tables: dict[str, pd.DataFrame], directory: Path = config.ARTIFACTS_DIR) -> None:
    """Write the aggregate SQL results as small CSVs (committed as evidence)."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.round(6).to_csv(directory / f"{config.SQL_ARTIFACT_PREFIX}{name}.csv", index=False)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def build_database(
    db_path: Path | str = config.DUCKDB_PATH,
    parquet_path: Path = config.CLEAN_DATA_PATH,
    report_dir: Path = config.ARTIFACTS_DIR,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load -> split -> model view -> SQL analyses -> save. Returns the result tables."""
    con = connect(db_path)
    try:
        load_loans(con, parquet_path)
        assign_sample_split(con)
        create_model_dataset_view(con)
        create_leakage_demo_view(con)
        tables = run_analyses(con)
    finally:
        con.close()

    save_sql_tables(tables, report_dir)
    if verbose:
        _print_report(tables, db_path, report_dir)
    return tables


def _print_report(tables, db_path, report_dir) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("=== Reconciliation ===")
        print(tables["dq_reconciliation"].to_string(index=False))
        print("\n=== Sample split (D-013) ===")
        print(tables["split_summary"].to_string(index=False))
        print(f"\nDuckDB database -> {db_path}")
        print(f"SQL result tables -> {report_dir / (config.SQL_ARTIFACT_PREFIX + '*.csv')}")


if __name__ == "__main__":
    build_database()
