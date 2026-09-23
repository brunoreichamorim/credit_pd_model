"""Smoke test: every dependency in requirements.txt can be imported.

If this fails, the environment is broken and nothing else in the project will run.
"""

import importlib

import pytest

REQUIRED_MODULES = [
    "pandas", "numpy", "pyarrow", "duckdb", "sklearn", "statsmodels",
    "scipy", "matplotlib", "plotly", "streamlit",
]


@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_dependency_importable(module_name):
    importlib.import_module(module_name)


def test_duckdb_runs_sql():
    """DuckDB works in-process, without a server: the basis of the SQL layer."""
    import duckdb

    assert duckdb.sql("SELECT AVG(x) FROM (VALUES (0), (1), (1), (0)) t(x)").fetchone()[0] == 0.5
