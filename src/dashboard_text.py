"""Stage 9: the dashboard's explanatory sentences (app.py; decision log D-029, D-030).

Short labels and settings live in src/config.py; the longer sentences live here so
config stays readable. No result number is typed here: every number shown on the
dashboard is read from a committed CSV or from config.
"""

from src import config

SCORING_OFF = """**Scoring is switched off in this online version.**

This page lets you try the model on one made-up loan, but only when the project runs on your own computer. It is off here on purpose:

- **It could be mistaken for a real credit decision.** It is not one: this is a learning project, not a bank's scoring system.
- **The model is a prototype on imperfect data.** It cannot use loan-to-value or debt-to-income, the core drivers of mortgage risk, because in this data those fields are missing almost only for loans that defaulted (D-017).
- **Its PDs only mean something inside this dataset**, because the definition of "default" is not documented (D-015).
- **Part of its skill rests on one field, `lump_sum_payment`**, whose effect cannot be separated from how the data was put together (D-017, D-022).
- **The hold-out test was used twice**, so its results are not fully unbiased (D-025, D-026).
- **The fitted model file is deliberately not published** (D-029).

To try it, clone the repository, build the model with `python -m src.model`, and start the app with scoring switched on (see README)."""

SCORING_SWITCH_HINT = (
    f"Scoring runs only when the environment variable `{config.SCORING_ENV_VAR}` is set to "
    f"`{config.SCORING_ENV_ON}` before the app starts (D-030)."
)
