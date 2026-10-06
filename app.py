"""Stage 9: Streamlit dashboard for the credit risk PD project (decision log D-029).

A read-only view of the results that Stages 2-8 wrote to artifacts/. Nothing is
refitted and no metric is recomputed here. The optional "Score a loan" page uses
the frozen in-scope model if artifacts/pd_model.joblib exists (it is gitignored).

Run from the project root:

    .venv\\Scripts\\python.exe -m streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src import dashboard
from src import grades
from src import holdout

DISCLAIMER = (
    "**Educational project.** This is not an official bank credit-scoring system, it does not "
    "make underwriting decisions and it makes no claim of regulatory compliance. The risk "
    "grades are illustrative internal grades for this project."
)
LIGHTS = {"green": "🟢 green", "amber": "🟠 amber", "red": "🔴 red", "pass": "🟢 pass", "report": "⚪ reported"}

st.set_page_config(page_title="Credit Risk PD Model", layout="wide")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@st.cache_data
def table(name: str) -> pd.DataFrame:
    """A committed artifact, read once per session."""
    return dashboard.load_artifact(name)


@st.cache_resource
def frozen_model():
    """The frozen in-scope model, or None if it has not been built locally."""
    return dashboard.load_model_if_available()


def show(df: pd.DataFrame, percent: tuple[str, ...] = (), decimals: int = 4) -> None:
    """A table with rates as percentages and other floats rounded."""
    columns = {}
    for col in df.columns:
        if col in percent:
            columns[col] = st.column_config.NumberColumn(format="percent")
        elif pd.api.types.is_float_dtype(df[col]):
            columns[col] = st.column_config.NumberColumn(format=f"%.{decimals}f")
    st.dataframe(df, column_config=columns, hide_index=True, width="stretch")


def with_lights(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Replace green/amber/red (and pass/report) with a coloured marker."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].map(lambda v: LIGHTS.get(v, v))
    return df


def figure(key: str) -> None:
    """One of the README figures listed in config.DASHBOARD_FIGURES."""
    path = config.FIGURES_DIR / config.DASHBOARD_FIGURES[key]
    if path.exists():
        st.image(str(path))
    else:
        st.info(f"Figure {path.name} is not available.")


def chart_layout(fig: go.Figure, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, margin=dict(t=30, b=10),
                      legend=dict(orientation="h", y=1.1))
    return fig


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview() -> None:
    st.title("Credit Risk PD Model")
    st.markdown(
        "An end-to-end Probability of Default (PD) project on loan-level data: data quality, "
        "SQL analysis, an interpretable logistic regression, hold-out validation, illustrative "
        "risk grades and monitoring. This dashboard only **shows** results that the pipeline "
        "has already produced (D-029)."
    )
    st.warning(DISCLAIMER)
    st.markdown(
        "Validation is **out-of-sample, not out-of-time**: `year` is 2019 for every loan, so a "
        f"stratified {1 - config.TEST_SIZE:.0%}/{config.TEST_SIZE:.0%} split on `Status` is used "
        "(D-004, D-013)."
    )

    st.subheader("Data quality (Stage 2)")
    show(table("dq_summary.csv"))
    st.markdown("Missing values in the raw file, with the default rate when missing and when present:")
    show(table("dq_missingness.csv"), decimals=4)

    st.subheader("Development / hold-out split (Stage 4, D-013, D-020)")
    show(table("sql_split_summary.csv"), percent=("share_loans", "default_rate"))

    st.subheader("Model scope (D-026)")
    st.markdown(f"Loans with `credit_type = {config.EQUI_LEVEL}` are outside the main model's scope.")
    show(table("validation_scope.csv"), percent=("observed_rate",))


def page_leakage() -> None:
    st.title("Leakage finding")
    st.error(
        "Several fields are missing (almost) only when the loan defaulted, and "
        f"`credit_type = {config.EQUI_LEVEL}` is almost always a default. This most likely reflects "
        "how the dataset was assembled rather than borrower risk; the cause is unknown (D-017). "
        "These fields are therefore **excluded from the main model** (D-011, D-017)."
    )

    st.subheader("Default rate when a field is missing vs present (Stage 4)")
    show(table("sql_dq_missingness.csv"),
         percent=("share_missing", "default_rate_if_missing", "default_rate_if_present"))
    figure("leakage_indicators")

    st.subheader("Leakage demonstration (D-024)")
    extra = ", ".join(f"`{f}`" for f in config.LEAKAGE_DEMO_EXTRA_FEATURES)
    st.markdown(
        "Two models built **only** to show the leakage. The full model uses the main features plus "
        f"the fields excluded under D-017 and `term` / `co-applicant_credit_type` (D-022): {extra}. "
        "The ablation uses only *whether* each leakage field is missing (plus `credit_type`). Both "
        f"are scored with {config.CV_FOLDS}-fold CV on the development sample. They are never used "
        "on the hold-out, for grades, monitoring or scoring in this dashboard."
    )
    cv = table("model_cv_metrics.csv")
    show(cv[cv["fold"].astype(str) == "mean"].drop(columns="fold"))
    st.caption(
        "The leakage models use all development rows, EQUI included; the main model uses in-scope "
        "development rows only (D-026), so the populations differ (D-024)."
    )
    figure("leakage_demo_auc")
    with st.expander("Per-fold CV results"):
        show(cv)

    st.subheader("Fields excluded from the main model")
    excluded = pd.DataFrame(config.EXCLUDED_FROM_MAIN_MODEL.items(), columns=["field", "reason"])
    show(excluded)


def page_model() -> None:
    st.title("Main model")
    st.markdown(
        "An unpenalised logistic regression (D-012) fitted on in-scope development loans only "
        f"(`credit_type` ≠ {config.EQUI_LEVEL}, D-026). Preprocessing is inside a scikit-learn "
        "Pipeline: log + standardise the two amounts, a missing indicator for income, and one-hot "
        "categoricals with the most frequent level as reference (D-023)."
    )
    st.markdown("**Features (D-022):** " + ", ".join(f"`{f}`" for f in config.MAIN_MODEL_FEATURES))

    st.subheader("Coefficients (Stage 5)")
    show(table("model_coefficients.csv"), percent=("cv_sign_share",))
    st.info(
        "**`loan_amount` sign (D-023).** On its own a larger loan goes with a *lower* default rate, "
        "because larger loans go to higher-income borrowers. With income in the model, its "
        "coefficient is positive: for the same income, a larger loan means higher leverage. It is "
        "kept, and the table still flags the mismatch with the expected sign."
    )
    figure("model_coefficients")

    st.subheader("Feature screening (D-022)")
    show(table("model_screening.csv"))

    st.subheader("`lpsm` sensitivity (D-022)")
    sensitivity = table("model_sensitivity.csv")
    show(sensitivity[sensitivity["fold"].astype(str) == "mean"].drop(columns="fold"))
    dropped = ", ".join(f"`{f}`" for f in config.SENSITIVITY_DROPPED_FEATURES)
    st.caption(
        f"Development-only {config.CV_FOLDS}-fold CV without {dropped}, on the same folds and rows; "
        "reported only and never acted on (D-022). Whether the `lpsm` effect is separate from the "
        "extraction pattern cannot be tested (D-017)."
    )


def page_validation() -> None:
    st.title("Validation (Stage 6)")
    st.markdown(
        "The frozen model scores the in-scope hold-out. The criteria were fixed before the "
        "hold-out was scored (D-025). After the D-026 scope change, the hold-out was looked at a "
        "second time, so its figures are no longer a fully unbiased estimate."
    )

    st.subheader("Discrimination")
    show(table("validation_metrics.csv"))
    st.markdown("Bootstrap confidence intervals on the hold-out:")
    show(table("validation_confidence_intervals.csv"))

    st.subheader("Pre-set criteria (D-025, heuristic thresholds)")
    criteria = table("validation_criteria.csv")
    criteria["rule"] = criteria["criterion"].map(dashboard.criterion_rules())
    show(with_lights(criteria, ("status",)))

    st.subheader("Calibration by PD decile (hold-out)")
    deciles = table("validation_calibration_deciles.csv")
    top = max(deciles["mean_pd"].max(), deciles["observed_rate"].max())
    fig = go.Figure([
        go.Scatter(x=[0, top], y=[0, top], mode="lines", name="perfect calibration",
                   line=dict(color=config.COLOR_INK_MUTED, dash="dash")),
        go.Scatter(x=deciles["mean_pd"], y=deciles["observed_rate"], mode="markers+lines",
                   name="PD decile", marker=dict(color=config.COLOR_PRIMARY, size=9),
                   customdata=deciles[["bin", "n_loans"]],
                   hovertemplate="decile %{customdata[0]}<br>mean PD %{x:.1%}<br>"
                                 "observed %{y:.1%}<br>%{customdata[1]:,} loans<extra></extra>"),
    ])
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(chart_layout(fig, "mean PD", "observed default rate"))
    show(deciles, percent=("mean_pd", "observed_rate", "gap"))
    show(table("validation_calibration.csv"), percent=("mean_pd", "observed_rate"))
    st.caption(
        "Hosmer-Lemeshow (`hl_p_value`) is reported without a pass/fail judgement: with tens of "
        "thousands of loans it rejects even trivial misfit (D-025)."
    )

    st.subheader("By segment (hold-out)")
    show(table("validation_segments.csv"), percent=("mean_pd", "observed_rate", "gap"))
    st.subheader("Coefficient stability (diagnostic refit on the hold-out, never used for scoring)")
    show(table("validation_coefficient_stability.csv"))


def page_grades() -> None:
    st.title("Illustrative risk grades (Stage 7)")
    scale = table("grades_scale.csv")
    n_loans = int(scale["n_loans"].sum())
    st.markdown(
        "Equal-count PD bins merged until each grade defaults significantly more than the one "
        f"below and holds at least {grades.min_grade_loans(n_loans):,} loans (the floor of "
        f"{config.GRADE_MIN_SHARE:.0%} × {n_loans:,} loans; D-014, D-027). Built from in-scope "
        "development loans only; no hold-out outcome is used. The grade PD is the mean model PD "
        "of the grade."
    )
    fig = go.Figure([
        go.Bar(x=scale["grade"], y=scale["grade_pd"], name="grade PD", marker_color=config.COLOR_PRIMARY),
        go.Bar(x=scale["grade"], y=scale["observed_rate"], name="observed default rate",
               marker_color=config.COLOR_SECONDARY),
    ])
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(chart_layout(fig, "grade", "rate (development)"))
    show(scale.drop(columns=["grade_rank", "n_defaults", "mean_pd"], errors="ignore"),
         percent=("pd_lower", "pd_upper", "grade_pd", "share", "observed_rate", "gap"))
    st.info(
        "**Sensitivity (D-027).** With the same rules, the number of grades depends on technical "
        f"settings: {len(scale)} with the adopted settings, 7 with boundaries to 9 decimals, 10 "
        "with 19 starting bins (`notebooks/04_risk_grades.ipynb` §2, where an `assert` checks the "
        "counts; D-027). The pre-set settings were kept, so the exact number of grades is "
        "documented as not robust. The shape is the same every time: one large grade A over the "
        "flat lower half of PDs, then rising grades."
    )

    st.subheader("Pre-set checks")
    show(with_lights(table("grades_checks.csv"), ("status",)))
    st.subheader("Out-of-fold check (development)")
    show(table("grades_oof_check.csv"), percent=("mean_pd", "observed_rate"))


def page_monitoring() -> None:
    st.title("Monitoring (Stage 8)")
    st.warning(
        "The development sample is the baseline, and the hold-out stands in for a \"next period\" "
        "(inputs and scores only, no outcomes). Because the split is random, a PSI close to zero "
        "is expected **by construction**: a green result shows that the monitoring works, but it "
        "is **not evidence of stability over time** (D-028)."
    )
    green, amber = config.PSI_THRESHOLDS
    st.markdown(f"PSI / CSI bands (heuristic): ≤ {green} green, ≤ {amber} amber, above that red.")

    psi = table("monitoring_psi.csv")
    fig = go.Figure(go.Bar(x=psi["psi"], y=psi["variable"], orientation="h",
                           marker_color=config.COLOR_PRIMARY, text=psi["psi"].map("{:.4f}".format),
                           textposition="outside", name="PSI"))
    for limit, label in ((green, "green limit"), (amber, "amber limit")):
        fig.add_vline(x=limit, line_dash="dash", line_color=config.COLOR_INK_MUTED,
                      annotation_text=f"{label} {limit}")
    fig.update_xaxes(range=[0, dashboard.psi_axis_max(psi["psi"])])
    st.plotly_chart(chart_layout(fig, "PSI (development vs hold-out)", ""))
    show(with_lights(psi, ("light",)))

    st.subheader("Bin shares by variable")
    bins = table("monitoring_psi_bins.csv")
    variable = st.selectbox("Variable", psi["variable"].tolist())
    chosen = bins[bins["variable"] == variable].sort_values("bin_order")
    fig = go.Figure([
        go.Bar(x=chosen["bin_label"], y=chosen["share_baseline"], name="development (baseline)",
               marker_color=config.COLOR_PRIMARY),
        go.Bar(x=chosen["bin_label"], y=chosen["share_monitored"], name="hold-out (monitored)",
               marker_color=config.COLOR_SECONDARY),
    ])
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(chart_layout(fig, "bin", "share of loans"))

    st.subheader("Watch list (D-028)")
    show(with_lights(table("monitoring_watch_list.csv"), ("light",)))
    st.subheader("Characteristic analysis")
    show(table("monitoring_characteristic.csv"))
    st.subheader("Grade backtest (development outcomes only; a reference, not a judgement)")
    st.caption(dashboard.backtest_light_rules())
    show(with_lights(table("monitoring_grade_backtest.csv"), ("gap_light", "binomial_light")),
         percent=("grade_pd", "observed_rate", "gap"))
    st.subheader("Scope")
    show(table("monitoring_scope.csv"), percent=("out_of_scope_share",))


def page_score() -> None:
    st.title("Score a loan")
    st.warning(
        "Illustrative only: this shows what the frozen model returns for one set of inputs. It is "
        "not a credit decision (D-029)."
    )
    pipeline = frozen_model()
    if pipeline is None:
        st.info(
            f"The fitted model ({config.MODEL_PATH.name}) is not available. It is not committed; "
            "build it locally with `python -m src.model` (needs the raw data). All other pages work "
            "without it."
        )
        return

    try:
        dashboard.check_model_coefficients(pipeline, table("model_coefficients.csv"))
    except ValueError as err:
        st.error(
            "The local model does not match the committed in-scope model (`model_coefficients.csv`); "
            f"rebuild it with `python -m src.model`. Not scoring. Details: {err}"
        )
        return

    known = dashboard.model_levels(pipeline)
    ranges = table("sql_model_input_ranges.csv")
    bounds = ranges.set_index("variable")
    q_lower, q_upper = (f"{q:.0%}" for q in config.INPUT_RANGE_QUANTILES)
    levels_table = table("dq_categorical_levels.csv")
    credit_levels = levels_table.loc[levels_table["column"] == config.CREDIT_TYPE_COL, "level"]

    def range_help(variable: str) -> str:
        b = bounds.loc[variable]
        return (f"In-scope development loans: {q_lower} to {q_upper} quantile {b['lower']:,.0f} to "
                f"{b['upper']:,.0f}; starts at the median (D-029).")

    with st.form("score"):
        credit_type = st.selectbox(
            "credit_type (scope check only, not a model feature)", credit_levels.tolist())
        loan_amount = st.number_input("loan_amount", min_value=0.0, value=float(bounds.loc["loan_amount", "median"]),
                                      help=range_help("loan_amount"))
        no_income = st.checkbox("Income not provided (scored as missing, D-023)")
        income = st.number_input("income", min_value=0.0,
                                 value=float(bounds.loc[config.INCOME_CLEAN_COL, "median"]),
                                 help=range_help(config.INCOME_CLEAN_COL))
        levels = {
            col: st.selectbox(col, known[col], index=known[col].index(config.REFERENCE_LEVELS[col]))
            for col in config.CATEGORICAL_FEATURES
        }
        submitted = st.form_submit_button("Score")

    if not submitted:
        return
    try:
        loan = dashboard.build_loan_frame(loan_amount, None if no_income else income, levels,
                                          credit_type, known)
    except ValueError as err:
        st.error(str(err))
        return

    pd_value = float(holdout.score(pipeline, loan).iloc[0])
    flagged = dashboard.out_of_range(loan_amount, None if no_income else income, ranges)
    if flagged:
        details = "; ".join(f"`{v}` outside {lo:,.0f} to {hi:,.0f}" for v, (lo, hi) in flagged.items())
        st.warning(
            f"**Extrapolation:** {details} (the {q_lower} to {q_upper} quantiles of in-scope development "
            "loans). The model has little data here, so this PD is less reliable (D-029)."
        )
    grade = dashboard.grade_for_pd(pd_value, table("grades_scale.csv"))
    left, mid, right = st.columns(3)
    left.metric("Model PD", f"{pd_value:.2%}")
    mid.metric("Illustrative grade", grade["grade"])
    right.metric("Grade PD", f"{grade['grade_pd']:.2%}")
    st.caption(f"Grade {grade['grade']} covers PDs from {grade['pd_lower']:.2%} to {grade['pd_upper']:.2%}; "
               f"its observed development default rate is {grade['observed_rate']:.1%} (D-027).")


PAGES = {
    "Overview": page_overview,
    "Leakage finding": page_leakage,
    "Model": page_model,
    "Validation": page_validation,
    "Risk grades": page_grades,
    "Monitoring": page_monitoring,
    "Score a loan": page_score,
}

choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption(DISCLAIMER)
missing = dashboard.missing_artifacts()
if missing:
    st.error(f"Missing artifacts: {', '.join(missing)}. Rebuild them with the stage runners.")
else:
    PAGES[choice]()
