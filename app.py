"""Stage 9: Streamlit dashboard for the credit risk PD project (decision log D-029, D-030).

A read-only view of the results that Stages 2-8 wrote to artifacts/. Nothing is
refitted, no metric is recomputed and no model is loaded: the loan-scoring page was
removed (D-030). The grade slider is a lookup in the committed grade scale.

Layout (D-030, sized for a 1920x1080 screen with the browser maximised): every page
has a title and a one-line subtitle, then one prominent chart in two thirds of the
width (its "?" says how to read it) next to a side panel with the headline tiles in a
2-column grid and the page's caveat. Detail tables sit in expanders, and the page ends
with its decision-log links and a "Next" button. Sentences live in
src/dashboard_text.py, charts in src/dashboard_charts.py, settings in src/config.py.
The theme is a fixed dark theme (.streamlit/config.toml).

Run from the project root:

    .venv\\Scripts\\python.exe -m streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src import dashboard
from src import dashboard_charts as charts
from src import dashboard_text as text

st.set_page_config(page_title="Credit risk PD model", page_icon=":material/analytics:", layout="wide")

PAGE_KEY = "page"  # session-state key of the sidebar radio
NUMBER_FORMATS = {
    "percent": st.column_config.NumberColumn(format="percent"),
    "count": st.column_config.NumberColumn(format="localized"),
    "decimal": st.column_config.NumberColumn(format=f"%.{config.DISPLAY_TABLE_DECIMALS}f"),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@st.cache_data
def table(name: str) -> pd.DataFrame:
    """A committed artifact, read once per session."""
    return dashboard.load_artifact(name)


@st.cache_data
def links(page: str) -> list[tuple[str, str]]:
    return dashboard.decision_links(config.DASHBOARD_PAGE_DECISIONS[page])


def show(df: pd.DataFrame) -> None:
    """A table with plain column labels, rates as percentages and counts with separators."""
    column_config = {label: NUMBER_FORMATS[kind] for label, kind in dashboard.column_formats(df).items()}
    st.dataframe(dashboard.readable_table(df), column_config=column_config, hide_index=True, width="stretch")


def details(title: str):
    return st.expander(f"Details: {title}", icon=":material/table_chart:")


def chart(page: str, key: str, fig: go.Figure) -> None:
    """A chart under its title; the title's "?" says how to read it."""
    title, how_to_read = text.PAGES[page]["charts"][key]
    st.subheader(title, help=how_to_read)
    st.plotly_chart(fig, key=f"chart_{page}_{key}")


def grid(items: list, columns: int = 2):
    """Yield (column, item) pairs that fill a grid row by row, so cards line up evenly."""
    for start in range(0, len(items), columns):
        for column, item in zip(st.columns(columns), items[start:start + columns]):
            yield column, item


def page_start(page: str):
    """Title and subtitle, then the two layout columns: (chart, side panel)."""
    st.title(dashboard.page_title(page))
    st.caption(text.PAGES[page]["subtitle"])
    return st.columns([2, 1], gap="large")


def caveat(page: str) -> None:
    if page in text.CAVEATS:
        box, message = text.CAVEATS[page]
        if box == "warning":
            st.warning(message, icon=":material/warning:")
        else:
            st.info(message, icon=":material/info:")


def side_panel(page: str) -> None:
    """The panel caption, the page's tiles in a 2-column grid (each with "?" help)."""
    st.caption(text.PANEL_CAPTIONS[page])
    for column, tile in grid(dashboard.page_tiles(page, table)):
        column.metric(tile.label, tile.value, help=tile.help, border=True)


def side_notes(page: str) -> None:
    """The page's caveat and its EQUI note, under the tiles."""
    caveat(page)
    if page in text.EQUI_CAPTION:
        st.caption(text.EQUI_CAPTION[page])


def go_to(page: str) -> None:
    st.session_state[PAGE_KEY] = config.DASHBOARD_PAGES[page]


def footer(page: str) -> None:
    """The page's decision-log entries (short ids, linked), then a button to the next page
    (on the last page, back to the first)."""
    items = " · ".join(f"[{heading.split(' ', 1)[0]}]({url})" for heading, url in links(page))
    st.markdown(f"**Decision log:** {items}")
    following = dashboard.next_page(page)
    if following:
        st.button(f"Next: {config.DASHBOARD_PAGES[following]}", icon=":material/arrow_forward:",
                  on_click=go_to, args=(following,), key=f"next_{page}")
    else:
        first = next(iter(config.DASHBOARD_PAGES))
        st.button(f"{text.BACK_TO_START} {config.DASHBOARD_PAGES[first]}", icon=":material/arrow_upward:",
                  on_click=go_to, args=(first,), key=f"back_{page}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview() -> None:
    st.title(dashboard.page_title("overview"))
    st.caption(text.PAGES["overview"]["subtitle"])
    steps = dashboard.overview_pipeline(table)
    for column, step in zip(st.columns(len(steps)), steps):
        with column.container(border=True):
            st.markdown(f"{step.icon} **{step.label}**")
            st.markdown(f"#### {step.value}")
            st.caption(step.note)

    main, side = st.columns([2, 1], gap="large")
    with main:
        chart("overview", "hero", charts.auc_comparison(table("model_cv_metrics.csv")))
    with side:
        st.subheader(text.KEY_FINDINGS)
        for column, finding in grid(dashboard.overview_findings(table)):
            with column.container(border=True):
                st.markdown(f"**{finding.title}**  \n{finding.text}")

    caveat("overview")
    with st.container(border=True):
        st.markdown(f"**{text.MODEL_CARD_TITLE}**\n\n" + "\n".join(
            f"- **{name}:** {line}" for name, line in text.MODEL_CARD.items()))
    with details("data quality"):
        show(table("dq_summary.csv"))
        show(table("dq_missingness.csv"))
    with details("development / hold-out split"):
        show(table("sql_split_summary.csv"))
    with details("model scope"):
        show(table("validation_scope.csv"))
    footer("overview")


def page_leakage() -> None:
    main, side = page_start("leakage")
    with main:
        overall = float(table("dq_summary.csv").set_index("check").loc["default_rate", "value"])
        chart("leakage", "indicators", charts.leakage_indicators(dashboard.leakage_chart_rows(table), overall))
    with side:
        side_panel("leakage")
        side_notes("leakage")
    with details("default rate when a field is missing vs present"):
        show(table("sql_dq_missingness.csv"))
    with details("leakage-demonstration cross-validation by fold"):
        show(table("model_cv_metrics.csv"))
    with details("fields excluded from the main model"):
        show(pd.DataFrame(config.EXCLUDED_FROM_MAIN_MODEL.items(), columns=["field", "reason"]))
    footer("leakage")


def page_model() -> None:
    main, side = page_start("model")
    with main:
        chart("model", "coefficients", charts.coefficients(table("model_coefficients.csv")))
        st.caption(text.LOAN_AMOUNT_CAPTION)
    with side:
        side_panel("model")
        st.markdown(f"**{text.MODEL_FIELDS_TITLE}**\n\n" + "\n".join(
            f"- {config.VARIABLE_LABELS.get(f, f)}" for f in config.MAIN_MODEL_FEATURES))
        side_notes("model")
    with details("coefficients"):
        show(table("model_coefficients.csv"))
    with details("feature screening"):
        show(table("model_screening.csv"))
    with details("cross-validation without " + ", ".join(config.SENSITIVITY_DROPPED_FEATURES)):
        show(table("model_sensitivity.csv"))
    footer("model")


def page_validation() -> None:
    main, side = page_start("validation")
    deciles = table("validation_calibration_deciles.csv")
    with main:
        calibration_tab, misses_tab = st.tabs([text.TAB_LABELS["calibration"], text.TAB_LABELS["misses"]])
        with calibration_tab:
            chart("validation", "calibration", charts.calibration(deciles))
        with misses_tab:
            view = st.segmented_control(text.MISS_SELECTOR, list(config.MISS_VIEWS),
                                        format_func=lambda v: config.MISS_VIEWS[v][0],
                                        default=next(iter(config.MISS_VIEWS)), required=True,
                                        key="miss_view")
            if config.MISS_VIEWS[view][1] == "deciles":
                fig = charts.feature_deciles(table("validation_feature_deciles.csv"), view)
            else:
                fig = charts.segments(table("validation_segments.csv"), view)
            chart("validation", "misses", fig)
    with side:
        side_panel("validation")
        side_notes("validation")
    with details("confidence intervals"):
        show(table("validation_confidence_intervals.csv"))
    with details("pre-set criteria"):
        criteria = table("validation_criteria.csv")
        criteria["rule"] = criteria["criterion"].map(dashboard.criterion_rules())
        show(criteria)
    with details("calibration"):
        show(deciles)
        show(table("validation_calibration.csv"))
        st.caption(text.HOSMER_LEMESHOW_CAPTION)
    with details("by feature decile and segment"):
        show(table("validation_feature_deciles.csv"))
        show(table("validation_segments.csv"))
    with details("coefficient stability (diagnostic refit on the hold-out, never used for scoring)"):
        show(table("validation_coefficient_stability.csv"))
    footer("validation")


def page_grades() -> None:
    main, side = page_start("grades")
    scale = table("grades_scale.csv")
    with main:
        label, help_text = text.GRADE_SLIDER
        start = round(float(scale["grade_pd"].iloc[0]) * 100, 1)
        pd_percent = st.slider(label, min_value=config.PD_MIN * 100, max_value=config.PD_MAX * 100, value=start,
                               step=config.GRADE_SLIDER_STEP, format="%.1f%%", help=help_text, key="grade_pd")
        selected = dashboard.grade_for_pd(pd_percent / 100, scale)
        rates_column, shares_column = st.columns([3, 2])
        with rates_column:
            chart("grades", "rates", charts.grade_rates(scale, table("grades_oof_check.csv"), str(selected["grade"])))
        with shares_column:
            chart("grades", "shares", charts.grade_shares(scale, str(selected["grade"])))
    with side:
        side_panel("grades")
        with st.container(border=True):
            st.markdown(
                f"**{text.SELECTED_GRADE}: {selected['grade']}**  \n"
                f"PD from {dashboard.format_rate(selected['pd_lower'])} to "
                f"{dashboard.format_rate(selected['pd_upper'])}  \n"
                f"Grade PD {dashboard.format_rate(selected['grade_pd'])}  \n"
                f"Observed {dashboard.format_rate(selected['observed_rate'])}"
            )
        side_notes("grades")
    with details("grade scale"):
        show(scale.drop(columns=["grade_rank", "n_defaults", "mean_pd"], errors="ignore"))
    with details("pre-set checks"):
        show(table("grades_checks.csv"))
    with details("out-of-fold check (development)"):
        show(table("grades_oof_check.csv"))
    footer("grades")


def page_monitoring() -> None:
    main, side = page_start("monitoring")
    psi = table("monitoring_psi.csv")
    with main:
        psi_tab, bins_tab = st.tabs([text.TAB_LABELS["psi"], text.TAB_LABELS["bins"]])
        with psi_tab:
            chart("monitoring", "psi", charts.psi(psi))
        with bins_tab:
            variable = st.selectbox(text.BIN_SELECTOR, psi["variable"].tolist(), key="bin_variable",
                                    format_func=lambda v: config.VARIABLE_LABELS.get(v, v))
            chart("monitoring", "bins", charts.bin_shares(table("monitoring_psi_bins.csv"), variable))
    with side:
        side_panel("monitoring")
        light = dashboard.largest_psi(psi)["light"]
        st.badge(light, icon=config.LIGHT_BADGE_ICONS[light], color=config.LIGHT_BADGE_COLORS[light])
        side_notes("monitoring")
    with details("PSI by variable"):
        show(psi)
    with details("watch list"):
        show(table("monitoring_watch_list.csv"))
    with details("characteristic analysis"):
        show(table("monitoring_characteristic.csv"))
    with details("grade backtest (development outcomes only; a reference, not a judgement)"):
        st.caption(dashboard.backtest_light_rules())
        show(table("monitoring_grade_backtest.csv"))
    with details("scope"):
        show(table("monitoring_scope.csv"))
    footer("monitoring")


PAGES = {
    "overview": page_overview,
    "leakage": page_leakage,
    "model": page_model,
    "validation": page_validation,
    "grades": page_grades,
    "monitoring": page_monitoring,
}
PAGE_BY_NAME = {name: key for key, name in config.DASHBOARD_PAGES.items()}

st.sidebar.subheader(text.SIDEBAR_TITLE)
choice = st.sidebar.radio("Page", list(config.DASHBOARD_PAGES.values()), key=PAGE_KEY,
                          label_visibility="collapsed")
st.sidebar.caption(text.DISCLAIMER)
missing = dashboard.missing_artifacts()
if missing:
    st.error(f"Missing artifacts: {', '.join(missing)}. Rebuild them with the stage runners.")
else:
    PAGES[PAGE_BY_NAME[choice]]()
