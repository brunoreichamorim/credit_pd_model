"""Stage 9: the dashboard's Plotly charts (app.py; decision log D-030).

Each function turns committed tables into a figure and nothing more: no Streamlit,
no new numbers. Colours come only from config.DASHBOARD_COLORS (documented dark
palette values), so tests can check both the data and the colours of every chart.
Sizes are set for a 1920x1080 screen with the browser maximised.

    auc_comparison      Overview hero: mean CV AUC of the main and leakage models
    leakage_indicators  Data leakage: default rate when missing vs present (figure 05 fields)
    coefficients        Model: what raises and lowers the PD (figure 07)
    calibration         Validation: mean PD vs observed default rate per PD decile
    feature_deciles     Validation: mean PD vs observed rate per decile of a feature (figure 11)
    segments            Validation: mean PD vs observed rate per level of a category
    grade_rates         Risk grades: grade PD, observed in-sample and out-of-fold (figure 12, left)
    grade_shares        Risk grades: share of loans per grade (figure 12, right)
    psi                 Monitoring: observed PSI next to the PSI expected with no shift
    bin_shares          Monitoring: share of loans per bin, development vs hold-out
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import config
from src import dashboard

COLORS = config.DASHBOARD_COLORS
MARKER = dict(size=config.CHART_MARKER_SIZE)


def _layout(fig: go.Figure, x_title: str, y_title: str,
            height: int = config.DASHBOARD_CHART_HEIGHT) -> go.Figure:
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, height=height,
                      margin=dict(t=30, b=10, r=40), legend=dict(orientation="h", y=1.12),
                      bargap=0.25, bargroupgap=0.05)
    return fig


def _reference_line(fig: go.Figure, x: float, text: str = "") -> None:
    """A dashed muted line under the marks, labelled at the bottom of the plot."""
    fig.add_vline(x=x, line_dash="dash", line_width=1, line_color=COLORS["muted"], layer="below",
                  annotation_text=text, annotation_position="bottom right",
                  annotation_font_color=COLORS["muted"])


def _gap_label(gap: float) -> str:
    """A gap (observed - PD) from the CSV, shown in percentage points."""
    return f"{gap * 100:+.1f} pp"


def _dumbbell(fig: go.Figure, labels, left, right) -> None:
    """Muted lines joining two dots per row; rows with a missing value are not joined."""
    xs, ys = [], []
    for label, a, b in zip(labels, left, right):
        if pd.notna(a) and pd.notna(b):
            xs += [a, b, None]
            ys += [label, label, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=COLORS["muted"], width=2),
                             showlegend=False, hoverinfo="skip"))


def auc_comparison(cv: pd.DataFrame) -> go.Figure:
    """Mean CV AUC of the main model (highlighted) and the two leakage models (muted),
    from model_cv_metrics.csv, against the 0.5 coin-flip line."""
    names = [config.CV_MODEL_MAIN, config.CV_MODEL_LEAKAGE_FULL, config.CV_MODEL_LEAKAGE_ABLATION]
    means = cv[cv["fold"].astype(str) == config.CV_MEAN_ROW].set_index("model")["auc"]
    values = [float(means[n]) for n in names]
    fig = go.Figure(go.Bar(
        x=values, y=[config.CV_MODEL_LABELS[n] for n in names], orientation="h", name="CV AUC",
        marker_color=[COLORS["series_1"] if n == config.CV_MODEL_MAIN else COLORS["muted"] for n in names],
        text=[dashboard.format_metric(v) for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: CV AUC %{x:.3f}<extra></extra>",
    ))
    _reference_line(fig, config.AUC_COIN_FLIP, f"{config.AUC_COIN_FLIP} coin flip")
    fig.update_xaxes(range=[0, config.CHART_RATE_AXIS_MAX])
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "mean cross-validated AUC", "", height=config.DASHBOARD_HERO_HEIGHT)


def leakage_indicators(rows: pd.DataFrame, overall_rate: float) -> go.Figure:
    """Dot plot from dashboard.leakage_chart_rows: per field, the default rate when it is
    missing and when it is present, joined by a line. The EQUI row has one dot only."""
    fig = go.Figure()
    _dumbbell(fig, rows["label"], rows["rate_if_true"], rows["rate_otherwise"])
    fig.add_trace(go.Scatter(x=rows["rate_otherwise"], y=rows["label"], mode="markers", name="when present",
                             marker=dict(color=COLORS["series_1"], **MARKER),
                             hovertemplate="%{y}, otherwise: default rate %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=rows["rate_if_true"], y=rows["label"], mode="markers",
                             name="when missing (or the category)",
                             marker=dict(color=COLORS["series_2"], **MARKER), customdata=rows["n_loans"],
                             hovertemplate="%{y}: default rate %{x:.1%}<br>%{customdata:,} loans<extra></extra>"))
    _reference_line(fig, overall_rate, "all loans")
    fig.update_xaxes(range=[0, config.CHART_RATE_AXIS_MAX], tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "default rate", "")


def coefficients(table: pd.DataFrame) -> go.Figure:
    """Coefficients (model_coefficients.csv) sorted by size, with readable names: positive
    raise the PD, negative lower it. The two largest carry their value; hover shows the
    technical name, odds ratio, p-value and CV sign share."""
    ordered = table.sort_values("coefficient").assign(
        label=lambda t: t["feature"].map(lambda f: config.COEFFICIENT_LABELS.get(f, f)))
    largest = set(ordered["coefficient"].abs().nlargest(2).index)
    fig = go.Figure()
    for name, mask, color in (("raises PD", ordered["coefficient"] > 0, COLORS["series_1"]),
                              ("lowers PD", ordered["coefficient"] <= 0, COLORS["series_2"])):
        part = ordered[mask]
        fig.add_trace(go.Bar(
            x=part["coefficient"], y=part["label"], orientation="h", name=name, marker_color=color,
            text=[f"{c:.2f}" if i in largest else "" for i, c in part["coefficient"].items()],
            textposition="outside", cliponaxis=False,
            customdata=np.column_stack([part["feature"], part["odds_ratio"],
                                        part["sm_p_value"].map(dashboard.format_p_value), part["cv_sign_share"]]),
            hovertemplate="%{y} (%{customdata[0]})<br>coefficient %{x:.3f}<br>odds ratio %{customdata[1]:.3f}"
                          "<br>p-value %{customdata[2]}<br>same sign in %{customdata[3]:.0%} of CV folds"
                          "<extra></extra>",
        ))
    _reference_line(fig, 0.0)
    fig.update_layout(barmode="relative")
    fig.update_yaxes(categoryorder="array", categoryarray=ordered["label"].tolist())
    return _layout(fig, "coefficient (log-odds)", "")


def calibration(deciles: pd.DataFrame) -> go.Figure:
    """Mean PD vs observed default rate per hold-out PD decile, with the diagonal."""
    top = max(deciles["mean_pd"].max(), deciles["observed_rate"].max())
    fig = go.Figure([
        go.Scatter(x=[0, top], y=[0, top], mode="lines", name="perfect calibration",
                   line=dict(color=COLORS["muted"], dash="dash", width=1)),
        go.Scatter(x=deciles["mean_pd"], y=deciles["observed_rate"], mode="markers+lines", name="PD decile",
                   marker=dict(color=COLORS["series_1"], **MARKER), line=dict(width=2),
                   customdata=deciles[["bin", "n_loans"]],
                   hovertemplate="decile %{customdata[0]}<br>mean PD %{x:.1%}<br>"
                                 "observed %{y:.1%}<br>%{customdata[1]:,} loans<extra></extra>"),
    ])
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, "mean PD", "observed default rate")


def feature_deciles(table: pd.DataFrame, variable: str) -> go.Figure:
    """Mean PD and observed default rate per hold-out decile of `variable`
    (validation_feature_deciles.csv, README figure 11). The largest gap is labelled."""
    rows = table[table["variable"] == variable].sort_values("bin")
    worst = rows["gap"].abs().idxmax()
    custom = rows[["bin_min", "bin_max", "n_loans"]]
    fig = go.Figure([
        go.Scatter(x=rows["bin"], y=rows["mean_pd"], mode="markers+lines", name="mean PD",
                   marker=dict(color=COLORS["series_1"], **MARKER), line=dict(width=2), customdata=custom,
                   hovertemplate="decile %{x} (%{customdata[0]:,.0f} to %{customdata[1]:,.0f})<br>"
                                 "mean PD %{y:.1%}<br>%{customdata[2]:,} loans<extra></extra>"),
        go.Scatter(x=rows["bin"], y=rows["observed_rate"], mode="markers+lines+text",
                   name="observed default rate",
                   marker=dict(color=COLORS["series_2"], **MARKER), line=dict(width=2, dash="dash"),
                   text=[_gap_label(g) if i == worst else "" for i, g in rows["gap"].items()],
                   textposition="top center",
                   hovertemplate="decile %{x}: observed %{y:.1%}<extra></extra>"),
    ])
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=".0%")
    label = config.VARIABLE_LABELS.get(variable, variable)
    return _layout(fig, f"{label} decile (1 = lowest)", "rate (hold-out)")


def segments(table: pd.DataFrame, variable: str) -> go.Figure:
    """Dot plot per level of `variable` (validation_segments.csv), sorted by gap: mean PD
    and observed default rate joined by a line. The largest gap is labelled."""
    rows = table[table["variable"] == variable].sort_values("gap")
    worst = rows["gap"].abs().idxmax()
    custom = rows[["n_loans", "gap", "auc"]]
    fig = go.Figure()
    _dumbbell(fig, rows["level"], rows["mean_pd"], rows["observed_rate"])
    fig.add_trace(go.Scatter(x=rows["mean_pd"], y=rows["level"], mode="markers", name="mean PD",
                             marker=dict(color=COLORS["series_1"], **MARKER), customdata=custom,
                             hovertemplate="%{y}: mean PD %{x:.1%}<br>%{customdata[0]:,} loans<br>"
                                           "AUC %{customdata[2]:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=rows["observed_rate"], y=rows["level"], mode="markers+text",
                             name="observed default rate",
                             marker=dict(color=COLORS["series_2"], **MARKER), customdata=custom,
                             text=[_gap_label(g) if i == worst else "" for i, g in rows["gap"].items()],
                             textposition="middle right",
                             hovertemplate="%{y}: observed %{x:.1%}<br>gap %{customdata[1]:+.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(type="category")
    return _layout(fig, "rate (hold-out)", config.VARIABLE_LABELS.get(variable, variable))


def _grade_band(fig: go.Figure, grades: list[str], selected: str | None) -> None:
    """A light band behind the selected grade (categorical axis: positions 0, 1, ...)."""
    if selected in grades:
        i = grades.index(selected)
        fig.add_vrect(x0=i - 0.5, x1=i + 0.5, fillcolor=COLORS["muted"],
                      opacity=config.CHART_BAND_OPACITY, line_width=0, layer="below")


def grade_rates(scale: pd.DataFrame, oof: pd.DataFrame, selected: str | None = None) -> go.Figure:
    """Figure 12, left: grade PD, observed in-sample rate (grades_scale.csv) and observed
    out-of-fold rate (grades_oof_check.csv, pooled rows) per grade."""
    grades = scale["grade"].tolist()
    pooled = oof[oof["sample"] == config.GRADE_OOF_POOLED].set_index("grade").reindex(grades)
    sizes = [config.CHART_SELECTED_MARKER_SIZE if g == selected else config.CHART_MARKER_SIZE for g in grades]
    fig = go.Figure([
        go.Scatter(x=grades, y=scale["grade_pd"], mode="markers+lines", name="grade PD",
                   marker=dict(color=COLORS["series_1"], size=sizes), line=dict(width=2),
                   hovertemplate="grade %{x}: grade PD %{y:.1%}<extra></extra>"),
        go.Scatter(x=grades, y=scale["observed_rate"], mode="markers+lines", name="observed, in-sample",
                   marker=dict(color=COLORS["series_2"], size=sizes), line=dict(width=2, dash="dash"),
                   hovertemplate="grade %{x}: observed %{y:.1%}<extra></extra>"),
        go.Scatter(x=grades, y=pooled["observed_rate"], mode="markers", name="observed, out-of-fold",
                   marker=dict(color=COLORS["muted"], symbol="x", **MARKER),
                   hovertemplate="grade %{x}: observed out-of-fold %{y:.1%}<extra></extra>"),
    ])
    _grade_band(fig, grades, selected)
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, "grade", "default rate (development)")


def grade_shares(scale: pd.DataFrame, selected: str | None = None) -> go.Figure:
    """Figure 12, right: share of development loans per grade (grades_scale.csv). Only the
    first grade carries its value; the rest show it on hover."""
    grades = scale["grade"].tolist()
    fig = go.Figure(go.Bar(
        x=grades, y=scale["share"], name="share of loans", marker_color=COLORS["series_1"],
        text=[dashboard.format_rate(s) if i == 0 else "" for i, s in enumerate(scale["share"])],
        textposition="outside", cliponaxis=False,
        hovertemplate="grade %{x}: %{y:.1%} of loans<extra></extra>",
    ))
    _grade_band(fig, grades, selected)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(showlegend=False)
    return _layout(fig, "grade", "share of loans (development)")


def psi(table: pd.DataFrame) -> go.Figure:
    """Per variable (monitoring_psi.csv): the observed PSI between development and hold-out,
    next to the PSI expected from sampling noise alone (no real shift). The axis fits the
    data, so no value is ever cut off; the heuristic limits are stated in the chart help."""
    rows = table.assign(label=table["variable"].map(lambda v: config.VARIABLE_LABELS.get(v, v)))
    fig = go.Figure()
    _dumbbell(fig, rows["label"], rows["psi_expected_no_shift"], rows["psi"])
    fig.add_trace(go.Scatter(x=rows["psi_expected_no_shift"], y=rows["label"], mode="markers",
                             name="expected with no shift", marker=dict(color=COLORS["muted"], **MARKER),
                             hovertemplate="%{y}: expected %{x:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=rows["psi"], y=rows["label"], mode="markers", name="observed PSI",
                             marker=dict(color=COLORS["series_1"], **MARKER), customdata=rows["light"],
                             hovertemplate="%{y}: PSI %{x:.4f} (%{customdata})<extra></extra>"))
    fig.update_xaxes(rangemode="tozero", tickformat=".4f")
    return _layout(fig, "PSI (development vs hold-out)", "")


def bin_shares(bins: pd.DataFrame, variable: str) -> go.Figure:
    """Share of loans per bin of `variable` (monitoring_psi_bins.csv): development vs hold-out."""
    chosen = bins[bins["variable"] == variable].sort_values("bin_order")
    fig = go.Figure([
        go.Bar(x=chosen["bin_label"], y=chosen["share_baseline"], name="development (baseline)",
               marker_color=COLORS["series_1"]),
        go.Bar(x=chosen["bin_label"], y=chosen["share_monitored"], name="hold-out (monitored)",
               marker_color=COLORS["series_2"]),
    ])
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, config.VARIABLE_LABELS.get(variable, variable), "share of loans")
