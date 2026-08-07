"""ShotFit public decision brief and model-validation application."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from shotfit.court import shot_translation_court

ROOT = Path(__file__).resolve().parent
APP_DATA = ROOT / "data" / "app"
FAMILY_COLUMNS = {
    "At the rim": "rim_extra",
    "Midrange": "midrange_extra",
    "Corner three": "corner_three_extra",
    "Above the break": "above_break_extra",
}

st.set_page_config(page_title="ShotFit", page_icon="🏀", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1080px; padding-top: 1.5rem;}
    [data-testid="stMetric"] {background: white; border: 1px solid #e7eaf0; padding: 1rem; border-radius: .8rem;}
    [data-testid="stMetricLabel"] {color: #687386;}
    .shotfit-kicker {text-transform: uppercase; letter-spacing: .12em; color: #687386; font-size: .78rem;}
    .shotfit-lead {font-size: 1.12rem; max-width: 760px; color: #364153;}
    .shotfit-role {border-left: 4px solid #1D428A; padding-left: 1rem;}
    .shotfit-footer {border-top: 1px solid #e7eaf0; margin-top: 2rem; padding-top: 1rem; color: #687386; font-size: .82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(bundle_signature: tuple[int, ...]):
    required = ["player_briefs.parquet", "player_hex_bins.parquet", "validation_summary.json", "monitoring.json", "model_metadata.json"]
    missing = [name for name in required if not (APP_DATA / name).exists()]
    if missing:
        st.error(f"App bundle missing: {', '.join(missing)}. Run `uv run python -m shotfit.cli export-app`.")
        st.stop()
    briefs = pd.read_parquet(APP_DATA / "player_briefs.parquet")
    hex_bins = pd.read_parquet(APP_DATA / "player_hex_bins.parquet")
    metrics = json.loads((APP_DATA / "validation_summary.json").read_text())
    monitoring = json.loads((APP_DATA / "monitoring.json").read_text())
    metadata = json.loads((APP_DATA / "model_metadata.json").read_text())
    return briefs, hex_bins, metrics, monitoring, metadata


bundle_files = ["player_briefs.parquet", "player_hex_bins.parquet", "validation_summary.json", "monitoring.json", "model_metadata.json"]
bundle_signature = tuple((APP_DATA / name).stat().st_mtime_ns for name in bundle_files)
briefs, hex_bins, metrics, monitoring, metadata = load_data(bundle_signature)
evaluation_season = metadata["evaluation_season"]
supporting_season = metadata["supporting_season"]
st.markdown("### SHOTFIT")
st.caption(f"Shooting translation brief · {evaluation_season} NBA regular season evaluation")

brief_tab, model_tab = st.tabs(["Basketball Brief", "Model & Validation"])

with brief_tab:
    st.info(
        f"Player estimates use {supporting_season} validation and {evaluation_season} evaluation shots. "
        f"Public eligibility and the team shown below are based on the {evaluation_season} regular season."
    )
    filter_team, filter_position, filter_player = st.columns([1.2, 1, 1.5])
    with filter_team:
        team = st.selectbox(f"Team ({evaluation_season})", ["All teams", *sorted(briefs.team_name.unique())])
    team_pool = briefs if team == "All teams" else briefs[briefs.team_name == team]
    with filter_position:
        position = st.selectbox("Position", ["All positions", *sorted(team_pool.position.unique())])
    player_pool = team_pool if position == "All positions" else team_pool[team_pool.position == position]
    with filter_player:
        player = st.selectbox("Player", sorted(player_pool.player_name), index=0)
    row = briefs.loc[briefs.player_name == player].iloc[0]
    st.caption(f"{row.team_name} · {row.position} · {evaluation_season} evaluation season")
    st.markdown('<div class="shotfit-kicker">Bottom line</div>', unsafe_allow_html=True)
    st.header(row.bottom_line)
    st.markdown(
        f'<div class="shotfit-lead">{row.player_name} produced <b>{row.extra_makes_per_100:+.1f} extra makes per 100 shots</b> compared with an average shooter receiving the same observable public shot profile.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**{row.confidence}**")
    left, middle, right = st.columns(3)
    left.metric("Extra makes", f"{row.extra_makes_per_100:+.1f}", "per 100 shots")
    middle.metric("Shots reviewed", f"{int(row.attempts):,}", f"across {int(row.seasons_reviewed)} out-of-sample seasons")
    right.metric("Repeated across areas?", row.repeat_label, f"positive in {int(row.positive_families)} of 4")
    st.subheader("Where the signal comes from")
    st.caption("Hex size shows shot volume. Blue locations finished above expectation; rust locations finished below expectation.")
    player_bins = hex_bins[hex_bins.player_id == row.player_id]
    st.plotly_chart(shot_translation_court(player_bins), width="stretch", config={"displayModeBar": False})
    with st.expander("View shot-area values as a table"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Shot area": list(FAMILY_COLUMNS),
                    "Extra makes per 100": [float(row[column]) for column in FAMILY_COLUMNS.values()],
                    "Shots": [int(row[f"{column.removesuffix('_extra')}_attempts"]) for column in FAMILY_COLUMNS.values()],
                }
            ).style.format({"Extra makes per 100": "{:+.1f}", "Shots": "{:,}"}),
            hide_index=True,
            width="stretch",
        )
    st.caption(row.strongest_evidence)
    role_col, next_col = st.columns(2, gap="large")
    with role_col:
        st.markdown('<div class="shotfit-role">', unsafe_allow_html=True)
        st.subheader("Best-supported role to investigate")
        st.markdown(f"**{row.role}**")
        st.write(row.role_description)
        st.markdown("</div>", unsafe_allow_html=True)
    with next_col:
        st.subheader("What staff should check next")
        st.markdown("1. **Film:** Does the shot profile survive movement and tighter release windows?\n2. **Tracking:** How much comes from defender distance and pass quality?\n3. **Scouting:** Does the suggested role match his decisions and comfort?")
    with st.expander("Confidence and limitations"):
        st.write(f"**Likely range:** {row.lower_80:+.1f} to {row.upper_80:+.1f} extra makes per 100 shots")
        st.write("Public shot records do not provide shot-level defender distance, pass quality, player balance, exact play design, health, or internal scouting context. Use this brief to focus film and tracking review—not as a final personnel grade.")

with model_tab:
    st.header("Model & Validation")
    st.write("ShotFit predicts whether an NBA field-goal attempt is made using only observable information available when the shot occurs. Player and team identity are excluded so the estimate represents an average shooter receiving the same public shot profile.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training", metrics["seasons"]["train"])
    c2.metric("Validation", metrics["seasons"]["validation"])
    c3.metric("Untouched test", metrics["seasons"]["test"])
    c4.metric("Selected model", "Logistic")
    st.subheader("Does complexity earn its place?")
    comparison = pd.DataFrame([{"Model": name, **values} for name, values in metrics["validation_models"].items()])
    st.dataframe(comparison.rename(columns={"log_loss": "Log loss", "brier_score": "Brier score", "roc_auc": "ROC AUC", "calibration_error": "Calibration error"}).style.format({"Log loss": "{:.4f}", "Brier score": "{:.4f}", "ROC AUC": "{:.4f}", "Calibration error": "{:.4f}"}), hide_index=True, width="stretch")
    st.caption(metrics["selection_rule"])
    st.subheader(f"Untouched {evaluation_season} results")
    test = metrics["test_metrics"]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Log loss", f"{test['log_loss']:.4f}")
    t2.metric("Brier score", f"{test['brier_score']:.4f}")
    t3.metric("ROC AUC", f"{test['roc_auc']:.3f}")
    t4.metric("Calibration error", f"{test['calibration_error']:.3f}")
    calibration = pd.DataFrame(metrics["calibration"])
    line = alt.Chart(calibration).mark_line(point=True, color="#1D428A").encode(x=alt.X("predicted:Q", title="Predicted make probability", scale=alt.Scale(domain=[0.2, 0.8])), y=alt.Y("observed:Q", title="Observed make rate", scale=alt.Scale(domain=[0.2, 0.8])), tooltip=[alt.Tooltip("predicted:Q", format=".3f"), alt.Tooltip("observed:Q", format=".3f"), "shots:Q"])
    perfect = alt.Chart(pd.DataFrame({"x": [0.2, 0.8], "y": [0.2, 0.8]})).mark_line(strokeDash=[5, 5], color="#9AA3B2").encode(x="x:Q", y="y:Q")
    st.altair_chart((perfect + line).properties(height=300), width="stretch")
    st.subheader("Sample-size adjustment")
    st.write("Small samples move toward the league average more strongly. Large samples retain more of the observed result. ShotFit reports the adjusted estimate and an 80% likely range rather than treating every hot or cold stretch as skill.")
    st.subheader("Production flow")
    st.code("nba_api → gzip raw cache → validated DuckDB → features → model → batch scores → app bundle", language=None)
    st.subheader("Operations")
    o1, o2, o3 = st.columns(3)
    o1.metric("Model version", metadata["model_version"])
    o2.metric("Qualified players", metadata["qualified_players"])
    o3.metric("Drift status", monitoring["status"].title())
    st.caption("The public app makes no runtime network calls. Drift is informational in v1; standardized feature shifts ≥0.20 or mean prediction shifts ≥0.03 trigger review.")
    st.markdown("**Known limits:** no shot-level defender distance, pass quality, movement, balance, play design, medical context, or internal scouting data.")

st.markdown('<div class="shotfit-footer">Use with film, tracking, and scouting context · Decision support, not a final personnel grade</div>', unsafe_allow_html=True)
