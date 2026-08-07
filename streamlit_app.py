"""ShotFit 2025–26 shot-making evidence board."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from shotfit.court import shot_translation_court

ROOT = Path(__file__).resolve().parent
APP_DATA = ROOT / "data" / "app"

st.set_page_config(page_title="ShotFit", page_icon="🏀", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 1.5rem;}
    [data-testid="stMetric"] {background: white; border: 1px solid #e7eaf0; padding: 1rem; border-radius: .8rem;}
    [data-testid="stMetricLabel"] {color: #687386;}
    .shotfit-kicker {text-transform: uppercase; letter-spacing: .12em; color: #687386; font-size: .78rem;}
    .shotfit-lead {font-size: 1.12rem; max-width: 820px; color: #364153;}
    .shotfit-intro {background: #f4f7fb; border-left: 4px solid #1d428a; border-radius: .55rem; padding: 1rem 1.15rem; margin-bottom: 1.2rem; color: #293548;}
    .shotfit-intro b {font-size: 1.05rem;}
    .shotfit-footer {border-top: 1px solid #e7eaf0; margin-top: 2rem; padding-top: 1rem; color: #687386; font-size: .82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(bundle_signature: tuple[int, ...]):
    required = [
        "player_briefs.parquet",
        "player_areas.parquet",
        "player_hex_bins.parquet",
        "validation_summary.json",
        "monitoring.json",
        "model_metadata.json",
        "evidence_standards.json",
    ]
    missing = [name for name in required if not (APP_DATA / name).exists()]
    if missing:
        st.error(f"App bundle missing: {', '.join(missing)}. Run `uv run python -m shotfit.cli export-app`.")
        st.stop()
    briefs = pd.read_parquet(APP_DATA / "player_briefs.parquet")
    areas = pd.read_parquet(APP_DATA / "player_areas.parquet")
    hex_bins = pd.read_parquet(APP_DATA / "player_hex_bins.parquet")
    metrics = json.loads((APP_DATA / "validation_summary.json").read_text())
    monitoring = json.loads((APP_DATA / "monitoring.json").read_text())
    metadata = json.loads((APP_DATA / "model_metadata.json").read_text())
    standards = json.loads((APP_DATA / "evidence_standards.json").read_text())
    return briefs, areas, hex_bins, metrics, monitoring, metadata, standards


bundle_files = [
    "player_briefs.parquet",
    "player_areas.parquet",
    "player_hex_bins.parquet",
    "validation_summary.json",
    "monitoring.json",
    "model_metadata.json",
    "evidence_standards.json",
]
bundle_signature = tuple((APP_DATA / name).stat().st_mtime_ns for name in bundle_files)
briefs, areas, hex_bins, metrics, monitoring, metadata, standards = load_data(bundle_signature)
evaluation_season = metadata["evaluation_season"]
validation_season = metadata["validation_season"]

st.markdown("### SHOTFIT")
st.caption(f"{evaluation_season} shot-making evidence board · public NBA regular season data")
st.markdown(
    "Which players produced the strongest evidence of shot-making above or below expectation, "
    "after accounting for where and how they shot?"
)

board_tab, brief_tab, model_tab = st.tabs(["Evidence Board", "Player Brief", "Model & Validation"])

with board_tab:
    st.header("Players worth a closer look")
    st.markdown(
        f'<div class="shotfit-intro"><b>Find unusual shooting results from the {evaluation_season} season.</b><br>'
        "ShotFit compares each player's makes with what an average NBA shooter would be expected to make from the same shot locations, shot types, and game situations.</div>",
        unsafe_allow_html=True,
    )
    st.caption("This is a starting point for review, not a player ranking. Use it to decide where film, tracking data, and scouting context are needed.")
    positive_count = int(briefs.evidence_label.eq("Strong positive evidence").sum())
    inconclusive_count = int(briefs.evidence_label.eq("Inconclusive evidence").sum())
    negative_count = int(briefs.evidence_label.eq("Strong negative evidence").sum())
    p1, p2, p3 = st.columns(3)
    p1.metric("Worth reviewing", positive_count, "Above expectation")
    p2.metric("No clear signal", inconclusive_count, "Could be normal variation")
    p3.metric("Potential concern", negative_count, "Below expectation")

    with st.expander("What do these groups mean?"):
        st.markdown(
            "- **Worth reviewing:** shooting results were clearly above what the shot difficulty model expected.\n"
            "- **No clear signal:** the available shots do not clearly separate performance from normal shooting variation.\n"
            "- **Potential concern:** shooting results were clearly below what the model expected.\n\n"
            "These groups describe shooting results, not overall player quality or future performance."
        )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        board_team = st.selectbox("Team", ["All teams", *sorted(briefs.team_name.unique())], key="board_team")
    with f2:
        board_position = st.selectbox("Position", ["All positions", *sorted(briefs.position.unique())], key="board_position")
    with f3:
        result_options = {
            "All results": None,
            "Outperformed expectations": "Strong positive evidence",
            "No clear difference": "Inconclusive evidence",
            "Underperformed expectations": "Strong negative evidence",
        }
        board_result = st.selectbox("Shooting result", list(result_options), key="board_result")
    with f4:
        board_profile = st.selectbox("Shot profile", ["All profiles", *sorted(briefs.shot_profile.unique())], key="board_profile")
    sort_options = {
        "Strongest evidence": ("lower_80", False),
        "Most shots reviewed": ("attempts", False),
        "Player name": ("player_name", True),
        "Team": ("team_name", True),
    }
    board_minimum = 250
    with st.expander("More filters"):
        extra1, extra2 = st.columns(2)
        with extra1:
            board_minimum = st.select_slider("Minimum shots reviewed", options=[250, 400, 600, 800, 1000], value=250)
        with extra2:
            board_sort = st.selectbox("Sort by", list(sort_options), key="board_sort")
    board = briefs[briefs.attempts >= board_minimum].copy()
    if board_team != "All teams":
        board = board[board.team_name == board_team]
    if board_position != "All positions":
        board = board[board.position == board_position]
    if result_options[board_result] is not None:
        board = board[board.evidence_label == result_options[board_result]]
    if board_profile != "All profiles":
        board = board[board.shot_profile == board_profile]
    sort_column, sort_ascending = sort_options[board_sort]
    board = board.sort_values(sort_column, ascending=sort_ascending)
    public_result = {
        "Strong positive evidence": "Worth reviewing",
        "Inconclusive evidence": "No clear signal",
        "Strong negative evidence": "Potential concern",
    }
    def board_reason(row):
        if row.evidence_label == "Inconclusive evidence":
            return f"No clear difference across {int(row.attempts):,} shots"
        direction = "Above" if row.evidence_label == "Strong positive evidence" else "Below"
        if row.strongest_supported_area != "No area with conclusive positive evidence":
            return f"{direction} expectation; strongest support from {row.strongest_supported_area.lower()}"
        return f"{direction} expectation on a {row.shot_profile.lower()} profile"
    board_display = board.assign(
        result=board.evidence_label.map(public_result),
        reason=board.apply(board_reason, axis=1),
    )[["player_name", "team_name", "position", "reason", "attempts", "result"]]
    board_display.columns = ["Player", "Team", "Position", "Why the player surfaced", "Shots reviewed", "Review group"]
    st.dataframe(board_display, hide_index=True, width="stretch", height=500)
    st.caption(f"Showing {len(board_display):,} of {len(briefs):,} qualified players. Choose a player in Player Brief to see the full evidence trail and likely shooting range.")

with brief_tab:
    st.header("Player Brief")
    st.caption(f"Results from the {evaluation_season} regular season. This is not a forecast or personnel grade.")
    filter_team, filter_position, filter_player = st.columns([1.2, 1, 1.5])
    with filter_team:
        team = st.selectbox("Team", ["All teams", *sorted(briefs.team_name.unique())], key="brief_team")
    team_pool = briefs if team == "All teams" else briefs[briefs.team_name == team]
    with filter_position:
        position = st.selectbox("Position", ["All positions", *sorted(team_pool.position.unique())], key="brief_position")
    player_pool = team_pool if position == "All positions" else team_pool[team_pool.position == position]
    with filter_player:
        player = st.selectbox("Player", sorted(player_pool.player_name), key="brief_player")
    row = briefs.loc[briefs.player_name == player].iloc[0]
    direction = f"{row.extra_makes_per_100:.1f} more" if row.extra_makes_per_100 >= 0 else f"{abs(row.extra_makes_per_100):.1f} fewer"
    st.caption(f"{row.team_name} · {row.position} · {evaluation_season} regular season")
    st.markdown('<div class="shotfit-kicker">Evidence summary</div>', unsafe_allow_html=True)
    brief_result = {
        "Strong positive evidence": "Outperformed expectations",
        "Inconclusive evidence": "No clear difference",
        "Strong negative evidence": "Underperformed expectations",
    }[row.evidence_label]
    st.header(brief_result)
    st.markdown(
        f'<div class="shotfit-lead">{row.player_name} made about <b>{direction} shots per 100 attempts</b> than expected based on shot location, shot type, and game situation. The likely range was <b>{row.lower_80:+.1f} to {row.upper_80:+.1f}</b>.</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Shooting difference", f"{row.extra_makes_per_100:+.1f}", "makes per 100 shots")
    m2.metric("Likely range", f"{row.lower_80:+.1f} to {row.upper_80:+.1f}")
    m3.metric("Shots reviewed", f"{int(row.attempts):,}", evaluation_season)
    a1, a2, a3 = st.columns(3)
    a1.metric("Actual makes", f"{int(row.actual_makes):,}")
    a2.metric("Expected makes", f"{row.expected_makes:,.1f}")
    a3.metric("Adjusted extra makes", f"{row.adjusted_extra_makes:+.1f}", "total")

    st.subheader("Shot distribution")
    st.markdown(f"**{row.shot_profile}** · {row.profile_statement}")
    distribution = pd.DataFrame(
        {
            "Area": ["Rim", "Midrange", "Three-point"],
            "Share of shots": [row.rim_share, row.midrange_share, row.three_share],
            "Position percentile": [row.rim_percentile, row.midrange_percentile, row.three_percentile],
        }
    )
    st.dataframe(distribution.style.format({"Share of shots": "{:.1%}", "Position percentile": "{:.0%}"}), hide_index=True, width="stretch")

    st.subheader("Where the result came from")
    st.caption("Hex size shows shot volume. Blue locations finished above expectation; rust locations finished below expectation.")
    player_bins = hex_bins[hex_bins.player_id == row.player_id]
    st.plotly_chart(shot_translation_court(player_bins), width="stretch", config={"displayModeBar": False})
    player_areas = areas[areas.player_id == row.player_id].copy().sort_values("attempts", ascending=False)
    area_display = player_areas.assign(
        range_80=player_areas.apply(lambda area: f"{area.lower_80:+.1f} to {area.upper_80:+.1f}", axis=1),
        adjusted=player_areas.extra_makes_per_100.map(lambda value: f"{value:+.1f}"),
    )[["shot_area", "attempts", "shot_share", "position_frequency_percentile", "actual_makes", "expected_makes", "adjusted", "range_80", "evidence_label"]]
    area_display.columns = ["Shot area", "Attempts", "Shot share", "Position frequency percentile", "Actual makes", "Expected makes", "Adjusted extra/100", "80% range", "Evidence"]
    st.dataframe(area_display.style.format({"Shot share": "{:.1%}", "Position frequency percentile": "{:.0%}", "Actual makes": "{:.0f}", "Expected makes": "{:.1f}"}), hide_index=True, width="stretch")
    if row.strongest_supported_area != "No area with conclusive positive evidence":
        st.success(f"Strongest supported area: {row.strongest_supported_area}.")
    else:
        st.info(row.strongest_supported_area + ".")
    if row.review_flag != "No high-volume area had strong negative evidence.":
        st.warning("Review flag: " + row.review_flag)
    else:
        st.caption(row.review_flag)
    with st.expander("Limits of this evidence"):
        st.write(
            "Public shot records do not include shot-level defender distance, pass quality, movement, balance, screen quality, play design, health, or internal role information. "
            "The estimates describe shot-making relative to observable context in this sample; they do not establish future performance, team fit, or what shots a player should take."
        )

with model_tab:
    st.header("Model & Validation")
    st.subheader("Decision question")
    st.write(
        f"Which players produced the strongest evidence of shot-making above or below expectation in {evaluation_season}, and where did the difference originate?"
    )
    st.write(
        "Expected makes are summed probabilities from a shot-context model that excludes player and team identity. Public player estimates use only the untouched test season."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training", metrics["seasons"]["train"])
    c2.metric("Validation", validation_season)
    c3.metric("Public evidence", evaluation_season)
    c4.metric("Selected model", "Logistic")

    st.subheader("Model comparison")
    comparison = pd.DataFrame([{"Model": name, **values} for name, values in metrics["validation_models"].items()])
    st.dataframe(comparison.rename(columns={"log_loss": "Log loss", "brier_score": "Brier score", "roc_auc": "ROC AUC", "calibration_error": "Calibration error"}).style.format({"Log loss": "{:.4f}", "Brier score": "{:.4f}", "ROC AUC": "{:.4f}", "Calibration error": "{:.4f}"}), hide_index=True, width="stretch")
    st.caption(metrics["selection_rule"])
    test = metrics["test_metrics"]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Test log loss", f"{test['log_loss']:.4f}")
    t2.metric("Test Brier score", f"{test['brier_score']:.4f}")
    t3.metric("Test ROC AUC", f"{test['roc_auc']:.3f}")
    t4.metric("Calibration error", f"{test['calibration_error']:.3f}")
    calibration = pd.DataFrame(metrics["calibration"])
    line = alt.Chart(calibration).mark_line(point=True, color="#1D428A").encode(x=alt.X("predicted:Q", title="Predicted make probability", scale=alt.Scale(domain=[0.2, 0.8])), y=alt.Y("observed:Q", title="Observed make rate", scale=alt.Scale(domain=[0.2, 0.8])), tooltip=[alt.Tooltip("predicted:Q", format=".3f"), alt.Tooltip("observed:Q", format=".3f"), "shots:Q"])
    perfect = alt.Chart(pd.DataFrame({"x": [0.2, 0.8], "y": [0.2, 0.8]})).mark_line(strokeDash=[5, 5], color="#9AA3B2").encode(x="x:Q", y="y:Q")
    st.altair_chart((perfect + line).properties(height=280), width="stretch")

    st.subheader("Evidence standards")
    st.write(standards["label_rule"])
    st.caption(standards["profile_rule"])
    threshold_table = pd.DataFrame(standards["threshold_analysis"]).rename(columns={"minimum_attempts": "Minimum attempts", "eligible_players": "Eligible players", "median_interval_width": "Median 80% width", "conclusive_share": "Conclusive share", "split_half_correlation": "Split-half correlation", "selected": "Selected"})
    st.dataframe(threshold_table.style.format({"Median 80% width": "{:.2f}", "Conclusive share": "{:.1%}", "Split-half correlation": "{:.2f}"}), hide_index=True, width="stretch")
    st.caption(f"The {metadata['minimum_test_attempts']}-attempt standard was selected on {validation_season} before applying labels to {evaluation_season}; it improves stability over 150 attempts while retaining substantially broader player coverage than 400 or 600.")
    sensitivity = pd.DataFrame(standards["interval_sensitivity"]).rename(columns={"interval": "Interval", "strong_positive": "Strong positive", "inconclusive": "Inconclusive", "strong_negative": "Strong negative"})
    st.dataframe(sensitivity, hide_index=True, width="stretch")
    st.caption("The 80% interval is used for screening. The sensitivity table shows how classifications become more conservative at 90% and 95%.")

    st.subheader("Sample-size adjustment")
    st.write("Actual-minus-expected rates are stabilized with an empirical-Bayes normal-normal model. Lower-volume results move more toward zero and receive wider intervals; larger samples retain more of the observed difference.")
    st.subheader("What ShotFit does not claim")
    st.markdown("- It does not forecast next-season shooting.\n- It does not recommend a role, acquisition, or shot-selection change.\n- It does not measure defender distance, pass quality, movement, balance, play design, health, or internal scouting context.")
    st.subheader("Operations")
    o1, o2, o3 = st.columns(3)
    o1.metric("Model version", metadata["model_version"])
    o2.metric("Qualified players", metadata["qualified_players"])
    o3.metric("Drift status", monitoring["status"].title())
    st.code("nba_api → gzip raw cache → DuckDB → features → model → batch evidence → app", language=None)
    st.caption("The public app loads only precomputed Parquet and JSON files and makes no runtime NBA.com requests.")

st.markdown('<div class="shotfit-footer">A starting point for film and tracking review. Not a forecast, role recommendation, or personnel grade.</div>', unsafe_allow_html=True)
