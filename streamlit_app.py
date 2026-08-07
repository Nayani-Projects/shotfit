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
    .stApp {background: #F7F7F5; color: #172033;}
    [data-testid="stMetric"] {background: white; border: 1px solid #e7eaf0; padding: 1rem; border-radius: .8rem;}
    [data-testid="stMetricLabel"] {color: #687386;}
    .shotfit-kicker {text-transform: uppercase; letter-spacing: .12em; color: #687386; font-size: .78rem;}
    .shotfit-lead {font-size: 1.12rem; max-width: 820px; color: #364153;}
    .shotfit-intro {background: #f4f7fb; border-left: 4px solid #1d428a; border-radius: .55rem; padding: 1rem 1.15rem; margin-bottom: 1.2rem; color: #293548;}
    .shotfit-intro b {font-size: 1.05rem;}
    .shotfit-result {border-left: 4px solid #4678B8; padding: .25rem 0 .25rem 1rem; margin: .75rem 0 1.25rem;}
    .shotfit-result.neutral {border-left-color: #A8AFB8;}
    .shotfit-result.negative {border-left-color: #B55D42;}
    .shotfit-result-label {font-size: .8rem; font-weight: 600; color: #667085; margin-bottom: .25rem;}
    .shotfit-result-name {font-size: 2rem; font-weight: 600; color: #172033; margin-bottom: .35rem;}
    .shotfit-result-copy {font-size: 1.08rem; max-width: 780px; color: #364153;}
    .shotfit-review-box {background: #F2F5F8; padding: 1rem 1.2rem; border-radius: .55rem; margin-top: 1rem;}
    .shotfit-area-row {border-top: 1px solid #E2E5E9; padding: .8rem 0;}
    .shotfit-area-row:last-child {border-bottom: 1px solid #E2E5E9;}
    .shotfit-area-name {font-weight: 600; color: #172033;}
    .shotfit-area-note {color: #667085;}
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

review_tab, model_tab = st.tabs(["Player Review", "Methodology"])

result_options = {
    "All results": None,
    "Worth reviewing": "Strong positive evidence",
    "No clear signal": "Inconclusive evidence",
    "Potential concern": "Strong negative evidence",
}
public_result = {
    "Strong positive evidence": "Outperformed expectations",
    "Inconclusive evidence": "No clear difference",
    "Strong negative evidence": "Underperformed expectations",
}
result_class = {
    "Strong positive evidence": "",
    "Inconclusive evidence": "neutral",
    "Strong negative evidence": "negative",
}


def area_status(area):
    if area.evidence_label == "Strong positive evidence":
        return "Above expectation"
    if area.evidence_label == "Strong negative evidence":
        return "Below expectation"
    return "Near expectation"


def area_note(area, main_area):
    if area.shot_area == main_area:
        return "Main source of the result"
    if area.evidence_label == "Strong negative evidence":
        return "Check this area on film"
    if area.evidence_label == "Strong positive evidence":
        return "Also contributed to the result"
    return "No unusual difference"


with review_tab:
    st.header("Player review")
    st.caption("Select a player to see what happened, where it happened, and what to check on film.")
    filter_team, filter_position, filter_result, filter_player = st.columns([1.15, 1, 1.25, 1.5])
    with filter_team:
        team = st.selectbox("Team", ["All teams", *sorted(briefs.team_name.unique())], key="brief_team")
    team_pool = briefs if team == "All teams" else briefs[briefs.team_name == team]
    with filter_position:
        position = st.selectbox("Position", ["All positions", *sorted(team_pool.position.unique())], key="brief_position")
    position_pool = team_pool if position == "All positions" else team_pool[team_pool.position == position]
    with filter_result:
        available_results = {
            label: internal
            for label, internal in result_options.items()
            if internal is None or position_pool.evidence_label.eq(internal).any()
        }
        shooting_result = st.selectbox("Shooting result", list(available_results), index=0, key="brief_result")
    player_pool = position_pool
    if available_results[shooting_result] is not None:
        player_pool = player_pool[player_pool.evidence_label == available_results[shooting_result]]
    player_pool = player_pool.sort_values("lower_80", ascending=False)
    with filter_player:
        player = st.selectbox("Player", player_pool.player_name.tolist(), key="brief_player")
    st.caption(f"{len(player_pool):,} players match these filters.")
    with st.expander("Browse matching players"):
        browse = player_pool[["player_name", "team_name", "position", "attempts"]].copy()
        browse.columns = ["Player", "Team", "Position", "Shots reviewed"]
        st.dataframe(browse, hide_index=True, width="stretch", height=260)

    row = briefs.loc[briefs.player_name == player].iloc[0]
    direction = f"{row.extra_makes_per_100:.1f} more" if row.extra_makes_per_100 >= 0 else f"{abs(row.extra_makes_per_100):.1f} fewer"
    player_areas = areas[areas.player_id == row.player_id].copy().sort_values("attempts", ascending=False)
    review_area = player_areas.iloc[0].shot_area
    if row.evidence_label == "Strong positive evidence":
        main_area = row.strongest_supported_area
        result_sentence = f"Most of the difference came from {main_area.lower()} shooting."
    elif row.evidence_label == "Strong negative evidence":
        negative_areas = player_areas[player_areas.evidence_label == "Strong negative evidence"]
        main_area = (negative_areas if not negative_areas.empty else player_areas).sort_values("extra_makes_per_100").iloc[0].shot_area
        result_sentence = f"The largest shortfall came from {main_area.lower()} shooting."
    else:
        main_area = "No clear area"
        result_sentence = "The difference was not large enough to separate from normal shooting variation."
    review_area = main_area if main_area != "No clear area" else review_area
    main_copy = review_area.lower()
    st.markdown(
        f'<div class="shotfit-result {result_class[row.evidence_label]}">'
        f'<div class="shotfit-result-label">{public_result[row.evidence_label]}</div>'
        f'<div class="shotfit-result-name">{row.player_name}</div>'
        f'<div class="shotfit-result-copy">{row.player_name} made about <b>{direction} shots per 100 attempts</b> than expected. '
        f'{result_sentence}</div></div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Shooting difference", f"{row.extra_makes_per_100:+.1f}", "makes per 100 shots")
    m2.metric("Shots reviewed", f"{int(row.attempts):,}", evaluation_season)
    m3.metric("Main source", main_area, "shot area")

    st.subheader("Where the result came from")
    st.caption("Larger markers represent more shots. Blue areas finished above expectation. Orange areas finished below expectation.")
    player_bins = hex_bins[hex_bins.player_id == row.player_id]
    court_col, note_col = st.columns([1.7, 0.8])
    with court_col:
        st.plotly_chart(shot_translation_court(player_bins), width="stretch", config={"displayModeBar": False})
    with note_col:
        st.markdown("#### What stands out")
        if row.evidence_label == "Inconclusive evidence":
            st.markdown(
                "- No shot area showed a clear difference from expectation.\n"
                f"- {review_area} accounted for the largest share of attempts.\n"
                f"- {int(row.attempts):,} shots provide the sample for this review."
            )
        else:
            st.markdown(
                f"- {main_area} was the main source of the result.\n"
                f"- {row.shot_profile.replace('-heavy', '').title()} shots made up the largest relative part of his shot mix.\n"
                f"- {int(row.attempts):,} shots provide the sample for this review."
            )

    st.subheader("Results by shot area")
    for area in player_areas.itertuples():
        st.markdown(
            f'<div class="shotfit-area-row"><span class="shotfit-area-name">{area.shot_area}</span> &nbsp; '
            f'{int(area.attempts):,} shots &nbsp; <b>{area_status(area)}</b><br>'
            f'<span class="shotfit-area-note">{area_note(area, main_area)}</span></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Shot mix")
    if row.shot_profile == "Balanced":
        st.write(f"{row.player_name}'s shot mix was typical for a {row.position.lower()}.")
    else:
        mix_name = row.shot_profile.replace("-heavy", "").lower()
        st.write(f"{row.player_name} took a larger share of his shots from {mix_name} areas than most {row.position.lower()}s.")
    mix_rows = [
        ("At the rim", row.rim_share, row.rim_percentile),
        ("Midrange", row.midrange_share, row.midrange_percentile),
        ("Three-point", row.three_share, row.three_percentile),
    ]
    for label, share, percentile in mix_rows:
        label_col, bar_col, note_col = st.columns([1, 3, 1.4])
        label_col.write(f"**{label}**  {share:.0%}")
        bar_col.progress(float(share))
        comparison = "Very high" if percentile >= 0.75 else "Low" if percentile <= 0.25 else "Typical"
        note_col.caption(f"{comparison} for a {row.position.lower()}")

    st.markdown('<div class="shotfit-review-box">', unsafe_allow_html=True)
    st.subheader("What to check on film")
    st.markdown(
        f"- What types of {main_copy} attempts produced the result?\n"
        "- How often was the closest defender late or absent?\n"
        "- Did the player's balance and shot preparation hold against tighter contests?"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("View calculations"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Actual makes", f"{int(row.actual_makes):,}")
        c2.metric("Expected makes", f"{row.expected_makes:,.1f}")
        c3.metric("Likely range", f"{row.lower_80:+.1f} to {row.upper_80:+.1f}")
        st.caption("The range and shooting difference account for sample size. Lower-volume results are pulled closer to zero.")
    with st.expander("Data limitations"):
        st.write(
            "Public shot records do not include shot-level defender distance, pass quality, movement, balance, screen quality, play design, health, or internal role information. "
            "The results describe this season. They do not establish future performance, team fit, or what shots a player should take."
        )

with model_tab:
    st.header("Methodology")
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
