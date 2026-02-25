"""
FPL 2024/25 Intelligence Dashboard
A comprehensive Streamlit dashboard for Fantasy Premier League data analysis,
powered by a V3 Position-Specific Meta-Ensemble with 3-GW average target.
4 positions x (XGBoost + RF + MLP + LSTM -> Meta-XGBoost)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
import os

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FPL Intelligence",
    page_icon="\u26bd",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
PLAYERS_CSV = os.path.join(BASE, "data", "processed", "players_clean.csv")
HISTORIES_CSV = os.path.join(BASE, "data", "raw", "player_histories.csv")
TEAMS_CSV = os.path.join(BASE, "data", "raw", "teams_raw.csv")
FIXTURES_CSV = os.path.join(BASE, "data", "raw", "fixtures_raw.csv")
PLAYERS_RAW_CSV = os.path.join(BASE, "data", "raw", "players_raw.csv")
DATA_JSON = os.path.join(BASE, "output", "data.json")
MODEL_JSON = os.path.join(BASE, "output", "model_data.json")
MODEL_PKL = os.path.join(BASE, "output", "model.pkl")
CHARTS_DIR = os.path.join(BASE, "output", "charts")

# ---------------------------------------------------------------------------
# FPL COLOUR PALETTE
# ---------------------------------------------------------------------------
FPL_PURPLE = "#37003c"
FPL_GREEN = "#00ff87"
FPL_DARK = "#1a1a2e"
FPL_LIGHT_PURPLE = "#6c2b7e"
FPL_TEAL = "#04f5ff"
POS_COLORS = {
    "Forward": "#e74c3c",
    "Midfielder": "#2ecc71",
    "Defender": "#3498db",
    "Goalkeeper": "#f39c12",
}

# Model colors for arena page
MODEL_COLORS = {
    "xgboost": "#00ff87",
    "random_forest": "#04f5ff",
    "neural_net": "#ff6384",
    "lstm": "#ffce56",
    "simple_avg": "#9966ff",
    "stacked_meta": "#ff9f40",
}

# ---------------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {FPL_PURPLE};
    }}
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label {{
        color: white !important;
    }}
    /* KPI card */
    .kpi-card {{
        background: linear-gradient(135deg, {FPL_PURPLE}, {FPL_LIGHT_PURPLE});
        border-left: 4px solid {FPL_GREEN};
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 12px;
        color: white;
    }}
    .kpi-card .kpi-label {{
        font-size: 0.85rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }}
    .kpi-card .kpi-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {FPL_GREEN};
    }}
    .kpi-card .kpi-sub {{
        font-size: 0.8rem;
        opacity: 0.7;
    }}
    /* Player card */
    .player-card {{
        background: linear-gradient(135deg, {FPL_PURPLE}, #2d1b4e);
        border: 2px solid {FPL_GREEN};
        border-radius: 14px;
        padding: 24px;
        color: white;
    }}
    .player-card h2 {{
        color: {FPL_GREEN} !important;
        margin-bottom: 2px;
    }}
    .player-card .stat-row {{
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }}
    .player-card .stat-label {{
        opacity: 0.7;
        font-size: 0.9rem;
    }}
    .player-card .stat-value {{
        font-weight: 600;
        color: {FPL_GREEN};
    }}
    /* Team badge */
    .team-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        background: {FPL_GREEN};
        color: {FPL_PURPLE};
        font-weight: 700;
        font-size: 0.85rem;
    }}
    /* Strategy cards */
    .strategy-card {{
        background: linear-gradient(135deg, {FPL_PURPLE}, {FPL_LIGHT_PURPLE});
        border-radius: 12px;
        padding: 20px;
        color: white;
        border-top: 3px solid {FPL_GREEN};
    }}
    .strategy-card h3 {{
        color: {FPL_GREEN} !important;
    }}
    /* Model arena card */
    .arena-card {{
        background: linear-gradient(135deg, {FPL_DARK}, {FPL_PURPLE});
        border-radius: 12px;
        padding: 20px;
        color: white;
        border-top: 3px solid;
        text-align: center;
        margin-bottom: 12px;
    }}
    .arena-card .arena-model-name {{
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .arena-card .arena-metric {{
        font-size: 0.8rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .arena-card .arena-value {{
        font-size: 1.5rem;
        font-weight: 700;
    }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# DATA LOADING (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_players():
    df = pd.read_csv(PLAYERS_CSV)
    numeric_cols = [
        "now_cost", "total_points", "goals_scored", "assists", "minutes",
        "clean_sheets", "goals_conceded", "bonus", "bps", "form",
        "points_per_game", "selected_by_percent", "ict_index", "influence",
        "creativity", "threat", "transfers_in", "transfers_out",
        "yellow_cards", "red_cards", "saves", "starts",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "value",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data
def load_histories():
    return pd.read_csv(HISTORIES_CSV)


@st.cache_data
def load_teams():
    return pd.read_csv(TEAMS_CSV)


@st.cache_data
def load_fixtures():
    return pd.read_csv(FIXTURES_CSV)


@st.cache_data
def load_players_raw():
    return pd.read_csv(PLAYERS_RAW_CSV)


@st.cache_data
def load_data_json():
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_model_json():
    with open(MODEL_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    """
    Load model.pkl — supports V3 (position-specific), V1 (stacked meta), and legacy formats.

    V3 format:
        {
            'model_version': 'v3',
            'positions': {
                'GK': {'xgboost': ..., 'random_forest': ..., 'meta_model': ..., ...},
                'DEF': {...}, 'MID': {...}, 'FWD': {...},
            },
            'features': [...],
            'meta_features': [...],
            'target_type': '3gw_avg',
            'min_minutes': 30,
        }
    """
    try:
        with open(MODEL_PKL, "rb") as f:
            data = pickle.load(f)

        # V3 position-specific format
        if data.get("model_version") == "v3":
            return data
        # V1 stacked meta-ensemble format
        if "meta_model" in data:
            return data
        # Legacy single-model format
        if "model" in data:
            return {
                "_legacy": True,
                "xgboost": data["model"],
                "features": data["features"],
                "params": data.get("params", {}),
            }
        return None
    except Exception:
        return None


def get_base_xgb_model():
    """Return an XGBoost base model and its feature list for predictions."""
    model_data = load_model()
    if model_data is None:
        return None, None
    # V3: use MID position model as default (largest dataset)
    if model_data.get("model_version") == "v3":
        positions = model_data.get("positions", {})
        mid_models = positions.get("MID", positions.get(list(positions.keys())[0] if positions else "MID", {}))
        return mid_models.get("xgboost"), model_data.get("features")
    # V1
    xgb_model = model_data.get("xgboost")
    features = model_data.get("features")
    return xgb_model, features


# ---------------------------------------------------------------------------
# HELPER: build features for a player (for prediction)
# ---------------------------------------------------------------------------
def build_player_features(
    player_row, hist_df, teams_raw, players_raw_df, fixtures_df,
    model_features, gw_number
):
    """
    Replicate improve_model.py feature engineering for a single player
    at a given future gameweek.
    """
    pid = player_row["id"]
    player_hist = hist_df[hist_df["player_id"] == pid].sort_values("round")
    if len(player_hist) < 3:
        return None

    player_to_team_id = dict(zip(players_raw_df["id"], players_raw_df["team"]))
    team_id = player_to_team_id.get(pid)
    if team_id is None:
        return None

    team_strength_attack_h = dict(zip(teams_raw["id"], teams_raw["strength_attack_home"]))
    team_strength_attack_a = dict(zip(teams_raw["id"], teams_raw["strength_attack_away"]))
    team_strength_defense_h = dict(zip(teams_raw["id"], teams_raw["strength_defence_home"]))
    team_strength_defense_a = dict(zip(teams_raw["id"], teams_raw["strength_defence_away"]))
    team_strength_overall_h = dict(zip(teams_raw["id"], teams_raw["strength_overall_home"]))
    team_strength_overall_a = dict(zip(teams_raw["id"], teams_raw["strength_overall_away"]))

    # Find fixture
    upcoming = fixtures_df[
        (fixtures_df["finished"] == False) & (fixtures_df["event"].notna())
    ].copy()
    if len(upcoming) == 0:
        return None
    upcoming["event"] = upcoming["event"].astype(int)
    gw_fixtures = upcoming[upcoming["event"] == gw_number]

    home_fix = gw_fixtures[gw_fixtures["team_h"] == team_id]
    away_fix = gw_fixtures[gw_fixtures["team_a"] == team_id]

    if len(home_fix) > 0:
        is_home = 1
        opp_id = int(home_fix.iloc[0]["team_a"])
    elif len(away_fix) > 0:
        is_home = 0
        opp_id = int(away_fix.iloc[0]["team_h"])
    else:
        return None

    row = {}

    # Rolling features for windows 3, 5, 8
    for w in [3, 5, 8]:
        recent = player_hist.tail(w)
        for col in [
            "total_points", "minutes", "goals_scored", "assists", "bonus",
            "bps", "ict_index", "influence", "creativity", "threat",
            "expected_goals", "expected_assists", "expected_goal_involvements",
            "clean_sheets", "goals_conceded", "saves",
        ]:
            if col in recent.columns:
                row[f"{col}_roll_{w}"] = recent[col].mean()

    # Consistency
    for w in [3, 5]:
        recent = player_hist.tail(w)
        pts_vals = recent["total_points"]
        std_val = pts_vals.std() if len(pts_vals) > 1 else 0
        row[f"pts_std_{w}"] = std_val
        roll_key = f"total_points_roll_{w}"
        mean_val = row.get(roll_key, 0)
        row[f"pts_cv_{w}"] = std_val / max(mean_val, 0.5) if mean_val else 0

    # Momentum
    row["momentum_3v8"] = row.get("total_points_roll_3", 0) - row.get("total_points_roll_8", 0)
    row["xg_momentum"] = row.get("expected_goals_roll_3", 0) - row.get("expected_goals_roll_5", 0)

    # Per-90 stats
    min5 = row.get("minutes_roll_5", 1)
    min5 = max(min5, 1)
    for col in ["goals_scored", "assists", "bonus", "expected_goals", "expected_assists"]:
        key5 = f"{col}_roll_5"
        row[f"{col}_per90"] = (row.get(key5, 0) / min5) * 90

    # Season cumulative
    row["season_pts"] = player_hist["total_points"].sum()
    row["season_goals"] = player_hist["goals_scored"].sum()
    row["season_assists"] = player_hist["assists"].sum()
    row["season_minutes"] = player_hist["minutes"].sum()
    row["season_xg"] = player_hist["expected_goals"].sum() if "expected_goals" in player_hist.columns else 0
    row["games_played"] = len(player_hist)
    row["ppg"] = round(row["season_pts"] / max(row["games_played"], 1), 2)

    # xG outperformance
    row["xg_overperformance"] = row.get("goals_scored_roll_5", 0) - row.get("expected_goals_roll_5", 0)
    row["xa_overperformance"] = row.get("assists_roll_5", 0) - row.get("expected_assists_roll_5", 0)

    # Home/away
    row["is_home"] = is_home

    # Played last
    if len(player_hist) > 0:
        row["played_last"] = 1 if player_hist.iloc[-1]["minutes"] > 0 else 0
    else:
        row["played_last"] = 0

    # Minutes trend
    row["minutes_trend"] = row.get("minutes_roll_3", 0) - row.get("minutes_roll_8", 0)

    # Transfer trend
    if "transfers_balance" in player_hist.columns:
        recent3 = player_hist.tail(3)
        row["transfer_trend"] = recent3["transfers_balance"].mean()
    else:
        row["transfer_trend"] = 0

    # Ownership change
    if "selected" in player_hist.columns and len(player_hist) >= 2:
        last_sel = player_hist["selected"].iloc[-1]
        prev_sel = player_hist["selected"].iloc[-2]
        row["ownership_change"] = (last_sel - prev_sel) / max(prev_sel, 1)
    else:
        row["ownership_change"] = 0

    # Opponent/own team strength
    row["opp_attack"] = (
        team_strength_attack_a.get(opp_id, 1100) if is_home
        else team_strength_attack_h.get(opp_id, 1100)
    )
    row["opp_defense"] = (
        team_strength_defense_a.get(opp_id, 1100) if is_home
        else team_strength_defense_h.get(opp_id, 1100)
    )
    row["opp_overall"] = (
        team_strength_overall_a.get(opp_id, 1100) if is_home
        else team_strength_overall_h.get(opp_id, 1100)
    )
    row["own_attack"] = (
        team_strength_attack_h.get(team_id, 1100) if is_home
        else team_strength_attack_a.get(team_id, 1100)
    )
    row["own_defense"] = (
        team_strength_defense_h.get(team_id, 1100) if is_home
        else team_strength_defense_a.get(team_id, 1100)
    )
    row["attack_advantage"] = row["own_attack"] - row["opp_defense"]
    row["defense_advantage"] = row["own_defense"] - row["opp_attack"]

    row["price"] = player_row["now_cost"]
    row["gw"] = gw_number

    # Position dummies
    for pos in ["Defender", "Forward", "Goalkeeper", "Midfielder"]:
        row[f"pos_{pos}"] = 1 if player_row["position"] == pos else 0

    # Build feature vector in model order
    feature_vector = []
    for feat in model_features:
        feature_vector.append(row.get(feat, 0))

    return np.array(feature_vector).reshape(1, -1)


def predict_player_points(player_row, n_gws=5):
    """Predict next N gameweeks for a player using the XGBoost base model."""
    xgb_model, features = get_base_xgb_model()
    if xgb_model is None or features is None:
        return None

    hist_df = load_histories()
    teams_raw_df = load_teams()
    players_raw_df = load_players_raw()
    fixtures_df = load_fixtures()

    upcoming = fixtures_df[
        (fixtures_df["finished"] == False) & (fixtures_df["event"].notna())
    ].copy()
    if len(upcoming) == 0:
        return None
    upcoming["event"] = upcoming["event"].astype(int)
    next_gws = sorted(upcoming["event"].unique())[:n_gws]

    results = []
    for gw in next_gws:
        fv = build_player_features(
            player_row, hist_df, teams_raw_df, players_raw_df, fixtures_df,
            features, gw,
        )
        if fv is not None:
            pred = float(xgb_model.predict(fv)[0])
            pred = max(pred, 0)

            # Determine opponent
            player_to_team_id = dict(zip(players_raw_df["id"], players_raw_df["team"]))
            team_id_to_name = dict(zip(teams_raw_df["id"], teams_raw_df["name"]))
            team_id = player_to_team_id.get(player_row["id"])
            gw_fix = upcoming[upcoming["event"] == gw]
            home_fix = gw_fix[gw_fix["team_h"] == team_id]
            away_fix = gw_fix[gw_fix["team_a"] == team_id]
            if len(home_fix) > 0:
                opp_name = team_id_to_name.get(int(home_fix.iloc[0]["team_a"]), "?")
                venue = "H"
            elif len(away_fix) > 0:
                opp_name = team_id_to_name.get(int(away_fix.iloc[0]["team_h"]), "?")
                venue = "A"
            else:
                opp_name = "?"
                venue = "?"

            results.append({
                "gw": int(gw),
                "predicted_points": round(pred, 2),
                "opponent": opp_name,
                "venue": venue,
            })
    return results


# ---------------------------------------------------------------------------
# HELPER: KPI card HTML
# ---------------------------------------------------------------------------
def kpi_card(label, value, sub=""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """


def arena_card(model_name, mae, r2, rank20, color="#00ff87", is_best=False):
    """Render a Model Arena KPI card for a single model."""
    border_color = "#FFD700" if is_best else color
    best_badge = '<span style="color:#FFD700; font-size:0.75rem;">BEST</span>' if is_best else ""
    return f"""
    <div class="arena-card" style="border-top-color: {border_color};">
        <div class="arena-model-name" style="color:{color};">{model_name} {best_badge}</div>
        <div class="arena-metric">MAE</div>
        <div class="arena-value" style="color:{color};">{mae:.3f}</div>
        <div class="arena-metric" style="margin-top:6px;">R&sup2;</div>
        <div class="arena-value" style="color:{color};">{r2:.3f}</div>
        <div class="arena-metric" style="margin-top:6px;">Top-20</div>
        <div class="arena-value" style="color:{color};">{rank20:.0%}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# PLOTLY TEMPLATE
# ---------------------------------------------------------------------------
FPL_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
)


# ===========================================================================
# SIDEBAR NAVIGATION
# ===========================================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <h1 style="color:{FPL_GREEN}; font-size:1.6rem; margin-bottom:0;">
                FPL Intelligence
            </h1>
            <p style="color:rgba(255,255,255,0.6); font-size:0.85rem;">
                V3 Position-Specific Ensemble
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Player Analysis",
            "xG Analysis",
            "Form & Trends",
            "Fixture Forecast",
            "Optimal Teams",
            "Model Arena",
            "Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f'<p style="color:rgba(255,255,255,0.5); font-size:0.75rem; text-align:center;">'
        f"V3 | 4 Positions | 3-GW Avg Target</p>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 1: OVERVIEW
# ===========================================================================
if page == "Overview":
    st.markdown(
        f'<h1 style="color:{FPL_GREEN};">FPL 2024/25 Intelligence Dashboard</h1>',
        unsafe_allow_html=True,
    )

    df = load_players()
    data = load_data_json()
    insights = data.get("insights", {})

    # --- KPI cards ---
    total = insights.get("total_players", len(df))
    active = insights.get("active_players", int((df["minutes"] > 0).sum()))
    top_scorer = insights.get("top_scorer", {})
    top_scorer_name = top_scorer.get("name", "N/A")
    top_scorer_pts = top_scorer.get("points", "N/A")
    top_scorer_team = top_scorer.get("team", "")

    # Best value
    best_value_players = []
    for pos_list in data.get("best_value", {}).values():
        best_value_players.extend(pos_list)
    if best_value_players:
        bvp = max(best_value_players, key=lambda x: x.get("value", 0))
        bvp_name = bvp["web_name"]
        bvp_val = bvp["value"]
    else:
        bvp_name = "N/A"
        bvp_val = ""

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Total Players", total, "In the game"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Active Players", active, "450+ minutes played"), unsafe_allow_html=True)
    with c3:
        st.markdown(
            kpi_card("Top Scorer", f"{top_scorer_name}", f"{top_scorer_pts} pts | {top_scorer_team}"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            kpi_card("Best Value", bvp_name, f"{bvp_val} pts/\u00a3m"),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Top 15 table ---
    st.subheader("Top 15 Players")
    top15 = pd.DataFrame(data.get("top15", []))
    if not top15.empty:
        # Highlight captain-worthy players (top 5 by points)
        def highlight_captain(row):
            if row.name < 5:
                return [f"background-color: rgba(0,255,135,0.15)"] * len(row)
            return [""] * len(row)

        styled = top15.style.apply(highlight_captain, axis=1).format(
            {"now_cost": "\u00a3{:.1f}m", "value": "{:.1f}"},
            na_rep="-",
        )
        st.dataframe(
            top15,
            use_container_width=True,
            height=560,
            column_config={
                "web_name": st.column_config.TextColumn("Player", width="medium"),
                "team_name": st.column_config.TextColumn("Team"),
                "position": st.column_config.TextColumn("Pos"),
                "now_cost": st.column_config.NumberColumn("Price", format="\u00a3%.1fm"),
                "total_points": st.column_config.NumberColumn("Points"),
                "goals_scored": st.column_config.NumberColumn("Goals"),
                "assists": st.column_config.NumberColumn("Assists"),
                "value": st.column_config.NumberColumn("Pts/\u00a3m", format="%.1f"),
            },
        )
    else:
        st.info("No top 15 data available.")

    st.markdown("---")

    # --- Distribution charts ---
    tab1, tab2 = st.tabs(["Points by Position", "Price vs Points"])

    active_df = df[df["minutes"] >= 450].copy()

    with tab1:
        fig = px.box(
            active_df,
            x="position",
            y="total_points",
            color="position",
            color_discrete_map=POS_COLORS,
            title="Points Distribution by Position (450+ mins)",
        )
        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            title_font_size=16,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.scatter(
            active_df,
            x="now_cost",
            y="total_points",
            color="position",
            color_discrete_map=POS_COLORS,
            hover_data=["web_name", "team_name", "value"],
            title="Price vs Total Points",
            labels={"now_cost": "Price (\u00a3m)", "total_points": "Total Points"},
        )
        fig.update_layout(template="plotly_dark", title_font_size=16)
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 2: PLAYER ANALYSIS
# ===========================================================================
elif page == "Player Analysis":
    st.markdown(
        f'<h1 style="color:{FPL_GREEN};">Player Analysis</h1>',
        unsafe_allow_html=True,
    )

    df = load_players()
    hist = load_histories()
    active_df = df[df["minutes"] > 0].copy()

    tab_search, tab_compare, tab_ask = st.tabs(
        ["Player Card & Prediction", "Player Comparison", "Ask the Model"]
    )

    # ----- TAB 1: Player Card + Prediction -----
    with tab_search:
        player_options = active_df.sort_values("total_points", ascending=False)
        display_names = (
            player_options["web_name"]
            + " ("
            + player_options["team_name"]
            + " - "
            + player_options["position"]
            + ")"
        )

        selected_display = st.selectbox(
            "Search player (type to filter, e.g. 'Sal' for Salah)",
            display_names.tolist(),
            index=0,
            placeholder="Type a name to search...",
        )

        if selected_display:
            idx = display_names.tolist().index(selected_display)
            player = player_options.iloc[idx]

            col_card, col_pred = st.columns([1, 1])

            with col_card:
                st.markdown(
                    f"""
                    <div class="player-card">
                        <h2>{player['web_name']}</h2>
                        <span class="team-badge">{player['team_name']}</span>
                        &nbsp; <span style="color:{POS_COLORS.get(player['position'], '#fff')}; font-weight:600;">
                            {player['position']}
                        </span>
                        <hr style="border-color:rgba(255,255,255,0.15);">
                        <div class="stat-row"><span class="stat-label">Price</span><span class="stat-value">\u00a3{player['now_cost']:.1f}m</span></div>
                        <div class="stat-row"><span class="stat-label">Total Points</span><span class="stat-value">{int(player['total_points'])}</span></div>
                        <div class="stat-row"><span class="stat-label">Form</span><span class="stat-value">{player['form']}</span></div>
                        <div class="stat-row"><span class="stat-label">Points/Game</span><span class="stat-value">{player['points_per_game']}</span></div>
                        <div class="stat-row"><span class="stat-label">Goals</span><span class="stat-value">{int(player['goals_scored'])}</span></div>
                        <div class="stat-row"><span class="stat-label">Assists</span><span class="stat-value">{int(player['assists'])}</span></div>
                        <div class="stat-row"><span class="stat-label">Minutes</span><span class="stat-value">{int(player['minutes'])}</span></div>
                        <div class="stat-row"><span class="stat-label">xG</span><span class="stat-value">{player['expected_goals']:.2f}</span></div>
                        <div class="stat-row"><span class="stat-label">xA</span><span class="stat-value">{player['expected_assists']:.2f}</span></div>
                        <div class="stat-row"><span class="stat-label">ICT Index</span><span class="stat-value">{player['ict_index']}</span></div>
                        <div class="stat-row"><span class="stat-label">Selected By</span><span class="stat-value">{player['selected_by_percent']}%</span></div>
                        <div class="stat-row"><span class="stat-label">Value</span><span class="stat-value">{player['value']:.1f} pts/\u00a3m</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_pred:
                st.markdown(f"#### Predicted Points (Next 5 GWs)")
                with st.spinner("Running prediction model..."):
                    predictions = predict_player_points(player, n_gws=5)

                if predictions:
                    pred_df = pd.DataFrame(predictions)
                    pred_df["label"] = pred_df.apply(
                        lambda r: f"GW{r['gw']}\nvs {r['opponent']} ({r['venue']})", axis=1
                    )

                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=pred_df["label"],
                            y=pred_df["predicted_points"],
                            marker_color=FPL_GREEN,
                            text=pred_df["predicted_points"].round(1),
                            textposition="outside",
                            textfont=dict(color="white", size=14),
                        )
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        yaxis_title="Predicted Points",
                        xaxis_title="",
                        title=f"Point Predictions for {player['web_name']}",
                        title_font_size=15,
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    total_pred = pred_df["predicted_points"].sum()
                    avg_pred = pred_df["predicted_points"].mean()
                    st.markdown(
                        f"**Total predicted:** {total_pred:.1f} pts | "
                        f"**Average:** {avg_pred:.1f} pts/GW"
                    )
                else:
                    st.warning(
                        "Could not generate predictions for this player. "
                        "They may not have enough history data or upcoming fixtures."
                    )

                # Gameweek points history
                st.markdown("#### Gameweek Points History")
                player_hist = hist[hist["player_id"] == player["id"]].sort_values("round")
                if not player_hist.empty:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(
                        go.Scatter(
                            x=player_hist["round"],
                            y=player_hist["total_points"],
                            mode="lines+markers",
                            line=dict(color=FPL_GREEN, width=2),
                            marker=dict(size=6),
                            name="Points",
                        )
                    )
                    avg_pts = player_hist["total_points"].mean()
                    fig_hist.add_hline(
                        y=avg_pts,
                        line_dash="dash",
                        line_color="rgba(255,255,255,0.4)",
                        annotation_text=f"Avg: {avg_pts:.1f}",
                    )
                    fig_hist.update_layout(
                        template="plotly_dark",
                        xaxis_title="Gameweek",
                        yaxis_title="Points",
                        height=300,
                        margin=dict(t=20),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

    # ----- TAB 2: Player Comparison -----
    with tab_compare:
        st.markdown("#### Compare 2-3 Players (Radar Chart)")

        compare_names = (
            active_df.sort_values("total_points", ascending=False)["web_name"]
            + " ("
            + active_df.sort_values("total_points", ascending=False)["team_name"]
            + ")"
        )

        selected_compare = st.multiselect(
            "Select 2-3 players to compare (type to search)",
            compare_names.tolist(),
            default=compare_names.tolist()[:2],
            max_selections=3,
            placeholder="Type a name to search...",
        )

        if len(selected_compare) >= 2:
            compare_players = []
            for s in selected_compare:
                name_part = s.split(" (")[0]
                p = active_df[active_df["web_name"] == name_part].iloc[0]
                compare_players.append(p)

            # Radar chart dimensions
            categories = [
                "Total Points", "Goals", "Assists", "Clean Sheets",
                "xG", "xA", "Form", "ICT Index",
            ]
            cols_for_radar = [
                "total_points", "goals_scored", "assists", "clean_sheets",
                "expected_goals", "expected_assists", "form", "ict_index",
            ]

            # Normalize to 0-100 scale
            max_vals = {}
            for c in cols_for_radar:
                max_v = active_df[c].max()
                max_vals[c] = max_v if max_v > 0 else 1

            radar_colors = [FPL_GREEN, "#ff6384", FPL_TEAL]

            fig = go.Figure()
            for i, p in enumerate(compare_players):
                vals = [(p[c] / max_vals[c]) * 100 for c in cols_for_radar]
                vals.append(vals[0])  # close the radar
                cats = categories + [categories[0]]
                fig.add_trace(
                    go.Scatterpolar(
                        r=vals,
                        theta=cats,
                        fill="toself",
                        name=p["web_name"],
                        line=dict(color=radar_colors[i % len(radar_colors)]),
                        opacity=0.7,
                    )
                )
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        gridcolor="rgba(255,255,255,0.15)",
                    ),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.15)"),
                ),
                template="plotly_dark",
                title="Player Comparison Radar",
                title_font_size=16,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Side by side stats table
            compare_df = pd.DataFrame(
                [{
                    "Player": p["web_name"],
                    "Team": p["team_name"],
                    "Position": p["position"],
                    "Price": f"\u00a3{p['now_cost']:.1f}m",
                    "Points": int(p["total_points"]),
                    "Goals": int(p["goals_scored"]),
                    "Assists": int(p["assists"]),
                    "Form": p["form"],
                    "xG": f"{p['expected_goals']:.2f}",
                    "xA": f"{p['expected_assists']:.2f}",
                    "PPG": p["points_per_game"],
                    "Value": f"{p['value']:.1f}",
                } for p in compare_players]
            )
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
        else:
            st.info("Select at least 2 players to compare.")

    # ----- TAB 3: Ask the Model -----
    with tab_ask:
        st.markdown("#### Ask the Model")
        st.markdown(
            "Select a player and get their predicted points for the next 5 gameweeks. "
            "**Start typing** to filter (e.g. type `H` to see Haaland, Havertz, etc.)"
        )

        # Show which model is being used
        model_data_check = load_model()
        if model_data_check is not None and not model_data_check.get("_legacy", False):
            st.info(
                "Using **XGBoost base model** from the Position-Specific Ensemble for individual predictions. "
                "Full ensemble (XGB + RF + MLP + LSTM -> Meta) is used for team selections."
            )
        else:
            st.info("Using **XGBoost** model for predictions.")

        # Searchable dropdown instead of text input
        ask_players = active_df.sort_values("total_points", ascending=False)
        ask_display = (
            ask_players["web_name"]
            + "  |  "
            + ask_players["team_name"]
            + "  |  "
            + ask_players["position"]
            + "  |  "
            + ask_players["total_points"].astype(int).astype(str) + " pts"
        ).tolist()

        ask_selected = st.selectbox(
            "Search player (type to filter)",
            options=[""] + ask_display,
            index=0,
            placeholder="Type a name... (e.g. Haaland, Salah, Palmer)",
        )

        if ask_selected and ask_selected != "":
            ask_idx = ask_display.index(ask_selected)
            match = ask_players.iloc[ask_idx]

            st.markdown(
                f"**{match['web_name']}** | {match['team_name']} | "
                f"{match['position']} | \u00a3{match['now_cost']:.1f}m | "
                f"{int(match['total_points'])} pts"
            )

            with st.spinner("Predicting..."):
                preds = predict_player_points(match, n_gws=5)

            if preds:
                pred_df = pd.DataFrame(preds)

                # Bar chart
                pred_df["label"] = pred_df.apply(
                    lambda r: f"GW{r['gw']}\nvs {r['opponent']} ({r['venue']})", axis=1
                )
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=pred_df["label"],
                        y=pred_df["predicted_points"],
                        marker_color=FPL_GREEN,
                        text=pred_df["predicted_points"].round(1),
                        textposition="outside",
                        textfont=dict(color="white", size=14),
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    yaxis_title="Predicted Points",
                    title=f"Predicted Points for {match['web_name']}",
                    title_font_size=15,
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    pred_df[["gw", "opponent", "venue", "predicted_points"]].rename(columns={
                        "gw": "Gameweek",
                        "predicted_points": "Predicted Pts",
                        "opponent": "Opponent",
                        "venue": "Venue",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
                total = pred_df["predicted_points"].sum()
                st.success(
                    f"Total predicted over {len(preds)} GWs: **{total:.1f} points** "
                    f"(avg {total / len(preds):.1f}/GW)"
                )
            else:
                st.warning("Could not generate predictions. Insufficient data or no upcoming fixtures.")


# ===========================================================================
# PAGE 3: xG ANALYSIS
# ===========================================================================
elif page == "xG Analysis":
    st.markdown(
        f'<h1 style="color:{FPL_GREEN};">Expected Goals & Assists Analysis</h1>',
        unsafe_allow_html=True,
    )

    df = load_players()
    data = load_data_json()
    active_df = df[(df["minutes"] >= 450) & (df["expected_goals"] > 0)].copy()

    # Position filter
    positions = st.multiselect(
        "Filter by Position",
        ["Forward", "Midfielder", "Defender", "Goalkeeper"],
        default=["Forward", "Midfielder", "Defender"],
    )
    filtered = active_df[active_df["position"].isin(positions)]

    tab_xg, tab_xa = st.tabs(["Goals vs xG", "Assists vs xA"])

    with tab_xg:
        st.subheader("Goals Scored vs Expected Goals (xG)")

        filtered["xg_diff"] = filtered["goals_scored"] - filtered["expected_goals"]

        fig = px.scatter(
            filtered,
            x="expected_goals",
            y="goals_scored",
            color="position",
            color_discrete_map=POS_COLORS,
            hover_data=["web_name", "team_name", "xg_diff"],
            size="total_points",
            size_max=18,
            title="Goals vs xG (above line = overperforming)",
            labels={
                "expected_goals": "Expected Goals (xG)",
                "goals_scored": "Actual Goals",
            },
        )
        max_val = max(filtered["expected_goals"].max(), filtered["goals_scored"].max()) + 1
        fig.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(dash="dash", color="rgba(255,255,255,0.4)"),
        )
        fig.update_layout(template="plotly_dark", title_font_size=16, height=550)
        st.plotly_chart(fig, use_container_width=True)

        # Tables
        col_over, col_under = st.columns(2)
        xg_analysis = data.get("xg_analysis", {})

        with col_over:
            st.markdown(f"##### Overperformers (Goals > xG)")
            over_df = pd.DataFrame(xg_analysis.get("overperformers", []))
            if not over_df.empty:
                st.dataframe(over_df, use_container_width=True, hide_index=True)

        with col_under:
            st.markdown(f"##### Underperformers (Goals < xG)")
            under_df = pd.DataFrame(xg_analysis.get("underperformers", []))
            if not under_df.empty:
                st.dataframe(under_df, use_container_width=True, hide_index=True)

    with tab_xa:
        st.subheader("Assists vs Expected Assists (xA)")

        active_xa = df[
            (df["minutes"] >= 450) & (df["expected_assists"] > 0)
        ].copy()
        active_xa = active_xa[active_xa["position"].isin(positions)]
        active_xa["xa_diff"] = active_xa["assists"] - active_xa["expected_assists"]

        fig = px.scatter(
            active_xa,
            x="expected_assists",
            y="assists",
            color="position",
            color_discrete_map=POS_COLORS,
            hover_data=["web_name", "team_name", "xa_diff"],
            size="total_points",
            size_max=18,
            title="Assists vs xA (above line = overperforming)",
            labels={
                "expected_assists": "Expected Assists (xA)",
                "assists": "Actual Assists",
            },
        )
        max_val_xa = max(
            active_xa["expected_assists"].max(), active_xa["assists"].max()
        ) + 1
        fig.add_shape(
            type="line", x0=0, y0=0, x1=max_val_xa, y1=max_val_xa,
            line=dict(dash="dash", color="rgba(255,255,255,0.4)"),
        )
        fig.update_layout(template="plotly_dark", title_font_size=16, height=550)
        st.plotly_chart(fig, use_container_width=True)

        xa_analysis = data.get("xa_analysis", {})
        col_oa, col_ua = st.columns(2)
        with col_oa:
            st.markdown("##### Overperformers (Assists > xA)")
            oa_df = pd.DataFrame(xa_analysis.get("overperformers", []))
            if not oa_df.empty:
                st.dataframe(oa_df, use_container_width=True, hide_index=True)
        with col_ua:
            st.markdown("##### Underperformers (Assists < xA)")
            ua_df = pd.DataFrame(xa_analysis.get("underperformers", []))
            if not ua_df.empty:
                st.dataframe(ua_df, use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 4: FORM & TRENDS
# ===========================================================================
elif page == "Form & Trends":
    st.markdown(
        f'<h1 style="color:{FPL_GREEN};">Form & Trends</h1>',
        unsafe_allow_html=True,
    )

    df = load_players()
    hist = load_histories()
    data = load_data_json()
    form_trends = data.get("form_trends", {})

    tab_rising, tab_history, tab_heatmap = st.tabs(
        ["Rising / Falling", "Player History", "Top 20 Heatmap"]
    )

    # ----- Rising / Falling -----
    with tab_rising:
        col_rise, col_fall = st.columns(2)

        with col_rise:
            st.subheader("Rising Players")
            rising = pd.DataFrame(form_trends.get("rising", []))
            if not rising.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        y=rising["web_name"],
                        x=rising["trend"],
                        orientation="h",
                        marker_color=FPL_GREEN,
                        text=rising["trend"].round(1),
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    title="Form Improvement (pts/GW)",
                    height=450,
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Trend (pts/GW difference)",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_fall:
            st.subheader("Falling Players")
            falling = pd.DataFrame(form_trends.get("falling", []))
            if not falling.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        y=falling["web_name"],
                        x=falling["trend"],
                        orientation="h",
                        marker_color="#e74c3c",
                        text=falling["trend"].round(1),
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    title="Form Decline (pts/GW)",
                    height=450,
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Trend (pts/GW difference)",
                )
                st.plotly_chart(fig, use_container_width=True)

    # ----- Player History -----
    with tab_history:
        st.subheader("Player Points Over Season")

        active_df = df[df["minutes"] > 0].sort_values("total_points", ascending=False)
        player_choice = st.selectbox(
            "Select a player",
            active_df["web_name"].tolist(),
            key="form_player_select",
        )

        if player_choice:
            pid = df[df["web_name"] == player_choice]["id"].values
            if len(pid) > 0:
                player_hist = hist[hist["player_id"] == pid[0]].sort_values("round")
                if not player_hist.empty:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=player_hist["round"],
                            y=player_hist["total_points"],
                            mode="lines+markers",
                            name="Points",
                            line=dict(color=FPL_GREEN, width=2),
                            marker=dict(size=7),
                        )
                    )
                    # Rolling average
                    if len(player_hist) >= 3:
                        player_hist["rolling_avg"] = (
                            player_hist["total_points"].rolling(3, min_periods=1).mean()
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=player_hist["round"],
                                y=player_hist["rolling_avg"],
                                mode="lines",
                                name="3-GW Rolling Avg",
                                line=dict(color=FPL_TEAL, width=2, dash="dash"),
                            )
                        )

                    fig.update_layout(
                        template="plotly_dark",
                        xaxis_title="Gameweek",
                        yaxis_title="Points",
                        title=f"{player_choice} - Points per Gameweek",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No history data for this player.")

    # ----- Heatmap -----
    with tab_heatmap:
        st.subheader("Top 20 Players - Gameweek Points Heatmap")

        top20_players = df.nlargest(20, "total_points")
        top20_ids = top20_players["id"].tolist()
        top20_hist = hist[hist["player_id"].isin(top20_ids)].copy()

        if not top20_hist.empty:
            name_map = dict(zip(df["id"], df["web_name"]))
            top20_hist["web_name"] = top20_hist["player_id"].map(name_map)

            pivot = top20_hist.pivot_table(
                index="web_name", columns="round", values="total_points",
                aggfunc="sum",
            )
            # Sort by total points
            pivot["total"] = pivot.sum(axis=1)
            pivot = pivot.sort_values("total", ascending=True).drop("total", axis=1)

            fig = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn",
                title="Points Heatmap (Top 20 Players)",
                labels=dict(x="Gameweek", y="Player", color="Points"),
                aspect="auto",
            )
            fig.update_layout(
                template="plotly_dark",
                title_font_size=16,
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No history data available.")


# ===========================================================================
# PAGE 5: FIXTURE FORECAST
# ===========================================================================
elif page == "Fixture Forecast":
    st.markdown(
        f'<h1 style="color:{FPL_GREEN};">Fixture Forecast</h1>',
        unsafe_allow_html=True,
    )

    data = load_data_json()
    forecast = data.get("fixture_forecast", {})
    ranking = forecast.get("ranking", [])
    gameweeks = forecast.get("gameweeks", [])

    tab_heatmap, tab_ranking, tab_picks = st.tabs(
        ["FDR Heatmap", "FDR Ranking", "Best Picks"]
    )

    # ----- FDR Heatmap -----
    with tab_heatmap:
        st.subheader("Fixture Difficulty Rating Heatmap")

        if ranking:
            teams_list = []
            for team_data in ranking:
                team_name = team_data["team"]
                row = {"Team": team_name}
                for fix in team_data.get("fixtures", []):
                    gw = fix["gw"]
                    row[f"GW{gw}"] = fix["fdr"]
                    row[f"GW{gw}_label"] = f"{fix['opponent']} ({fix['venue']})"
                teams_list.append(row)

            # Build heatmap data
            gw_cols = [f"GW{g}" for g in gameweeks]
            label_cols = [f"GW{g}_label" for g in gameweeks]
            team_names = [t["Team"] for t in teams_list]

            z_data = []
            hover_text = []
            for t in teams_list:
                row_z = [t.get(c, 0) for c in gw_cols]
                row_h = [t.get(c, "") for c in label_cols]
                z_data.append(row_z)
                hover_text.append(row_h)

            # FDR color scale: 1=green (easy), 5=red (hard)
            colorscale = [
                [0.0, "#1e8449"],
                [0.25, "#2ecc71"],
                [0.5, "#f1c40f"],
                [0.75, "#e67e22"],
                [1.0, "#c0392b"],
            ]

            fig = go.Figure(
                data=go.Heatmap(
                    z=z_data,
                    x=[f"GW{g}" for g in gameweeks],
                    y=team_names,
                    text=hover_text,
                    texttemplate="%{text}",
                    textfont=dict(size=10, color="white"),
                    colorscale=colorscale,
                    zmin=1,
                    zmax=5,
                    colorbar=dict(title="FDR"),
                )
            )
            fig.update_layout(
                template="plotly_dark",
                title="Fixture Difficulty Heatmap (1=Easy, 5=Hard)",
                title_font_size=16,
                height=max(400, len(team_names) * 35),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fixture forecast data available.")

    # ----- FDR Ranking -----
    with tab_ranking:
        st.subheader("Easiest Upcoming Fixtures (by Average FDR)")

        if ranking:
            rank_df = pd.DataFrame([{
                "Team": r["team"],
                "Avg FDR": r["avg_fdr"],
                "Easy Fixtures": r["n_easy"],
            } for r in ranking])

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=rank_df["Team"],
                    y=rank_df["Avg FDR"],
                    marker_color=[
                        FPL_GREEN if v <= 2.5 else "#f1c40f" if v <= 3.0 else "#e74c3c"
                        for v in rank_df["Avg FDR"]
                    ],
                    text=rank_df["Avg FDR"].round(1),
                    textposition="outside",
                )
            )
            fig.update_layout(
                template="plotly_dark",
                title="Average Fixture Difficulty (Lower = Easier)",
                title_font_size=16,
                yaxis_title="Average FDR",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ----- Best Picks -----
    with tab_picks:
        st.subheader("Best Picks by Position (Based on Fixtures)")

        best_value = data.get("best_value", {})
        if best_value:
            for pos in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
                if pos in best_value:
                    st.markdown(
                        f"##### {pos}s "
                        f'<span style="color:{POS_COLORS.get(pos, "#fff")}">&#9679;</span>',
                        unsafe_allow_html=True,
                    )
                    pos_df = pd.DataFrame(best_value[pos])
                    st.dataframe(
                        pos_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "web_name": "Player",
                            "team_name": "Team",
                            "now_cost": st.column_config.NumberColumn("Price", format="\u00a3%.1fm"),
                            "total_points": "Points",
                            "value": st.column_config.NumberColumn("Pts/\u00a3m", format="%.1f"),
                        },
                    )


# ===========================================================================
# PAGE 6: OPTIMAL TEAMS
# ===========================================================================
elif page == "Optimal Teams":
    st.markdown(
        f'<h1 style="color:{FPL_GREEN};">Optimal Teams</h1>',
        unsafe_allow_html=True,
    )

    data = load_data_json()
    model_data = load_model_json()

    tab_strategies, tab_gw_team, tab_wildcard = st.tabs(
        ["Team Strategies", "Best Team per GW", "Wildcard Advisor"]
    )

    # ----- Team Strategies -----
    with tab_strategies:
        st.subheader("Optimal Team Strategies")
        optimal_teams = data.get("optimal_teams", {})

        strategy_names = {"best": "Max Points", "value": "Max Value", "budget": "Budget"}

        cols = st.columns(3)
        for i, (key, label) in enumerate(strategy_names.items()):
            with cols[i]:
                team_info = optimal_teams.get(key, {})
                total_cost = team_info.get("total_cost", 0)
                total_pts = team_info.get("total_points", 0)
                team_label = team_info.get("label", label)

                st.markdown(
                    f"""
                    <div class="strategy-card">
                        <h3>{team_label}</h3>
                        <p style="color:{FPL_GREEN}; font-size:1.5rem; font-weight:700;">
                            {int(total_pts)} pts
                        </p>
                        <p>Cost: \u00a3{total_cost:.1f}m</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Show details in tabs
        strategy_tab = st.radio(
            "View team details:",
            list(strategy_names.values()),
            horizontal=True,
        )

        strategy_key = {v: k for k, v in strategy_names.items()}.get(strategy_tab, "best")
        team_info = optimal_teams.get(strategy_key, {})
        players_list = team_info.get("players", [])

        if players_list:
            team_df = pd.DataFrame(players_list)
            pos_order = {"Goalkeeper": 0, "Defender": 1, "Midfielder": 2, "Forward": 3}
            team_df["_order"] = team_df["position"].map(pos_order)
            team_df = team_df.sort_values("_order").drop("_order", axis=1)

            fig = go.Figure()
            for pos in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
                pos_data = team_df[team_df["position"] == pos]
                fig.add_trace(
                    go.Bar(
                        y=pos_data["web_name"],
                        x=pos_data["total_points"],
                        orientation="h",
                        name=pos,
                        marker_color=POS_COLORS.get(pos, "#888"),
                        text=[
                            f"{r['total_points']}pts | \u00a3{r['now_cost']}m"
                            for _, r in pos_data.iterrows()
                        ],
                        textposition="outside",
                    )
                )
            fig.update_layout(
                template="plotly_dark",
                title=f"{strategy_tab} - Player Points",
                barmode="stack",
                height=550,
                yaxis=dict(autorange="reversed"),
                xaxis_title="Total Points",
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                team_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "web_name": "Player",
                    "team_name": "Team",
                    "position": "Pos",
                    "now_cost": st.column_config.NumberColumn("Price", format="\u00a3%.1fm"),
                    "total_points": "Points",
                    "goals_scored": "Goals",
                    "assists": "Assists",
                    "clean_sheets": "CS",
                },
            )

    # ----- Best Team per GW -----
    with tab_gw_team:
        st.subheader("Model Predicted Best Team per Gameweek (3-GW Avg)")

        best_teams = model_data.get("best_teams", {})
        if best_teams:
            gw_options = sorted(best_teams.keys(), key=lambda x: int(x))
            selected_gw = st.select_slider(
                "Select Gameweek",
                options=gw_options,
                value=gw_options[0],
            )

            gw_team = best_teams.get(str(selected_gw), {})
            gw_players = gw_team.get("players", [])
            captain = gw_team.get("captain", "")
            total_pts = gw_team.get("total_with_captain", 0)
            total_cost = gw_team.get("total_cost", 0)

            st.markdown(
                f"**GW{selected_gw}** | Captain: **{captain}** | "
                f"Total: **{total_pts:.0f} pts** | Cost: \u00a3{total_cost:.1f}m"
            )

            if gw_players:
                gw_df = pd.DataFrame(gw_players)

                # Determine if we have individual model prediction columns
                has_model_cols = all(
                    col in gw_df.columns for col in ["xgb_pts", "rf_pts", "mlp_pts"]
                )

                fig = go.Figure()
                colors = []
                labels = []
                for _, r in gw_df.iterrows():
                    is_cap = r.get("is_captain", False)
                    color = "#FFD700" if is_cap else POS_COLORS.get(r["position"], "#888")
                    colors.append(color)
                    cap_tag = "(C) " if is_cap else ""
                    labels.append(
                        f"{cap_tag}{r['web_name']} vs {r['opponent']} ({r['venue']})"
                    )

                fig.add_trace(
                    go.Bar(
                        y=labels,
                        x=gw_df["predicted_points"],
                        orientation="h",
                        marker_color=colors,
                        text=gw_df["predicted_points"].round(1),
                        textposition="outside",
                        name="Ensemble",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    title=f"Predicted Best Team GW{selected_gw}",
                    height=550,
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Predicted Points",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show detailed table with individual model predictions
                if has_model_cols:
                    st.markdown("##### Individual Model Predictions & Agreement")

                    display_df = gw_df.copy()

                    # Compute model agreement: how close are the individual predictions
                    model_pred_cols = ["xgb_pts", "rf_pts", "mlp_pts"]
                    available_model_cols = [c for c in model_pred_cols if c in display_df.columns]

                    if available_model_cols:
                        display_df["model_std"] = display_df[available_model_cols].std(axis=1)
                        display_df["agreement"] = display_df["model_std"].apply(
                            lambda x: "High" if x < 0.3 else ("Medium" if x < 0.7 else "Low")
                        )

                    col_config = {
                        "web_name": "Player",
                        "team_name": "Team",
                        "position": "Pos",
                        "opponent": "Opp",
                        "venue": "H/A",
                        "predicted_points": st.column_config.NumberColumn(
                            "Ensemble Pts", format="%.2f"
                        ),
                        "xgb_pts": st.column_config.NumberColumn(
                            "XGBoost", format="%.2f"
                        ),
                        "rf_pts": st.column_config.NumberColumn(
                            "Rand. Forest", format="%.2f"
                        ),
                        "mlp_pts": st.column_config.NumberColumn(
                            "MLP", format="%.2f"
                        ),
                    }
                    if "agreement" in display_df.columns:
                        col_config["agreement"] = "Agreement"

                    show_cols = [
                        c for c in [
                            "web_name", "team_name", "position", "opponent", "venue",
                            "predicted_points", "xgb_pts", "rf_pts", "mlp_pts", "agreement",
                        ] if c in display_df.columns
                    ]

                    st.dataframe(
                        display_df[show_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config=col_config,
                    )
                else:
                    # Fallback: just show the basic table
                    st.dataframe(
                        gw_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "web_name": "Player",
                            "team_name": "Team",
                            "position": "Pos",
                            "opponent": "Opp",
                            "venue": "H/A",
                            "predicted_points": st.column_config.NumberColumn(
                                "Pred. Pts", format="%.2f"
                            ),
                            "now_cost": st.column_config.NumberColumn(
                                "Price", format="\u00a3%.1fm"
                            ),
                        },
                    )
        else:
            st.info("No predicted team data available.")

    # ----- Wildcard Advisor -----
    with tab_wildcard:
        st.subheader("Wildcard Advisor")

        wildcard = model_data.get("wildcard", {})
        if wildcard:
            wc_players = wildcard.get("players", [])
            wc_cost = wildcard.get("total_cost", 0)
            wc_pts = wildcard.get("total_predicted_5gw", 0)
            wc_avg = wildcard.get("avg_per_gw", 0)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    kpi_card("5-GW Total", f"{wc_pts:.0f} pts", "Predicted"),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    kpi_card("Average / GW", f"{wc_avg:.1f} pts", "Per gameweek"),
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    kpi_card("Squad Cost", f"\u00a3{wc_cost:.1f}m", "Total budget"),
                    unsafe_allow_html=True,
                )

            if wc_players:
                wc_df = pd.DataFrame(wc_players)
                pos_order = {"Goalkeeper": 0, "Defender": 1, "Midfielder": 2, "Forward": 3}
                wc_df["_order"] = wc_df["position"].map(pos_order)
                wc_df = wc_df.sort_values("_order").drop("_order", axis=1)

                # Support both old 'total_predicted' and new 'predicted_points' column names
                pts_col = "predicted_points" if "predicted_points" in wc_df.columns else "total_predicted"

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        y=wc_df["web_name"] + " (" + wc_df["team_name"] + ")",
                        x=wc_df[pts_col],
                        orientation="h",
                        marker_color=[
                            POS_COLORS.get(p, "#888") for p in wc_df["position"]
                        ],
                        text=[
                            f"{r[pts_col]:.1f}pts | \u00a3{r['now_cost']}m"
                            for _, r in wc_df.iterrows()
                        ],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    title="Wildcard Team - Predicted Points (Next 5 GWs)",
                    height=550,
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Total Predicted Points",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    wc_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "web_name": "Player",
                        "team_name": "Team",
                        "position": "Pos",
                        "now_cost": st.column_config.NumberColumn("Price", format="\u00a3%.1fm"),
                        pts_col: st.column_config.NumberColumn(
                            "Pred. Points (5 GWs)", format="%.1f"
                        ),
                    },
                )
        else:
            st.info("No wildcard data available.")


# ===========================================================================
# PAGE 7: MODEL ARENA (NEW)
# ===========================================================================
elif page == "Model Arena":
    st.markdown(
        f'<h1 style="color:{FPL_GREEN};">Model Arena</h1>',
        unsafe_allow_html=True,
    )
    model_data = load_model_json()
    metrics = model_data.get("metrics", {})
    target_type = metrics.get("target", "single GW")
    min_mins = metrics.get("min_minutes", "N/A")
    st.markdown(
        f'<p style="color:rgba(255,255,255,0.7); font-size:1.05rem;">'
        f"Head-to-head comparison of all 6 models | Target: <b>{target_type}</b> | Min minutes: <b>{min_mins}</b></p>",
        unsafe_allow_html=True,
    )

    best_model = metrics.get("best_model", "")

    # Collect model metrics
    model_keys = ["xgboost", "random_forest", "neural_net", "lstm", "simple_avg", "stacked_meta"]
    model_display_names = {
        "xgboost": "XGBoost",
        "random_forest": "Random Forest",
        "neural_net": "MLP Neural Net",
        "lstm": "LSTM",
        "simple_avg": "Simple Average",
        "stacked_meta": "Stacked Meta",
    }

    arena_models = []
    for mk in model_keys:
        m = metrics.get(mk, {})
        if m:
            arena_models.append({
                "key": mk,
                "name": model_display_names.get(mk, mk),
                "mae": m.get("mae", 0),
                "r2": m.get("r2", 0),
                "rank20": m.get("rank20", 0),
                "color": MODEL_COLORS.get(mk, "#ffffff"),
            })

    if arena_models:
        # --- KPI Cards Row ---
        cols = st.columns(len(arena_models))
        for i, m in enumerate(arena_models):
            with cols[i]:
                is_best = (m["key"] == best_model)
                st.markdown(
                    arena_card(
                        m["name"], m["mae"], m["r2"], m["rank20"],
                        color=m["color"], is_best=is_best,
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # --- Comparison Table ---
        st.subheader("Full Comparison Table")
        comp_df = pd.DataFrame(arena_models)
        comp_df = comp_df.rename(columns={
            "name": "Model",
            "mae": "MAE",
            "r2": "R\u00b2",
            "rank20": "Top-20 Ranking",
        })
        comp_df["Top-20 Ranking"] = comp_df["Top-20 Ranking"].apply(lambda x: f"{x:.0%}")

        # Highlight best model row
        st.dataframe(
            comp_df[["Model", "MAE", "R\u00b2", "Top-20 Ranking"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "MAE": st.column_config.NumberColumn("MAE", format="%.3f"),
                "R\u00b2": st.column_config.NumberColumn("R\u00b2", format="%.3f"),
                "Top-20 Ranking": st.column_config.TextColumn("Top-20 Ranking"),
            },
        )

        st.markdown("---")

        # --- Charts ---
        tab_mae, tab_r2, tab_rank, tab_pos, tab_images = st.tabs(
            ["MAE Comparison", "R\u00b2 Comparison", "Top-20 Ranking", "Per-Position", "Detailed Charts"]
        )

        names = [m["name"] for m in arena_models]
        colors = [m["color"] for m in arena_models]

        with tab_mae:
            st.subheader("Mean Absolute Error (MAE) - Lower is Better")
            mae_vals = [m["mae"] for m in arena_models]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=mae_vals,
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in mae_vals],
                    textposition="outside",
                    textfont=dict(color="white", size=13),
                )
            )
            # Highlight the best MAE with a horizontal line
            best_mae = min(mae_vals)
            fig.add_hline(
                y=best_mae, line_dash="dash", line_color="#FFD700",
                annotation_text=f"Best: {best_mae:.3f}",
                annotation_font_color="#FFD700",
            )
            fig.update_layout(
                template="plotly_dark",
                title="MAE Comparison Across All Models",
                title_font_size=16,
                yaxis_title="MAE",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_r2:
            st.subheader("R\u00b2 Score - Higher is Better")
            r2_vals = [m["r2"] for m in arena_models]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=r2_vals,
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in r2_vals],
                    textposition="outside",
                    textfont=dict(color="white", size=13),
                )
            )
            best_r2 = max(r2_vals)
            fig.add_hline(
                y=best_r2, line_dash="dash", line_color="#FFD700",
                annotation_text=f"Best: {best_r2:.3f}",
                annotation_font_color="#FFD700",
            )
            fig.update_layout(
                template="plotly_dark",
                title="R\u00b2 Comparison Across All Models",
                title_font_size=16,
                yaxis_title="R\u00b2",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_rank:
            st.subheader("Top-20 Ranking Accuracy - Higher is Better")
            rank_vals = [m["rank20"] for m in arena_models]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=[v * 100 for v in rank_vals],
                    marker_color=colors,
                    text=[f"{v:.0%}" for v in rank_vals],
                    textposition="outside",
                    textfont=dict(color="white", size=13),
                )
            )
            best_rank = max(rank_vals) * 100
            fig.add_hline(
                y=best_rank, line_dash="dash", line_color="#FFD700",
                annotation_text=f"Best: {max(rank_vals):.0%}",
                annotation_font_color="#FFD700",
            )
            fig.update_layout(
                template="plotly_dark",
                title="Top-20 Ranking Accuracy Across All Models",
                title_font_size=16,
                yaxis_title="Accuracy (%)",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_pos:
            st.subheader("Per-Position Performance")
            per_pos = metrics.get("per_position", {})

            if per_pos:
                pos_labels = list(per_pos.keys())
                pos_maes = [per_pos[p].get("meta_mae", 0) for p in pos_labels]
                pos_r2s = [per_pos[p].get("meta_r2", 0) for p in pos_labels]
                pos_train = [per_pos[p].get("train_size", 0) for p in pos_labels]
                pos_test = [per_pos[p].get("test_size", 0) for p in pos_labels]
                pos_colors_list = ["#f39c12", "#3498db", "#2ecc71", "#e74c3c"]

                # KPI cards per position
                cols = st.columns(len(pos_labels))
                for i, pos in enumerate(pos_labels):
                    with cols[i]:
                        st.markdown(
                            f"""<div class="arena-card" style="border-top-color: {pos_colors_list[i]};">
                                <div class="arena-model-name" style="color:{pos_colors_list[i]};">{pos}</div>
                                <div class="arena-metric">MAE</div>
                                <div class="arena-value" style="color:{pos_colors_list[i]};">{pos_maes[i]:.3f}</div>
                                <div class="arena-metric" style="margin-top:6px;">R&sup2;</div>
                                <div class="arena-value" style="color:{pos_colors_list[i]};">{pos_r2s[i]:.3f}</div>
                                <div class="arena-metric" style="margin-top:6px;">Train / Test</div>
                                <div class="arena-value" style="color:{pos_colors_list[i]}; font-size:1rem;">{pos_train[i]:,} / {pos_test[i]:,}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")

                # Bar charts
                fig = make_subplots(rows=1, cols=2, subplot_titles=["MAE per Position", "R² per Position"])
                fig.add_trace(
                    go.Bar(x=pos_labels, y=pos_maes, marker_color=pos_colors_list,
                           text=[f"{v:.3f}" for v in pos_maes], textposition="outside",
                           textfont=dict(color="white", size=13)),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Bar(x=pos_labels, y=pos_r2s, marker_color=pos_colors_list,
                           text=[f"{v:.3f}" for v in pos_r2s], textposition="outside",
                           textfont=dict(color="white", size=13)),
                    row=1, col=2,
                )
                fig.update_layout(template="plotly_dark", height=450, showlegend=False)
                fig.update_yaxes(title_text="MAE", row=1, col=1)
                fig.update_yaxes(title_text="R²", row=1, col=2)
                st.plotly_chart(fig, use_container_width=True)

                # Position R² chart image
                pos_r2_path = os.path.join(CHARTS_DIR, "v3_position_r2.png")
                if os.path.exists(pos_r2_path):
                    st.markdown("##### Detailed: R² per Position per Model")
                    st.image(pos_r2_path, use_container_width=True)
            else:
                st.info("Per-position metrics not available. Run model_v3.py to generate.")

        with tab_images:
            st.subheader("Detailed Model Charts")

            chart_files = {
                "Final Model Comparison": "final_model_comparison.png",
                "Gameweek MAE Breakdown": "final_gw_mae.png",
                "Position R² Breakdown": "v3_position_r2.png",
                "Meta Feature Importance": "meta_feature_importance.png",
            }

            for title, filename in chart_files.items():
                chart_path = os.path.join(CHARTS_DIR, filename)
                if os.path.exists(chart_path):
                    st.markdown(f"##### {title}")
                    st.image(chart_path, use_container_width=True)
                    st.markdown("---")
                else:
                    st.info(f"Chart not found: {filename}")

    else:
        st.info(
            "No multi-model metrics available. "
            "Run the stacked meta-ensemble training pipeline to generate model comparison data."
        )


# ===========================================================================
# PAGE 8: MODEL PERFORMANCE
# ===========================================================================
elif page == "Model Performance":
    st.markdown(
        f'<h1 style="color:{FPL_GREEN};">Model Performance</h1>',
        unsafe_allow_html=True,
    )

    model_data = load_model_json()
    metrics = model_data.get("metrics", {})

    # Determine model type
    model_type = metrics.get("model_type", "XGBoost Regressor")

    st.markdown(
        f'<p style="color:{FPL_TEAL}; font-size:1.1rem; font-weight:600;">{model_type}</p>',
        unsafe_allow_html=True,
    )

    # --- Metrics KPI ---
    # Try new format first, fall back to old format
    stacked_meta = metrics.get("stacked_meta", {})
    xgb_metrics = metrics.get("xgboost", {})
    lstm_metrics = metrics.get("lstm", {})

    if stacked_meta:
        # Multi-model format (V3 or V1)
        target_type = metrics.get("target", "single GW points")
        is_v3 = "per_position" in metrics
        positions_list = metrics.get("positions", [])
        min_mins = metrics.get("min_minutes", "N/A")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            best_model_name = metrics.get("best_model", "N/A")
            best_model_data = metrics.get(best_model_name, {})
            st.markdown(
                kpi_card(
                    "Best MAE",
                    f"{best_model_data.get('mae', 'N/A')}",
                    f"{best_model_name.upper()}"
                ),
                unsafe_allow_html=True,
            )
        with c2:
            best_r2_name = metrics.get("best_r2_model", "")
            best_r2_data = metrics.get(best_r2_name, {})
            all_r2 = [m.get('r2', 0) for m in [metrics.get(k, {}) for k in ['xgboost', 'random_forest', 'neural_net', 'lstm', 'stacked_meta']] if m]
            st.markdown(
                kpi_card(
                    "Best R\u00b2",
                    f"{max(all_r2):.3f}" if all_r2 else "N/A",
                    f"{best_r2_name.upper() if best_r2_name else 'N/A'}"
                ),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                kpi_card(
                    "Target",
                    "3-GW Avg" if is_v3 else "Single GW",
                    f"Min {min_mins} min" if is_v3 else "All minutes"
                ),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                kpi_card(
                    "Architecture",
                    f"4 Pos" if is_v3 else "6 Models",
                    f"{', '.join(positions_list)}" if positions_list else "XGB+RF+MLP+LSTM+Avg+Meta"
                ),
                unsafe_allow_html=True,
            )
    else:
        # Old single-model format
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                kpi_card("Test MAE", f"{metrics.get('test_mae', 'N/A')}", "Mean Absolute Error"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                kpi_card("Test RMSE", f"{metrics.get('test_rmse', 'N/A')}", "Root Mean Sq Error"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                kpi_card("R\u00b2 Score", f"{metrics.get('test_r2', 'N/A')}", "Variance Explained"),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                kpi_card("Features", f"{metrics.get('n_features', 'N/A')}", "Model inputs"),
                unsafe_allow_html=True,
            )

    st.markdown("---")

    tab_feat, tab_scatter, tab_limits = st.tabs(
        ["Feature Importance", "Predicted vs Actual", "Model Details"]
    )

    # ----- Feature Importance -----
    with tab_feat:
        st.subheader("Top Feature Importances")

        # Show meta feature importance chart if available
        meta_fi_path = os.path.join(CHARTS_DIR, "meta_feature_importance.png")
        if os.path.exists(meta_fi_path):
            st.markdown("##### Meta-Model Feature Importance")
            st.image(meta_fi_path, use_container_width=True)
            st.markdown("---")

        # Also show the base model feature importance from JSON
        top_features = model_data.get("top_features", [])

        if top_features:
            st.markdown("##### Base XGBoost Feature Importance")
            feat_df = pd.DataFrame(top_features)
            feat_df = feat_df.sort_values("importance", ascending=True).tail(25)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=feat_df["feature"],
                    x=feat_df["importance"],
                    orientation="h",
                    marker_color=FPL_GREEN,
                    text=feat_df["importance"].round(4),
                    textposition="outside",
                )
            )
            fig.update_layout(
                template="plotly_dark",
                title="XGBoost Feature Importance (Gain)",
                title_font_size=16,
                height=700,
                xaxis_title="Importance",
                margin=dict(l=200),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ----- Predicted vs Actual -----
    with tab_scatter:
        st.subheader("Predicted vs Actual (Test Set)")

        # Try to show static charts first (more reliable with the new multi-model setup)
        scatter_path = os.path.join(CHARTS_DIR, "model_accuracy.png")
        all_scatter_path = os.path.join(CHARTS_DIR, "all_models_scatter.png")

        shown_static = False
        if os.path.exists(all_scatter_path):
            st.markdown("##### All Models - Predicted vs Actual")
            st.image(all_scatter_path, use_container_width=True)
            shown_static = True

        if os.path.exists(scatter_path) and not shown_static:
            st.image(scatter_path, use_container_width=True)
            shown_static = True

        if not shown_static:
            # Fall back to generating interactive scatter from the XGBoost base model
            try:
                hist = load_histories()
                df = load_players()
                teams_raw = load_teams()
                players_raw = load_players_raw()
                xgb_model, feat_list = get_base_xgb_model()

                if xgb_model is not None:
                    pos_map = dict(zip(df["id"], df["position"]))
                    player_to_team_id = dict(zip(players_raw["id"], players_raw["team"]))

                    team_str_attack_h = dict(zip(teams_raw["id"], teams_raw["strength_attack_home"]))
                    team_str_attack_a = dict(zip(teams_raw["id"], teams_raw["strength_attack_away"]))
                    team_str_defense_h = dict(zip(teams_raw["id"], teams_raw["strength_defence_home"]))
                    team_str_defense_a = dict(zip(teams_raw["id"], teams_raw["strength_defence_away"]))
                    team_str_overall_h = dict(zip(teams_raw["id"], teams_raw["strength_overall_home"]))
                    team_str_overall_a = dict(zip(teams_raw["id"], teams_raw["strength_overall_away"]))

                    # Replicate feature engineering
                    h = hist.sort_values(["player_id", "round"]).copy()
                    h["position"] = h["player_id"].map(pos_map)

                    # Rolling
                    for w in [3, 5, 8]:
                        for col in [
                            "total_points", "minutes", "goals_scored", "assists", "bonus",
                            "bps", "ict_index", "influence", "creativity", "threat",
                            "expected_goals", "expected_assists", "expected_goal_involvements",
                            "clean_sheets", "goals_conceded", "saves",
                        ]:
                            if col in h.columns:
                                h[f"{col}_roll_{w}"] = h.groupby("player_id")[col].transform(
                                    lambda x: x.rolling(w, min_periods=1).mean().shift(1)
                                )

                    for w in [3, 5]:
                        h[f"pts_std_{w}"] = h.groupby("player_id")["total_points"].transform(
                            lambda x: x.rolling(w, min_periods=2).std().shift(1)
                        )
                        roll_key = f"total_points_roll_{w}"
                        if roll_key in h.columns:
                            h[f"pts_cv_{w}"] = h[f"pts_std_{w}"] / h[roll_key].clip(0.5)

                    h["momentum_3v8"] = h.get("total_points_roll_3", 0) - h.get("total_points_roll_8", 0)
                    h["xg_momentum"] = h.get("expected_goals_roll_3", 0) - h.get("expected_goals_roll_5", 0)

                    for col in ["goals_scored", "assists", "bonus", "expected_goals", "expected_assists"]:
                        k5 = f"{col}_roll_5"
                        if k5 in h.columns and "minutes_roll_5" in h.columns:
                            h[f"{col}_per90"] = (h[k5] / h["minutes_roll_5"].clip(1)) * 90

                    h["season_pts"] = h.groupby("player_id")["total_points"].transform(lambda x: x.cumsum().shift(1))
                    h["season_goals"] = h.groupby("player_id")["goals_scored"].transform(lambda x: x.cumsum().shift(1))
                    h["season_assists"] = h.groupby("player_id")["assists"].transform(lambda x: x.cumsum().shift(1))
                    h["season_minutes"] = h.groupby("player_id")["minutes"].transform(lambda x: x.cumsum().shift(1))
                    h["season_xg"] = h.groupby("player_id")["expected_goals"].transform(lambda x: x.cumsum().shift(1))
                    h["games_played"] = h.groupby("player_id").cumcount()
                    h["ppg"] = (h["season_pts"] / h["games_played"].clip(1)).round(2)

                    h["xg_overperformance"] = h.get("goals_scored_roll_5", 0) - h.get("expected_goals_roll_5", 0)
                    h["xa_overperformance"] = h.get("assists_roll_5", 0) - h.get("expected_assists_roll_5", 0)

                    h["is_home"] = h["was_home"].astype(int)
                    h["played_last"] = h.groupby("player_id")["minutes"].transform(
                        lambda x: (x.shift(1) > 0).astype(int)
                    )
                    h["minutes_trend"] = h.get("minutes_roll_3", 0) - h.get("minutes_roll_8", 0)
                    h["transfer_trend"] = h.groupby("player_id")["transfers_balance"].transform(
                        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
                    )
                    h["ownership_change"] = h.groupby("player_id")["selected"].transform(
                        lambda x: x.pct_change().shift(1)
                    )

                    h["opp_team_id"] = h["opponent_team"]
                    h["own_team_id"] = h["player_id"].map(player_to_team_id)
                    h["opp_attack"] = h.apply(
                        lambda r: team_str_attack_a.get(r["opp_team_id"], 1100) if r["is_home"]
                        else team_str_attack_h.get(r["opp_team_id"], 1100), axis=1)
                    h["opp_defense"] = h.apply(
                        lambda r: team_str_defense_a.get(r["opp_team_id"], 1100) if r["is_home"]
                        else team_str_defense_h.get(r["opp_team_id"], 1100), axis=1)
                    h["opp_overall"] = h.apply(
                        lambda r: team_str_overall_a.get(r["opp_team_id"], 1100) if r["is_home"]
                        else team_str_overall_h.get(r["opp_team_id"], 1100), axis=1)
                    h["own_attack"] = h.apply(
                        lambda r: team_str_attack_h.get(r["own_team_id"], 1100) if r["is_home"]
                        else team_str_attack_a.get(r["own_team_id"], 1100), axis=1)
                    h["own_defense"] = h.apply(
                        lambda r: team_str_defense_h.get(r["own_team_id"], 1100) if r["is_home"]
                        else team_str_defense_a.get(r["own_team_id"], 1100), axis=1)
                    h["attack_advantage"] = h["own_attack"] - h["opp_defense"]
                    h["defense_advantage"] = h["own_defense"] - h["opp_attack"]

                    h["price"] = h["value"] / 10
                    h["gw"] = h["round"]
                    h["target"] = h["total_points"]

                    pos_dummies = pd.get_dummies(h["position"], prefix="pos")
                    h = pd.concat([h, pos_dummies], axis=1)

                    # Filter valid data
                    valid_features = [f for f in feat_list if f in h.columns]
                    model_df = h[h["gw"] >= 5].copy()
                    model_df = model_df.dropna(subset=valid_features)
                    model_df = model_df.replace([np.inf, -np.inf], 0)

                    max_gw = model_df["gw"].max()
                    test_gw_start = max_gw - 4
                    test_data = model_df[model_df["gw"] > test_gw_start]

                    if len(test_data) > 0 and len(valid_features) == len(feat_list):
                        X_test = test_data[feat_list].fillna(0)
                        y_actual = test_data["target"]
                        y_pred = xgb_model.predict(X_test)

                        fig = make_subplots(rows=1, cols=2, subplot_titles=[
                            "Predicted vs Actual", "Error Distribution"
                        ])

                        fig.add_trace(
                            go.Scatter(
                                x=y_actual,
                                y=y_pred,
                                mode="markers",
                                marker=dict(color=FPL_GREEN, opacity=0.3, size=5),
                                name="Predictions",
                            ),
                            row=1, col=1,
                        )
                        max_v = max(y_actual.max(), y_pred.max())
                        fig.add_trace(
                            go.Scatter(
                                x=[0, max_v],
                                y=[0, max_v],
                                mode="lines",
                                line=dict(color="red", dash="dash"),
                                name="Perfect",
                            ),
                            row=1, col=1,
                        )

                        errors = y_actual.values - y_pred
                        fig.add_trace(
                            go.Histogram(
                                x=errors,
                                nbinsx=50,
                                marker_color=FPL_GREEN,
                                opacity=0.7,
                                name="Errors",
                            ),
                            row=1, col=2,
                        )
                        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)

                        fig.update_layout(
                            template="plotly_dark",
                            height=450,
                            showlegend=False,
                        )
                        fig.update_xaxes(title_text="Actual Points", row=1, col=1)
                        fig.update_yaxes(title_text="Predicted Points", row=1, col=1)
                        fig.update_xaxes(title_text="Error (Actual - Predicted)", row=1, col=2)
                        fig.update_yaxes(title_text="Count", row=1, col=2)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(
                            "Cannot generate scatter plot. Feature mismatch or insufficient data."
                        )
                else:
                    st.warning("Model not loaded.")
            except Exception as e:
                st.warning(f"Could not generate interactive scatter: {e}")

    # ----- Model Details -----
    with tab_limits:
        st.subheader("Model Details")

        col_info, col_params = st.columns(2)

        with col_info:
            st.markdown("##### Training Information")

            # Check if we have multi-model metrics
            model_keys_check = ["xgboost", "random_forest", "neural_net", "lstm", "simple_avg", "stacked_meta"]
            has_multi = any(metrics.get(k) for k in model_keys_check)

            if has_multi:
                st.markdown(f"**Model Architecture:** {model_type}")
                st.markdown("")

                # Build a comprehensive table of all model metrics
                rows = []
                for mk in model_keys_check:
                    m = metrics.get(mk, {})
                    if m:
                        display_name = {
                            "xgboost": "XGBoost",
                            "random_forest": "Random Forest",
                            "neural_net": "MLP Neural Net",
                            "lstm": "LSTM",
                            "simple_avg": "Simple Average",
                            "stacked_meta": "Stacked Meta",
                        }.get(mk, mk)
                        is_best = (mk == metrics.get("best_model", ""))
                        name_display = f"{display_name} *" if is_best else display_name
                        rows.append({
                            "Model": name_display,
                            "MAE": f"{m.get('mae', 'N/A'):.3f}" if isinstance(m.get('mae'), (int, float)) else "N/A",
                            "R\u00b2": f"{m.get('r2', 'N/A'):.3f}" if isinstance(m.get('r2'), (int, float)) else "N/A",
                            "Top-20": f"{m.get('rank20', 'N/A'):.0%}" if isinstance(m.get('rank20'), (int, float)) else "N/A",
                        })

                if rows:
                    detail_df = pd.DataFrame(rows)
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                    st.caption("* = Best performing model (by MAE)")

                # Per-position breakdown (V3)
                per_pos = metrics.get("per_position", {})
                if per_pos:
                    st.markdown("")
                    st.markdown("##### Per-Position Results (Meta-Ensemble)")
                    pos_rows = []
                    for pos, pd_data in per_pos.items():
                        pos_rows.append({
                            "Position": pos,
                            "MAE": f"{pd_data.get('meta_mae', 0):.3f}",
                            "R\u00b2": f"{pd_data.get('meta_r2', 0):.3f}",
                            "Train": f"{pd_data.get('train_size', 0):,}",
                            "Test": f"{pd_data.get('test_size', 0):,}",
                        })
                    st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

                # Training info
                st.markdown("")
                train_size = metrics.get("train_size", "N/A")
                test_size = metrics.get("test_size", "N/A")
                n_feat = metrics.get("n_base_features", metrics.get("n_features", "N/A"))
                n_meta = metrics.get("n_meta_features", "N/A")
                target_info = metrics.get("target", "single GW")
                min_min = metrics.get("min_minutes", "N/A")
                st.markdown(f"""
                | Training Detail | Value |
                |-----------------|-------|
                | Target | {target_info} |
                | Min Minutes | {min_min} |
                | Train Size | {train_size:,} rows |
                | Test Size | {test_size:,} rows |
                | Base Features | {n_feat} |
                | Meta Features | {n_meta} |
                """)
            else:
                # Old single-model format
                st.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | Algorithm | XGBoost Regressor |
                | Train MAE | {metrics.get('train_mae', 'N/A')} |
                | Test MAE | {metrics.get('test_mae', 'N/A')} |
                | Test RMSE | {metrics.get('test_rmse', 'N/A')} |
                | Test R\u00b2 | {metrics.get('test_r2', 'N/A')} |
                | Train Size | {metrics.get('train_size', 'N/A')} rows |
                | Test Size | {metrics.get('test_size', 'N/A')} rows |
                | Features | {metrics.get('n_features', 'N/A')} |
                | Train GWs | {metrics.get('train_gws', 'N/A')} |
                | Test GWs | {metrics.get('test_gws', 'N/A')} |
                """)

                if "ranking_accuracy_top20" in metrics:
                    st.markdown(f"""
                    | Ranking Metric | Value |
                    |----------------|-------|
                    | Top-20 Accuracy | {metrics.get('ranking_accuracy_top20', 'N/A')} |
                    | Top-10 Precision | {metrics.get('ranking_accuracy_top10', 'N/A')} |
                    | Captain Accuracy | {metrics.get('captain_accuracy', 'N/A')} |
                    """)

        with col_params:
            st.markdown("##### V3 Architecture")
            is_v3 = "per_position" in metrics

            if is_v3:
                st.markdown("""
                **3 Key Improvements:**
                1. **Filter < 30 min** - Removes sub appearances (noise reduction)
                2. **3-GW Average Target** - Predicts mean of current + next 2 GWs (smoother target)
                3. **Per-Position Models** - GK/DEF/MID/FWD each get their own ensemble

                **Per Position:**
                - XGBoost (600 trees, 5-fold CV)
                - Random Forest (200 trees, 5-fold CV)
                - MLP Neural Net (3 layers, 5-fold CV)
                - LSTM with Attention (128 hidden, 2 layers)
                - Meta-XGBoost stacking on OOF predictions
                """)
            else:
                best_params = metrics.get("best_params", {})
                if best_params:
                    for k, v in best_params.items():
                        st.markdown(f"- **{k}**: {v}")
                else:
                    model_pkl_data = load_model()
                    if model_pkl_data is not None:
                        params = model_pkl_data.get("params", {})
                        if params:
                            st.markdown("##### XGBoost Parameters")
                            for k, v in params.items():
                                st.markdown(f"- **{k}**: {v}")

            # Show meta feature importance chart
            meta_fi_path = os.path.join(CHARTS_DIR, "meta_feature_importance.png")
            if os.path.exists(meta_fi_path):
                st.markdown("---")
                st.markdown("##### Meta Feature Importance")
                st.image(meta_fi_path, use_container_width=True)

            # Show position R² chart
            pos_r2_path = os.path.join(CHARTS_DIR, "v3_position_r2.png")
            if os.path.exists(pos_r2_path):
                st.markdown("---")
                st.markdown("##### R² per Position per Model")
                st.image(pos_r2_path, use_container_width=True)

        st.markdown("---")
        st.markdown("##### Model Limitations & Notes")
        st.markdown(
            """
            - **FPL is inherently unpredictable**: Injuries, rotation, weather, VAR decisions,
              and managerial choices create high variance that no model can fully capture.
            - **3-GW average target** smooths out single-GW randomness, improving MAE significantly.
              However, R\u00b2 may appear lower because the target variance is also reduced.
            - **Position-specific models** allow each position to learn its own scoring patterns
              (clean sheets for DEF/GK, goals/assists for MID/FWD). This especially helps ranking.
            - **The model excels at ranking** (Top-20 accuracy 36%+) rather than exact point
              prediction -- identifying which players to pick is what matters for FPL.
            - **MAE of ~1.35 is in the "excellent" tier** (industry benchmark: excellent = 1.4-1.8).
            - **No injury/suspension data**: The model does not account for real-time availability.
            - **Min 30 minutes filter**: Players with < 30 min per GW are excluded from training
              to reduce noise from sub appearances.
            - **Rolling averages smooth outliers**: One-off hauls and blanks are averaged out.
            - **Budget constraints**: Optimal team selections respect FPL rules but may not
              match your current team.
            """
        )
