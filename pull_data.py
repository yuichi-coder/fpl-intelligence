"""
Pull Fantasy Premier League data from the public API
"""
import requests
import json
import pandas as pd
import os

BASE_URL = "https://fantasy.premierleague.com/api"

def pull_bootstrap_data():
    """Pull main FPL data (players, teams, gameweeks)"""
    print("Pulling main FPL data...")
    response = requests.get(f"{BASE_URL}/bootstrap-static/")
    response.raise_for_status()
    data = response.json()

    # Save raw JSON
    with open("data/raw/bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Extract and save players
    players = pd.DataFrame(data["elements"])
    players.to_csv("data/raw/players_raw.csv", index=False)
    print(f"  -> {len(players)} players pulled")

    # Extract and save teams
    teams = pd.DataFrame(data["teams"])
    teams.to_csv("data/raw/teams_raw.csv", index=False)
    print(f"  -> {len(teams)} teams pulled")

    # Extract and save gameweeks
    gameweeks = pd.DataFrame(data["events"])
    gameweeks.to_csv("data/raw/gameweeks_raw.csv", index=False)
    print(f"  -> {len(gameweeks)} gameweeks pulled")

    # Extract position types
    positions = pd.DataFrame(data["element_types"])
    positions.to_csv("data/raw/positions_raw.csv", index=False)
    print(f"  -> {len(positions)} positions pulled")

    return players, teams, gameweeks, positions

def pull_fixtures():
    """Pull all fixtures data"""
    print("Pulling fixtures...")
    response = requests.get(f"{BASE_URL}/fixtures/")
    response.raise_for_status()
    fixtures = pd.DataFrame(response.json())
    fixtures.to_csv("data/raw/fixtures_raw.csv", index=False)
    print(f"  -> {len(fixtures)} fixtures pulled")
    return fixtures

def pull_player_histories(player_ids, players_df):
    """Pull gameweek-by-gameweek history for top players"""
    print(f"Pulling gameweek history for {len(player_ids)} players...")
    all_history = []

    for i, pid in enumerate(player_ids):
        try:
            resp = requests.get(f"{BASE_URL}/element-summary/{pid}/", timeout=10)
            resp.raise_for_status()
            history = resp.json().get("history", [])
            for h in history:
                h["player_id"] = pid
            all_history.extend(history)
            if (i + 1) % 20 == 0:
                print(f"  -> {i+1}/{len(player_ids)} done")
        except Exception as e:
            print(f"  -> Failed for player {pid}: {e}")

    if all_history:
        hist_df = pd.DataFrame(all_history)
        hist_df.to_csv("data/raw/player_histories.csv", index=False)
        print(f"  -> {len(hist_df)} gameweek records saved")
        return hist_df
    return pd.DataFrame()

def clean_players(players, teams, positions):
    """Clean and prepare player data for analysis"""
    print("Cleaning player data...")

    # Map team names
    team_map = dict(zip(teams["id"], teams["name"]))
    players["team_name"] = players["team"].map(team_map)

    # Map position names
    pos_map = dict(zip(positions["id"], positions["singular_name"]))
    players["position"] = players["element_type"].map(pos_map)

    # Select key columns
    key_cols = [
        "id", "first_name", "second_name", "web_name", "team_name", "position",
        "now_cost", "total_points", "goals_scored", "assists",
        "minutes", "clean_sheets", "goals_conceded",
        "bonus", "bps", "form", "points_per_game",
        "selected_by_percent", "ict_index",
        "influence", "creativity", "threat",
        "transfers_in", "transfers_out",
        "yellow_cards", "red_cards", "saves",
        "starts", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded"
    ]

    # Only keep columns that exist
    available_cols = [c for c in key_cols if c in players.columns]
    clean = players[available_cols].copy()

    # Convert cost (it's stored as x10, e.g., 100 = £10.0m)
    clean["now_cost"] = clean["now_cost"] / 10

    # Convert numeric columns
    numeric_cols = ["form", "points_per_game", "selected_by_percent",
                    "ict_index", "influence", "creativity", "threat",
                    "expected_goals", "expected_assists",
                    "expected_goal_involvements", "expected_goals_conceded"]
    for col in numeric_cols:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    # Add value metric (points per million)
    clean["value"] = round(clean["total_points"] / clean["now_cost"], 2)
    clean.loc[clean["now_cost"] == 0, "value"] = 0

    # Save
    clean.to_csv("data/processed/players_clean.csv", index=False)
    print(f"  -> Cleaned data saved: {len(clean)} players, {len(clean.columns)} columns")

    return clean

if __name__ == "__main__":
    print("=" * 50)
    print("FPL Data Pull")
    print("=" * 50)

    players, teams, gameweeks, positions = pull_bootstrap_data()
    fixtures = pull_fixtures()
    clean = clean_players(players, teams, positions)

    # Pull gameweek history for top 80 players by points
    top_ids = clean[clean['minutes'] >= 900].nlargest(80, 'total_points')['id'].tolist()
    pull_player_histories(top_ids, clean)

    print("\n" + "=" * 50)
    print("DONE! Data saved to:")
    print("  Raw:       data/raw/")
    print("  Cleaned:   data/processed/")
    print("=" * 50)

    # Quick preview
    print(f"\nTop 10 players by total points:")
    top = clean.nlargest(10, "total_points")[["web_name", "team_name", "position", "now_cost", "total_points", "value"]]
    print(top.to_string(index=False))
