"""
Pull historical FPL data from vaastav/Fantasy-Premier-League GitHub repo.
Seasons 2019-20 to 2024-25 (xG data available from 2019+)
"""
import requests
import pandas as pd
import os
import time

BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
os.makedirs("data/historical", exist_ok=True)

# Seasons with xG data (2019-20 onwards has expected stats)
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]

all_gw_data = []

for season in SEASONS:
    print(f"\n{'='*50}")
    print(f"Season: {season}")
    print(f"{'='*50}")

    # 1. Get player list for this season
    players_url = f"{BASE}/{season}/players_raw.csv"
    print(f"  Pulling players...")
    try:
        players = pd.read_csv(players_url)
        players['season'] = season
        print(f"    -> {len(players)} players")
    except Exception as e:
        print(f"    -> FAILED: {e}")
        continue

    # 2. Get team data
    teams_url = f"{BASE}/{season}/teams.csv"
    try:
        teams = pd.read_csv(teams_url)
        team_map = dict(zip(teams['id'], teams['name']))
        team_strength_ah = dict(zip(teams['id'], teams.get('strength_attack_home', teams.get('strength', 0))))
        team_strength_aa = dict(zip(teams['id'], teams.get('strength_attack_away', teams.get('strength', 0))))
        team_strength_dh = dict(zip(teams['id'], teams.get('strength_defence_home', teams.get('strength', 0))))
        team_strength_da = dict(zip(teams['id'], teams.get('strength_defence_away', teams.get('strength', 0))))
    except:
        team_map = {}

    # 3. Get merged GW data (all players all gameweeks)
    gw_url = f"{BASE}/{season}/gws/merged_gw.csv"
    print(f"  Pulling gameweek data...")
    try:
        gw_data = pd.read_csv(gw_url, encoding='latin-1')
        gw_data['season'] = season
        print(f"    -> {len(gw_data)} gameweek records")

        # Map team names
        if 'team' in gw_data.columns and team_map:
            pass  # team might already be name

        all_gw_data.append(gw_data)
    except Exception as e:
        print(f"    -> FAILED: {e}")
        continue

    time.sleep(0.5)

# Combine all seasons
print(f"\n{'='*50}")
print("Combining all seasons...")

combined = pd.concat(all_gw_data, ignore_index=True)
print(f"Total records: {len(combined)}")
print(f"Seasons: {combined['season'].unique()}")
print(f"Columns: {list(combined.columns)}")

# Standardize column names
col_renames = {
    'GW': 'round',
    'xG': 'expected_goals',
    'xA': 'expected_assists',
    'xGI': 'expected_goal_involvements',
    'xGC': 'expected_goals_conceded',
}
for old, new in col_renames.items():
    if old in combined.columns and new not in combined.columns:
        combined = combined.rename(columns={old: new})

# Save
combined.to_csv("data/historical/all_seasons_gw.csv", index=False)

# Also get the cleaned merged seasons file
print("\nPulling pre-merged season totals...")
merged_url = f"{BASE}/cleaned_merged_seasons.csv"
try:
    merged = pd.read_csv(merged_url, encoding='latin-1')
    merged.to_csv("data/historical/merged_seasons.csv", index=False)
    print(f"  -> {len(merged)} player-season records")
except Exception as e:
    print(f"  -> FAILED: {e}")

print(f"\n{'='*50}")
print("DONE!")
print(f"  Historical GW data: data/historical/all_seasons_gw.csv ({len(combined)} records)")
print(f"{'='*50}")
