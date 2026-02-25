"""Pull gameweek history for all active players"""
import requests
import pandas as pd
import time

BASE_URL = "https://fantasy.premierleague.com/api"

df = pd.read_csv('data/processed/players_clean.csv')
active_ids = df[df['minutes'] >= 450]['id'].tolist()
print(f"Pulling history for {len(active_ids)} players...")

all_history = []
failed = []

for i, pid in enumerate(active_ids):
    try:
        resp = requests.get(f"{BASE_URL}/element-summary/{pid}/", timeout=10)
        resp.raise_for_status()
        history = resp.json().get("history", [])
        for h in history:
            h["player_id"] = pid
        all_history.extend(history)
    except Exception as e:
        failed.append(pid)

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(active_ids)} done ({len(all_history)} records)")
        time.sleep(0.5)  # be nice to the API

hist_df = pd.DataFrame(all_history)
hist_df.to_csv("data/raw/player_histories.csv", index=False)
print(f"\nDONE: {len(hist_df)} records for {hist_df['player_id'].nunique()} players")
if failed:
    print(f"Failed: {len(failed)} players")
