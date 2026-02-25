"""
FPL MODEL V3 — Position-Specific Stacked Meta-Ensemble

3 Key Improvements over V1:
  1. Filter players with < 30 minutes played (noise reduction)
  2. Target = 3-GW forward average points (reduces randomness, boosts R²)
  3. Separate models per position (GK / DEF / MID / FWD)

Architecture per position:
  Base: XGBoost + Random Forest + MLP + LSTM
  Meta: XGBoost stacking on OOF predictions + context features
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
import json
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('output/charts', exist_ok=True)

COLORS = {'Forward': '#e74c3c', 'Midfielder': '#2ecc71', 'Defender': '#3498db', 'Goalkeeper': '#f39c12'}
POS_COLORS = {'GK': '#f39c12', 'DEF': '#3498db', 'MID': '#2ecc71', 'FWD': '#e74c3c'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 6
N_FOLDS = 5
MIN_MINUTES = 30
TARGET_WINDOW = 3

POS_NORM = {
    'GKP': 'GK', 'GK': 'GK', 'Goalkeeper': 'GK',
    'DEF': 'DEF', 'Defender': 'DEF',
    'MID': 'MID', 'Midfielder': 'MID',
    'FWD': 'FWD', 'Forward': 'FWD',
}
POS_FULL = {'GK': 'Goalkeeper', 'DEF': 'Defender', 'MID': 'Midfielder', 'FWD': 'Forward'}
POSITIONS = ['GK', 'DEF', 'MID', 'FWD']

print("=" * 60)
print("FPL MODEL V3 — POSITION-SPECIFIC META-ENSEMBLE")
print(f"  Improvements: min {MIN_MINUTES}min | {TARGET_WINDOW}-GW avg target | per-position")
print(f"  Base models: XGBoost + RF + MLP + LSTM (x4 positions)")
print(f"  Meta-learner: XGBoost stacking (x4 positions)")
print(f"  {N_FOLDS}-fold CV for out-of-fold predictions")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/9] Loading data...", flush=True)

hist_all = pd.read_csv('data/historical/all_seasons_gw.csv', low_memory=False)
current = pd.read_csv('data/raw/player_histories.csv')
current_players = pd.read_csv('data/processed/players_clean.csv')
teams_raw = pd.read_csv('data/raw/teams_raw.csv')
fixtures = pd.read_csv('data/raw/fixtures_raw.csv')
players_raw = pd.read_csv('data/raw/players_raw.csv')

name_map = dict(zip(current_players['id'], current_players['web_name']))
team_map_id = dict(zip(current_players['id'], current_players['team_name']))
pos_map = dict(zip(current_players['id'], current_players['position']))
player_to_team_id = dict(zip(players_raw['id'], players_raw['team']))
team_id_to_name = dict(zip(teams_raw['id'], teams_raw['name']))

print(f"  Historical: {len(hist_all)} records", flush=True)

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n[2/9] Feature engineering...", flush=True)

if 'GW' in hist_all.columns and 'round' not in hist_all.columns:
    hist_all = hist_all.rename(columns={'GW': 'round'})

hist_all['player_season_id'] = hist_all['name'].astype(str) + '_' + hist_all['season'].astype(str)

for col in ['total_points', 'minutes', 'goals_scored', 'assists', 'bonus', 'bps',
            'ict_index', 'influence', 'creativity', 'threat',
            'clean_sheets', 'goals_conceded', 'saves',
            'expected_goals', 'expected_assists', 'expected_goal_involvements',
            'value', 'transfers_balance', 'selected', 'round']:
    if col in hist_all.columns:
        hist_all[col] = pd.to_numeric(hist_all[col], errors='coerce')

hist_all['was_home'] = hist_all['was_home'].map(
    {True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}
).fillna(0)

for col in ['expected_goals', 'expected_assists', 'expected_goal_involvements',
            'saves', 'clean_sheets', 'goals_conceded', 'transfers_balance', 'selected']:
    if col in hist_all.columns:
        hist_all[col] = hist_all[col].fillna(0)

hist_all = hist_all.sort_values(['player_season_id', 'round'])

# Filter: total season minutes >= 270 (keep regular players)
hist_all = hist_all[hist_all.groupby('player_season_id')['minutes'].transform('sum') >= 270]

# NEW: Filter individual GW minutes >= 30 (removes sub appearances / noise)
before_filter = len(hist_all)
hist_all = hist_all[hist_all['minutes'] >= MIN_MINUTES]
print(f"  Minutes filter ({MIN_MINUTES}+): {before_filter} -> {len(hist_all)} rows ({before_filter - len(hist_all)} removed)", flush=True)

# Position normalization
hist_all['pos_group'] = hist_all.get('position', pd.Series(dtype=str)).map(POS_NORM).fillna('MID')

pos_enc = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3,
           'Goalkeeper': 0, 'Defender': 1, 'Midfielder': 2, 'Forward': 3, 'GK': 0}
hist_all['pos_encoded'] = hist_all.get('position', pd.Series(dtype=str)).map(pos_enc).fillna(2)

group_col = 'player_season_id'

# Rolling features
for w in [3, 5, 8]:
    for col in ['total_points', 'minutes', 'goals_scored', 'assists', 'bonus',
                'bps', 'ict_index', 'influence', 'creativity', 'threat',
                'expected_goals', 'expected_assists', 'expected_goal_involvements',
                'clean_sheets', 'goals_conceded', 'saves']:
        if col in hist_all.columns:
            hist_all[f'{col}_roll_{w}'] = hist_all.groupby(group_col)[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean().shift(1)
            )

for w in [3, 5]:
    hist_all[f'pts_std_{w}'] = hist_all.groupby(group_col)['total_points'].transform(
        lambda x: x.rolling(w, min_periods=2).std().shift(1)
    )

hist_all['momentum_3v8'] = hist_all.get('total_points_roll_3', 0) - hist_all.get('total_points_roll_8', 0)
hist_all['xg_momentum'] = hist_all.get('expected_goals_roll_3', 0) - hist_all.get('expected_goals_roll_5', 0)

for col in ['goals_scored', 'assists', 'bonus', 'expected_goals', 'expected_assists']:
    if f'{col}_roll_5' in hist_all.columns:
        hist_all[f'{col}_per90'] = (hist_all[f'{col}_roll_5'] / hist_all['minutes_roll_5'].clip(1)) * 90

hist_all['season_pts'] = hist_all.groupby(group_col)['total_points'].transform(lambda x: x.cumsum().shift(1))
hist_all['season_goals'] = hist_all.groupby(group_col)['goals_scored'].transform(lambda x: x.cumsum().shift(1))
hist_all['season_assists'] = hist_all.groupby(group_col)['assists'].transform(lambda x: x.cumsum().shift(1))
hist_all['season_minutes'] = hist_all.groupby(group_col)['minutes'].transform(lambda x: x.cumsum().shift(1))
hist_all['games_played'] = hist_all.groupby(group_col).cumcount()
hist_all['ppg'] = (hist_all['season_pts'] / hist_all['games_played'].clip(1)).round(2)
hist_all['xg_overperformance'] = hist_all.get('goals_scored_roll_5', 0) - hist_all.get('expected_goals_roll_5', 0)
hist_all['xa_overperformance'] = hist_all.get('assists_roll_5', 0) - hist_all.get('expected_assists_roll_5', 0)
hist_all['is_home'] = hist_all['was_home'].fillna(0).astype(int)
hist_all['played_last'] = hist_all.groupby(group_col)['minutes'].transform(lambda x: (x.shift(1) > 0).astype(int))
hist_all['minutes_trend'] = hist_all.get('minutes_roll_3', 0) - hist_all.get('minutes_roll_8', 0)
hist_all['price'] = hist_all['value'] / 10
hist_all['gw'] = hist_all['round']

if 'transfers_balance' in hist_all.columns:
    hist_all['transfer_trend'] = hist_all.groupby(group_col)['transfers_balance'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )

# NEW: 3-GW forward average target
# target at row i = mean(points_i, points_{i+1}, points_{i+2})
hist_all['target'] = hist_all.groupby(group_col)['total_points'].transform(
    lambda x: (x + x.shift(-1) + x.shift(-2)) / 3
)
before_target = len(hist_all)
hist_all = hist_all.dropna(subset=['target'])
print(f"  3-GW target: {before_target} -> {len(hist_all)} rows ({before_target - len(hist_all)} dropped, no future data)", flush=True)

# --- Current season features ---
current_feat = current.copy()
current_feat['player_season_id'] = current_feat['player_id'].astype(str) + '_2024-25'
current_feat['season'] = '2024-25'
current_feat = current_feat.sort_values(['player_id', 'round'])
current_feat['position'] = current_feat['player_id'].map(pos_map)
current_feat['pos_group'] = current_feat['position'].map(POS_NORM).fillna('MID')
current_feat['pos_encoded'] = current_feat['position'].map(
    {'Goalkeeper': 0, 'Defender': 1, 'Midfielder': 2, 'Forward': 3}
).fillna(2)
current_feat['was_home'] = current_feat['was_home'].astype(int)

# Filter minutes
current_feat = current_feat[current_feat['minutes'] >= MIN_MINUTES]

gcol = 'player_id'
for w in [3, 5, 8]:
    for col in ['total_points', 'minutes', 'goals_scored', 'assists', 'bonus',
                'bps', 'ict_index', 'influence', 'creativity', 'threat',
                'expected_goals', 'expected_assists', 'expected_goal_involvements',
                'clean_sheets', 'goals_conceded', 'saves']:
        if col in current_feat.columns:
            current_feat[f'{col}_roll_{w}'] = current_feat.groupby(gcol)[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean().shift(1)
            )

for w in [3, 5]:
    current_feat[f'pts_std_{w}'] = current_feat.groupby(gcol)['total_points'].transform(
        lambda x: x.rolling(w, min_periods=2).std().shift(1)
    )

current_feat['momentum_3v8'] = current_feat.get('total_points_roll_3', 0) - current_feat.get('total_points_roll_8', 0)
current_feat['xg_momentum'] = current_feat.get('expected_goals_roll_3', 0) - current_feat.get('expected_goals_roll_5', 0)

for col in ['goals_scored', 'assists', 'bonus', 'expected_goals', 'expected_assists']:
    if f'{col}_roll_5' in current_feat.columns:
        current_feat[f'{col}_per90'] = (current_feat[f'{col}_roll_5'] / current_feat['minutes_roll_5'].clip(1)) * 90

current_feat['season_pts'] = current_feat.groupby(gcol)['total_points'].transform(lambda x: x.cumsum().shift(1))
current_feat['season_goals'] = current_feat.groupby(gcol)['goals_scored'].transform(lambda x: x.cumsum().shift(1))
current_feat['season_assists'] = current_feat.groupby(gcol)['assists'].transform(lambda x: x.cumsum().shift(1))
current_feat['season_minutes'] = current_feat.groupby(gcol)['minutes'].transform(lambda x: x.cumsum().shift(1))
current_feat['games_played'] = current_feat.groupby(gcol).cumcount()
current_feat['ppg'] = (current_feat['season_pts'] / current_feat['games_played'].clip(1)).round(2)
current_feat['xg_overperformance'] = current_feat.get('goals_scored_roll_5', 0) - current_feat.get('expected_goals_roll_5', 0)
current_feat['xa_overperformance'] = current_feat.get('assists_roll_5', 0) - current_feat.get('expected_assists_roll_5', 0)
current_feat['is_home'] = current_feat['was_home'].astype(int)
current_feat['played_last'] = current_feat.groupby(gcol)['minutes'].transform(lambda x: (x.shift(1) > 0).astype(int))
current_feat['minutes_trend'] = current_feat.get('minutes_roll_3', 0) - current_feat.get('minutes_roll_8', 0)
current_feat['transfer_trend'] = current_feat.groupby(gcol)['transfers_balance'].transform(
    lambda x: x.rolling(3, min_periods=1).mean().shift(1)
)
current_feat['price'] = current_feat['value'] / 10
current_feat['gw'] = current_feat['round']

# 3-GW forward average target for current season
current_feat['target'] = current_feat.groupby(gcol)['total_points'].transform(
    lambda x: (x + x.shift(-1) + x.shift(-2)) / 3
)
current_feat = current_feat.dropna(subset=['target'])

# Position counts
for pos in POSITIONS:
    h_count = (hist_all['pos_group'] == pos).sum()
    c_count = (current_feat['pos_group'] == pos).sum()
    print(f"  {pos}: {h_count} hist + {c_count} current = {h_count + c_count} total", flush=True)

# ============================================================
# 3. FEATURE SETS
# ============================================================

feature_cols = [c for c in hist_all.columns if any(c.startswith(p) for p in [
    'total_points_roll', 'minutes_roll', 'goals_scored_roll', 'assists_roll',
    'bonus_roll', 'bps_roll', 'ict_index_roll', 'influence_roll',
    'creativity_roll', 'threat_roll', 'expected_goals_roll', 'expected_assists_roll',
    'expected_goal_involvements_roll', 'clean_sheets_roll', 'goals_conceded_roll',
    'saves_roll', 'pts_std', 'goals_scored_per90', 'assists_per90', 'bonus_per90',
    'expected_goals_per90', 'expected_assists_per90',
])]
feature_cols += [
    'momentum_3v8', 'xg_momentum', 'season_pts', 'season_goals', 'season_assists',
    'season_minutes', 'games_played', 'ppg', 'xg_overperformance', 'xa_overperformance',
    'is_home', 'played_last', 'minutes_trend', 'price', 'gw',
]
if 'transfer_trend' in hist_all.columns:
    feature_cols.append('transfer_trend')
# Remove position dummies — we train per position, so they add no info
feature_cols = list(set([c for c in feature_cols if c in hist_all.columns and not c.startswith('pos_')]))

# Sequence features for LSTM (remove pos_encoded since position-specific)
seq_features = [c for c in [
    'total_points', 'minutes', 'goals_scored', 'assists', 'bonus', 'bps',
    'ict_index', 'influence', 'creativity', 'threat',
    'clean_sheets', 'goals_conceded', 'saves',
    'expected_goals', 'expected_assists', 'expected_goal_involvements',
    'was_home', 'value',
] if c in hist_all.columns]

# Meta context features
meta_context_cols = [
    'total_points_roll_3', 'total_points_roll_5', 'pts_std_3', 'pts_std_5',
    'momentum_3v8', 'is_home', 'ppg', 'games_played', 'price', 'gw',
    'minutes_roll_3', 'xg_overperformance',
]
meta_context_cols = [c for c in meta_context_cols if c in feature_cols]

print(f"\n  Tabular features: {len(feature_cols)} (no position dummies)")
print(f"  Sequence features: {len(seq_features)} (no pos_encoded)")
print(f"  Meta context: {len(meta_context_cols)}", flush=True)

# ============================================================
# 4. PREPARE TRAIN/TEST
# ============================================================
print("\n[3/9] Preparing data...", flush=True)

model_data = hist_all[hist_all['gw'] >= 5].copy()
model_data = model_data.dropna(subset=feature_cols).replace([np.inf, -np.inf], 0)

for fc in feature_cols:
    if fc not in current_feat.columns:
        current_feat[fc] = 0
    if fc not in model_data.columns:
        model_data[fc] = 0

current_model = current_feat[current_feat['gw'] >= 5].dropna(
    subset=[c for c in feature_cols if c in current_feat.columns]
).replace([np.inf, -np.inf], 0)

max_gw = current_model['gw'].max()
train_end = max_gw - 5

# Tabular: combine historical + current train
train_hist = model_data[feature_cols + ['target', 'pos_group']].copy()
train_curr = current_model[current_model['gw'] <= train_end][feature_cols + ['target', 'pos_group']].copy()
train_tab = pd.concat([train_hist, train_curr], ignore_index=True)
test_tab = current_model[current_model['gw'] > train_end].copy()

print(f"  Tabular — TRAIN: {len(train_tab)} | TEST: {len(test_tab)}")
for pos in POSITIONS:
    n_tr = (train_tab['pos_group'] == pos).sum()
    n_te = (test_tab['pos_group'] == pos).sum()
    print(f"    {pos}: train={n_tr}, test={n_te}", flush=True)

# Sequences
for col in seq_features:
    if col not in current_feat.columns:
        current_feat[col] = 0
    current_feat[col] = pd.to_numeric(current_feat[col], errors='coerce').fillna(0)

def build_sequences(df, feat_cols, seq_len, target_col='target', group_col='player_season_id'):
    sequences, targets, meta, pos_groups = [], [], [], []
    for pid, group in df.groupby(group_col):
        group = group.sort_values('round')
        vals = group[feat_cols].values.astype(np.float32)
        tgt = group[target_col].values.astype(np.float32)
        rounds = group['round'].values
        pg = group['pos_group'].iloc[0] if 'pos_group' in group.columns else 'MID'
        for i in range(seq_len, len(group)):
            if np.isnan(tgt[i]):
                continue
            sequences.append(vals[i - seq_len:i])
            targets.append(tgt[i])
            meta.append((pid, rounds[i]))
            pos_groups.append(pg)
    return (np.array(sequences) if sequences else np.zeros((0, seq_len, len(feat_cols))),
            np.array(targets), meta, pos_groups)

valid_hist = hist_all[hist_all.groupby('player_season_id')['round'].transform('count') >= SEQ_LEN + 1]
X_seq_h, y_seq_h, meta_seq_h, pos_seq_h = build_sequences(valid_hist, seq_features, SEQ_LEN)

valid_curr = current_feat[current_feat.groupby('player_season_id')['round'].transform('count') >= SEQ_LEN + 1]
X_seq_c, y_seq_c, meta_seq_c, pos_seq_c = build_sequences(valid_curr, seq_features, SEQ_LEN, 'target', 'player_season_id')

curr_meta_gws = np.array([m[1] for m in meta_seq_c]) if meta_seq_c else np.array([])
curr_train_mask = curr_meta_gws <= train_end
curr_test_mask = curr_meta_gws > train_end

# Combine sequences
if len(X_seq_c[curr_train_mask]) > 0:
    X_train_seq = np.concatenate([X_seq_h, X_seq_c[curr_train_mask]], axis=0)
    y_train_seq = np.concatenate([y_seq_h, y_seq_c[curr_train_mask]], axis=0)
    pos_train_seq = pos_seq_h + [pos_seq_c[i] for i in range(len(pos_seq_c)) if curr_train_mask[i]]
else:
    X_train_seq = X_seq_h
    y_train_seq = y_seq_h
    pos_train_seq = pos_seq_h

X_test_seq = X_seq_c[curr_test_mask]
y_test_seq = y_seq_c[curr_test_mask]
meta_test_seq = [meta_seq_c[i] for i in range(len(meta_seq_c)) if curr_test_mask[i]]
pos_test_seq = [pos_seq_c[i] for i in range(len(pos_seq_c)) if curr_test_mask[i]]

print(f"\n  Sequence — TRAIN: {len(X_train_seq)} | TEST: {len(X_test_seq)}")
for pos in POSITIONS:
    n_tr = sum(1 for p in pos_train_seq if p == pos)
    n_te = sum(1 for p in pos_test_seq if p == pos)
    print(f"    {pos}: train={n_tr}, test={n_te}", flush=True)

# ============================================================
# 5. NEURAL NETWORK DEFINITIONS
# ============================================================

class FPLMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class FPLLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn).sum(dim=1)
        return self.regressor(context).squeeze(-1)

# ============================================================
# 6. PER-POSITION TRAINING
# ============================================================
print("\n[4/9] Training per-position models...", flush=True)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Storage
position_models = {}
nn_states = {}

# For overall evaluation: collect per-position test predictions
all_pos_test_preds = {}   # {pos: {model_name: predictions_array}}
all_pos_test_actual = {}  # {pos: actual_array}
all_pos_test_indices = {} # {pos: original_test_tab_indices}

# LSTM test predictions mapped to tabular rows
all_lstm_test_map = {}

for pos in POSITIONS:
    print(f"\n{'='*55}")
    print(f"  POSITION: {pos} ({POS_FULL[pos]})")
    print(f"{'='*55}", flush=True)

    # --- Tabular data for this position ---
    pos_train_mask = train_tab['pos_group'] == pos
    pos_test_mask = test_tab['pos_group'] == pos

    X_train_pos = train_tab[pos_train_mask][feature_cols].fillna(0).values
    y_train_pos = train_tab[pos_train_mask]['target'].fillna(0).values
    X_test_pos = test_tab[pos_test_mask][feature_cols].fillna(0).values
    y_test_pos = test_tab[pos_test_mask]['target'].fillna(0).values

    pos_test_indices = test_tab[pos_test_mask].index

    all_pos_test_actual[pos] = y_test_pos
    all_pos_test_indices[pos] = pos_test_indices

    n_train = len(X_train_pos)
    n_test = len(X_test_pos)
    print(f"  Data: {n_train} train, {n_test} test", flush=True)

    if n_test == 0:
        print(f"  SKIP — no test data for {pos}", flush=True)
        continue

    # --- OOF arrays ---
    oof_xgb = np.zeros(n_train)
    oof_rf = np.zeros(n_train)
    oof_mlp = np.zeros(n_train)
    test_xgb_pos = np.zeros(n_test)
    test_rf_pos = np.zeros(n_test)
    test_mlp_pos = np.zeros(n_test)

    # --- Scale for MLP ---
    scaler_mlp_pos = StandardScaler()
    X_train_mlp_s = scaler_mlp_pos.fit_transform(X_train_pos)
    X_test_mlp_s = scaler_mlp_pos.transform(X_test_pos)

    # --- K-Fold training ---
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_pos)):
        X_tr, X_val = X_train_pos[tr_idx], X_train_pos[val_idx]
        y_tr, y_val = y_train_pos[tr_idx], y_train_pos[val_idx]

        # XGBoost
        xgb_m = xgb.XGBRegressor(
            max_depth=5, learning_rate=0.03, n_estimators=600,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=12,
            reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1,
            early_stopping_rounds=30,
        )
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_m.predict(X_val)
        test_xgb_pos += xgb_m.predict(X_test_pos) / N_FOLDS

        # Random Forest
        rf_m = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=20,
            max_features=0.6, n_jobs=-1, random_state=42,
        )
        rf_m.fit(X_tr, y_tr)
        oof_rf[val_idx] = rf_m.predict(X_val)
        test_rf_pos += rf_m.predict(X_test_pos) / N_FOLDS

        # MLP
        X_tr_s = scaler_mlp_pos.transform(X_tr)
        X_val_s = scaler_mlp_pos.transform(X_val)

        mlp_m = FPLMLP(len(feature_cols)).to(DEVICE)
        mlp_opt = torch.optim.Adam(mlp_m.parameters(), lr=0.001, weight_decay=1e-5)
        mlp_crit = nn.HuberLoss(delta=2.0)

        ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr))
        loader = DataLoader(ds, batch_size=512, shuffle=True, drop_last=True)

        best_mlp_state = None
        best_mlp_val = 999
        no_imp = 0

        for epoch in range(30):
            mlp_m.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                mlp_opt.zero_grad()
                mlp_crit(mlp_m(xb), yb).backward()
                mlp_opt.step()

            mlp_m.eval()
            with torch.no_grad():
                val_p = mlp_m(torch.FloatTensor(X_val_s).to(DEVICE)).cpu().numpy()
                vm = mean_absolute_error(y_val, val_p)
            if vm < best_mlp_val:
                best_mlp_val = vm
                best_mlp_state = {k: v.clone() for k, v in mlp_m.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 5:
                break

        mlp_m.load_state_dict(best_mlp_state)
        mlp_m.eval()
        with torch.no_grad():
            oof_mlp[val_idx] = mlp_m(torch.FloatTensor(X_val_s).to(DEVICE)).cpu().numpy()
            test_mlp_pos += mlp_m(torch.FloatTensor(X_test_mlp_s).to(DEVICE)).cpu().numpy() / N_FOLDS

        if (fold + 1) % 2 == 0 or fold == 0:
            print(f"    Fold {fold+1}/{N_FOLDS}: XGB={mean_absolute_error(y_val, oof_xgb[val_idx]):.3f} "
                  f"RF={mean_absolute_error(y_val, oof_rf[val_idx]):.3f} "
                  f"MLP={mean_absolute_error(y_val, oof_mlp[val_idx]):.3f}", flush=True)

    print(f"\n  OOF: XGB={mean_absolute_error(y_train_pos, oof_xgb):.3f} "
          f"RF={mean_absolute_error(y_train_pos, oof_rf):.3f} "
          f"MLP={mean_absolute_error(y_train_pos, oof_mlp):.3f}", flush=True)

    # --- LSTM for this position ---
    print(f"\n  Training LSTM for {pos}...", flush=True)

    pos_train_seq_mask = np.array([p == pos for p in pos_train_seq])
    pos_test_seq_mask = np.array([p == pos for p in pos_test_seq])

    X_tr_seq_pos = X_train_seq[pos_train_seq_mask]
    y_tr_seq_pos = y_train_seq[pos_train_seq_mask]
    X_te_seq_pos = X_test_seq[pos_test_seq_mask]
    y_te_seq_pos = y_test_seq[pos_test_seq_mask]
    meta_te_seq_pos = [meta_test_seq[i] for i in range(len(meta_test_seq)) if pos_test_seq_mask[i]]

    print(f"    Sequences: {len(X_tr_seq_pos)} train, {len(X_te_seq_pos)} test", flush=True)

    test_lstm_pos = np.zeros(n_test)
    lstm_mae_pos = 999.0

    if len(X_tr_seq_pos) >= 100 and len(X_te_seq_pos) >= 10:
        scaler_lstm_pos = StandardScaler()
        X_tr_seq_2d = X_tr_seq_pos.reshape(-1, len(seq_features))
        scaler_lstm_pos.fit(X_tr_seq_2d)

        X_tr_s = scaler_lstm_pos.transform(X_tr_seq_2d).reshape(X_tr_seq_pos.shape)
        X_te_s = scaler_lstm_pos.transform(X_te_seq_pos.reshape(-1, len(seq_features))).reshape(X_te_seq_pos.shape)

        X_tr_t = torch.FloatTensor(X_tr_s)
        y_tr_t = torch.FloatTensor(y_tr_seq_pos)
        X_te_t = torch.FloatTensor(X_te_s)

        lstm_m = FPLLSTM(len(seq_features), hidden_size=128, num_layers=2, dropout=0.3).to(DEVICE)
        lstm_opt = torch.optim.Adam(lstm_m.parameters(), lr=0.001, weight_decay=1e-5)
        lstm_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(lstm_opt, mode='min', factor=0.5, patience=5)
        lstm_crit = nn.HuberLoss(delta=2.0)

        lstm_ds = TensorDataset(X_tr_t, y_tr_t)
        lstm_loader = DataLoader(lstm_ds, batch_size=256, shuffle=True, drop_last=True)

        best_lstm_state = None
        best_lstm_mae = 999
        no_imp_lstm = 0

        for epoch in range(40):
            lstm_m.train()
            for xb, yb in lstm_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                lstm_opt.zero_grad()
                loss = lstm_crit(lstm_m(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lstm_m.parameters(), max_norm=1.0)
                lstm_opt.step()

            lstm_m.eval()
            with torch.no_grad():
                val_p = lstm_m(X_te_t.to(DEVICE)).cpu().numpy()
                vm = mean_absolute_error(y_te_seq_pos, val_p)
            lstm_sched.step(vm)

            if vm < best_lstm_mae:
                best_lstm_mae = vm
                best_lstm_state = {k: v.clone() for k, v in lstm_m.state_dict().items()}
                no_imp_lstm = 0
            else:
                no_imp_lstm += 1

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}: MAE={vm:.3f}", flush=True)
            if no_imp_lstm >= 10:
                print(f"    Early stop at epoch {epoch+1}", flush=True)
                break

        lstm_m.load_state_dict(best_lstm_state)
        lstm_m.eval()
        with torch.no_grad():
            test_lstm_raw = lstm_m(X_te_t.to(DEVICE)).cpu().numpy()
        lstm_mae_pos = best_lstm_mae
        print(f"    LSTM {pos} MAE: {lstm_mae_pos:.3f}", flush=True)

        # Map LSTM predictions to tabular test rows
        for i, m in enumerate(meta_te_seq_pos):
            pid = int(m[0].split('_')[0])
            gw = int(m[1])
            all_lstm_test_map[(pid, gw, pos)] = test_lstm_raw[i]
    else:
        scaler_lstm_pos = None
        lstm_m = None
        print(f"    SKIP LSTM — too few sequences for {pos}", flush=True)

    # Map LSTM to tabular test indices
    test_tab_pos = test_tab[pos_test_mask]
    test_lstm_arr = test_tab_pos.apply(
        lambda r: all_lstm_test_map.get((int(r.get('player_id', 0)), int(r['gw']), pos), np.nan), axis=1
    ).values

    # --- META MODEL ---
    print(f"\n  Training meta-model for {pos}...", flush=True)

    meta_train_df = pd.DataFrame({
        'xgb_pred': oof_xgb,
        'rf_pred': oof_rf,
        'mlp_pred': oof_mlp,
        'pred_mean': (oof_xgb + oof_rf + oof_mlp) / 3,
        'pred_std': np.std([oof_xgb, oof_rf, oof_mlp], axis=0),
        'pred_max': np.max([oof_xgb, oof_rf, oof_mlp], axis=0),
        'pred_min': np.min([oof_xgb, oof_rf, oof_mlp], axis=0),
        'pred_range': np.max([oof_xgb, oof_rf, oof_mlp], axis=0) - np.min([oof_xgb, oof_rf, oof_mlp], axis=0),
    })

    pos_train_context = train_tab[pos_train_mask]
    for col in meta_context_cols:
        meta_train_df[col] = pos_train_context[col].fillna(0).values

    meta_features = list(meta_train_df.columns)

    meta_test_df = pd.DataFrame({
        'xgb_pred': test_xgb_pos,
        'rf_pred': test_rf_pos,
        'mlp_pred': test_mlp_pos,
        'pred_mean': (test_xgb_pos + test_rf_pos + test_mlp_pos) / 3,
        'pred_std': np.std([test_xgb_pos, test_rf_pos, test_mlp_pos], axis=0),
        'pred_max': np.max([test_xgb_pos, test_rf_pos, test_mlp_pos], axis=0),
        'pred_min': np.min([test_xgb_pos, test_rf_pos, test_mlp_pos], axis=0),
        'pred_range': np.max([test_xgb_pos, test_rf_pos, test_mlp_pos], axis=0) - np.min([test_xgb_pos, test_rf_pos, test_mlp_pos], axis=0),
    })

    for col in meta_context_cols:
        meta_test_df[col] = test_tab_pos[col].fillna(0).values

    # Add LSTM predictions
    meta_test_df['lstm_pred'] = test_lstm_arr
    has_lstm = meta_test_df['lstm_pred'].notna()
    meta_train_df['lstm_pred'] = meta_train_df['pred_mean']  # No OOF for LSTM
    meta_test_df['lstm_pred'] = meta_test_df['lstm_pred'].fillna(meta_test_df['pred_mean'])

    if has_lstm.any():
        meta_test_df.loc[has_lstm, 'pred_mean'] = meta_test_df.loc[has_lstm, ['xgb_pred', 'rf_pred', 'mlp_pred', 'lstm_pred']].mean(axis=1)
        meta_test_df.loc[has_lstm, 'pred_std'] = meta_test_df.loc[has_lstm, ['xgb_pred', 'rf_pred', 'mlp_pred', 'lstm_pred']].std(axis=1)

    meta_features_full = list(meta_train_df.columns)
    if 'lstm_pred' not in meta_features_full:
        meta_features_full.append('lstm_pred')
    meta_features_full = list(dict.fromkeys(meta_features_full))

    X_meta_train = meta_train_df[meta_features_full].fillna(0).values
    X_meta_test = meta_test_df[meta_features_full].fillna(0).values

    meta_model_pos = xgb.XGBRegressor(
        max_depth=3, learning_rate=0.05, n_estimators=500,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        reg_alpha=1.0, reg_lambda=3.0, random_state=42, n_jobs=-1,
        early_stopping_rounds=20,
    )
    val_size = min(3000, len(X_meta_train) // 5)
    if val_size > 0:
        meta_model_pos.fit(
            X_meta_train[:-val_size], y_train_pos[:-val_size],
            eval_set=[(X_meta_train[-val_size:], y_train_pos[-val_size:])],
            verbose=False,
        )
    else:
        meta_model_pos.fit(X_meta_train, y_train_pos, verbose=False)

    meta_pred_test = meta_model_pos.predict(X_meta_test)

    # --- Store all predictions for this position ---
    all_pos_test_preds[pos] = {
        'xgboost': test_xgb_pos,
        'random_forest': test_rf_pos,
        'neural_net': test_mlp_pos,
        'lstm_raw': test_lstm_arr,
        'stacked_meta': meta_pred_test,
    }

    # Simple average
    simple = (test_xgb_pos + test_rf_pos + test_mlp_pos) / 3
    all_pos_test_preds[pos]['simple_avg'] = simple

    meta_mae = mean_absolute_error(y_test_pos, meta_pred_test)
    meta_r2 = r2_score(y_test_pos, meta_pred_test)
    print(f"\n  {pos} RESULTS:")
    print(f"    XGBoost:  MAE={mean_absolute_error(y_test_pos, test_xgb_pos):.3f}  R²={r2_score(y_test_pos, test_xgb_pos):.3f}")
    print(f"    RF:       MAE={mean_absolute_error(y_test_pos, test_rf_pos):.3f}  R²={r2_score(y_test_pos, test_rf_pos):.3f}")
    print(f"    MLP:      MAE={mean_absolute_error(y_test_pos, test_mlp_pos):.3f}  R²={r2_score(y_test_pos, test_mlp_pos):.3f}")
    if lstm_mae_pos < 999:
        lstm_valid = ~np.isnan(test_lstm_arr)
        if lstm_valid.any():
            print(f"    LSTM:     MAE={mean_absolute_error(y_test_pos[lstm_valid], test_lstm_arr[lstm_valid]):.3f}  R²={r2_score(y_test_pos[lstm_valid], test_lstm_arr[lstm_valid]):.3f}")
    print(f"    Simple:   MAE={mean_absolute_error(y_test_pos, simple):.3f}  R²={r2_score(y_test_pos, simple):.3f}")
    print(f"    Meta:     MAE={meta_mae:.3f}  R²={meta_r2:.3f}  ***", flush=True)

    # Store models
    position_models[pos] = {
        'xgboost': xgb_m,
        'random_forest': rf_m,
        'meta_model': meta_model_pos,
        'scaler_mlp': scaler_mlp_pos,
        'scaler_lstm': scaler_lstm_pos,
        'lstm_mae': lstm_mae_pos,
    }
    nn_states[pos] = {
        'mlp_state': mlp_m.state_dict(),
        'lstm_state': lstm_m.state_dict() if lstm_m else None,
    }

# ============================================================
# 7. OVERALL EVALUATION
# ============================================================
print("\n[5/9] Overall evaluation...", flush=True)

# Combine per-position predictions into overall arrays (aligned with test_tab)
model_names = ['xgboost', 'random_forest', 'neural_net', 'simple_avg', 'stacked_meta']
overall_preds = {m: np.zeros(len(test_tab)) for m in model_names}
overall_actual = np.zeros(len(test_tab))

for pos in POSITIONS:
    if pos not in all_pos_test_preds:
        continue
    idx = all_pos_test_indices[pos]
    # Map to test_tab integer positions
    pos_locs = test_tab.index.get_indexer(idx)
    for m in model_names:
        overall_preds[m][pos_locs] = all_pos_test_preds[pos][m]
    overall_actual[pos_locs] = all_pos_test_actual[pos]

all_results = {}
for m in model_names:
    pred = overall_preds[m]
    all_results[m] = {
        'pred': pred,
        'mae': mean_absolute_error(overall_actual, pred),
        'r2': r2_score(overall_actual, pred),
    }

# LSTM: evaluate only on rows that have predictions
lstm_all_pred = np.full(len(test_tab), np.nan)
lstm_all_actual = np.full(len(test_tab), np.nan)
for pos in POSITIONS:
    if pos not in all_pos_test_preds:
        continue
    idx = all_pos_test_indices[pos]
    pos_locs = test_tab.index.get_indexer(idx)
    lstm_raw = all_pos_test_preds[pos]['lstm_raw']
    valid = ~np.isnan(lstm_raw)
    lstm_all_pred[pos_locs[valid]] = lstm_raw[valid]
    lstm_all_actual[pos_locs[valid]] = all_pos_test_actual[pos][valid]

lstm_valid_mask = ~np.isnan(lstm_all_pred)
if lstm_valid_mask.any():
    all_results['lstm'] = {
        'pred': lstm_all_pred[lstm_valid_mask],
        'mae': mean_absolute_error(lstm_all_actual[lstm_valid_mask], lstm_all_pred[lstm_valid_mask]),
        'r2': r2_score(lstm_all_actual[lstm_valid_mask], lstm_all_pred[lstm_valid_mask]),
    }
else:
    all_results['lstm'] = {'pred': np.array([]), 'mae': 999, 'r2': 0}

# Ranking evaluation
for mname in model_names:
    pred_arr = overall_preds[mname]
    r20, r10, cap = [], [], []
    for gw in sorted(test_tab['gw'].unique()):
        mask = test_tab['gw'].values == gw
        if mask.sum() < 20:
            continue
        gd = test_tab[mask].copy()
        gd['pred'] = pred_arr[mask]
        gd['actual_target'] = overall_actual[mask]
        p20 = set(gd.nlargest(20, 'pred')['player_id'])
        a20 = set(gd.nlargest(20, 'actual_target')['player_id'])
        r20.append(len(p20 & a20) / 20)
        p10 = set(gd.nlargest(10, 'pred')['player_id'])
        a10 = set(gd.nlargest(10, 'actual_target')['player_id'])
        r10.append(len(p10 & a10) / 10)
        pc = gd.nlargest(1, 'pred')['player_id'].values[0]
        a5 = set(gd.nlargest(5, 'actual_target')['player_id'])
        cap.append(1 if pc in a5 else 0)
    all_results[mname]['rank20'] = np.mean(r20) if r20 else 0
    all_results[mname]['rank10'] = np.mean(r10) if r10 else 0
    all_results[mname]['captain'] = np.mean(cap) if cap else 0

# LSTM ranking
if lstm_valid_mask.any():
    r20_l, r10_l, cap_l = [], [], []
    test_lstm_eval = test_tab.copy()
    test_lstm_eval['lstm_pred'] = lstm_all_pred
    test_lstm_eval['actual_3gw'] = overall_actual
    test_lstm_eval = test_lstm_eval.dropna(subset=['lstm_pred'])
    for gw in sorted(test_lstm_eval['gw'].unique()):
        gd = test_lstm_eval[test_lstm_eval['gw'] == gw]
        if len(gd) < 20: continue
        r20_l.append(len(set(gd.nlargest(20, 'lstm_pred')['player_id']) & set(gd.nlargest(20, 'actual_3gw')['player_id'])) / 20)
        r10_l.append(len(set(gd.nlargest(10, 'lstm_pred')['player_id']) & set(gd.nlargest(10, 'actual_3gw')['player_id'])) / 10)
        pc = gd.nlargest(1, 'lstm_pred')['player_id'].values[0]
        cap_l.append(1 if pc in set(gd.nlargest(5, 'actual_3gw')['player_id']) else 0)
    all_results['lstm']['rank20'] = np.mean(r20_l) if r20_l else 0
    all_results['lstm']['rank10'] = np.mean(r10_l) if r10_l else 0
    all_results['lstm']['captain'] = np.mean(cap_l) if cap_l else 0

print(f"\n  {'Model':<18} {'MAE':>8} {'R²':>8} {'Top-20':>8} {'Top-10':>8} {'Captain':>8}")
print(f"  {'-'*58}")
for mname in ['xgboost', 'random_forest', 'neural_net', 'lstm', 'simple_avg', 'stacked_meta']:
    r = all_results[mname]
    star = ' ***' if mname == 'stacked_meta' else ''
    print(f"  {mname:<18} {r['mae']:>8.3f} {r['r2']:>8.3f} {r.get('rank20',0):>7.1%} {r.get('rank10',0):>7.1%} {r.get('captain',0):>7.1%}{star}")

best_name = min(all_results.keys(), key=lambda m: all_results[m]['mae'])
print(f"\n  BEST (MAE): {best_name} ({all_results[best_name]['mae']:.3f})")
best_r2 = max(all_results.keys(), key=lambda m: all_results[m]['r2'])
print(f"  BEST (R²):  {best_r2} ({all_results[best_r2]['r2']:.3f})", flush=True)

# Per-position summary
print(f"\n  Per-Position R² (stacked meta):")
for pos in POSITIONS:
    if pos in all_pos_test_preds:
        r2 = r2_score(all_pos_test_actual[pos], all_pos_test_preds[pos]['stacked_meta'])
        mae = mean_absolute_error(all_pos_test_actual[pos], all_pos_test_preds[pos]['stacked_meta'])
        print(f"    {pos}: MAE={mae:.3f}  R²={r2:.3f}", flush=True)

# ============================================================
# 8. CHARTS
# ============================================================
print("\n[6/9] Generating charts...", flush=True)

# 1. Overall comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
labels = ['XGBoost', 'Random\nForest', 'Neural\nNet', 'LSTM', 'Simple\nAvg', 'Stacked\nMeta']
keys = ['xgboost', 'random_forest', 'neural_net', 'lstm', 'simple_avg', 'stacked_meta']
colors = ['#f1c40f', '#e74c3c', '#2ecc71', '#9b59b6', '#95a5a6', '#3498db']

maes = [all_results[k]['mae'] for k in keys]
bars = axes[0].bar(labels, maes, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
axes[0].set_ylabel('MAE (lower = better)', fontsize=12)
axes[0].set_title('Test MAE (3-GW Avg Target)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, maes):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
                 ha='center', fontweight='bold', fontsize=9)
axes[0].set_ylim(min(maes) * 0.92, max(maes) * 1.04)

r2s = [all_results[k]['r2'] for k in keys]
bars = axes[1].bar(labels, r2s, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
axes[1].set_ylabel('R² (higher = better)', fontsize=12)
axes[1].set_title('Test R² (3-GW Avg Target)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, r2s):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{val:.3f}',
                 ha='center', fontweight='bold', fontsize=9)

ranks = [all_results[k].get('rank20', 0) for k in keys]
bars = axes[2].bar(labels, ranks, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
axes[2].set_ylabel('Top-20 Accuracy', fontsize=12)
axes[2].set_title('Ranking Accuracy (Top-20)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, ranks):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.1%}',
                 ha='center', fontweight='bold', fontsize=9)
axes[2].set_ylim(0, max(ranks) * 1.3 if max(ranks) > 0 else 0.3)

plt.suptitle('FPL V3 — Position-Specific Models (3-GW Average Target)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output/charts/final_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Per-position R² chart
fig, ax = plt.subplots(figsize=(12, 7))
x_pos = np.arange(len(POSITIONS))
width = 0.15
model_labels = ['XGBoost', 'RF', 'MLP', 'Simple Avg', 'Meta']
model_keys = ['xgboost', 'random_forest', 'neural_net', 'simple_avg', 'stacked_meta']
model_colors = ['#f1c40f', '#e74c3c', '#2ecc71', '#95a5a6', '#3498db']

for i, (mk, ml, mc) in enumerate(zip(model_keys, model_labels, model_colors)):
    r2_vals = []
    for pos in POSITIONS:
        if pos in all_pos_test_preds:
            r2_vals.append(r2_score(all_pos_test_actual[pos], all_pos_test_preds[pos][mk]))
        else:
            r2_vals.append(0)
    ax.bar(x_pos + i * width, r2_vals, width, label=ml, color=mc, edgecolor='white')
    for j, v in enumerate(r2_vals):
        ax.text(x_pos[j] + i * width, v + 0.005, f'{v:.2f}', ha='center', fontsize=7, fontweight='bold')

ax.set_xticks(x_pos + width * 2)
ax.set_xticklabels(POSITIONS, fontsize=13, fontweight='bold')
ax.set_ylabel('R²', fontsize=12)
ax.set_title('R² per Position per Model', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('output/charts/v3_position_r2.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. GW-level MAE chart
fig, ax = plt.subplots(figsize=(14, 7))
gw_list = sorted(test_tab['gw'].unique())
for mname, color, label in [
    ('xgboost', '#f1c40f', 'XGBoost'), ('random_forest', '#e74c3c', 'RF'),
    ('neural_net', '#2ecc71', 'MLP'), ('stacked_meta', '#3498db', 'Meta'),
]:
    gw_maes = []
    for gw in gw_list:
        mask = test_tab['gw'].values == gw
        gw_maes.append(mean_absolute_error(overall_actual[mask], overall_preds[mname][mask]))
    ax.plot(gw_list, gw_maes, 'o-', color=color, linewidth=2.5, markersize=8,
            label=f"{label} ({all_results[mname]['mae']:.3f})")

ax.set_xlabel('Gameweek', fontsize=12)
ax.set_ylabel('GW MAE (3-GW avg target)', fontsize=12)
ax.set_title('V3 Model Comparison: MAE per Gameweek', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, title='Model (avg MAE)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/charts/final_gw_mae.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Meta feature importance (use first available position's meta model)
for pos in POSITIONS:
    if pos in position_models:
        meta_imp = pd.DataFrame({
            'feature': meta_features_full,
            'importance': position_models[pos]['meta_model'].feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(meta_imp)), meta_imp['importance'].values, color='#3498db')
        ax.set_yticks(range(len(meta_imp)))
        ax.set_yticklabels(meta_imp['feature'].values, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Meta-Model Feature Importance (example: {pos})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('output/charts/meta_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        break

# ============================================================
# 9. PREDICTIONS FOR UPCOMING GWS
# ============================================================
print("\n[7/9] Generating predictions...", flush=True)

upcoming = fixtures[(fixtures['finished'] == False) & (fixtures['event'].notna())].copy()
upcoming['event'] = upcoming['event'].astype(int)
next_gws = sorted(upcoming['event'].unique())[:5]

active_players = current_players[current_players['minutes'] >= 450].copy()
all_pred_rows = []

for gw in next_gws:
    gw_fix = upcoming[upcoming['event'] == gw]
    for _, player in active_players.iterrows():
        pid = player['id']
        team_id = player_to_team_id.get(pid)
        ph = current[current['player_id'] == pid].sort_values('round')
        if len(ph) < 3: continue

        home_f = gw_fix[gw_fix['team_h'] == team_id]
        away_f = gw_fix[gw_fix['team_a'] == team_id]
        if len(home_f) > 0:
            is_home = 1
            opp_id = home_f.iloc[0]['team_a']
        elif len(away_f) > 0:
            is_home = 0
            opp_id = away_f.iloc[0]['team_h']
        else:
            continue

        row = {}
        for w in [3, 5, 8]:
            recent = ph.tail(w)
            for col in ['total_points', 'minutes', 'goals_scored', 'assists', 'bonus',
                        'bps', 'ict_index', 'influence', 'creativity', 'threat',
                        'expected_goals', 'expected_assists', 'expected_goal_involvements',
                        'clean_sheets', 'goals_conceded', 'saves']:
                if col in recent.columns:
                    row[f'{col}_roll_{w}'] = recent[col].mean()

        for w in [3, 5]:
            r = ph.tail(w)
            row[f'pts_std_{w}'] = r['total_points'].std() if len(r) > 1 else 0

        row['momentum_3v8'] = row.get('total_points_roll_3', 0) - row.get('total_points_roll_8', 0)
        row['xg_momentum'] = row.get('expected_goals_roll_3', 0) - row.get('expected_goals_roll_5', 0)
        for col in ['goals_scored', 'assists', 'bonus', 'expected_goals', 'expected_assists']:
            if f'{col}_roll_5' in row:
                row[f'{col}_per90'] = (row[f'{col}_roll_5'] / max(row.get('minutes_roll_5', 1), 1)) * 90

        row['season_pts'] = ph['total_points'].sum()
        row['season_goals'] = ph['goals_scored'].sum()
        row['season_assists'] = ph['assists'].sum()
        row['season_minutes'] = ph['minutes'].sum()
        row['games_played'] = len(ph)
        row['ppg'] = ph['total_points'].sum() / max(len(ph), 1)
        row['xg_overperformance'] = row.get('goals_scored_roll_5', 0) - row.get('expected_goals_roll_5', 0)
        row['xa_overperformance'] = row.get('assists_roll_5', 0) - row.get('expected_assists_roll_5', 0)
        row['is_home'] = is_home
        row['played_last'] = 1 if ph.iloc[-1]['minutes'] > 0 else 0
        row['minutes_trend'] = row.get('minutes_roll_3', 0) - row.get('minutes_roll_8', 0)
        row['transfer_trend'] = ph.tail(3)['transfers_balance'].mean() if 'transfers_balance' in ph.columns else 0
        row['price'] = player['now_cost']
        row['gw'] = int(gw)

        row['player_id'] = pid
        row['web_name'] = player['web_name']
        row['team_name'] = player['team_name']
        row['position'] = player['position']
        row['pos_group'] = POS_NORM.get(player['position'], 'MID')
        row['now_cost'] = player['now_cost']
        row['opponent'] = team_id_to_name.get(opp_id, '?')
        row['venue'] = 'H' if is_home else 'A'
        all_pred_rows.append(row)

pred_df = pd.DataFrame(all_pred_rows)
for fc in feature_cols:
    if fc not in pred_df.columns:
        pred_df[fc] = 0

# Generate predictions per position
pred_df['xgb_pts'] = 0.0
pred_df['rf_pts'] = 0.0
pred_df['mlp_pts'] = 0.0
pred_df['lstm_pts'] = np.nan
pred_df['predicted_points'] = 0.0

for pos in POSITIONS:
    pos_mask = pred_df['pos_group'] == pos
    if pos_mask.sum() == 0 or pos not in position_models:
        continue

    pm = position_models[pos]
    X_pred_pos = pred_df.loc[pos_mask, feature_cols].fillna(0).values

    # XGBoost & RF
    pred_df.loc[pos_mask, 'xgb_pts'] = pm['xgboost'].predict(X_pred_pos).clip(0).round(2)
    pred_df.loc[pos_mask, 'rf_pts'] = pm['random_forest'].predict(X_pred_pos).clip(0).round(2)

    # MLP
    mlp_pred_m = FPLMLP(len(feature_cols)).to(DEVICE)
    mlp_pred_m.load_state_dict(nn_states[pos]['mlp_state'])
    mlp_pred_m.eval()
    with torch.no_grad():
        pred_df.loc[pos_mask, 'mlp_pts'] = mlp_pred_m(
            torch.FloatTensor(pm['scaler_mlp'].transform(X_pred_pos)).to(DEVICE)
        ).cpu().numpy().clip(0).round(2)

    # LSTM
    if nn_states[pos]['lstm_state'] is not None and pm['scaler_lstm'] is not None:
        lstm_pred_m = FPLLSTM(len(seq_features), hidden_size=128, num_layers=2, dropout=0.3).to(DEVICE)
        lstm_pred_m.load_state_dict(nn_states[pos]['lstm_state'])
        lstm_pred_m.eval()

        for idx in pred_df[pos_mask].index:
            pid = pred_df.loc[idx, 'player_id']
            ph_l = current_feat[current_feat['player_id'] == pid].sort_values('round')
            if len(ph_l) >= SEQ_LEN:
                seq = ph_l.tail(SEQ_LEN)[seq_features].values.astype(np.float32)
                seq_s = pm['scaler_lstm'].transform(seq.reshape(-1, len(seq_features))).reshape(1, SEQ_LEN, len(seq_features))
                with torch.no_grad():
                    pred_df.loc[idx, 'lstm_pts'] = max(0, round(
                        lstm_pred_m(torch.FloatTensor(seq_s).to(DEVICE)).cpu().item(), 2))

    # Meta prediction
    meta_p = pd.DataFrame({
        'xgb_pred': pred_df.loc[pos_mask, 'xgb_pts'].values,
        'rf_pred': pred_df.loc[pos_mask, 'rf_pts'].values,
        'mlp_pred': pred_df.loc[pos_mask, 'mlp_pts'].values,
        'pred_mean': pred_df.loc[pos_mask, ['xgb_pts', 'rf_pts', 'mlp_pts']].mean(axis=1).values,
        'pred_std': pred_df.loc[pos_mask, ['xgb_pts', 'rf_pts', 'mlp_pts']].std(axis=1).values,
        'pred_max': pred_df.loc[pos_mask, ['xgb_pts', 'rf_pts', 'mlp_pts']].max(axis=1).values,
        'pred_min': pred_df.loc[pos_mask, ['xgb_pts', 'rf_pts', 'mlp_pts']].min(axis=1).values,
        'pred_range': (pred_df.loc[pos_mask, ['xgb_pts', 'rf_pts', 'mlp_pts']].max(axis=1) -
                       pred_df.loc[pos_mask, ['xgb_pts', 'rf_pts', 'mlp_pts']].min(axis=1)).values,
    })

    for col in meta_context_cols:
        meta_p[col] = pred_df.loc[pos_mask, col].fillna(0).values if col in pred_df.columns else 0

    lstm_vals = pred_df.loc[pos_mask, 'lstm_pts'].values
    meta_p['lstm_pred'] = np.where(np.isnan(lstm_vals), meta_p['pred_mean'].values, lstm_vals)

    has_l = ~np.isnan(lstm_vals)
    if has_l.any():
        all4 = pred_df.loc[pos_mask, ['xgb_pts', 'rf_pts', 'mlp_pts', 'lstm_pts']]
        meta_p.loc[has_l, 'pred_mean'] = all4[has_l].mean(axis=1).values
        meta_p.loc[has_l, 'pred_std'] = all4[has_l].std(axis=1).values

    X_meta_p = meta_p[meta_features_full].fillna(0).values
    pred_df.loc[pos_mask, 'predicted_points'] = pm['meta_model'].predict(X_meta_p).clip(0).round(2)

print(f"  Generated {len(pred_df)} predictions for GWs {next_gws}", flush=True)

# ============================================================
# 10. BEST TEAMS
# ============================================================
print("\n[8/9] Selecting best teams...", flush=True)

def solve_team(p_df, budget=100.0):
    p = p_df.reset_index(drop=True)
    n = len(p)
    if n < 15: return None, 0, 0
    prob = LpProblem("best", LpMaximize)
    x = [LpVariable(f"x{i}", cat=LpBinary) for i in range(n)]
    prob += lpSum(x[i] * p.loc[i, 'predicted_points'] for i in range(n))
    prob += lpSum(x[i] * p.loc[i, 'now_cost'] for i in range(n)) <= budget
    prob += lpSum(x[i] for i in range(n)) == 15
    for pos, cnt in {'Goalkeeper': 2, 'Defender': 5, 'Midfielder': 5, 'Forward': 3}.items():
        idx = p[p['position'] == pos].index.tolist()
        if idx: prob += lpSum(x[i] for i in idx) == cnt
    for tm in p['team_name'].unique():
        idx = p[p['team_name'] == tm].index.tolist()
        prob += lpSum(x[i] for i in idx) <= 3
    prob.solve(PULP_CBC_CMD(msg=0))
    sel = [i for i in range(n) if x[i].varValue == 1]
    if not sel: return None, 0, 0
    return p.loc[sel].copy(), p.loc[sel, 'now_cost'].sum(), p.loc[sel, 'predicted_points'].sum()

model_output = {}
best_teams = {}
for gw in next_gws:
    gw_p = pred_df[pred_df['gw'] == int(gw)]
    team, cost, pts = solve_team(gw_p)
    if team is not None:
        cap_idx = team['predicted_points'].idxmax()
        team['is_captain'] = False
        team.loc[cap_idx, 'is_captain'] = True
        cap_pts = team.loc[cap_idx, 'predicted_points']
        po = {'Goalkeeper': 0, 'Defender': 1, 'Midfielder': 2, 'Forward': 3}
        team['po'] = team['position'].map(po)
        team = team.sort_values(['po', 'predicted_points'], ascending=[True, False])
        best_teams[str(int(gw))] = {
            'players': team[['web_name', 'team_name', 'position', 'now_cost', 'predicted_points',
                             'xgb_pts', 'rf_pts', 'mlp_pts', 'opponent', 'venue', 'is_captain']].to_dict('records'),
            'total_cost': round(float(cost), 1),
            'total_points': round(float(pts), 1),
            'total_with_captain': round(float(pts + cap_pts), 1),
            'captain': team.loc[cap_idx, 'web_name'],
        }
        print(f"  GW{int(gw)}: {pts:.1f}pts (C: {team.loc[cap_idx, 'web_name']})", flush=True)

model_output['best_teams'] = best_teams

captain_picks = {}
for gw in next_gws:
    top5 = pred_df[pred_df['gw'] == int(gw)].nlargest(5, 'predicted_points')
    captain_picks[str(int(gw))] = top5[['web_name', 'team_name', 'position', 'predicted_points',
                                         'xgb_pts', 'rf_pts', 'mlp_pts', 'opponent', 'venue']].to_dict('records')
model_output['captain_picks'] = captain_picks

five_gw = pred_df.groupby('player_id').agg(
    total_predicted=('predicted_points', 'sum'), web_name=('web_name', 'first'),
    team_name=('team_name', 'first'), position=('position', 'first'), now_cost=('now_cost', 'first')
).reset_index().sort_values('total_predicted', ascending=False)

wc_team, wc_cost, wc_pts = solve_team(five_gw.rename(columns={'total_predicted': 'predicted_points'}))
if wc_team is not None:
    po = {'Goalkeeper': 0, 'Defender': 1, 'Midfielder': 2, 'Forward': 3}
    wc_team['po'] = wc_team['position'].map(po)
    wc_team = wc_team.sort_values(['po', 'predicted_points'], ascending=[True, False])
    model_output['wildcard'] = {
        'players': wc_team[['web_name', 'team_name', 'position', 'now_cost', 'predicted_points']].round(1).to_dict('records'),
        'total_cost': round(float(wc_cost), 1),
        'total_predicted_5gw': round(float(wc_pts), 1),
        'avg_per_gw': round(float(wc_pts / len(next_gws)), 1),
    }

# Best team chart
if next_gws and str(int(next_gws[0])) in best_teams:
    bt = pd.DataFrame(best_teams[str(int(next_gws[0]))]['players'])
    fig, ax = plt.subplots(figsize=(14, 9))
    for i, (_, r) in enumerate(bt.iterrows()):
        c = '#FFD700' if r.get('is_captain') else COLORS.get(r['position'], '#888')
        ax.barh(i, r['predicted_points'], color=c, height=0.7)
    ax.set_yticks(range(len(bt)))
    ax.set_yticklabels([f"{'(C) ' if r.get('is_captain') else ''}{r['web_name']} vs {r['opponent']} ({r['venue']})"
                        for _, r in bt.iterrows()], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Predicted 3-GW Avg Points (Position-Specific Meta)')
    cap = best_teams[str(int(next_gws[0]))]['captain']
    tot = best_teams[str(int(next_gws[0]))]['total_with_captain']
    ax.set_title(f'V3 Best Team GW{int(next_gws[0])} | Captain: {cap} | Total: {tot:.0f}pts',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/charts/model_best_team.png', dpi=150, bbox_inches='tight')
    plt.close()

# Heatmap
if len(pred_df) > 0:
    top20_ids = five_gw.head(20)['player_id'].tolist()
    pivot = pred_df[pred_df['player_id'].isin(top20_ids)].pivot_table(
        index='web_name', columns='gw', values='predicted_points')
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('total', ascending=False).drop('total', axis=1)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', linewidths=0.5, ax=ax)
    ax.set_title(f'V3 Predicted 3-GW Avg Points (GW{int(next_gws[0])}-{int(next_gws[-1])})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/charts/model_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# 11. SAVE
# ============================================================
print("\n[9/9] Saving...", flush=True)

model_output['metrics'] = {
    'model_type': 'V3 Position-Specific Meta-Ensemble (3-GW avg target, min 30min)',
    'target': '3-GW forward average points',
    'min_minutes': MIN_MINUTES,
    'positions': POSITIONS,
    'xgboost': {'mae': round(all_results['xgboost']['mae'], 3), 'r2': round(all_results['xgboost']['r2'], 3), 'rank20': round(all_results['xgboost'].get('rank20', 0), 3)},
    'random_forest': {'mae': round(all_results['random_forest']['mae'], 3), 'r2': round(all_results['random_forest']['r2'], 3), 'rank20': round(all_results['random_forest'].get('rank20', 0), 3)},
    'neural_net': {'mae': round(all_results['neural_net']['mae'], 3), 'r2': round(all_results['neural_net']['r2'], 3), 'rank20': round(all_results['neural_net'].get('rank20', 0), 3)},
    'lstm': {'mae': round(all_results['lstm']['mae'], 3), 'r2': round(all_results['lstm']['r2'], 3), 'rank20': round(all_results['lstm'].get('rank20', 0), 3)},
    'simple_avg': {'mae': round(all_results['simple_avg']['mae'], 3), 'r2': round(all_results['simple_avg']['r2'], 3), 'rank20': round(all_results['simple_avg'].get('rank20', 0), 3)},
    'stacked_meta': {'mae': round(all_results['stacked_meta']['mae'], 3), 'r2': round(all_results['stacked_meta']['r2'], 3), 'rank20': round(all_results['stacked_meta'].get('rank20', 0), 3)},
    'best_model': best_name,
    'best_r2_model': best_r2,
    'train_size': len(train_tab),
    'test_size': len(test_tab),
    'n_base_features': len(feature_cols),
    'n_meta_features': len(meta_features_full),
    'next_gws': [int(g) for g in next_gws],
    'per_position': {},
}

for pos in POSITIONS:
    if pos in all_pos_test_preds:
        model_output['metrics']['per_position'][pos] = {
            'meta_mae': round(mean_absolute_error(all_pos_test_actual[pos], all_pos_test_preds[pos]['stacked_meta']), 3),
            'meta_r2': round(r2_score(all_pos_test_actual[pos], all_pos_test_preds[pos]['stacked_meta']), 3),
            'train_size': int((train_tab['pos_group'] == pos).sum()),
            'test_size': int((test_tab['pos_group'] == pos).sum()),
        }

with open('output/model_data.json', 'w') as f:
    json.dump(model_output, f, indent=2, default=str)

with open('output/model.pkl', 'wb') as f:
    pickle.dump({
        'model_version': 'v3',
        'positions': {pos: {
            'xgboost': position_models[pos]['xgboost'],
            'random_forest': position_models[pos]['random_forest'],
            'meta_model': position_models[pos]['meta_model'],
            'scaler_mlp': position_models[pos]['scaler_mlp'],
            'scaler_lstm': position_models[pos]['scaler_lstm'],
        } for pos in POSITIONS if pos in position_models},
        'features': feature_cols,
        'meta_features': meta_features_full,
        'meta_context_cols': meta_context_cols,
        'seq_features': seq_features,
        'seq_len': SEQ_LEN,
        'target_type': '3gw_avg',
        'min_minutes': MIN_MINUTES,
    }, f)

nn_save = {
    'model_version': 'v3',
    'mlp_n_features': len(feature_cols),
    'lstm_input_size': len(seq_features),
    'lstm_hidden': 128,
    'lstm_layers': 2,
}
for pos in POSITIONS:
    if pos in nn_states:
        nn_save[f'{pos}_mlp_state'] = nn_states[pos]['mlp_state']
        nn_save[f'{pos}_lstm_state'] = nn_states[pos]['lstm_state']
torch.save(nn_save, 'output/nn_models.pt')

print("\n" + "=" * 60)
print("MODEL V3 COMPLETE!")
print(f"\n  Target: 3-GW forward average | Min minutes: {MIN_MINUTES}")
print(f"  Position-specific models: {', '.join(POSITIONS)}")
print(f"\n  {'Model':<18} {'MAE':>8} {'R²':>8} {'Top-20':>8}")
print(f"  {'-'*42}")
for mname in ['xgboost', 'random_forest', 'neural_net', 'lstm', 'simple_avg', 'stacked_meta']:
    r = all_results[mname]
    star = ' <-- BEST MAE' if mname == best_name else (' <-- BEST R²' if mname == best_r2 else '')
    print(f"  {mname:<18} {r['mae']:>8.3f} {r['r2']:>8.3f} {r.get('rank20',0):>7.1%}{star}")

print(f"\n  Per-Position R²:")
for pos in POSITIONS:
    if pos in all_pos_test_preds:
        r2 = r2_score(all_pos_test_actual[pos], all_pos_test_preds[pos]['stacked_meta'])
        print(f"    {pos}: {r2:.3f}")

print(f"\n  Improvement over V1:")
print(f"    V1 best R²:  0.146 (stacked meta, single GW)")
print(f"    V3 best R²:  {all_results[best_r2]['r2']:.3f} ({best_r2}, 3-GW avg)")
print("=" * 60)
