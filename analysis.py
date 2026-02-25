"""
FPL 2024/25 Data Analysis
Generates all charts + JSON data for the HTML dashboard
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.patches import Patch
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, value
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
os.makedirs('output/charts', exist_ok=True)

COLORS = {'Forward': '#e74c3c', 'Midfielder': '#2ecc71', 'Defender': '#3498db', 'Goalkeeper': '#f39c12'}
POSITION_ORDER = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv('data/processed/players_clean.csv')
df_active = df[df['minutes'] >= 450].copy()
print(f"  Total: {len(df)} players | Active (450+ min): {len(df_active)}")

results = {}

# ============================================================
# 1. TOP 15 PLAYERS
# ============================================================
print("Chart 1: Top 15 players...")
fig, ax = plt.subplots(figsize=(12, 7))
top15 = df_active.nlargest(15, 'total_points')
bar_colors = [COLORS.get(pos, '#95a5a6') for pos in top15['position']]

ax.barh(range(len(top15)), top15['total_points'], color=bar_colors)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels([f"{row['web_name']} ({row['team_name']})" for _, row in top15.iterrows()])
ax.set_xlabel('Total Points')
ax.set_title('Top 15 FPL Players by Total Points (2024/25)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, (_, row) in enumerate(top15.iterrows()):
    ax.text(row['total_points'] + 1, i, str(int(row['total_points'])), va='center', fontweight='bold')
legend_elements = [Patch(facecolor=c, label=p) for p, c in COLORS.items()]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig('output/charts/top15.png', dpi=150, bbox_inches='tight')
plt.close()

results['top15'] = top15[['web_name', 'team_name', 'position', 'now_cost', 'total_points', 'goals_scored', 'assists', 'value']].to_dict('records')

# ============================================================
# 2. DISTRIBUTION BY POSITION
# ============================================================
print("Chart 2: Distribution by position...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=df_active, x='position', y='total_points', order=POSITION_ORDER, ax=axes[0])
axes[0].set_title('Points Distribution by Position', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Total Points')
sns.boxplot(data=df_active, x='position', y='now_cost', order=POSITION_ORDER, ax=axes[1])
axes[1].set_title('Price Distribution by Position', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Price (£m)')
plt.tight_layout()
plt.savefig('output/charts/distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 3. PRICE VS POINTS
# ============================================================
print("Chart 3: Price vs Points...")
fig, ax = plt.subplots(figsize=(12, 8))
for pos in POSITION_ORDER:
    subset = df_active[df_active['position'] == pos]
    ax.scatter(subset['now_cost'], subset['total_points'], label=pos, alpha=0.6, s=60, color=COLORS[pos])
for _, row in df_active.nlargest(10, 'total_points').iterrows():
    ax.annotate(row['web_name'], (row['now_cost'], row['total_points']), textcoords='offset points', xytext=(5, 5), fontsize=9)
ax.set_xlabel('Price (£m)', fontsize=12)
ax.set_ylabel('Total Points', fontsize=12)
ax.set_title('Price vs Points: Is Expensive Always Better?', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('output/charts/price_vs_points.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 4. TEAM PERFORMANCE
# ============================================================
print("Chart 4: Team performance...")
team_stats = df_active.groupby('team_name').agg(
    avg_points=('total_points', 'mean'),
    total_goals=('goals_scored', 'sum'),
    total_assists=('assists', 'sum'),
    avg_price=('now_cost', 'mean'),
    player_count=('id', 'count')
).round(2).sort_values('avg_points', ascending=False)

fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(range(len(team_stats)), team_stats['avg_points'], color=sns.color_palette('viridis', len(team_stats)))
ax.set_xticks(range(len(team_stats)))
ax.set_xticklabels(team_stats.index, rotation=45, ha='right')
ax.set_ylabel('Average Points per Player')
ax.set_title('Average FPL Points per Active Player by Team', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/charts/teams.png', dpi=150, bbox_inches='tight')
plt.close()

results['teams'] = team_stats.reset_index().to_dict('records')

# ============================================================
# 5. CORRELATION HEATMAP
# ============================================================
print("Chart 5: Correlation heatmap...")
corr_cols = ['total_points', 'now_cost', 'goals_scored', 'assists', 'minutes',
             'clean_sheets', 'bonus', 'ict_index', 'influence', 'creativity',
             'threat', 'expected_goals', 'expected_assists', 'value']
available_corr = [c for c in corr_cols if c in df_active.columns]

fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df_active[available_corr].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Matrix of Key FPL Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/charts/correlation.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 6. VALUE ANALYSIS
# ============================================================
print("Chart 6: Value analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, pos in zip(axes.flatten(), POSITION_ORDER):
    subset = df_active[df_active['position'] == pos]
    ax.scatter(subset['now_cost'], subset['value'], alpha=0.6, color=COLORS[pos], s=50)
    for _, row in subset.nlargest(3, 'value').iterrows():
        ax.annotate(row['web_name'], (row['now_cost'], row['value']), fontsize=9, fontweight='bold')
    ax.set_xlabel('Price (£m)')
    ax.set_ylabel('Value (pts/£m)')
    ax.set_title(f'{pos}s - Value vs Price', fontweight='bold')
plt.suptitle('Who Gives You the Most Points Per Million?', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('output/charts/value.png', dpi=150, bbox_inches='tight')
plt.close()

# Best value per position
best_value = {}
for pos in POSITION_ORDER:
    bv = df_active[df_active['position'] == pos].nlargest(5, 'value')
    best_value[pos] = bv[['web_name', 'team_name', 'now_cost', 'total_points', 'value']].to_dict('records')
results['best_value'] = best_value

# ============================================================
# 7. PCA
# ============================================================
print("Chart 7: PCA...")
pca_features = ['total_points', 'goals_scored', 'assists', 'minutes', 'clean_sheets',
                'bonus', 'ict_index', 'influence', 'creativity', 'threat',
                'now_cost', 'expected_goals', 'expected_assists']
available_pca = [c for c in pca_features if c in df_active.columns]

X = df_active[available_pca].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Scree + cumulative
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Scree Plot', fontweight='bold')

cumvar = np.cumsum(pca.explained_variance_ratio_)
axes[1].plot(range(1, len(cumvar) + 1), cumvar, 'o-', color='steelblue')
axes[1].axhline(y=0.8, color='red', linestyle='--', label='80%')
axes[1].axhline(y=0.9, color='orange', linestyle='--', label='90%')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig('output/charts/pca_variance.png', dpi=150, bbox_inches='tight')
plt.close()

# Loadings heatmap
n_components = 3
loadings = pd.DataFrame(pca.components_[:n_components].T, columns=[f'PC{i+1}' for i in range(n_components)], index=available_pca)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('PCA Component Loadings', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/charts/pca_loadings.png', dpi=150, bbox_inches='tight')
plt.close()

# PCA scatter
fig, ax = plt.subplots(figsize=(14, 10))
for pos in POSITION_ORDER:
    mask = df_active['position'].values == pos
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=pos, alpha=0.6, s=60, color=COLORS[pos])
for idx in np.argsort(X_pca[:, 0])[-10:]:
    ax.annotate(df_active.iloc[idx]['web_name'], (X_pca[idx, 0], X_pca[idx, 1]), fontsize=9, fontweight='bold')
ax.set_xlabel(f'PC1 - Overall Quality ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 - Defensive vs Attacking ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA: Player Profiles in 2D Space', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig('output/charts/pca_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

results['pca_variance'] = [round(float(v), 4) for v in pca.explained_variance_ratio_[:5]]

# ============================================================
# 8. CLUSTERING
# ============================================================
print("Chart 8: Clustering...")
X_clust = X_pca[:, :3]

# Elbow
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clust)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(2, 11), inertias, 'o-', color='steelblue', linewidth=2)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method - Optimal Number of Clusters', fontweight='bold')
plt.tight_layout()
plt.savefig('output/charts/elbow.png', dpi=150, bbox_inches='tight')
plt.close()

# K-Means
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df_active['cluster'] = kmeans.fit_predict(X_clust)
cluster_colors = sns.color_palette('Set1', k)

fig, ax = plt.subplots(figsize=(14, 10))
for c in range(k):
    mask = df_active['cluster'].values == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {c}', alpha=0.6, s=60, color=cluster_colors[c])
for idx in np.argsort(X_pca[:, 0])[-8:]:
    ax.annotate(df_active.iloc[idx]['web_name'], (X_pca[idx, 0], X_pca[idx, 1]), fontsize=9, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title(f'K-Means Clustering (k={k}) on PCA Components', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('output/charts/clusters.png', dpi=150, bbox_inches='tight')
plt.close()

# Radar chart
radar_cols = ['total_points', 'goals_scored', 'assists', 'clean_sheets', 'now_cost', 'value']
available_radar = [c for c in radar_cols if c in df_active.columns]
cluster_means = df_active.groupby('cluster')[available_radar].mean()
cluster_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

angles = np.linspace(0, 2 * np.pi, len(available_radar), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
for c in range(k):
    values = cluster_norm.loc[c].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {c}', color=cluster_colors[c])
    ax.fill(angles, values, alpha=0.1, color=cluster_colors[c])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(available_radar, fontsize=11)
ax.set_title('Cluster Profiles - Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig('output/charts/radar.png', dpi=150, bbox_inches='tight')
plt.close()

# Cluster data
cluster_data = []
for c in range(k):
    cluster_players = df_active[df_active['cluster'] == c]
    top5 = cluster_players.nlargest(5, 'total_points')
    profile = cluster_means.loc[c].round(2).to_dict()
    cluster_data.append({
        'id': int(c),
        'count': int(len(cluster_players)),
        'profile': profile,
        'top_players': top5[['web_name', 'team_name', 'position', 'total_points', 'now_cost']].to_dict('records')
    })
results['clusters'] = cluster_data

# ============================================================
# 9. xG ANALYSIS — Overperformers & Underperformers
# ============================================================
print("Chart 9: xG Analysis...")

# Only players with goals or xG > 0
xg_df = df_active[(df_active['goals_scored'] > 0) | (df_active['expected_goals'] > 0)].copy()
xg_df['xg_diff'] = xg_df['goals_scored'] - xg_df['expected_goals']
xg_df['xa_diff'] = xg_df['assists'] - xg_df['expected_assists']
xg_df['xgi_diff'] = (xg_df['goals_scored'] + xg_df['assists']) - xg_df['expected_goal_involvements']

# --- Chart 9a: Goals vs xG scatter ---
fig, ax = plt.subplots(figsize=(12, 10))
max_val = max(xg_df['goals_scored'].max(), xg_df['expected_goals'].max()) + 2
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=2, label='Perfect line (Goals = xG)')

for pos in ['Forward', 'Midfielder', 'Defender']:
    subset = xg_df[xg_df['position'] == pos]
    ax.scatter(subset['expected_goals'], subset['goals_scored'],
               label=pos, alpha=0.6, s=60, color=COLORS[pos])

# Label biggest overperformers (above the line)
overperformers = xg_df.nlargest(8, 'xg_diff')
for _, row in overperformers.iterrows():
    ax.annotate(f"{row['web_name']}", (row['expected_goals'], row['goals_scored']),
                textcoords='offset points', xytext=(5, 5), fontsize=9,
                fontweight='bold', color='#2ecc71')

# Label biggest underperformers (below the line)
underperformers = xg_df.nsmallest(5, 'xg_diff')
for _, row in underperformers.iterrows():
    ax.annotate(f"{row['web_name']}", (row['expected_goals'], row['goals_scored']),
                textcoords='offset points', xytext=(5, -10), fontsize=9,
                fontweight='bold', color='#e74c3c')

ax.fill_between([0, max_val], [0, max_val], [max_val, max_val], alpha=0.05, color='green', label='Overperforming zone')
ax.fill_between([0, max_val], [0, 0], [0, max_val], alpha=0.05, color='red', label='Underperforming zone')

ax.set_xlabel('Expected Goals (xG)', fontsize=12)
ax.set_ylabel('Actual Goals', fontsize=12)
ax.set_title('Goals vs Expected Goals (xG): Who Is Overperforming?', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('output/charts/xg_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Chart 9b: Top overperformers & underperformers bar chart ---
top_over = xg_df.nlargest(10, 'xg_diff')[['web_name', 'team_name', 'position', 'goals_scored', 'expected_goals', 'xg_diff']]
top_under = xg_df.nsmallest(10, 'xg_diff')[['web_name', 'team_name', 'position', 'goals_scored', 'expected_goals', 'xg_diff']]
xg_bar = pd.concat([top_over, top_under]).sort_values('xg_diff')

fig, ax = plt.subplots(figsize=(12, 10))
bar_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in xg_bar['xg_diff']]
ax.barh(range(len(xg_bar)), xg_bar['xg_diff'], color=bar_colors)
ax.set_yticks(range(len(xg_bar)))
ax.set_yticklabels([f"{row['web_name']} ({row['team_name']})" for _, row in xg_bar.iterrows()])
ax.axvline(x=0, color='white', linewidth=1.5)
ax.set_xlabel('Goals - xG (Difference)', fontsize=12)
ax.set_title('xG Difference: Overperformers (Green) vs Underperformers (Red)', fontsize=14, fontweight='bold')

for i, (_, row) in enumerate(xg_bar.iterrows()):
    sign = '+' if row['xg_diff'] > 0 else ''
    ax.text(row['xg_diff'] + (0.1 if row['xg_diff'] >= 0 else -0.1), i,
            f"{sign}{row['xg_diff']:.1f}", va='center', fontweight='bold',
            ha='left' if row['xg_diff'] >= 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig('output/charts/xg_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Chart 9c: Assists vs xA scatter ---
xa_df = df_active[(df_active['assists'] > 0) | (df_active['expected_assists'] > 0)].copy()
xa_df['xa_diff'] = xa_df['assists'] - xa_df['expected_assists']

fig, ax = plt.subplots(figsize=(12, 10))
max_val_a = max(xa_df['assists'].max(), xa_df['expected_assists'].max()) + 2
ax.plot([0, max_val_a], [0, max_val_a], 'k--', alpha=0.4, linewidth=2, label='Perfect line (Assists = xA)')

for pos in ['Forward', 'Midfielder', 'Defender']:
    subset = xa_df[xa_df['position'] == pos]
    ax.scatter(subset['expected_assists'], subset['assists'],
               label=pos, alpha=0.6, s=60, color=COLORS[pos])

for _, row in xa_df.nlargest(6, 'xa_diff').iterrows():
    ax.annotate(row['web_name'], (row['expected_assists'], row['assists']),
                textcoords='offset points', xytext=(5, 5), fontsize=9,
                fontweight='bold', color='#2ecc71')

for _, row in xa_df.nsmallest(4, 'xa_diff').iterrows():
    ax.annotate(row['web_name'], (row['expected_assists'], row['assists']),
                textcoords='offset points', xytext=(5, -10), fontsize=9,
                fontweight='bold', color='#e74c3c')

ax.fill_between([0, max_val_a], [0, max_val_a], [max_val_a, max_val_a], alpha=0.05, color='green')
ax.fill_between([0, max_val_a], [0, 0], [0, max_val_a], alpha=0.05, color='red')
ax.set_xlabel('Expected Assists (xA)', fontsize=12)
ax.set_ylabel('Actual Assists', fontsize=12)
ax.set_title('Assists vs Expected Assists (xA): Who Is Overperforming?', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('output/charts/xa_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Chart 9d: Combined xGI analysis per position ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, pos in zip(axes.flatten(), ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']):
    subset = xg_df[xg_df['position'] == pos].copy() if pos != 'Goalkeeper' else df_active[df_active['position'] == pos].copy()
    if pos == 'Goalkeeper':
        ax.text(0.5, 0.5, 'Goalkeepers\n(No xG data relevant)', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='#888')
        ax.set_title(f'{pos}s', fontweight='bold')
        continue

    subset['xgi_diff'] = (subset['goals_scored'] + subset['assists']) - subset['expected_goal_involvements']
    subset = subset.sort_values('xgi_diff', ascending=True)

    top_bottom = pd.concat([subset.head(5), subset.tail(5)])
    bar_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in top_bottom['xgi_diff']]
    ax.barh(range(len(top_bottom)), top_bottom['xgi_diff'], color=bar_colors)
    ax.set_yticks(range(len(top_bottom)))
    ax.set_yticklabels(top_bottom['web_name'], fontsize=8)
    ax.axvline(x=0, color='gray', linewidth=1)
    ax.set_title(f'{pos}s - xGI Difference', fontweight='bold')
    ax.set_xlabel('(Goals+Assists) - xGI')

plt.suptitle('Expected Goal Involvement Difference by Position', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('output/charts/xgi_position.png', dpi=150, bbox_inches='tight')
plt.close()

# Save xG data for HTML
xg_over = xg_df.nlargest(10, 'xg_diff')
xg_under = xg_df.nsmallest(10, 'xg_diff')
results['xg_analysis'] = {
    'overperformers': xg_over[['web_name', 'team_name', 'position', 'goals_scored', 'expected_goals', 'xg_diff']].round(2).to_dict('records'),
    'underperformers': xg_under[['web_name', 'team_name', 'position', 'goals_scored', 'expected_goals', 'xg_diff']].round(2).to_dict('records'),
}

xa_over = xa_df.nlargest(10, 'xa_diff')
xa_under = xa_df.nsmallest(10, 'xa_diff')
results['xa_analysis'] = {
    'overperformers': xa_over[['web_name', 'team_name', 'position', 'assists', 'expected_assists', 'xa_diff']].round(2).to_dict('records'),
    'underperformers': xa_under[['web_name', 'team_name', 'position', 'assists', 'expected_assists', 'xa_diff']].round(2).to_dict('records'),
}

# ============================================================
# 10. OPTIMAL TEAM SELECTOR (Linear Programming)
# ============================================================
print("Chart 10: Optimal Team Selector...")

def solve_optimal_team(players_df, budget=100.0, optimize_col='total_points', label='Season Points'):
    """
    Solve the FPL optimal team selection using Integer Linear Programming.
    Rules: 2 GK, 5 DEF, 5 MID, 3 FWD, max 3 per team, budget £100m
    """
    players = players_df.reset_index(drop=True)
    n = len(players)

    prob = LpProblem(f"FPL_Optimal_{label}", LpMaximize)
    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]

    # Objective: maximize points
    prob += lpSum(x[i] * players.loc[i, optimize_col] for i in range(n))

    # Budget constraint
    prob += lpSum(x[i] * players.loc[i, 'now_cost'] for i in range(n)) <= budget

    # Squad size: exactly 15
    prob += lpSum(x[i] for i in range(n)) == 15

    # Position constraints
    pos_limits = {'Goalkeeper': 2, 'Defender': 5, 'Midfielder': 5, 'Forward': 3}
    for pos, count in pos_limits.items():
        indices = players[players['position'] == pos].index.tolist()
        prob += lpSum(x[i] for i in indices) == count

    # Max 3 players per team
    for team in players['team_name'].unique():
        indices = players[players['team_name'] == team].index.tolist()
        prob += lpSum(x[i] for i in indices) <= 3

    prob.solve()

    selected = [i for i in range(n) if x[i].varValue == 1]
    team = players.loc[selected].copy()
    total_cost = team['now_cost'].sum()
    total_pts = team[optimize_col].sum()

    return team, total_cost, total_pts

# Use active players only
opt_players = df_active[df_active['minutes'] >= 900].copy()

# --- Optimal Team by Total Points ---
best_team, best_cost, best_pts = solve_optimal_team(opt_players, budget=100.0, optimize_col='total_points', label='Points')

# --- Optimal Team by Value (points per million) ---
value_team, value_cost, value_pts = solve_optimal_team(opt_players, budget=100.0, optimize_col='value', label='Value')

# --- Optimal Budget Team (£85m) ---
budget_team, budget_cost, budget_pts = solve_optimal_team(opt_players, budget=85.0, optimize_col='total_points', label='Budget')

# Chart: Best Team visualization
def plot_team(team_df, title, filename, total_cost, total_pts):
    fig, ax = plt.subplots(figsize=(14, 9))

    # Sort by position order then points
    pos_order_map = {'Goalkeeper': 0, 'Defender': 1, 'Midfielder': 2, 'Forward': 3}
    team_sorted = team_df.copy()
    team_sorted['pos_order'] = team_sorted['position'].map(pos_order_map)
    team_sorted = team_sorted.sort_values(['pos_order', 'total_points'], ascending=[True, False])

    y_positions = range(len(team_sorted))
    bar_colors = [COLORS.get(pos, '#95a5a6') for pos in team_sorted['position']]

    bars = ax.barh(y_positions, team_sorted['total_points'], color=bar_colors, height=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row['web_name']} ({row['team_name']}) — £{row['now_cost']}m"
                        for _, row in team_sorted.iterrows()], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Total Points', fontsize=12)
    ax.set_title(f'{title}\nTotal: {total_pts:.0f} pts | Cost: £{total_cost:.1f}m', fontsize=14, fontweight='bold')

    for i, (_, row) in enumerate(team_sorted.iterrows()):
        ax.text(row['total_points'] + 0.5, i, f"{int(row['total_points'])}pts", va='center', fontsize=9, fontweight='bold')

    # Position separators
    legend_elements = [Patch(facecolor=c, label=p) for p, c in COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'output/charts/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

plot_team(best_team, 'Optimal FPL Team (Max Points, £100m Budget)', 'optimal_best.png', best_cost, best_pts)
plot_team(value_team, 'Optimal FPL Team (Max Value, £100m Budget)', 'optimal_value.png', value_cost, value_pts)
plot_team(budget_team, 'Optimal FPL Team (Max Points, £85m Budget)', 'optimal_budget.png', budget_cost, budget_pts)

# --- Chart: Comparison of the 3 strategies ---
fig, ax = plt.subplots(figsize=(10, 6))
strategies = ['Max Points\n(£100m)', 'Max Value\n(£100m)', 'Budget\n(£85m)']
points = [best_pts, value_team['total_points'].sum(), budget_pts]
costs = [best_cost, value_cost, budget_cost]

x_pos = range(3)
bars = ax.bar(x_pos, points, color=['#00ff87', '#3498db', '#f39c12'], width=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(strategies, fontsize=11)
ax.set_ylabel('Total Points', fontsize=12)
ax.set_title('Optimal Team Strategies Compared', fontsize=14, fontweight='bold')

for i, (pt, co) in enumerate(zip(points, costs)):
    ax.text(i, pt + 5, f"{pt:.0f} pts\n£{co:.1f}m", ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('output/charts/optimal_compare.png', dpi=150, bbox_inches='tight')
plt.close()

# Save data for HTML
def team_to_dict(team_df):
    pos_order_map = {'Goalkeeper': 0, 'Defender': 1, 'Midfielder': 2, 'Forward': 3}
    team_sorted = team_df.copy()
    team_sorted['pos_order'] = team_sorted['position'].map(pos_order_map)
    team_sorted = team_sorted.sort_values(['pos_order', 'total_points'], ascending=[True, False])
    return team_sorted[['web_name', 'team_name', 'position', 'now_cost', 'total_points',
                         'goals_scored', 'assists', 'clean_sheets']].to_dict('records')

results['optimal_teams'] = {
    'best': {
        'players': team_to_dict(best_team),
        'total_cost': round(float(best_cost), 1),
        'total_points': round(float(best_pts), 0),
        'label': 'Maximum Points (£100m)'
    },
    'value': {
        'players': team_to_dict(value_team),
        'total_cost': round(float(value_cost), 1),
        'total_points': round(float(value_team['total_points'].sum()), 0),
        'label': 'Maximum Value (£100m)'
    },
    'budget': {
        'players': team_to_dict(budget_team),
        'total_cost': round(float(budget_cost), 1),
        'total_points': round(float(budget_pts), 0),
        'label': 'Budget Team (£85m)'
    }
}

# ============================================================
# 11. FORM TREND DETECTION
# ============================================================
print("Chart 11: Form Trend Detection...")

hist_path = 'data/raw/player_histories.csv'
if os.path.exists(hist_path):
    hist = pd.read_csv(hist_path)

    # Map player names
    name_map = dict(zip(df['id'], df['web_name']))
    team_map_id = dict(zip(df['id'], df['team_name']))
    pos_map_id = dict(zip(df['id'], df['position']))
    hist['web_name'] = hist['player_id'].map(name_map)
    hist['team_name'] = hist['player_id'].map(team_map_id)
    hist['position'] = hist['player_id'].map(pos_map_id)

    # Get the latest completed gameweek
    max_gw = hist['round'].max()
    recent_5 = hist[hist['round'] > max_gw - 5].copy()
    earlier_5 = hist[(hist['round'] > max_gw - 10) & (hist['round'] <= max_gw - 5)].copy()

    # Calculate form for last 5 vs previous 5
    recent_form = recent_5.groupby('player_id').agg(
        recent_pts=('total_points', 'sum'),
        recent_goals=('goals_scored', 'sum'),
        recent_assists=('assists', 'sum'),
        recent_minutes=('minutes', 'sum'),
        recent_gw_count=('round', 'count')
    ).reset_index()

    earlier_form = earlier_5.groupby('player_id').agg(
        earlier_pts=('total_points', 'sum'),
        earlier_goals=('goals_scored', 'sum'),
        earlier_assists=('assists', 'sum'),
        earlier_minutes=('minutes', 'sum'),
        earlier_gw_count=('round', 'count')
    ).reset_index()

    form_df = recent_form.merge(earlier_form, on='player_id', how='inner')
    form_df['pts_per_gw_recent'] = (form_df['recent_pts'] / form_df['recent_gw_count']).round(2)
    form_df['pts_per_gw_earlier'] = (form_df['earlier_pts'] / form_df['earlier_gw_count']).round(2)
    form_df['trend'] = (form_df['pts_per_gw_recent'] - form_df['pts_per_gw_earlier']).round(2)
    form_df['web_name'] = form_df['player_id'].map(name_map)
    form_df['team_name'] = form_df['player_id'].map(team_map_id)
    form_df['position'] = form_df['player_id'].map(pos_map_id)

    # Only players who played in both periods
    form_df = form_df[(form_df['recent_minutes'] > 100) & (form_df['earlier_minutes'] > 100)]

    # --- Chart 11a: Trending UP vs DOWN ---
    rising = form_df.nlargest(12, 'trend')
    falling = form_df.nsmallest(12, 'trend')
    trend_bar = pd.concat([falling, rising]).sort_values('trend')

    fig, ax = plt.subplots(figsize=(12, 10))
    bar_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in trend_bar['trend']]
    ax.barh(range(len(trend_bar)), trend_bar['trend'], color=bar_colors)
    ax.set_yticks(range(len(trend_bar)))
    ax.set_yticklabels([f"{row['web_name']} ({row['team_name']})" for _, row in trend_bar.iterrows()])
    ax.axvline(x=0, color='white', linewidth=1.5)
    ax.set_xlabel('Points per GW Change (Last 5 vs Previous 5)', fontsize=12)
    ax.set_title(f'Form Trend: Rising vs Falling Players (GW{max_gw-9} → GW{max_gw})', fontsize=14, fontweight='bold')

    for i, (_, row) in enumerate(trend_bar.iterrows()):
        sign = '+' if row['trend'] > 0 else ''
        ax.text(row['trend'] + (0.05 if row['trend'] >= 0 else -0.05), i,
                f"{sign}{row['trend']:.1f}", va='center', fontweight='bold',
                ha='left' if row['trend'] >= 0 else 'right', fontsize=9)

    plt.tight_layout()
    plt.savefig('output/charts/form_trend.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Chart 11b: Points over time for top trending players ---
    top_trending = pd.concat([rising.head(5), falling.head(5)])['player_id'].tolist()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Rising players
    ax = axes[0]
    for pid in rising.head(5)['player_id']:
        player_hist = hist[hist['player_id'] == pid].sort_values('round')
        # Rolling average
        player_hist['rolling_pts'] = player_hist['total_points'].rolling(3, min_periods=1).mean()
        ax.plot(player_hist['round'], player_hist['rolling_pts'], 'o-', linewidth=2,
                label=name_map.get(pid, str(pid)), markersize=4, alpha=0.8)
    ax.set_xlabel('Gameweek')
    ax.set_ylabel('Points (3-GW Rolling Avg)')
    ax.set_title('Rising Players - Points Trend Over Season', fontweight='bold', color='#2ecc71')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Falling players
    ax = axes[1]
    for pid in falling.head(5)['player_id']:
        player_hist = hist[hist['player_id'] == pid].sort_values('round')
        player_hist['rolling_pts'] = player_hist['total_points'].rolling(3, min_periods=1).mean()
        ax.plot(player_hist['round'], player_hist['rolling_pts'], 'o-', linewidth=2,
                label=name_map.get(pid, str(pid)), markersize=4, alpha=0.8)
    ax.set_xlabel('Gameweek')
    ax.set_ylabel('Points (3-GW Rolling Avg)')
    ax.set_title('Falling Players - Points Trend Over Season', fontweight='bold', color='#e74c3c')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/charts/form_lines.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Chart 11c: Heatmap - top 20 players points by gameweek ---
    top20_ids = df_active.nlargest(20, 'total_points')['id'].tolist()
    top20_hist = hist[hist['player_id'].isin(top20_ids)].copy()

    pivot = top20_hist.pivot_table(index='web_name', columns='round', values='total_points', aggfunc='sum').fillna(0)
    # Sort by total
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('total', ascending=False).drop('total', axis=1)

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(pivot, cmap='RdYlGn', ax=ax, linewidths=0.5, linecolor='#2a2d35',
                cbar_kws={'label': 'Points'}, vmin=0, vmax=15)
    ax.set_title('Top 20 Players - Points Heatmap by Gameweek', fontsize=14, fontweight='bold')
    ax.set_xlabel('Gameweek')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig('output/charts/form_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save data for HTML
    results['form_trends'] = {
        'max_gw': int(max_gw),
        'rising': rising[['web_name', 'team_name', 'position', 'recent_pts', 'earlier_pts',
                          'pts_per_gw_recent', 'pts_per_gw_earlier', 'trend']].to_dict('records'),
        'falling': falling[['web_name', 'team_name', 'position', 'recent_pts', 'earlier_pts',
                            'pts_per_gw_recent', 'pts_per_gw_earlier', 'trend']].to_dict('records'),
    }
    print(f"  -> Form trends calculated (GW{max_gw-9} to GW{max_gw})")
else:
    print("  -> Skipping: No player history data found. Run pull_data.py first.")

# ============================================================
# 12. FIXTURE DIFFICULTY FORECAST
# ============================================================
print("Chart 12: Fixture Difficulty Forecast...")

fixtures = pd.read_csv('data/raw/fixtures_raw.csv')
teams_raw = pd.read_csv('data/raw/teams_raw.csv')
team_name_map = dict(zip(teams_raw['id'], teams_raw['name']))
team_short_map = dict(zip(teams_raw['id'], teams_raw['short_name']))

# Get upcoming fixtures (not finished, with an event/gameweek)
upcoming = fixtures[(fixtures['finished'] == False) & (fixtures['event'].notna())].copy()
upcoming['event'] = upcoming['event'].astype(int)

# Get the next 5 gameweeks
next_gws = sorted(upcoming['event'].unique())[:5]
upcoming_5 = upcoming[upcoming['event'].isin(next_gws)]

# Calculate FDR per team for next 5 GWs
team_fdr = {}
team_fixtures_detail = {}

for team_id in teams_raw['id']:
    team_name = team_name_map[team_id]
    team_short = team_short_map[team_id]

    # Home games
    home = upcoming_5[upcoming_5['team_h'] == team_id][['event', 'team_a', 'team_h_difficulty']].copy()
    home.columns = ['gw', 'opponent_id', 'fdr']
    home['venue'] = 'H'

    # Away games
    away = upcoming_5[upcoming_5['team_a'] == team_id][['event', 'team_h', 'team_a_difficulty']].copy()
    away.columns = ['gw', 'opponent_id', 'fdr']
    away['venue'] = 'A'

    team_fix = pd.concat([home, away]).sort_values('gw')
    team_fix['opponent'] = team_fix['opponent_id'].map(team_short_map)

    total_fdr = team_fix['fdr'].sum()
    avg_fdr = team_fix['fdr'].mean() if len(team_fix) > 0 else 5
    n_easy = len(team_fix[team_fix['fdr'] <= 2])

    team_fdr[team_name] = {
        'total_fdr': total_fdr,
        'avg_fdr': round(avg_fdr, 2),
        'n_fixtures': len(team_fix),
        'n_easy': n_easy,
        'fixtures': team_fix[['gw', 'opponent', 'venue', 'fdr']].to_dict('records')
    }

# Sort by easiest schedule
fdr_ranking = sorted(team_fdr.items(), key=lambda x: x[1]['avg_fdr'])

# --- Chart 12a: FDR ranking bar chart ---
fig, ax = plt.subplots(figsize=(14, 8))
teams_sorted = [t[0] for t in fdr_ranking]
fdrs_sorted = [t[1]['avg_fdr'] for t in fdr_ranking]

# Color by difficulty
bar_colors = []
for fdr in fdrs_sorted:
    if fdr <= 2.5:
        bar_colors.append('#2ecc71')
    elif fdr <= 3.0:
        bar_colors.append('#f39c12')
    else:
        bar_colors.append('#e74c3c')

ax.barh(range(len(teams_sorted)), fdrs_sorted, color=bar_colors)
ax.set_yticks(range(len(teams_sorted)))
ax.set_yticklabels(teams_sorted)
ax.set_xlabel('Average Fixture Difficulty Rating (Lower = Easier)')
ax.set_title(f'Fixture Difficulty Next {len(next_gws)} Gameweeks (GW{next_gws[0]}-{next_gws[-1]})', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.axvline(x=3.0, color='white', linestyle='--', alpha=0.5, label='Average (3.0)')
ax.legend()

for i, fdr in enumerate(fdrs_sorted):
    ax.text(fdr + 0.02, i, f'{fdr:.2f}', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('output/charts/fixture_fdr.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Chart 12b: Fixture heatmap ---
fig, ax = plt.subplots(figsize=(16, 10))

# Build heatmap matrix
heatmap_data = []
for team_name in teams_sorted:
    row = {}
    for gw in next_gws:
        fix = [f for f in team_fdr[team_name]['fixtures'] if f['gw'] == gw]
        if fix:
            row[f'GW{gw}'] = fix[0]['fdr']
        else:
            row[f'GW{gw}'] = np.nan
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, index=teams_sorted)

# Also build annotation with opponent names
annot_data = []
for team_name in teams_sorted:
    row = {}
    for gw in next_gws:
        fix = [f for f in team_fdr[team_name]['fixtures'] if f['gw'] == gw]
        if fix:
            row[f'GW{gw}'] = f"{fix[0]['opponent']} ({fix[0]['venue']})"
        else:
            row[f'GW{gw}'] = '-'
    annot_data.append(row)

annot_df = pd.DataFrame(annot_data, index=teams_sorted)

sns.heatmap(heatmap_df, annot=annot_df.values, fmt='', cmap='RdYlGn_r',
            linewidths=1, linecolor='#2a2d35', ax=ax,
            cbar_kws={'label': 'Fixture Difficulty (1=Easy, 5=Hard)'},
            vmin=1, vmax=5)
ax.set_title(f'Fixture Difficulty Heatmap (GW{next_gws[0]}-{next_gws[-1]})', fontsize=14, fontweight='bold')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('output/charts/fixture_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Chart 12c: Best picks based on fixtures + form ---
# Players from teams with easy fixtures who are in form
easy_teams = [t[0] for t in fdr_ranking[:7]]  # Top 7 easiest fixtures
fixture_picks = df_active[df_active['team_name'].isin(easy_teams)].copy()
fixture_picks['fixture_score'] = fixture_picks['team_name'].map(
    {t[0]: round(5 - t[1]['avg_fdr'], 2) for t in fdr_ranking}
)
fixture_picks['combined_score'] = (fixture_picks['form'].fillna(0) * 2 + fixture_picks['fixture_score'] * 3).round(2)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, pos in zip(axes.flatten(), ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']):
    subset = fixture_picks[fixture_picks['position'] == pos].nlargest(8, 'combined_score')
    bar_colors = ['#2ecc71' if s > subset['combined_score'].median() else '#3498db' for s in subset['combined_score']]
    ax.barh(range(len(subset)), subset['combined_score'], color=bar_colors)
    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels([f"{row['web_name']} ({row['team_name']})" for _, row in subset.iterrows()], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f'Best {pos}s to Buy (Form + Easy Fixtures)', fontweight='bold')
    ax.set_xlabel('Combined Score (Form x2 + Fixture x3)')

plt.suptitle(f'Recommended Picks: GW{next_gws[0]}-{next_gws[-1]}', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('output/charts/fixture_picks.png', dpi=150, bbox_inches='tight')
plt.close()

# Save data for HTML
results['fixture_forecast'] = {
    'gameweeks': [int(gw) for gw in next_gws],
    'ranking': [{
        'team': t[0],
        'avg_fdr': t[1]['avg_fdr'],
        'n_easy': t[1]['n_easy'],
        'fixtures': t[1]['fixtures']
    } for t in fdr_ranking],
    'picks': {}
}

for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
    picks = fixture_picks[fixture_picks['position'] == pos].nlargest(5, 'combined_score')
    results['fixture_forecast']['picks'][pos] = picks[['web_name', 'team_name', 'now_cost', 'form', 'total_points', 'combined_score']].round(2).to_dict('records')

# ============================================================
# 13. TRANSFER ROI ANALYSIS
# ============================================================
print("Chart 13: Transfer ROI Analysis...")

if os.path.exists(hist_path):
    hist = pd.read_csv(hist_path)
    hist['web_name'] = hist['player_id'].map(name_map)
    hist['team_name'] = hist['player_id'].map(team_map_id)
    hist['position'] = hist['player_id'].map(pos_map_id)
    hist['price'] = hist['value'] / 10  # convert to millions

    # --- A) Price movers: who rose/fell most ---
    price_change = hist.groupby('player_id').agg(
        start_price=('price', 'first'),
        end_price=('price', 'last'),
        total_pts=('total_points', 'sum'),
        total_transfers_in=('transfers_in', 'sum'),
        total_transfers_out=('transfers_out', 'sum'),
    ).reset_index()
    price_change['price_change'] = (price_change['end_price'] - price_change['start_price']).round(1)
    price_change['pts_per_price'] = (price_change['total_pts'] / price_change['end_price']).round(2)
    price_change['web_name'] = price_change['player_id'].map(name_map)
    price_change['team_name'] = price_change['player_id'].map(team_map_id)
    price_change['position'] = price_change['player_id'].map(pos_map_id)
    price_change['net_transfers'] = price_change['total_transfers_in'] - price_change['total_transfers_out']

    # --- Chart 13a: Price change vs actual points ---
    fig, ax = plt.subplots(figsize=(12, 8))
    for pos in ['Forward', 'Midfielder', 'Defender']:
        subset = price_change[price_change['position'] == pos]
        ax.scatter(subset['price_change'], subset['total_pts'],
                   label=pos, alpha=0.6, s=60, color=COLORS[pos])

    # Label interesting players
    for _, row in price_change.nlargest(5, 'price_change').iterrows():
        ax.annotate(row['web_name'], (row['price_change'], row['total_pts']),
                    textcoords='offset points', xytext=(5, 5), fontsize=9, fontweight='bold', color='#2ecc71')

    for _, row in price_change.nsmallest(5, 'price_change').iterrows():
        ax.annotate(row['web_name'], (row['price_change'], row['total_pts']),
                    textcoords='offset points', xytext=(5, -10), fontsize=9, fontweight='bold', color='#e74c3c')

    ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    ax.set_xlabel('Price Change (£m)', fontsize=12)
    ax.set_ylabel('Total Points', fontsize=12)
    ax.set_title('Price Change vs Actual Points: Did Price Rises Deliver?', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('output/charts/transfer_price_vs_pts.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Chart 13b: Biggest price risers and fallers ---
    risers = price_change.nlargest(10, 'price_change')
    fallers = price_change.nsmallest(10, 'price_change')
    price_bar = pd.concat([fallers, risers]).sort_values('price_change')

    fig, ax = plt.subplots(figsize=(12, 10))
    bar_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in price_bar['price_change']]
    ax.barh(range(len(price_bar)), price_bar['price_change'], color=bar_colors)
    ax.set_yticks(range(len(price_bar)))
    ax.set_yticklabels([f"{row['web_name']} ({row['team_name']})" for _, row in price_bar.iterrows()])
    ax.axvline(x=0, color='white', linewidth=1.5)
    ax.set_xlabel('Price Change (£m)')
    ax.set_title('Biggest Price Movers This Season', fontsize=14, fontweight='bold')

    for i, (_, row) in enumerate(price_bar.iterrows()):
        sign = '+£' if row['price_change'] > 0 else '-£'
        ax.text(row['price_change'] + (0.02 if row['price_change'] >= 0 else -0.02), i,
                f"{sign}{abs(row['price_change'])}m ({int(row['total_pts'])}pts)",
                va='center', fontweight='bold', fontsize=8,
                ha='left' if row['price_change'] >= 0 else 'right')

    plt.tight_layout()
    plt.savefig('output/charts/transfer_price_movers.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- Chart 13c: Knee-jerk detector ---
    # Find GWs where player had big transfer IN followed by bad return
    knee_jerks = []
    for pid in hist['player_id'].unique():
        ph = hist[hist['player_id'] == pid].sort_values('round')
        for i, (_, row) in enumerate(ph.iterrows()):
            if row['transfers_in'] > 200000:  # big transfer in
                # Check next 3 GW performance
                next_rows = ph[ph['round'] > row['round']].head(3)
                if len(next_rows) >= 2:
                    next_avg = next_rows['total_points'].mean()
                    if next_avg < 3:  # bad return
                        knee_jerks.append({
                            'web_name': row['web_name'],
                            'team_name': row.get('team_name', ''),
                            'gw': int(row['round']),
                            'gw_points': int(row['total_points']),
                            'transfers_in': int(row['transfers_in']),
                            'next_3gw_avg': round(next_avg, 1),
                        })

    knee_jerks_df = pd.DataFrame(knee_jerks).sort_values('transfers_in', ascending=False).head(15) if knee_jerks else pd.DataFrame()

    if len(knee_jerks_df) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        labels = [f"{row['web_name']} GW{row['gw']} ({row['gw_points']}pts)" for _, row in knee_jerks_df.iterrows()]
        transfers = knee_jerks_df['transfers_in'].values / 1000  # in thousands
        next_avg = knee_jerks_df['next_3gw_avg'].values

        x = range(len(knee_jerks_df))
        width = 0.35
        bars1 = ax.bar([i - width/2 for i in x], transfers, width, label='Transfers In (thousands)', color='#3498db', alpha=0.8)
        ax2 = ax.twinx()
        bars2 = ax2.bar([i + width/2 for i in x], next_avg, width, label='Next 3 GW Avg Points', color='#e74c3c', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Transfers In (thousands)', color='#3498db')
        ax2.set_ylabel('Next 3 GW Avg Points', color='#e74c3c')
        ax.set_title('Knee-Jerk Alert: Huge Transfers In → Poor Returns After', fontsize=14, fontweight='bold')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig('output/charts/transfer_kneejerk.png', dpi=150, bbox_inches='tight')
        plt.close()

    # --- Chart 13d: Ownership vs Points (who's overhyped?) ---
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_df = df_active[df_active['selected_by_percent'] > 2].copy()
    for pos in ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']:
        subset = plot_df[plot_df['position'] == pos]
        ax.scatter(subset['selected_by_percent'], subset['total_points'],
                   label=pos, alpha=0.6, s=60, color=COLORS[pos])

    # Overhyped: high ownership, low points
    overhyped = plot_df[plot_df['selected_by_percent'] > 15].nsmallest(5, 'value')
    for _, row in overhyped.iterrows():
        ax.annotate(f"{row['web_name']} (OVERHYPED)", (row['selected_by_percent'], row['total_points']),
                    textcoords='offset points', xytext=(5, -10), fontsize=9, color='#e74c3c', fontweight='bold')

    # Under-owned gems: low ownership, high points
    underowned = plot_df[plot_df['selected_by_percent'] < 5].nlargest(5, 'total_points')
    for _, row in underowned.iterrows():
        ax.annotate(f"{row['web_name']} (HIDDEN)", (row['selected_by_percent'], row['total_points']),
                    textcoords='offset points', xytext=(5, 5), fontsize=9, color='#2ecc71', fontweight='bold')

    ax.set_xlabel('Ownership %', fontsize=12)
    ax.set_ylabel('Total Points', fontsize=12)
    ax.set_title('Ownership vs Points: Are Popular Picks Actually Good?', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('output/charts/transfer_ownership.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save data for HTML
    results['transfer_roi'] = {
        'risers': risers[['web_name', 'team_name', 'position', 'start_price', 'end_price',
                          'price_change', 'total_pts']].round(1).to_dict('records'),
        'fallers': fallers[['web_name', 'team_name', 'position', 'start_price', 'end_price',
                            'price_change', 'total_pts']].round(1).to_dict('records'),
        'knee_jerks': knee_jerks_df.to_dict('records') if len(knee_jerks_df) > 0 else [],
    }

    print(f"  -> Transfer ROI: {len(risers)} risers, {len(fallers)} fallers, {len(knee_jerks_df)} knee-jerks")
else:
    print("  -> Skipping: No player history data found.")

# ============================================================
# 14. INSIGHTS
# ============================================================
print("Generating insights...")
best = df_active.nlargest(1, 'total_points').iloc[0]
results['insights'] = {
    'top_scorer': {'name': best['web_name'], 'team': best['team_name'], 'points': int(best['total_points'])},
    'total_players': int(len(df)),
    'active_players': int(len(df_active)),
}

# Overvalued
overvalued = df_active[df_active['selected_by_percent'] > 15].nsmallest(3, 'value')
results['insights']['overvalued'] = overvalued[['web_name', 'team_name', 'selected_by_percent', 'value']].to_dict('records')

# Hidden gems
gems = df_active[df_active['selected_by_percent'] < 5].nlargest(5, 'value')
results['insights']['hidden_gems'] = gems[['web_name', 'team_name', 'selected_by_percent', 'value', 'total_points', 'now_cost']].to_dict('records')

# Position stats
pos_stats = df_active.groupby('position').agg(
    count=('id', 'count'),
    avg_points=('total_points', 'mean'),
    avg_price=('now_cost', 'mean'),
    avg_value=('value', 'mean')
).round(2)
results['position_stats'] = pos_stats.reset_index().to_dict('records')

# Save JSON
with open('output/data.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 50)
print("DONE! All outputs saved to output/")
print("  Charts: output/charts/ (11 PNG files)")
print("  Data:   output/data.json")
print("  Open:   output/index.html in your browser")
print("=" * 50)
