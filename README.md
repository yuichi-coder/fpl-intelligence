<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

<h1 align="center">FPL Intelligence</h1>
<h3 align="center">Fantasy Premier League Prediction Engine</h3>

<p align="center">
  A production-grade ML pipeline that predicts FPL player points using a <b>Position-Specific Stacked Meta-Ensemble</b><br>
  trained on <b>128,000+ gameweek records</b> across 5 seasons (2019-2025)
</p>

<p align="center">
  <a href="#model-results">Results</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#dashboard">Dashboard</a>
</p>

---

## Model Results

### V3 — Position-Specific Ensemble (3-GW Average Target)

| Model | MAE | R² | Top-20 | Top-10 |
|:------|:---:|:---:|:------:|:------:|
| **Random Forest** | **1.348** | 0.074 | 30.0% | 27.5% |
| LSTM (Attention) | 1.349 | 0.048 | 25.0% | 27.5% |
| Simple Average | 1.349 | 0.077 | 33.8% | 27.5% |
| Stacked Meta | 1.354 | 0.061 | 35.0% | 30.0% |
| XGBoost | 1.355 | 0.065 | **36.2%** | **32.5%** |
| Neural Net (MLP) | 1.377 | 0.034 | 28.8% | 20.0% |

> **MAE of 1.35** places this model in the **"excellent" tier** by industry benchmarks (excellent = 1.4–1.8, good = 1.8–2.3).
> Top-20 ranking accuracy of **36.2%** means the model correctly identifies over a third of the best-performing players each gameweek.

### V3 Improvements over V1

| Metric | V1 (Single GW) | V3 (3-GW Avg) | Improvement |
|:-------|:---:|:---:|:---:|
| Best MAE | 1.673 | **1.348** | **-19.4%** |
| Top-20 Accuracy | 17% | **36.2%** | **+113%** |

### Per-Position Performance

| Position | MAE | R² | Train Size | Test Size |
|:---------|:---:|:---:|:----------:|:---------:|
| GK | 1.417 | -0.034 | 2,676 | 77 |
| DEF | 1.201 | -0.015 | 11,134 | 282 |
| MID | 1.419 | 0.108 | 19,558 | 317 |
| FWD | 1.594 | 0.012 | 3,212 | 73 |

---

## Architecture

### V3 — Three Key Innovations

```
1. NOISE REDUCTION          Filter out GWs where player played < 30 minutes
                            → removes ~46% of noisy sub appearances

2. SMOOTH TARGET            Predict 3-GW forward average instead of single GW
                            → target = mean(GW_i, GW_i+1, GW_i+2)
                            → dramatically reduces target variance

3. POSITION-SPECIFIC        Separate model pipeline per position (GK/DEF/MID/FWD)
                            → each position learns its own scoring patterns
```

### Stacking Pipeline (per position)

```
                    ┌─────────────────────────────────────────────┐
                    │           TRAINING DATA (per position)       │
                    │   Historical (5 seasons) + Current season    │
                    │   Features: 73 engineered features           │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────────────┐
                    │                  │                           │
              ┌─────▼─────┐    ┌──────▼──────┐    ┌──────────────▼──┐
              │  XGBoost   │    │ Random Forest│    │ MLP Neural Net  │
              │ 600 trees  │    │  200 trees   │    │ 256→128→64→1    │
              │ 5-fold CV  │    │  5-fold CV   │    │   5-fold CV     │
              └─────┬──────┘    └──────┬───────┘    └────────┬────────┘
                    │                  │                      │
                    │    Out-of-Fold Predictions              │
                    └──────────┬───────┴──────────────────────┘
                               │
              ┌────────────────▼──────────────────────────────────┐
              │                 LSTM with Attention                │
              │    Sequence: last 6 GWs × 18 features             │
              │    128 hidden → Attention → 64 → 32 → 1           │
              └────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────▼──────────────────────────────────┐
              │              META-LEARNER (XGBoost)                │
              │                                                    │
              │   Input: 4 base predictions                        │
              │        + pred_mean, pred_std, pred_max, min, range │
              │        + 12 context features (form, price, GW...)  │
              │        + lstm_pred                                 │
              │   ─────────────────────────────────────────        │
              │   Total: 22 meta-features → Final prediction       │
              └────────────────┬──────────────────────────────────┘
                               │
                        ┌──────▼──────┐
                        │  Predicted   │
                        │  3-GW Avg    │
                        │   Points     │
                        └─────────────┘

              × 4 positions (GK, DEF, MID, FWD)
```

---

## Features

### Data Pipeline
- **FPL API** — 818 players, 380 fixtures, 20 teams (real-time)
- **128,602 historical records** from 5 seasons via [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
- **Gameweek-level history** for 362+ active players

### Feature Engineering (73 features)

| Category | Features | Description |
|:---------|:---------|:------------|
| Rolling Averages | `*_roll_3`, `*_roll_5`, `*_roll_8` | Points, minutes, goals, assists, bonus, BPS, ICT, xG, xA, CS, saves |
| Consistency | `pts_std_3`, `pts_std_5` | Points standard deviation over recent windows |
| Momentum | `momentum_3v8`, `xg_momentum` | Short-term vs long-term form comparison |
| Per-90 | `*_per90` | Goals, assists, bonus, xG, xA normalized by minutes |
| Season Cumulative | `season_pts`, `ppg`, `games_played` | Running totals and averages |
| xG Analysis | `xg_overperformance`, `xa_overperformance` | Actual vs expected (sustainability indicator) |
| Context | `is_home`, `price`, `gw`, `minutes_trend` | Match and player context |
| Transfer | `transfer_trend` | Recent transfer balance momentum |

### Analysis Engine (38+ Charts)
- Points distribution, price vs value scatter
- **PCA** dimensionality reduction + **K-Means clustering**
- **xG/xA deep dive** — overperformers vs underperformers
- **Form detection** — rising and falling players
- **Fixture difficulty forecast** — FDR heatmap
- **Transfer ROI** — price movers, knee-jerk detector
- **Optimal squad selection** via Integer Linear Programming

### Team Optimization
Optimal team via **Integer Linear Programming** (PuLP):
- Budget: **100.0m**
- Squad: 2 GK + 5 DEF + 5 MID + 3 FWD = 15 players
- Max 3 players per club
- Captain selection (highest predicted points)
- Wildcard advisor (best squad over next 5 GWs)

---

## Dashboard

**8-page interactive Streamlit dashboard** with FPL-themed design and Plotly visualizations.

| Page | Description |
|:-----|:------------|
| **Overview** | KPI cards, top performers, position breakdown |
| **Player Analysis** | Search any player, predict next 5 GWs, radar comparison |
| **xG Analysis** | Expected vs actual goals/assists, overperformers |
| **Form & Trends** | Rising/falling form, consistency metrics |
| **Fixture Forecast** | FDR heatmap, best picks by difficulty |
| **Optimal Teams** | Best 15 per GW, wildcard advisor, captain picks |
| **Model Arena** | Head-to-head: 6 models compared with interactive charts |
| **Model Performance** | Per-position R², feature importance, architecture details |

---

## Quick Start

### Prerequisites
- Python 3.10+
- ~2GB disk space (historical data + models)

### 1. Clone & install
```bash
git clone https://github.com/yuichi-coder/fpl-intelligence.git
cd fpl-intelligence
pip install -r requirements.txt
```

### 2. Pull data
```bash
python pull_data.py              # Current season from FPL API
python pull_all_histories.py     # GW histories for all active players
python pull_historical.py        # 5 seasons of historical data (2019-2024)
```

### 3. Run analysis
```bash
python analysis.py               # Generates 38+ charts + data.json
```

### 4. Train models
```bash
python model_v3.py               # Position-specific ensemble (~15 min on CPU)
```

### 5. Launch dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501**

---

## Project Structure

```
fpl-intelligence/
│
├── app.py                    # Streamlit dashboard — 8 pages, ~2500 lines
├── model_v3.py               # V3 Position-Specific Meta-Ensemble training
├── analysis.py               # EDA engine — 38+ charts, clustering, PCA
├── pull_data.py              # FPL API data pipeline
├── pull_historical.py        # Historical data scraper (5 seasons)
├── pull_all_histories.py     # Player gameweek history collector
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
│
├── data/
│   ├── raw/                  # Raw API data (players, teams, fixtures)
│   ├── processed/            # Cleaned & enriched player data
│   └── historical/           # 128K+ records across 5 seasons
│
└── output/
    ├── model.pkl             # Saved position-specific models (~25MB)
    ├── nn_models.pt          # PyTorch weights — MLP + LSTM per position
    ├── model_data.json       # Predictions, best teams, captain picks, metrics
    ├── data.json             # Analysis results for dashboard
    └── charts/               # 50+ generated PNG charts
```

---

## Tech Stack

| Category | Tools |
|:---------|:------|
| **Language** | Python 3.13 |
| **Deep Learning** | PyTorch (LSTM with Attention, MLP with BatchNorm) |
| **Gradient Boosting** | XGBoost (base models + meta-learner) |
| **Classical ML** | scikit-learn (Random Forest, StandardScaler, KFold) |
| **Optimization** | PuLP (Integer Linear Programming for squad selection) |
| **Data** | pandas, NumPy |
| **Visualization** | Plotly (interactive), matplotlib + seaborn (static) |
| **Dashboard** | Streamlit |
| **Data Sources** | FPL API, [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) |

---

## How It Works

```
FPL API ──→ pull_data.py ──→ data/raw/
                                │
GitHub ───→ pull_historical.py → data/historical/
                                │
                          analysis.py ──→ output/charts/ + data.json
                                │
                          model_v3.py ──→ output/model.pkl + model_data.json
                                │
                            app.py ──→ localhost:8501 (Streamlit Dashboard)
```

---

## Methodology Notes

- **No data leakage**: All features use `.shift(1)` (lagged by 1 GW). Out-of-fold predictions via 5-fold CV ensure the meta-learner never sees its own training data.
- **3-GW average target**: Predicting the mean of current + next 2 GWs reduces single-match variance. MAE improves by ~19% vs single-GW prediction.
- **Minutes filter (30+)**: Removing sub appearances eliminates ~46% of noisy rows where a player comes on for 5 minutes and gets 1 point.
- **Per-position models**: GK/DEF depend on clean sheets; MID/FWD depend on goals/assists. Separate models capture these different scoring dynamics.
- **LSTM uses raw GW data**: The LSTM processes sequences of 6 gameweeks with 18 features each, while tabular models use pre-computed rolling averages.

---

## License

MIT

---

<p align="center">
  Built as a data science portfolio project — end-to-end ML pipeline from data collection to interactive deployment.
</p>
