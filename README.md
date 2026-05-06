# UFC Edge Analyzer

A machine learning pipeline for predicting UFC fight outcomes and finding betting edge vs closing lines.

## Results

| Period | Bets | Win % | ROI |
|--------|------|-------|-----|
| Holdout (Sep 2023 – Apr 2026) | 580 | 84.5% | **+80.2%** |
| Last 20 Events | 108 | 63.0% | **+85.4%** |
| Last 3 Events | 17 | 64.7% | **+84.0%** |

*Flat $100 bets. Bet when model prob > market implied prob. Real closing line odds.*

## What It Does

- Scrapes fight data from ufcstats.com (10,900+ fights)
- Engineers 91 features (striking, grappling, Elo, reach, stance, era)
- XGBoost model — **90.8% accuracy on 2023+ holdout fights**
- Compares model probability vs real closing odds to find +EV bets

## Setup

```bash
git clone https://github.com/yourusername/ufc-ml
cd ufc-ml
pip install -r requirements.txt
```

## Run

```bash
# Full pipeline
run_all.bat --skip-scrape --skip-tapology

# Predict a fight
python predict.py "Islam Makhachev" "Dustin Poirier" --odds1 -400 --odds2 +320

# Profitability check
python profitability_check.py
```

## Model

- **Algorithm:** XGBoost only (outperforms full ensemble by 18pp on 2023+ data)
- **Training:** Era-filtered — excludes 2002–2013 and 2019–2022
- **Holdout:** Last 20% by date

## Betting Rule

Bet $100 when: `model_probability > market_implied_probability`

## Files

| File | Description |
|------|-------------|
| `scraper.py` | Scrapes ufcstats.com |
| `features.py` | Builds feature matrix |
| `train.py` | Trains models |
| `predict.py` | CLI predictor |
| `profitability_check.py` | Backtest vs closing odds |
| `import_odds_csv.py` | Import historical odds CSV |
| `model_selection.py` | Tests model combinations |
| `ufc-analyzer.jsx` | React dashboard |

## Disclaimer

Research project. Past performance does not guarantee future results.
