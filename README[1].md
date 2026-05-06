# UFC Edge Analyzer

Live at: https://spritzer134420.github.io/UFC-Model/

## Weekly workflow

After each fight card:

```bash
# 1. Scrape latest results
python scraper.py

# 2. Retrain model
run_all.bat --skip-tapology

# 3. Update card with next week's fights in predict_card.py, then run:
python predict_card.py

# 4. Copy predictions to this repo and push
copy C:\Users\Garrett\ufc_ml\card_predictions.json .
git add card_predictions.json
git commit -m "UFC XXX predictions"
git push
```

## How it works

- Model predicts win probability based on fighter stats (ELO, sig strike accuracy, takedown defense, recency features)
- Edge = model probability minus market implied probability
- Only bets within -150/+150 odds range — backtest shows +80.8% ROI on favs, +18.6% ROI on dogs in that range
- Live odds pulled from The Odds API (updates on page load)
- Edit any fight's odds manually to recalculate edge live
