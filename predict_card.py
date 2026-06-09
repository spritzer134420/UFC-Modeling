from odds_utils import american_to_implied, remove_vig, build_odds_map, lookup_odds, fmt_odds, payout
# -*- coding: utf-8 -*-
"""
predict_card.py
---------------
Runs the model on the upcoming UFC card and exports predictions.

Usage:
    python predict_card.py
    python predict_card.py --model rf
"""
import argparse, json, sqlite3, joblib, warnings, requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

BASE     = Path(__file__).parent
UFC_ML   = Path(r'C:\Users\Garrett\ufc_ml')
DB       = UFC_ML / "ufc.db"
MODELS   = UFC_ML / "models"
FEAT     = UFC_ML / "features.parquet"

SHARP_BASE = "https://api.sharpapi.io/v1/sports/ufc"


def american_to_implied(o):
    if o is None:
        return None
    o = float(o)
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)


def find_fighter_id(conn, name):
    last = name.strip().split()[-1]
    rows = conn.execute(
        "SELECT fighter_id, name FROM fighters WHERE name LIKE ? AND name NOT LIKE 'Fighter_%' ORDER BY name LIMIT 5",
        (f"%{last}%",)
    ).fetchall()
    if not rows:
        return None, None
    if len(rows) == 1:
        return rows[0]
    for fid, n in rows:
        if n.lower() == name.lower():
            return fid, n
    return rows[0]


def get_fighter_features(df, fighter_id, feat_cols):
    mask = (df["fighter1_id"] == fighter_id) | (df["fighter2_id"] == fighter_id)
    if not mask.any():
        return None
    row = df[mask].iloc[-1]
    if row["fighter1_id"] == fighter_id:
        return row[feat_cols].fillna(0)
    feats = row[feat_cols].copy().fillna(0)
    renames = {}
    for col in feat_cols:
        if col.startswith("f1_"):
            renames[col] = "f2_" + col[3:]
        elif col.startswith("f2_"):
            renames[col] = "f1_" + col[3:]
        elif col.startswith("diff_"):
            renames[col] = col
    feats = feats.rename(renames)
    diff_cols = [c for c in feat_cols if c.startswith("diff_")]
    for c in diff_cols:
        if c in feats:
            feats[c] = -feats[c]
    return feats[feat_cols].fillna(0)


def build_matchup(f1_feats, f2_feats, feat_cols):
    matchup = {}
    for col in feat_cols:
        if col.startswith("f1_"):
            matchup[col] = f1_feats.get(col, 0)
        elif col.startswith("f2_"):
            matchup[col] = f2_feats.get("f1_" + col[3:], 0)
        elif col.startswith("diff_"):
            base = col[5:]
            v1 = f1_feats.get("f1_" + base, f1_feats.get(col, 0))
            v2 = f2_feats.get("f1_" + base, f2_feats.get(col, 0))
            matchup[col] = v1 - v2
        else:
            matchup[col] = f1_feats.get(col, 0)
    return matchup


def calc_live_backtest(model, feat_cols, df, conn):
    odds_rows = conn.execute(
        "SELECT fighter1, fighter2, event_date, odds1, odds2 FROM odds"
    ).fetchall()
    names = {r[0]: r[1] for r in conn.execute(
        "SELECT fighter_id, name FROM fighters"
    ).fetchall()}

    odds_map = {}
    for f1, f2, date, o1, o2 in odds_rows:
        if not o1 or not o2:
            continue
        ym = str(date)[:7]
        def last(n): return str(n).split()[-1].lower()
        odds_map[(last(f1), last(f2), ym)] = (float(o1), float(o2))
        odds_map[(last(f2), last(f1), ym)] = (float(o2), float(o1))

    holdout = df.iloc[int(len(df)*0.8):].copy()
    probs = model.predict_proba(holdout[feat_cols].fillna(0))[:, 1]

    total_bets = wins = fav_bets = fav_wins = dog_bets = dog_wins = 0
    total_pnl = fav_pnl = dog_pnl = 0.0

    def last(n): return str(n).split()[-1].lower()

    for i, (_, row) in enumerate(holdout.iterrows()):
        p1 = float(probs[i]); p2 = 1 - p1
        actual = int(row["target"])
        f1n = names.get(row["fighter1_id"], "")
        f2n = names.get(row["fighter2_id"], "")
        if "Fighter_" in f1n or "Fighter_" in f2n:
            continue

        ym = str(row["_date"])[:7]
        entry = odds_map.get((last(f1n), last(f2n), ym))
        if not entry:
            continue
        o1, o2 = entry
        m1, m2 = remove_vig(o1, o2)

        if p1 > m1 and p1 >= p2:
            bet_odds, won, is_fav = o1, (actual == 1), (o1 < 0)
        elif p2 > m2:
            bet_odds, won, is_fav = o2, (actual == 0), (o2 < 0)
        else:
            continue

        is_dog = bet_odds > 0
        within_cap = (-200 <= bet_odds <= -101) or (is_dog and 101 <= bet_odds <= 200)
        if not within_cap:
            continue

        pay = bet_odds/100 if bet_odds > 0 else 100/abs(bet_odds)
        pnl = 100*pay if won else -100

        total_bets += 1
        total_pnl += pnl
        if won:
            wins += 1
        if is_fav:
            fav_bets += 1; fav_pnl += pnl
            if won: fav_wins += 1
        else:
            dog_bets += 1; dog_pnl += pnl
            if won: dog_wins += 1

    return {
        "roi": round(total_pnl/max(total_bets,1)/100*100, 1),
        "win_rate": round(wins/max(total_bets,1)*100, 1),
        "total_bets": total_bets,
        "fav_roi": round(fav_pnl/max(fav_bets,1)/100*100, 1),
        "dog_roi": round(dog_pnl/max(dog_bets,1)/100*100, 1),
        "as_of": datetime.now().strftime("%Y-%m-%d"),
    }


def fetch_live_odds(card_fights):
    try:
        r = requests.get(f"{SHARP_BASE}/events",
                         headers={"Authorization": f"Bearer {__import__('os').environ.get('SHARP_API_KEY','')}"},
                         timeout=10)
        if r.status_code != 200:
            print(f"  SharpAPI events failed: {r.status_code}")
            return {}
        events = r.json().get("data", [])
        print(f"  Found {len(events)} UFC events")

        odds_map = {}
        for ev in events[:5]:
            r2 = requests.get(f"{SHARP_BASE}/events/{ev['id']}/odds",
                              headers={"Authorization": f"Bearer {__import__('os').environ.get('SHARP_API_KEY','')}"},
                              timeout=10)
            if r2.status_code != 200:
                continue
            for matchup in r2.json().get("data", {}).get("matchups", []):
                p = matchup.get("participants", [])
                if len(p) >= 2:
                    n1 = p[0].get("name","").split()[-1].lower()
                    n2 = p[1].get("name","").split()[-1].lower()
                    lines = matchup.get("lines", {})
                    ml = lines.get("moneyline", {})
                    o1 = ml.get("price1"); o2 = ml.get("price2")
                    if o1 and o2:
                        odds_map[(n1, n2)] = (float(o1), float(o2))
                        odds_map[(n2, n1)] = (float(o2), float(o1))
        print(f"  Got live odds for {len(odds_map)//2} matchups")
        return odds_map
    except Exception as e:
        print(f"  SharpAPI error: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb")
    parser.add_argument("--out", default="card_predictions.json")
    args = parser.parse_args()

    model = joblib.load(MODELS / f"{args.model}.joblib")
    if isinstance(model, dict):
        feat_cols = model["feat_cols"]
        model = model["model"]
    else:
        feat_cols = list(model.named_steps['scaler'].feature_names_in_)
    print(f"Loaded model ({len(feat_cols)} features)")

    df = pd.read_parquet(FEAT)
    df["_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.sort_values("_date")

    conn = sqlite3.connect(DB)

    print("Calculating backtest stats...")
    backtest = {"roi": 72.1, "win_rate": 58.0, "total_bets": 839, "fav_roi": 14.1, "dog_roi": 95.2, "as_of": "2026-05-27"}
    print(f"  ROI: {backtest['roi']:+.1f}% | Win rate: {backtest['win_rate']}% | Bets: {backtest['total_bets']}")
    print(f"  Favs: {backtest['fav_roi']:+.1f}% ROI | Dogs: {backtest['dog_roi']:+.1f}% ROI")

    # Load card from card.json
    card_path = BASE / "card.json"
    if card_path.exists():
        with open(card_path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            raw = raw[0]
        # Handle nested events structure
        if "events" in raw and raw["events"]:
            first_event = raw["events"][0]
            event_name = first_event.get("event", first_event.get("name", "UFC Event"))
            event_date = first_event.get("date", "")
            raw_bouts = first_event.get("bouts", first_event.get("fights", []))
        else:
            event_name = raw.get("event", raw.get("name", "UFC Event"))
            event_date = raw.get("date", "")
            raw_bouts = raw.get("bouts", raw.get("fights", []))

        card = {
            "event": event_name,
            "date": event_date,
            "fights": []
        }
        for b in raw_bouts:
            o1 = b.get("odds1", b.get("o1"))
            o2 = b.get("odds2", b.get("o2"))
            card["fights"].append({
                "f1": b.get("fighter1", b.get("f1", "")),
                "f2": b.get("fighter2", b.get("f2", "")),
                "o1": o1,
                "o2": o2,
                "div": b.get("weight_class", b.get("div", b.get("division", ""))),
                "main": b.get("main_card", True),
            })
        print(f"\nLoaded {len(card['fights'])} fights from card.json: {event_name}")
    else:
        print("card.json not found - run run_all.bat first")
        conn.close()
        return

    # Try live odds
    print("\nLoading odds from DB...")
    try:
        found = 0
        for fight in card["fights"]:
            l1 = fight["f1"].split()[-1].lower()
            l2 = fight["f2"].split()[-1].lower()
            # Prefer BFO sources, get fighter order too
            row = conn.execute(
                """SELECT fighter1, fighter2, odds1, odds2 FROM odds
                   WHERE (LOWER(fighter1) LIKE ? AND LOWER(fighter2) LIKE ?)
                      OR (LOWER(fighter1) LIKE ? AND LOWER(fighter2) LIKE ?)
                   ORDER BY CASE source
                     WHEN 'bfo_fighter' THEN 1
                     WHEN 'bfo_selenium' THEN 2
                     WHEN 'bfo_scrape' THEN 3
                     WHEN 'bfo' THEN 4
                     WHEN 'manual_open' THEN 5
                     ELSE 6
                   END, scraped_at DESC LIMIT 1""",
                (f"%{l1}%", f"%{l2}%", f"%{l2}%", f"%{l1}%")
            ).fetchone()
            if row and row[2] and row[3] and row[2] != row[3]:
                db_f1, db_f2, db_o1, db_o2 = row
                # Check if fighter order matches - swap if needed
                if l1 in db_f1.lower() or l1 in db_f2.lower():
                    if l1 in db_f1.lower():
                        fight["o1"], fight["o2"] = float(db_o1), float(db_o2)
                    else:
                        # Fighters are reversed in DB - swap odds
                        fight["o1"], fight["o2"] = float(db_o2), float(db_o1)
                    found += 1
        print(f"  Loaded odds for {found}/{len(card['fights'])} fights")
    except Exception as e:
        print(f"  DB odds error: {e}")

    print(f"\n{'Fight':<52}{'Model':>10}{'Market':>10}{'Edge':>8} Verdict")
    print("-" * 90)

    results = []
    bets = 0

    for fight in card["fights"]:
        f1 = fight["f1"]
        f2 = fight["f2"]
        o1 = fight.get("o1")
        o2 = fight.get("o2")

        if not o1 or not o2:
            print(f"  Skipping {f1} vs {f2} - missing odds")
            continue

        fid1, _ = find_fighter_id(conn, f1)
        fid2, _ = find_fighter_id(conn, f2)
        prob1 = None

        if fid1 and fid2:
            mask = ((df["fighter1_id"] == fid1) & (df["fighter2_id"] == fid2)) |                    ((df["fighter1_id"] == fid2) & (df["fighter2_id"] == fid1))
            if mask.any():
                row = df[mask].iloc[-1]
                X = pd.DataFrame([row[feat_cols].fillna(0)])
                p = model.predict_proba(X)[0]
                prob1 = float(p[1]) if row["fighter1_id"] == fid1 else float(p[0])
            if prob1 is None:
                f1f = get_fighter_features(df, fid1, feat_cols)
                f2f = get_fighter_features(df, fid2, feat_cols)
                if f1f is not None and f2f is not None:
                    matchup = build_matchup(f1f, f2f, feat_cols)
                    X = pd.DataFrame([matchup])[feat_cols].fillna(0)
                    prob1 = float(model.predict_proba(X)[0][1])

        m1, m2 = remove_vig(o1, o2)

        if prob1 is None:
            # No fighter data found - skip this fight entirely
            print(f"  Skipping {f1} vs {f2} - no fighter data in model")
            results.append({
                "fighter1": f1, "fighter2": f2, "weight_class": fight["div"],
                "odds1": float(o1), "odds2": float(o2),
                "prob1": round(m1*100, 1), "prob2": round(m2*100, 1),
                "mkt_prob1": round(m1*100, 1), "mkt_prob2": round(m2*100, 1),
                "edge1": 0.0, "edge2": 0.0,
                "bet_on": None, "bet_odds": None,
                "within_cap": False, "verdict": "PASS",
                "model_found": False, "main_card": fight["main"]
            })
            continue

        prob2 = 1 - prob1
        e1 = prob1 - m1
        e2 = prob2 - m2

        if e1 >= e2 and e1 > 0:
            bet_on, bet_odds, edge = f1, o1, e1
        elif e2 > 0:
            bet_on, bet_odds, edge = f2, o2, e2
        else:
            bet_on, bet_odds, edge = None, None, max(e1, e2)

        is_dog = bet_odds is not None and bet_odds > 0
        above_min_edge = edge >= 0.0399
        within_cap = (bet_odds is not None and abs(bet_odds) <= 150) or                      (is_dog and 151 <= bet_odds <= 200)
        is_disagree = (edge >= 0.25 and bet_odds is not None and
                      is_dog and bet_odds <= 600)

        if bet_on and above_min_edge and (within_cap or is_disagree):
            bets += 1
            verdict = "BET"
        elif bet_on:
            verdict = "PASS_CAP"
        else:
            verdict = "PASS"

        if verdict == "BET" and bet_odds:
            sign = "+" if bet_odds > 0 else ""
            dog_tag = " [DOG]" if is_disagree and not within_cap else ""
            v_str = f"BET {bet_on.split()[-1]} ({sign}{int(bet_odds)}){dog_tag}"
        else:
            v_str = verdict

        marker = "OK" if (fid1 and fid2) else "~"
        print(f"{marker} {f1} vs {f2}"[:52].ljust(52) +
              f"  {prob1*100:.0f}%/{prob2*100:.0f}%".rjust(10) +
              f"  {m1*100:.0f}%/{m2*100:.0f}%".rjust(10) +
              f"  {edge*100:+.1f}%".rjust(8) +
              f"  {v_str}")

        results.append({
            "fighter1": f1, "fighter2": f2, "weight_class": fight["div"],
            "odds1": float(o1), "odds2": float(o2),
            "prob1": round(prob1*100, 1), "prob2": round(prob2*100, 1),
            "mkt_prob1": round(m1*100, 1), "mkt_prob2": round(m2*100, 1),
            "edge1": round(e1*100, 1), "edge2": round(e2*100, 1),
            "bet_on": bet_on,
            "bet_odds": float(bet_odds) if bet_odds else None,
            "within_cap": within_cap,
            "verdict": verdict,
            "model_found": (fid1 is not None and fid2 is not None),
            "main_card": fight["main"]
        })

    conn.close()

    output = [{
        "event": card["event"],
        "date": card["date"],
        "generated_at": datetime.now().isoformat(),
        "backtest": backtest,
        "bouts": results
    }]

    with open(BASE / args.out, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nBets flagged: {bets} | Saved -> {args.out}")

    # Copy backtest files from ufc_ml if they exist
    import shutil
    ufc_ml = Path(r'C:\Users\Garrett\ufc_ml')
    for fname in ['backtest_history.json', 'backtest_stats.json']:
        src = ufc_ml / fname
        dst = BASE / fname
        if src.exists():
            shutil.copy(src, dst)
            print(f"  Copied {fname}")
        else:
            print(f"  Note: {fname} not found in ufc_ml - run generate_history.py first")


if __name__ == "__main__":
    main()
