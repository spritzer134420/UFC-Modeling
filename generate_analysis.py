import json
import os
import urllib.request
import urllib.error

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not API_KEY:
    raise SystemExit("ANTHROPIC_API_KEY not set. Run: set ANTHROPIC_API_KEY=sk-ant-...")

PREDICTIONS_FILE = "card_predictions.json"

def get_analysis(fighter, opponent, weight_class, prob, mkt_prob, edge, verdict, bet_odds):
    odds_str = f"+{int(bet_odds)}" if bet_odds > 0 else str(int(bet_odds))
    prompt = (
        f"Write 2-3 sentences analyzing why {fighter} is interesting against {opponent} "
        f"in a UFC {weight_class}. Focus only on fighter attributes: style, experience, "
        f"finishing ability, recent form. Do not mention odds or betting."
    )
    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"].strip()
    except Exception as e:
        print(f"  API error for {fighter}: {e}")
        return f"Model gives {fighter} {prob:.0f}% vs market's {mkt_prob:.0f}% — a {edge:.1f}% edge."

def main():
    with open(PREDICTIONS_FILE, encoding="utf-8") as f:
        predictions = json.load(f)

    card = predictions[0]
    bouts = card["bouts"]
    event = card.get("event", "UFC Event")

    print(f"Generating analysis for {event}...")
    print(f"Found {len(bouts)} bouts\n")

    for bout in bouts:
        fighter1 = bout["fighter1"]
        fighter2 = bout["fighter2"]
        verdict = bout.get("verdict", "")
        bet_on = bout.get("bet_on", fighter1)
        opponent = fighter2 if bet_on == fighter1 else fighter1
        prob = bout["prob1"] if bet_on == fighter1 else bout["prob2"]
        mkt_prob = bout["mkt_prob1"] if bet_on == fighter1 else bout["mkt_prob2"]
        edge = abs(bout.get("edge1", bout.get("edge2", 0)))
        bet_odds = bout.get("bet_odds", 0)
        weight_class = bout.get("weight_class", "bout")

        if verdict == "BET":
            print(f"  Generating: {fighter1} vs {fighter2} (BET {bet_on})...")
            analysis = get_analysis(bet_on, opponent, weight_class, prob, mkt_prob, edge, verdict, bet_odds)
            bout["analysis"] = analysis
            prob_dec = prob / 100
            if bet_odds > 200 and edge >= 25:
                min_mkt_prob = max(prob_dec - 0.25, 0.01)
                sweet_odds = int((1 - min_mkt_prob) / min_mkt_prob * 100)
                sweet_spot = f"+{sweet_odds} to +600"
            elif bet_odds > 0:
                sweet_spot = "+101 to +200"
            else:
                min_mkt_prob = max(prob_dec - 0.04, 0.01)
                sweet_odds = -int(min_mkt_prob / (1 - min_mkt_prob) * 100)
                sweet_spot = f"{sweet_odds} to -101"
            bout["analysis_footer"] = (
                f"Model gives {bet_on} {prob:.0f}% vs market's {mkt_prob:.0f}% — "
                f"a {edge:.1f}% edge. Bet range: {sweet_spot}."
            )
            print(f"    Done.")
        else:
            bout["analysis"] = ""
            bout["analysis_footer"] = (
                f"Model gives {bet_on} {prob:.0f}% vs market's {mkt_prob:.0f}% — "
                f"{edge:.1f}% edge. Below bet threshold."
            )

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    bets = [b for b in bouts if b.get("verdict") == "BET"]
    print(f"\nDone. {len(bets)} bets with analysis saved to {PREDICTIONS_FILE}")
    for b in bets:
        print(f"  {b['bet_on']}: {b['analysis'][:80]}...")

if __name__ == "__main__":
    main()
