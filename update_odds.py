#!/usr/bin/env python3
"""
update_odds.py
Fetches fresh DraftKings odds from The Odds API,
updates card_predictions.json with new odds and recalculated verdicts.
Run by GitHub Action twice daily.
"""
import json, os, sys
from urllib.request import urlopen
from urllib.error import URLError

ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')
ODDS_API_URL = (
    'https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds'
    '?regions=us&markets=h2h&oddsFormat=american&bookmakers=draftkings'
    f'&apiKey={ODDS_API_KEY}'
)

def imp(o):
    o = float(o)
    return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)

def remove_vig(o1, o2):
    i1, i2 = imp(o1), imp(o2)
    t = i1 + i2
    return i1/t, i2/t

def calc_verdict(prob1, prob2, odds1, odds2):
    m1, m2 = remove_vig(odds1, odds2)
    e1 = prob1/100 - m1
    e2 = prob2/100 - m2

    if e1 >= e2 and e1 > 0:
        bet_on, bet_odds, edge = 'f1', odds1, e1
    elif e2 > 0:
        bet_on, bet_odds, edge = 'f2', odds2, e2
    else:
        return None, None, None, 'PASS', max(e1, e2)

    is_dog = bet_odds > 0
    within_cap = (abs(bet_odds) <= 150) or (is_dog and 151 <= bet_odds <= 200)
    is_disagree = edge >= 0.25 and is_dog and bet_odds <= 600

    if edge >= 0.0399 and (within_cap or is_disagree):
        verdict = 'BET'
    elif bet_on:
        verdict = 'PASS_CAP'
    else:
        verdict = 'PASS'

    return bet_on, bet_odds, edge, verdict, edge

def last(name):
    return name.strip().split()[-1].lower()

def main():
    # Load current predictions
    with open('card_predictions.json', 'r') as f:
        data = json.load(f)

    event = data[0]
    bouts = event['bouts']

    # Fetch live odds
    try:
        with urlopen(ODDS_API_URL) as r:
            games = json.loads(r.read())
    except URLError as e:
        print(f'Failed to fetch odds: {e}')
        sys.exit(0)

    print(f'Fetched {len(games)} games from The Odds API')

    # Build odds map by last name
    odds_map = {}
    for game in games:
        dk = next((b for b in game.get('bookmakers', []) if b['key'] == 'draftkings'), None)
        if not dk: continue
        market = next((m for m in dk.get('markets', []) if m['key'] == 'h2h'), None)
        if not market or len(market['outcomes']) < 2: continue
        o = market['outcomes']
        n1, n2 = last(o[0]['name']), last(o[1]['name'])
        odds_map[(n1, n2)] = (float(o[0]['price']), float(o[1]['price']))
        odds_map[(n2, n1)] = (float(o[1]['price']), float(o[0]['price']))

    # Update each bout
    updated = 0
    for bout in bouts:
        f1l = last(bout['fighter1'])
        f2l = last(bout['fighter2'])
        entry = odds_map.get((f1l, f2l))
        if not entry:
            print(f'  No odds found: {bout["fighter1"]} vs {bout["fighter2"]}')
            continue

        new_o1, new_o2 = entry
        bout['odds1'] = new_o1
        bout['odds2'] = new_o2

        # Recalculate verdict
        if bout.get('prob1') and bout.get('prob2'):
            bet_side, bet_odds, edge, verdict, _ = calc_verdict(
                bout['prob1'], bout['prob2'], new_o1, new_o2
            )
            bout['verdict'] = verdict
            if bet_side == 'f1':
                bout['bet_on'] = bout['fighter1']
                bout['bet_odds'] = new_o1
            elif bet_side == 'f2':
                bout['bet_on'] = bout['fighter2']
                bout['bet_odds'] = new_o2
            else:
                bout['bet_on'] = None
                bout['bet_odds'] = None

            m1, m2 = remove_vig(new_o1, new_o2)
            bout['edge1'] = round((bout['prob1']/100 - m1)*100, 1)
            bout['edge2'] = round((bout['prob2']/100 - m2)*100, 1)
            bout['mkt_prob1'] = round(m1*100, 1)
            bout['mkt_prob2'] = round(m2*100, 1)
            bout['within_cap'] = (abs(new_o1 if bet_side=='f1' else new_o2) <= 150) if bet_side else False

        updated += 1
        print(f'  Updated: {bout["fighter1"]} vs {bout["fighter2"]} | {new_o1}/{new_o2} | {verdict}')

    print(f'Updated {updated}/{len(bouts)} bouts')

    # Save
    with open('card_predictions.json', 'w') as f:
        json.dump(data, f, indent=2)
    print('Saved card_predictions.json')

if __name__ == '__main__':
    main()
