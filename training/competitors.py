"""
Strong AI-style competitor bots for the round-robin — the kind of opponents
other people's AI would plausibly build for this task. NOT submitted; used only
to stress-test our bot against similar-caliber play and find its leaks.

  eqpro  — clean equity + pot-odds + position (the "typical good AI heuristic")
  tag    — tight-aggressive: premium ranges, aggressive c-bets, folds the rest
  gto    — balanced/polarized: value-bets big AND bluffs ~30%, bluff-catches at
           pot odds (the hardest to exploit — tests if WE are exploitable)
  adapt  — exploitative: tracks opponents (aggression + fold rate) and adjusts
"""

import random
import eval7

_RANKS = "23456789TJQKA"
_FULL = [eval7.Card(r + s) for r in _RANKS for s in "shdc"]
_CSTR = {str(c): c for c in _FULL}
_RV = {r: i + 2 for i, r in enumerate(_RANKS)}


def _equity(hole, board, n_opp, iters=120):
    if n_opp <= 0:
        return 1.0
    h = [_CSTR[c] for c in hole]
    b = [_CSTR[c] for c in board]
    known = set(hole) | set(board)
    deck = [c for c in _FULL if str(c) not in known]
    need = 5 - len(b)
    dn = n_opp * 2 + need
    if dn > len(deck):
        return 0.5
    win = 0.0
    for _ in range(iters):
        s = random.sample(deck, dn)
        comm = b + s[n_opp * 2:]
        my = eval7.evaluate(h + comm)
        bo = 0
        for i in range(n_opp):
            o = eval7.evaluate(s[2 * i:2 * i + 2] + comm)
            if o > bo:
                bo = o
        win += 1.0 if my > bo else (0.5 if my == bo else 0.0)
    return win / iters


def _chen(cards):
    r1, r2 = cards[0][0], cards[1][0]
    suited = cards[0][1] == cards[1][1]
    v1, v2 = _RV[r1], _RV[r2]
    hi, lo = max(v1, v2), min(v1, v2)
    base = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}.get(hi, hi / 2.0)
    if r1 == r2:
        return max(base * 2, 5)
    sc = base + (2 if suited else 0)
    sc -= {0: 0, 1: 1, 2: 2, 3: 4}.get(hi - lo - 1, 5)
    if hi - lo - 1 <= 1 and hi <= 11:
        sc += 1
    return sc


def _n_opp(state):
    me = state["seat_to_act"]
    return sum(1 for p in state["players"] if not p["is_folded"] and p["seat"] != me)


def _late(state):
    players = state["players"]
    n = len(players)
    me = state["seat_to_act"]
    sb = None
    for a in state.get("action_log", []):
        if a.get("action") == "small_blind":
            sb = a["seat"]
    if sb is None:
        return 0.5
    btn = sb if n == 2 else (sb - 1) % n
    act = [p["seat"] for p in players if not p["is_folded"]]
    if me not in act or len(act) <= 1:
        return 0.5
    order = sorted(act, key=lambda x: ((x - btn - 1) % n))
    return order.index(me) / (len(order) - 1)


def _raise_to(state, frac):
    pot, owed, cb = state["pot"], state["amount_owed"], state["current_bet"]
    mx = state["your_stack"] + state["your_bet_this_street"]
    tgt = int(round((cb + frac * (pot + owed)) if owed > 0 else frac * pot))
    if tgt >= mx:
        return {"action": "all_in"}
    tgt = max(tgt, state["min_raise_to"])
    return {"action": "all_in"} if tgt >= mx else {"action": "raise", "amount": tgt}


def _opp_id(state):
    me = state["seat_to_act"]
    for p in state["players"]:
        if p["seat"] != me and not p["is_folded"]:
            return p.get("bot_id")
    return None


def _read(state, bot_id):
    """(aggression_factor, fold_factor) of bot_id from the match log; (None,None)."""
    if not bot_id:
        return None, None
    aggr = pas = folds = tot = 0
    for e in state.get("match_action_log", []):
        if e.get("bot_id") != bot_id:
            continue
        a = e.get("action")
        if a in ("fold", "call", "check", "raise", "all_in"):
            tot += 1
            if a in ("raise", "all_in"):
                aggr += 1
            elif a in ("call", "check"):
                pas += 1
            if a == "fold":
                folds += 1
    af = aggr / (aggr + pas) if (aggr + pas) >= 8 else None
    ff = folds / tot if tot >= 8 else None
    return af, ff


# ---------------------------------------------------------------------------
# Competitors
# ---------------------------------------------------------------------------

def eqpro(state):
    s = state["street"]; owed = state["amount_owed"]; pot = state["pot"]
    cc = state["can_check"]; n = _n_opp(state); late = _late(state)
    if s == "preflop":
        ch = _chen(state["your_cards"]); facing = state["current_bet"] > 100
        if ch >= 12:
            return _raise_to(state, 1.0)
        if not facing:
            if ch >= 8 - 3 * late:
                return _raise_to(state, 1.0)
            if cc:
                return {"action": "check"}
            return {"action": "call"} if (ch >= 6 and owed <= 100) else {"action": "fold"}
        if ch >= 9 and owed <= 0.10 * state["your_stack"]:
            return {"action": "call"}
        return {"action": "check"} if cc else {"action": "fold"}
    eq = _equity(state["your_cards"], state["community_cards"], n)
    req = owed / (pot + owed) if pot + owed > 0 else 1.0
    if cc:
        return _raise_to(state, 0.7) if eq >= 0.68 else {"action": "check"}
    if eq >= 0.82:
        return _raise_to(state, 0.8)
    return {"action": "call"} if eq >= req + 0.05 else {"action": "fold"}


def tag(state):
    s = state["street"]; owed = state["amount_owed"]; pot = state["pot"]
    cc = state["can_check"]; n = _n_opp(state); late = _late(state)
    if s == "preflop":
        ch = _chen(state["your_cards"]); facing = state["current_bet"] > 100
        if ch >= 11:
            return _raise_to(state, 1.2)
        if not facing and (ch >= 9 or (ch >= 8 and late > 0.5)):
            return _raise_to(state, 1.0)
        return {"action": "check"} if cc else {"action": "fold"}
    eq = _equity(state["your_cards"], state["community_cards"], n)
    req = owed / (pot + owed) if pot + owed > 0 else 1.0
    if cc:
        return _raise_to(state, 0.75) if eq >= 0.60 else {"action": "check"}
    if eq >= 0.78:
        return _raise_to(state, 0.85)
    return {"action": "call"} if eq >= req + 0.08 else {"action": "fold"}


def gto(state):
    s = state["street"]; owed = state["amount_owed"]; pot = state["pot"]
    cc = state["can_check"]; n = _n_opp(state); late = _late(state)
    if s == "preflop":
        ch = _chen(state["your_cards"]); facing = state["current_bet"] > 100
        if ch >= 10:
            return _raise_to(state, 1.1)
        if not facing and ch >= 6 - 2 * late:
            return _raise_to(state, 1.0)
        if facing and ch >= 9 and owed <= 0.12 * state["your_stack"]:
            return {"action": "call"}
        if facing and ch >= 5 and random.random() < 0.15:    # 3-bet bluff
            return _raise_to(state, 1.0)
        return {"action": "check"} if cc else {"action": "fold"}
    eq = _equity(state["your_cards"], state["community_cards"], n)
    req = owed / (pot + owed) if pot + owed > 0 else 1.0
    hu = n == 1
    if cc:
        if eq >= 0.75:
            return _raise_to(state, 0.8)                      # polarized value
        if hu and eq < 0.30 and random.random() < 0.33:
            return _raise_to(state, 0.8)                      # bluff
        return {"action": "check"}
    if eq >= 0.85:
        return _raise_to(state, 0.9)
    if hu and eq < 0.25 and random.random() < 0.15:
        return _raise_to(state, 0.9)                          # bluff-raise
    return {"action": "call"} if eq >= req else {"action": "fold"}   # bluff-catch at odds


def adapt(state):
    s = state["street"]; owed = state["amount_owed"]; pot = state["pot"]
    cc = state["can_check"]; n = _n_opp(state); late = _late(state)
    _, ff = _read(state, _opp_id(state))
    if s == "preflop":
        ch = _chen(state["your_cards"]); facing = state["current_bet"] > 100
        if ch >= 10:
            return _raise_to(state, 1.1)
        if not facing and ch >= 8 - 3 * late:
            return _raise_to(state, 1.0)
        if facing and ch >= 9 and owed <= 0.12 * state["your_stack"]:
            return {"action": "call"}
        return {"action": "check"} if cc else {"action": "fold"}
    eq = _equity(state["your_cards"], state["community_cards"], n)
    req = owed / (pot + owed) if pot + owed > 0 else 1.0
    hu = n == 1
    if cc:
        if eq >= 0.66:
            return _raise_to(state, 0.75)
        if hu and ff is not None and ff > 0.3 and eq < 0.40 and random.random() < min(0.6, ff):
            return _raise_to(state, 0.7)                      # exploit foldy opp
        return {"action": "check"}
    if eq >= 0.80:
        return _raise_to(state, 0.85)
    margin = 0.03 if (ff is not None and ff < 0.20) else 0.07  # call wider vs stations
    return {"action": "call"} if eq >= req + margin else {"action": "fold"}


BOTS = {"eqpro": eqpro, "tag": tag, "gto": gto, "adapt": adapt}
