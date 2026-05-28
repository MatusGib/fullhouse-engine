"""
Fullhouse Hackathon bot — PHASE 1 BASELINE (heuristic / equity).

Strategy in one breath:
  - Preflop: Chen-formula hand scoring + position + facing-a-raise logic.
  - Postflop: Monte-Carlo equity (eval7) vs. live opponents, compared to the
    pot-odds price. Value-bet when ahead, call when priced in, fold otherwise.

This is the SAFE qualifier bot: it never crashes (wrapped in try/except),
stays far under the 2s budget (bounded MC), and uses no forbidden imports.
The CFR blueprint comes later and will slot in behind this same file, using
the equity logic here as its off-tree fallback.
"""

import random
import math
import eval7

BOT_NAME = "FullHouseCFR"
BOT_AVATAR = "robot_1"

# ---------------------------------------------------------------------------
# Module-level precompute (runs once at import; the warmup call absorbs it)
# ---------------------------------------------------------------------------

_RANKS = "23456789TJQKA"
_SUITS = "shdc"
FULL_DECK = [eval7.Card(r + s) for r in _RANKS for s in _SUITS]
CARD_STR = {str(c): c for c in FULL_DECK}     # "As" -> eval7.Card

RANK_VAL = {r: i + 2 for i, r in enumerate(_RANKS)}   # "2"->2 ... "A"->14

DEFAULT_BB = 100


# ---------------------------------------------------------------------------
# Preflop: Chen formula  (a well-known starting-hand score; AA=20, 72o≈-1)
# ---------------------------------------------------------------------------

def chen_score(cards):
    r1, r2 = cards[0][0], cards[1][0]
    s1, s2 = cards[0][1], cards[1][1]
    v1, v2 = RANK_VAL[r1], RANK_VAL[r2]
    hi, lo = max(v1, v2), min(v1, v2)

    # Base value from the higher card.
    base = {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}.get(hi, hi / 2.0)

    if r1 == r2:                       # pocket pair
        return max(base * 2.0, 5.0)

    score = base
    if s1 == s2:                       # suited
        score += 2.0

    gap = hi - lo - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)

    # 0/1-gap connectors below Q get a small straight bonus.
    if gap <= 1 and hi <= 11:
        score += 1.0

    return math.floor(score + 0.5)     # round half up


# ---------------------------------------------------------------------------
# Postflop: Monte-Carlo equity vs. N random opponent hands
# ---------------------------------------------------------------------------

def equity(hole_strs, board_strs, n_opp, iters):
    if n_opp <= 0:
        return 1.0
    hole = [CARD_STR[c] for c in hole_strs]
    board = [CARD_STR[c] for c in board_strs]
    known = set(hole_strs) | set(board_strs)
    deck = [c for c in FULL_DECK if str(c) not in known]

    need_board = 5 - len(board)
    draw_n = n_opp * 2 + need_board
    if draw_n > len(deck):
        return 0.5  # pathological; bail safe

    wins = ties = 0.0
    for _ in range(iters):
        s = random.sample(deck, draw_n)
        comm = board + s[n_opp * 2:]
        my = eval7.evaluate(hole + comm)
        best_opp = 0
        for i in range(n_opp):
            o = eval7.evaluate(s[2 * i:2 * i + 2] + comm)
            if o > best_opp:
                best_opp = o
        if my > best_opp:
            wins += 1
        elif my == best_opp:
            ties += 1
    return (wins + ties / 2.0) / iters


# ---------------------------------------------------------------------------
# Sizing helpers — engine wants the TOTAL bet ("raise to"), not raise-by.
# ---------------------------------------------------------------------------

def _clamp_to(state, target_total):
    """Turn a desired total bet into a legal raise/all_in action."""
    max_total = state["your_stack"] + state["your_bet_this_street"]
    target_total = int(round(target_total))
    if target_total >= max_total:
        return {"action": "all_in"}
    target_total = max(target_total, state["min_raise_to"])
    if target_total >= max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": target_total}


def _bet(state, frac):
    """Open/c-bet a fraction of the pot when it's checked to us."""
    return _clamp_to(state, frac * state["pot"])


def _reraise(state, frac):
    """Raise a fraction of the (pot + call) on top of the current bet."""
    return _clamp_to(state, state["current_bet"] + frac * (state["pot"] + state["amount_owed"]))


# ---------------------------------------------------------------------------
# Table reads
# ---------------------------------------------------------------------------

def _big_blind(state):
    for a in state.get("action_log", []):
        if a.get("action") == "big_blind" and a.get("amount"):
            return a["amount"]
    return DEFAULT_BB


def _num_opponents(state):
    me = state["seat_to_act"]
    return sum(1 for p in state["players"]
               if not p["is_folded"] and p["seat"] != me)


def _lateness(state):
    """0.0 = earliest to act, 1.0 = button. Derived from logged blind seats."""
    players = state["players"]
    n = len(players)
    me = state["seat_to_act"]
    sb = None
    for a in state.get("action_log", []):
        if a.get("action") == "small_blind":
            sb = a["seat"]
    if sb is None:
        return 0.5
    button = sb if n == 2 else (sb - 1) % n
    active = [p["seat"] for p in players if not p["is_folded"]]
    if me not in active or len(active) <= 1:
        return 0.5
    order = sorted(active, key=lambda s: ((s - button - 1) % n))  # first..button
    return order.index(me) / (len(order) - 1)


# ---------------------------------------------------------------------------
# Preflop decision
# ---------------------------------------------------------------------------

def _preflop(state):
    bb = _big_blind(state)
    chen = chen_score(state["your_cards"])
    late = _lateness(state)
    owed = state["amount_owed"]
    pot = state["pot"]
    stack = state["your_stack"]
    can_check = state["can_check"]
    current_bet = state["current_bet"]
    facing_raise = current_bet > bb

    # Short-stack push/fold: with <=10bb, raising small is a trap — jam or fold.
    if stack <= 10 * bb and not can_check:
        return {"action": "all_in"} if chen >= 7 else {"action": "fold"}

    if not facing_raise:
        # Open or take a free look.
        open_thresh = 9 - 4 * late          # ~9 UTG, ~5 on the button
        if chen >= open_thresh:
            target = 3 * bb
            if pot > 2 * bb:                # limpers in front -> size up
                target += pot - 2 * bb
            return _clamp_to(state, target)
        if can_check:
            return {"action": "check"}
        if owed <= bb and chen >= 7 - 3 * late:   # cheap limp with playable hands
            return {"action": "call"}
        return {"action": "fold"}

    # Facing a raise: 3-bet premiums, flat strong, fold the rest.
    reraise_thresh = 13 - 2 * late
    call_thresh = 9 - 3 * late
    if chen >= reraise_thresh:
        return _reraise(state, 1.0)
    if chen >= call_thresh and owed <= 0.12 * (stack + state["your_bet_this_street"]):
        return {"action": "call"}
    if can_check:
        return {"action": "check"}
    return {"action": "fold"}


# ---------------------------------------------------------------------------
# Postflop decision
# ---------------------------------------------------------------------------

def _postflop(state):
    n_opp = _num_opponents(state)
    iters = 250 if n_opp <= 2 else 140
    eq = equity(state["your_cards"], state["community_cards"], n_opp, iters)

    owed = state["amount_owed"]
    pot = state["pot"]
    can_check = state["can_check"]

    if can_check:
        # No bet to us: value-bet when ahead, otherwise take the free card.
        if eq >= 0.80:
            return _bet(state, 0.75)
        if eq >= 0.62:
            return _bet(state, 0.55)
        return {"action": "check"}

    # Facing a bet. equity() is measured vs. a RANDOM hand, but a player who
    # bets — especially big, on a late street — has a range much stronger than
    # random. Calling on raw equity-vs-random therefore over-calls (e.g. bottom
    # pair vs. a pot-sized river bet). So we lift the equity bar we need to
    # call, scaled by the bet size (relative to pot) and the street (ranges get
    # more defined street by street, and the river has no cards left to draw).
    required = owed / (pot + owed) if (pot + owed) > 0 else 1.0
    if eq >= 0.82:
        return _reraise(state, 0.9)                 # strong enough to raise for value
    street_factor = {"flop": 0.16, "turn": 0.24, "river": 0.30}.get(state["street"], 0.24)
    bet_frac = owed / pot if pot > 0 else 1.0        # call size vs. the current pot
    range_penalty = street_factor * min(bet_frac / 0.6, 1.2)
    bar = required + 0.03 + range_penalty
    if eq >= bar:
        return {"action": "call"}
    return {"action": "fold"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def decide(game_state: dict) -> dict:
    try:
        if game_state.get("street") == "preflop":
            return _preflop(game_state)
        return _postflop(game_state)
    except Exception:
        # Never crash: take a free check if we can, else fold.
        if game_state.get("can_check"):
            return {"action": "check"}
        return {"action": "fold"}
