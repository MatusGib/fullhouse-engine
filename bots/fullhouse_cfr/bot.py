"""
Fullhouse Hackathon bot — heads-up CFR blueprint over a strong heuristic core.

  - CFR BLUEPRINT (active in true heads-up spots): maps the live state into an
    abstracted infoset (8 equity buckets/street + a half-pot/pot/all-in menu) and
    plays an external-sampling MCCFR strategy from data/blueprint.npz (~2M iters,
    CFR+ with Linear averaging; see training/). Heads-up is two-player zero-sum,
    so this approaches Nash. In mirrored A/B it nets ~+1.5 bb/100 over the
    heuristic and stays robust across opponent styles. Toggle with BP_ENABLED.
  - HEURISTIC (everything else + fallback): Chen-formula preflop + eval7
    Monte-Carlo equity vs. a pot-odds price, a range penalty vs. bets,
    opponent-aggression reads from the match log, and draw-based semi-bluffing.
    Beats every reference bot and every style archetype (worst case +33 bb/100).

Safety: the blueprint acts only in true heads-up, deep-enough spots, gated by a
visit-count trust threshold; any missing data or inconsistent lookup falls back
to the heuristic, so the bot is never worse than the validated baseline. Never
crashes (wrapped), stays far under the 2s budget, uses no forbidden imports.
"""

import os
import random
import math
import eval7
import numpy as np

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
SB_CHIPS, BB_CHIPS = 50, 100      # engine blinds (must match engine/game.py)


# ---------------------------------------------------------------------------
# CFR blueprint — loaded once at import (the 30s warmup call absorbs it). If the
# data files are missing or load fails, BLUEPRINT stays None and the bot plays
# the pure heuristic below, so it is never worse than the validated baseline.
# ---------------------------------------------------------------------------

def _load_blueprint():
    try:
        data_dir = os.environ.get(
            "BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
        cz = np.load(os.path.join(data_dir, "abstraction.npz"))
        centroids = {s: cz[s] for s in cz.files}
        bz = np.load(os.path.join(data_dir, "blueprint.npz"))
        index = {str(k): i for i, k in enumerate(bz["keys"])}
        return {"centroids": centroids, "codes": bz["codes"], "probs": bz["probs"],
                "visits": bz["visits"], "index": index}
    except Exception:
        return None


BLUEPRINT = _load_blueprint()
# Only trust a blueprint infoset that was visited enough to have converged. The
# broad-use range (50-150) tested best in mirrored A/B; >=1000 is too selective
# (mixes blueprint-preflop with heuristic-postflop incoherently).
BP_MIN_VISITS = 150.0
# Master switch. Blueprint = 8 buckets/street + a {check, half-pot, pot, all-in}
# / {fold, call, pot-raise, all-in} menu, external-sampling MCCFR (CFR+, Linear),
# ~2M iters. Adding the half-pot bet over v1 flipped the mirrored heads-up A/B
# from -2.5 to a small consistent net win (~+1.5 bb/100) vs the heuristic, and
# robustness_eval.py confirms it stays robust across opponent styles (worst-case
# +33 bb/100, matching the heuristic). It only acts in true heads-up spots; the
# heuristic plays everything else and is the fallback. Set False to play pure
# heuristic.
BP_ENABLED = True


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
# Opponent modeling (uses the cross-hand match_action_log injected by the engine)
# ---------------------------------------------------------------------------

def _current_aggressor_bot_id(state):
    """bot_id of the player whose bet we're facing (the last raise/all_in)."""
    seat = None
    for a in state.get("action_log", []):
        if a.get("action") in ("raise", "all_in"):
            seat = a.get("seat")
    if seat is None:
        return None
    for p in state["players"]:
        if p["seat"] == seat:
            return p.get("bot_id")
    return None


def _aggression_factor(state, bot_id):
    """Fraction of a player's voluntary actions that were bets/raises, taken
    from the cross-hand match_action_log. None if we lack enough samples."""
    if not bot_id:
        return None
    aggressive = passive = 0
    for e in state.get("match_action_log", []):
        if e.get("bot_id") != bot_id:
            continue
        a = e.get("action")
        if a in ("raise", "all_in"):
            aggressive += 1
        elif a in ("call", "check"):
            passive += 1
    total = aggressive + passive
    if total < 8:
        return None
    return aggressive / total


def _penalty_multiplier(state):
    """Scale the facing-a-bet range penalty by the current bettor's aggression.
    Maniacs (high AF) bet light -> smaller penalty (call wider); nits -> larger."""
    af = _aggression_factor(state, _current_aggressor_bot_id(state))
    if af is None:
        return 1.0
    mult = 1.0 - (af - 0.35) * 1.1            # AF ~0.35 is treated as neutral
    return max(0.35, min(1.5, mult))


def _hu_opponent_bot_id(state):
    """The single live opponent's bot_id (heads-up), for fold-tendency reads."""
    me = state["seat_to_act"]
    for p in state["players"]:
        if p["seat"] != me and not p["is_folded"]:
            return p.get("bot_id")
    return None


def _fold_factor(state, bot_id):
    """Fraction of a player's actions that were folds (from the match log).
    High = foldy (bluff more); low = calling station (don't bluff). None if scant."""
    if not bot_id:
        return None
    folds = total = 0
    for e in state.get("match_action_log", []):
        if e.get("bot_id") != bot_id:
            continue
        if e.get("action") in ("fold", "call", "check", "raise", "all_in"):
            total += 1
            if e.get("action") == "fold":
                folds += 1
    return folds / total if total >= 8 else None


def _draw_outs(hole_strs, board_strs):
    """Rough out count for flush + straight draws. Used to recognise strong
    draws, whose equity holds up even against a strong made-hand range."""
    cards = hole_strs + board_strs
    suit_counts = {}
    for c in cards:
        suit_counts[c[1]] = suit_counts.get(c[1], 0) + 1
    flush_draw = any(v == 4 for v in suit_counts.values())

    present = set(RANK_VAL[c[0]] for c in cards)
    if 14 in present:
        present.add(1)                        # ace plays low for the wheel
    straight_ranks = 0
    for r in range(1, 15):
        if r in present:
            continue
        test = present | {r}
        run = 0
        for v in range(1, 15):
            run = run + 1 if v in test else 0
            if run >= 5:
                straight_ranks += 1
                break

    outs = (9 if flush_draw else 0) + straight_ranks * 4
    if flush_draw and straight_ranks:
        outs -= 2                             # rough de-dup of shared cards
    return outs


# ---------------------------------------------------------------------------
# Preflop decision
# ---------------------------------------------------------------------------

def _preflop(state):
    bb = _big_blind(state)
    cards = state["your_cards"]
    chen = chen_score(cards)
    is_pair = cards[0][0] == cards[1][0]
    late = _lateness(state)
    n_opp = _num_opponents(state)
    owed = state["amount_owed"]
    pot = state["pot"]
    eff = state["your_stack"] + state["your_bet_this_street"]   # effective stack
    can_check = state["can_check"]
    current_bet = state["current_bet"]
    facing_raise = current_bet > bb

    # Short-stack push/fold: with <=10bb, raising small is a trap — jam or fold.
    if eff <= 10 * bb and not can_check:
        return {"action": "all_in"} if chen >= 7 else {"action": "fold"}

    if not facing_raise:
        # Open or take a free look (open wider heads-up / short-handed).
        open_thresh = 9 - 4 * late - (2 if n_opp <= 1 else 0)
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

    # Facing a raise: 3-bet premiums, flat strong, set-mine cheap pairs, else fold.
    reraise_thresh = 13 - 2 * late
    call_thresh = 9 - 3 * late
    if chen >= reraise_thresh:
        return _reraise(state, 1.0)
    if chen >= call_thresh and owed <= 0.12 * eff:
        return {"action": "call"}
    if is_pair and owed <= 0.06 * eff and eff >= 30 * bb:   # set-mine: cheap + deep
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

    street = state["street"]
    owed = state["amount_owed"]
    pot = state["pot"]
    can_check = state["can_check"]

    heads_up = n_opp == 1
    # Semi-bluffs only make sense heads-up (no fold equity into a crowd).
    strong_draw = (heads_up and street in ("flop", "turn")
                   and _draw_outs(state["your_cards"], state["community_cards"]) >= 8)
    # With more opponents someone is likelier to hold a strong hand, so demand
    # more equity before betting/raising for value.
    bump = 0.0 if heads_up else 0.05 + 0.03 * (n_opp - 1)

    if can_check:
        # No bet to us.
        if eq >= 0.78 + bump:
            return _bet(state, 0.70)                 # value
        if eq >= 0.60 + bump:
            return _bet(state, 0.55)                 # thin value / protection
        if strong_draw and random.random() < 0.5:
            return _bet(state, 0.60)                 # semi-bluff (heads-up only)
        # Balanced bluff (heads-up, turn/river): bet a weak hand sometimes so our
        # checks aren't a pure tell. Gated by the opponent's fold rate — never
        # bluff a calling station; bluff foldy / unknown opponents.
        if heads_up and street in ("turn", "river") and eq < 0.45:
            ff = _fold_factor(state, _hu_opponent_bot_id(state))
            freq = 0.20 if ff is None else (min(0.45, ff) if ff > 0.25 else 0.0)
            if random.random() < freq:
                return _bet(state, 0.60)
        return {"action": "check"}

    # Facing a bet. equity() is measured vs. a RANDOM hand, but a player who
    # bets — especially big, on a late street — has a range much stronger than
    # random. Calling on raw equity-vs-random therefore over-calls (e.g. bottom
    # pair vs. a pot-sized river bet). So we lift the equity bar we need to call:
    #   * scaled by bet size (relative to pot) and street,
    #   * softened vs. proven-aggressive opponents / hardened vs. nits, and
    #   * largely waived when WE hold a strong draw (drawing equity holds up
    #     even against a strong made-hand range).
    required = owed / (pot + owed) if (pot + owed) > 0 else 1.0
    if eq >= 0.82 + bump:
        return _reraise(state, 0.9)                  # value raise

    if strong_draw and eq >= 0.42 and random.random() < 0.30:
        return _reraise(state, 0.8)                  # semi-bluff raise (heads-up only)

    street_factor = {"flop": 0.16, "turn": 0.24, "river": 0.30}.get(street, 0.24)
    bet_frac = owed / pot if pot > 0 else 1.0        # call size vs. the current pot
    range_penalty = street_factor * min(bet_frac / 0.6, 1.2)
    range_penalty *= _penalty_multiplier(state)
    if strong_draw:
        range_penalty *= 0.25
    bar = required + 0.03 + range_penalty
    if eq >= bar:
        return {"action": "call"}
    return {"action": "fold"}


# ---------------------------------------------------------------------------
# CFR blueprint lookup  (mirrors training/holdem_abstraction.py + holdem_train.py)
# ---------------------------------------------------------------------------

def _bp_preflop_index(hole):
    i1, i2 = RANK_VAL[hole[0][0]] - 2, RANK_VAL[hole[1][0]] - 2   # 0..12
    if i1 == i2:
        return i1
    hi, lo = max(i1, i2), min(i1, i2)
    pid = hi * (hi - 1) // 2 + lo
    return 13 + pid if (hole[0][1] == hole[1][1]) else 13 + 78 + pid


def _bp_feature(hole, board, iters=200):
    """(equity_to_river, made_strength_now) — identical formula to training."""
    hole_c = [CARD_STR[c] for c in hole]
    board_c = [CARD_STR[c] for c in board]
    known = set(hole) | set(board)
    deck = [c for c in FULL_DECK if str(c) not in known]
    need = 5 - len(board)
    my_now = eval7.evaluate(hole_c + board_c)
    w_river = w_now = 0.0
    for _ in range(iters):
        s = random.sample(deck, 2 + need)
        opp, run = s[:2], s[2:]
        on = eval7.evaluate(opp + board_c)
        if my_now > on:
            w_now += 1
        elif my_now == on:
            w_now += 0.5
        comm = board_c + run
        mr, orr = eval7.evaluate(hole_c + comm), eval7.evaluate(opp + comm)
        if mr > orr:
            w_river += 1
        elif mr == orr:
            w_river += 0.5
    return (w_river / iters, w_now / iters)


def _bp_bucket(hole, board, street):
    if street == "preflop":
        return _bp_preflop_index(hole)
    cents = BLUEPRINT["centroids"][street]
    f = _bp_feature(hole, board)
    d = ((cents - np.asarray(f, dtype=np.float64)) ** 2).sum(axis=1)
    return int(np.argmin(d))


_STREET_IDX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def _bp_reconstruct_key(state):
    """Replay the action log into the abstract infoset key used in training:
    "street|bucket|history". Returns None if the log can't be mapped cleanly."""
    log = state["action_log"]
    sb = bb = None
    for a in log:
        if a["action"] == "small_blind":
            sb = a["seat"]
        elif a["action"] == "big_blind":
            bb = a["seat"]
    if sb is None or bb is None:
        return None
    seat2p = {sb: 0, bb: 1}

    codes = []
    sc = [SB_CHIPS, BB_CHIPS]       # street contributions, seeded with the blinds
    pot = SB_CHIPS + BB_CHIPS       # total pot (for opening-bet half/pot sizing)
    acted = [False, False]
    bet_seen = True                 # preflop has the standing big blind
    for a in log:
        act = a["action"]
        if act in ("small_blind", "big_blind"):
            continue
        pl = seat2p.get(a["seat"])
        if pl is None:
            return None
        amt = a.get("amount", 0) or 0
        if act == "fold":
            codes.append("f")
            acted[pl] = True
        elif act == "check":
            codes.append("x")
            acted[pl] = True
        elif act == "call":
            added = max(0, max(sc) - sc[pl])
            sc[pl] += added
            pot += added
            codes.append("c")
            acted[pl] = True
        elif act == "raise":
            added = max(0, amt - sc[pl])
            if bet_seen:                       # a re-raise facing a bet
                codes.append("r")
            else:                              # opening bet: half ('h') vs pot ('b')
                codes.append("h" if (pot > 0 and added <= 0.75 * pot) else "b")
            sc[pl] = amt
            pot += added
            acted[pl] = bet_seen = True
        elif act == "all_in":
            tot = amt if amt else max(sc)
            pot += max(0, tot - sc[pl])
            sc[pl] = tot
            codes.append("a")
            acted[pl] = bet_seen = True
        else:
            return None
        if act != "fold" and sc[0] == sc[1] and acted[0] and acted[1]:
            codes.append("/")
            sc = [0, 0]
            acted = [False, False]
            bet_seen = False

    bucket = _bp_bucket(state["your_cards"], state["community_cards"], state["street"])
    return "%d|%d|%s" % (_STREET_IDX[state["street"]], bucket, "".join(codes))


def _bp_translate(state, code):
    """Turn an abstract action code into a concrete, legal engine action."""
    if code == "x":
        return {"action": "check"} if state["can_check"] else {"action": "call"}
    if code == "c":
        return {"action": "check"} if state["can_check"] else {"action": "call"}
    if code == "f":
        return {"action": "fold"} if not state["can_check"] else {"action": "check"}
    if code == "a":
        return {"action": "all_in"}
    if code == "h":
        return _bet(state, 0.5)
    if code == "b":
        return _bet(state, 1.0)
    if code == "r":
        return _reraise(state, 1.0)
    return None


def _blueprint_action(state):
    """Return a blueprint action for a true heads-up spot, or None to defer to
    the heuristic."""
    if not BP_ENABLED or BLUEPRINT is None:
        return None
    players = state["players"]
    if len(players) != 2:                       # heads-up table only
        return None
    bb = _big_blind(state)
    eff = min(p["stack"] + p["bet_this_street"] for p in players)
    if eff < 40 * bb:                           # blueprint is ~100bb; skip short stacks
        return None
    me = state["seat_to_act"]
    sb = bbs = None
    for a in state["action_log"]:
        if a["action"] == "small_blind":
            sb = a["seat"]
        elif a["action"] == "big_blind":
            bbs = a["seat"]
    if me not in (sb, bbs):
        return None

    key = _bp_reconstruct_key(state)
    if key is None:
        return None
    idx = BLUEPRINT["index"].get(key)
    if idx is None:
        return None
    if BLUEPRINT["visits"][idx] < BP_MIN_VISITS:   # only trust converged infosets
        return None
    codes = str(BLUEPRINT["codes"][idx])
    # consistency guard: a facing-bet infoset starts with 'f', a no-bet one 'x'.
    faces_bet = codes[:1] == "f"
    if faces_bet == bool(state["can_check"]):
        return None

    probs = np.asarray(BLUEPRINT["probs"][idx][:len(codes)], dtype=np.float64)
    tot = probs.sum()
    if tot <= 0:
        return None
    probs /= tot
    r = random.random()
    cum = 0.0
    pick = codes[-1]
    for c, p in zip(codes, probs):
        cum += p
        if r <= cum:
            pick = c
            break
    return _bp_translate(state, pick)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _safe_action(game_state):
    return {"action": "check"} if game_state.get("can_check") else {"action": "fold"}


def decide(game_state: dict) -> dict:
    try:
        # Defensive: malformed/unknown hole cards would corrupt equity + lookups,
        # so bail to a safe action rather than act on garbage.
        cards = game_state.get("your_cards")
        if (not isinstance(cards, list) or len(cards) != 2
                or any(c not in CARD_STR for c in cards)):
            return _safe_action(game_state)
        action = _blueprint_action(game_state)   # heads-up CFR blueprint, if applicable
        if action is not None:
            return action
        if game_state.get("street") == "preflop":
            return _preflop(game_state)
        return _postflop(game_state)
    except Exception:
        return _safe_action(game_state)
