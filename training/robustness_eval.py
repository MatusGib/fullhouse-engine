"""
Generalization / robustness check — does the bot hold up across opponent STYLES,
not just the 5 reference bots?

The real competition opponents are unknown bots by other people. To guard against
overfitting, we measure our bot heads-up (mirrored, so variance cancels) against a
spread of crude-but-distinct archetypes:

  nit      — very tight, folds most hands, only continues with strength
  lag      — loose-aggressive, raises/bluffs a lot
  station  — calls almost everything, never folds, rarely raises
  random   — uniformly random legal-ish action (chaotic control)

IMPORTANT: these exist only to *detect* weaknesses (getting crushed by a style is
a red flag). We do NOT tune the bot to beat them — that would just be overfitting
to our own archetypes. A robust bot beats every style by a clear margin.

Run:  python training/robustness_eval.py [hands] [--blueprint]
"""

import os
import sys
import importlib.util
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.game import PokerEngine, STARTING_STACK   # noqa: E402

BB = 100


def load_my_bot(enable_blueprint=False):
    spec = importlib.util.spec_from_file_location(
        "mybot", os.path.join(ROOT, "bots", "fullhouse_cfr", "bot.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if m.BLUEPRINT is not None:
        m.BP_ENABLED = enable_blueprint
    return m.decide


# ---------------------------------------------------------------------------
# Archetype opponents (crude, deliberately distinct styles)
# ---------------------------------------------------------------------------

def _ranks(state):
    return [c[0] for c in state["your_cards"]]


def _premium(state):
    r = _ranks(state)
    hi = "AKQJT"
    return r[0] == r[1] or (r[0] in hi and r[1] in hi) or "A" in r


def nit(state):
    owed, can_check, pot = state["amount_owed"], state["can_check"], state["pot"]
    if state["street"] == "preflop":
        if _premium(state):
            if can_check:
                return {"action": "raise", "amount": min(3 * BB, state["your_stack"])}
            return {"action": "call"} if owed <= 4 * BB else {"action": "fold"}
        return {"action": "check"} if can_check else {"action": "fold"}
    board = [c[0] for c in state["community_cards"]]
    made = _ranks(state)[0] == _ranks(state)[1] or any(x in board for x in _ranks(state))
    if can_check:
        return {"action": "raise", "amount": state["min_raise_to"]} if made else {"action": "check"}
    return {"action": "call"} if (made and owed <= 0.5 * pot) else {"action": "fold"}


def lag(state):
    can_check = state["can_check"]
    stack = state["your_stack"]
    roll = random.random()
    if can_check:
        if roll < 0.55:
            return {"action": "raise", "amount": min(state["min_raise_to"] * 2, stack)}
        return {"action": "check"}
    if roll < 0.4:
        return {"action": "raise", "amount": min(state["min_raise_to"] * 2, stack)}
    if roll < 0.8:
        return {"action": "call"}
    return {"action": "fold"}


def station(state):
    if state["can_check"]:
        return {"action": "check"}
    # calls almost anything but the most enormous bets
    return {"action": "call"} if state["amount_owed"] <= 3 * state["pot"] else {"action": "fold"}


def maniac(state):
    stack = state["your_stack"]
    if random.random() < 0.8:
        return {"action": "raise", "amount": min(state["min_raise_to"] * 3, stack)}
    return {"action": "check"} if state["can_check"] else {"action": "call"}


def rnd(state):
    opts = ["check"] if state["can_check"] else ["fold", "call"]
    if state["your_stack"] > state["amount_owed"]:
        opts.append("raise")
    pick = random.choice(opts)
    if pick == "raise":
        return {"action": "raise", "amount": state["min_raise_to"]}
    return {"action": pick}


ARCHETYPES = {"nit": nit, "lag": lag, "station": station, "maniac": maniac, "random": rnd}


# ---------------------------------------------------------------------------
# Mirrored heads-up evaluation
# ---------------------------------------------------------------------------

def hu_mirrored(decide_me, decide_opp, hands, seed):
    delta = 0
    for h in range(hands):
        for me_btn in (True, False):
            seats = {0: decide_me, 1: decide_opp} if me_btn else {0: decide_opp, 1: decide_me}
            me_seat = 0 if me_btn else 1
            eng = PokerEngine("r%d_%d" % (h, int(me_btn)), ["0", "1"],
                              dealer_seat=0, seed=seed + h)
            state = eng.start_hand()
            while state.get("type") == "action_request":
                seat = state["seat_to_act"]
                try:
                    action = seats[seat](state)
                except Exception:
                    action = {"action": "fold"}
                state = eng.apply_action(seat, action)
            delta += state["final_stacks"][str(me_seat)] - STARTING_STACK
    return delta / (2 * hands) / BB * 100      # bb/100


def main():
    hands = 2000
    use_bp = "--blueprint" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        hands = int(args[0])
    me = load_my_bot(enable_blueprint=use_bp)
    print("Robustness: our bot (%s) vs. archetypes, mirrored HU, %d hands each\n"
          % ("blueprint ON" if use_bp else "heuristic", hands))
    print("  opponent   bb/100")
    worst = 1e9
    for name, fn in ARCHETYPES.items():
        random.seed(2024)
        r = hu_mirrored(me, fn, hands, seed=4242)
        worst = min(worst, r)
        print("  %-9s  %+.2f" % (name, r))
    print("\n  worst-case style: %+.2f bb/100  -> %s"
          % (worst, "robust (beats every style)" if worst > 0 else "WEAK vs some style"))


if __name__ == "__main__":
    main()
