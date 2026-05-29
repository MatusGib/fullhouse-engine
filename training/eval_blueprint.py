"""
A/B evaluation of the heads-up CFR blueprint.

Loads bots/fullhouse_cfr/bot.py twice — once normally (blueprint ON) and once
with BLUEPRINT forced to None (heuristic ONLY) — and plays them heads-up,
in-process, through the real engine. The chip delta isolates exactly what the
blueprint adds on top of the heuristic. Also prints the learned preflop opening
ranges so you can eyeball them against poker sense (AA/AK raise, 72o folds).

Run:  python training/eval_blueprint.py [hands]
"""

import os
import sys
import importlib.util
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.game import PokerEngine, STARTING_STACK   # noqa: E402

BOT_PATH = os.path.join(ROOT, "bots", "fullhouse_cfr", "bot.py")


def load_bot(name, disable_blueprint=False):
    spec = importlib.util.spec_from_file_location(name, BOT_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if disable_blueprint:
        m.BLUEPRINT = None
    return m


def show_preflop_range(bot):
    if bot.BLUEPRINT is None:
        print("  (no blueprint loaded)")
        return
    idx = bot.BLUEPRINT["index"]
    hands = [("AA", ["As", "Ad"]), ("KK", ["Ks", "Kd"]), ("AKs", ["As", "Ks"]),
             ("AKo", ["As", "Kd"]), ("QQ", ["Qs", "Qd"]), ("JTs", ["Js", "Ts"]),
             ("T9o", ["Ts", "9d"]), ("A5s", ["As", "5s"]), ("72o", ["7s", "2d"])]
    print("  hand   key            strategy (SB first to act)")
    for name, h in hands:
        key = "0|%d|" % bot._bp_preflop_index(h)
        i = idx.get(key)
        if i is None:
            print("  %-5s  %-13s  unvisited" % (name, key))
            continue
        codes = str(bot.BLUEPRINT["codes"][i])
        probs = bot.BLUEPRINT["probs"][i][:len(codes)]
        strat = {c: round(float(p), 2) for c, p in zip(codes, probs)}
        print("  %-5s  %-13s  %s" % (name, key, strat))


def play(bot_a, bot_b, hands, seed):
    """Mirrored heads-up: each deal is played twice with the button swapped, so
    card + position variance cancels (identical bots -> ~0). Returns bot_a's
    total chip delta over the 2*hands games."""
    delta_a = 0
    for h in range(hands):
        for a_button in (True, False):
            seats = {0: bot_a, 1: bot_b} if a_button else {0: bot_b, 1: bot_a}
            a_seat = 0 if a_button else 1
            eng = PokerEngine("h%d_%d" % (h, int(a_button)), ["0", "1"],
                              dealer_seat=0, seed=seed + h)
            state = eng.start_hand()
            while state.get("type") == "action_request":
                seat = state["seat_to_act"]
                state = eng.apply_action(seat, seats[seat].decide(state))
            delta_a += state["final_stacks"][str(a_seat)] - STARTING_STACK
    return delta_a


def main():
    hands = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    blueprint_bot = load_bot("bot_bp", disable_blueprint=False)
    heuristic_bot = load_bot("bot_heur", disable_blueprint=True)

    loaded = blueprint_bot.BLUEPRINT is not None
    n_info = len(blueprint_bot.BLUEPRINT["index"]) if loaded else 0
    print("blueprint loaded: %s  (%d infosets)\n" % (loaded, n_info))

    print("Learned preflop open ranges:")
    show_preflop_range(blueprint_bot)

    # Sweep the runtime trust threshold. Same hands + reseeded RNG each time, so
    # differences are purely the threshold. thr=inf disables the blueprint and
    # must read ~0 (sanity check that the A/B harness is unbiased).
    print("\nHead-to-head vs. heuristic-only over %d HU hands, by trust threshold:" % hands)
    print("  BP_MIN_VISITS     chip delta     bb/100")
    for thr in (50, 150, 400, 1000, 1e9):
        blueprint_bot.BP_MIN_VISITS = thr
        random.seed(123)
        delta = play(blueprint_bot, heuristic_bot, hands, seed=777)
        tag = "inf (heuristic)" if thr > 1e8 else str(int(thr))
        print("  %-15s   %+8d     %+.2f" % (tag, delta, delta / (2 * hands)))


if __name__ == "__main__":
    main()
