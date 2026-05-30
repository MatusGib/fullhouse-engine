"""
Round-robin tournament: our bot vs. a field of strong AI-style competitors.

Two formats, both with the cross-hand match log injected (so adaptive bots + our
opponent reads work):
  * heads-up, mirrored (each deal played both ways -> variance cancels): a clean
    bb/100 matrix of who beats whom.
  * 6-max bake-off with persistent stacks (the real qualifier format): total chip
    delta, rotating seats across matches to balance position.

Our bot is loaded with the BLUEPRINT OFF (the heuristic workhorse we're hardening).

Run:  python training/tournament.py [hu_hands] [sixmax_matches]
"""

import os
import sys
import importlib.util
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.game import PokerEngine, STARTING_STACK   # noqa: E402
import competitors as C                                # noqa: E402

BB = 100


def load_ours(modname, bp_on):
    spec = importlib.util.spec_from_file_location(
        modname, os.path.join(ROOT, "bots", "fullhouse_cfr", "bot.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if m.BLUEPRINT is not None:
        m.BP_ENABLED = bp_on
    return m.decide


SPOT_LOG = None     # set to a dict by --spots mode; accumulates leak stats


def _chips_added(state, action):
    """Chips our bot voluntarily puts in with this action, from the pre-action
    state. Used to weight where in the hand we actually committed money."""
    a = action.get("action")
    owed = state.get("amount_owed", 0) or 0
    stack = state.get("your_stack", 0) or 0
    bts = state.get("your_bet_this_street", 0) or 0
    if a == "call":
        return min(owed, stack)
    if a == "all_in":
        return stack
    if a == "raise":
        amt = action.get("amount") or 0
        return max(0, min(amt - bts, stack))
    return 0                                   # check / fold add nothing


def _spot_context(state):
    """A readable 'street position facing' tag for leak attribution."""
    street = state["street"]
    n = len(state["players"])
    me = state["seat_to_act"]
    sb = None
    for a in state.get("action_log", []):
        if a.get("action") == "small_blind":
            sb = a["seat"]
    if sb is None:
        button = None
    elif n == 2:
        button = sb                       # heads-up: SB is the button
    else:
        button = (sb - 1) % n
    if street == "preflop":
        pos = "btn/SB" if me == button else "BB"
    else:
        pos = "IP" if me == button else "OOP"
    facing = "check" if state.get("can_check") else "vsbet"
    return "%-7s %-6s %-5s" % (street, pos, facing)


def _run_hand(eng, seat_to_decide, ids, mlog, hand_num, track_name=None):
    state = eng.start_hand()
    state["match_action_log"] = mlog[-200:]
    steps = 0
    hand_spots = []     # (tag, chips_added) for our voluntary decisions this hand
    while state.get("type") == "action_request":
        seat = state["seat_to_act"]
        tracking = (track_name is not None and SPOT_LOG is not None
                    and ids[seat] == track_name)
        ctx = _spot_context(state) if tracking else None
        try:
            action = seat_to_decide[seat](state)
            if not isinstance(action, dict) or "action" not in action:
                action = {"action": "fold"}
        except Exception:
            action = {"action": "fold"}
        if tracking:
            tag = ctx + " " + str(action.get("action"))
            hand_spots.append((tag, _chips_added(state, action)))
        mlog.append({"hand_num": hand_num, "seat": seat, "bot_id": ids[seat],
                     "action": action.get("action"), "amount": action.get("amount")})
        state = eng.apply_action(seat, action)
        if state.get("type") == "action_request":
            state["match_action_log"] = mlog[-200:]
        steps += 1
        if steps > 1000:
            break
    if track_name is not None and SPOT_LOG is not None and hand_spots:
        net = state["final_stacks"].get(track_name, STARTING_STACK) - STARTING_STACK
        # Attribute the hand's net to spots in proportion to chips committed there,
        # so the result localises to WHERE we put money — not smeared over the
        # cheap preflop open. Pure check/fold hands (no investment) split evenly.
        total_inv = sum(c for _, c in hand_spots)
        for tag, c in hand_spots:
            w = (c / total_inv) if total_inv > 0 else (1.0 / len(hand_spots))
            e = SPOT_LOG.setdefault(tag, [0.0, 0])
            e[0] += net * w
            e[1] += 1
    return state


def play_hu(name_a, dec_a, name_b, dec_b, hands, seed):
    """Mirrored heads-up. Returns A's bb/100."""
    delta = 0
    mlog = []
    for h in range(hands):
        for a_btn in (True, False):
            ids = [name_a, name_b] if a_btn else [name_b, name_a]
            decide = {0: dec_a, 1: dec_b} if a_btn else {0: dec_b, 1: dec_a}
            a_seat = 0 if a_btn else 1
            eng = PokerEngine("g", ids, dealer_seat=0, seed=seed + h)
            final = _run_hand(eng, decide, ids, mlog, h)["final_stacks"]
            delta += final[ids[a_seat]] - STARTING_STACK
    return delta / (2 * hands) / BB * 100


def play_6max(field, hands, seed):
    """One persistent-stack match among `field` {name: decide}. Returns chip deltas."""
    names = list(field)
    stacks = {n: STARTING_STACK for n in names}
    mlog = []
    dealer = 0
    for h in range(hands):
        alive = [n for n in names if stacks[n] > 0]
        if len(alive) < 2:
            break
        eng = PokerEngine("h%d" % h, alive, dealer_seat=dealer % len(alive),
                          starting_stacks={n: stacks[n] for n in alive}, seed=seed + h)
        decide = {i: field[alive[i]] for i in range(len(alive))}
        final = _run_hand(eng, decide, alive, mlog, h)["final_stacks"]
        for n, s in final.items():
            stacks[n] = s
        dealer += 1
    return {n: stacks[n] - STARTING_STACK for n in names}


def play_ring(field, hands, seed):
    """Cash ring: every hand starts fresh at 100bb (no busting -> low variance,
    measures true per-hand EV). Match log accumulates across hands for adaptation.
    Returns each bot's bb/100."""
    names = list(field)
    n = len(names)
    mlog = []
    deltas = {x: 0 for x in names}
    for h in range(hands):
        eng = PokerEngine("h%d" % h, names, dealer_seat=h % n, seed=seed + h)
        decide = {i: field[names[i]] for i in range(n)}
        final = _run_hand(eng, decide, names, mlog, h)["final_stacks"]
        for x in names:
            deltas[x] += final[x] - STARTING_STACK
    return {x: deltas[x] / hands / BB * 100 for x in names}


def diagnose(opp_name, hands, seed):
    """Heads-up vs one opponent with per-spot chip attribution. Prints the
    spots where our bot bleeds the most (most-negative net first).

    Each hand's net chip result is split across the spots we acted in IN
    PROPORTION to the chips we committed there, so a row's tot_chips is the share
    of our win/loss attributable to that spot. This localises leaks to where money
    actually went (a cheap preflop open barely registers; a big river call carries
    its full weight). Rows now sum to ~the overall total. 'n' is how many times we
    were in that spot. NOTE: chip-flow can't see opportunity cost — over-folding a
    cheap spot looks fine here because folding only forfeits chips already in.
    """
    global SPOT_LOG
    ours = load_ours("ours_diag_%s" % opp_name, False)
    opp = C.BOTS[opp_name]
    SPOT_LOG = {}
    mlog = []
    net_total = 0
    for h in range(hands):
        for a_btn in (True, False):
            ids = ["ours", opp_name] if a_btn else [opp_name, "ours"]
            decide = {0: ours, 1: opp} if a_btn else {0: opp, 1: ours}
            eng = PokerEngine("g", ids, dealer_seat=0, seed=seed + h)
            st = _run_hand(eng, decide, ids, mlog, h, track_name="ours")
            net_total += st["final_stacks"]["ours"] - STARTING_STACK
    overall = net_total / (2 * hands) / BB * 100
    print("\n=== leak diagnosis: ours vs %s  (%d mirrored hands, overall %+.2f bb/100) ==="
          % (opp_name, hands, overall))
    print("  %-27s %5s %10s %9s" % ("street  pos    facing action", "n", "tot_chips", "bb/100"))
    rows = sorted(SPOT_LOG.items(), key=lambda kv: kv[1][0])   # most negative first
    for tag, (tot, n) in rows:
        bb100 = tot / n / BB * 100 if n else 0
        print("  %-27s %5d %10d %+9.2f" % (tag, n, int(round(tot)), bb100))
    SPOT_LOG = None


def main():
    raw = sys.argv[1:]
    spots = "--spots" in raw
    raw = [a for a in raw if a != "--spots"]
    hu_hands = int(raw[0]) if len(raw) > 0 else 500
    sixmax_matches = int(raw[1]) if len(raw) > 1 else 6

    if spots:
        # Focused leak hunt vs the elite bots we lose to. Reproducible RNG so the
        # numbers are stable across runs (judge by rank, not the noise floor).
        random.seed(424242)
        for opp in ("eqpro", "adapt", "tag"):
            diagnose(opp, hu_hands, seed=1000)
        return

    random.seed(424242)        # reproducible: bots' MC/bluff RNG fixed across runs
    field = {"ours": load_ours("ours_off", False), **C.BOTS}
    names = list(field)

    # ---- heads-up round-robin (mirrored) ----
    print("Heads-up round-robin — bb/100 (row vs col), %d mirrored hands\n" % hu_hands)
    hu = {a: {} for a in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = play_hu(a, field[a], b, field[b], hu_hands, seed=1000)
            hu[a][b] = r
            hu[b][a] = -r
    hdr = "        " + "".join("%8s" % n for n in names)
    print(hdr)
    for a in names:
        row = "%-7s " % a + "".join(("%+8.1f" % hu[a][b]) if b in hu[a] else "%8s" % "-"
                                    for b in names)
        print(row)
    print("\nmean HU bb/100:")
    for a in names:
        vals = [hu[a][b] for b in names if b in hu[a]]
        print("  %-7s %+.1f" % (a, sum(vals) / len(vals)))

    if sixmax_matches <= 0:
        return
    # ---- 6-max ring (fresh stacks each hand -> low-variance per-hand EV) ----
    ring_hands = sixmax_matches * 1000
    print("\n6-max ring (cash, fresh 100bb each hand) — bb/100 over %d hands" % ring_hands)
    res = play_ring(field, ring_hands, seed=9000)
    for n in sorted(res, key=lambda x: -res[x]):
        print("  %-7s %+.2f" % (n, res[n]))


if __name__ == "__main__":
    main()
