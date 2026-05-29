"""
Heads-up No-Limit Hold'em blueprint via external-sampling MCCFR.

This scales the CFR machinery proven on Kuhn/Leduc to real Hold'em by adding the
two things that make Hold'em tractable (see holdem_abstraction.py for the first):

  1. CARD ABSTRACTION  — 169 preflop hands + 8 equity buckets/street (imported).
  2. ACTION ABSTRACTION — a tiny bet menu so the betting tree stays finite:
        facing no bet : check / bet 0.75-pot / all-in
        facing a bet  : fold / call / pot-raise / all-in   (raises capped)

Heads-up only: it is a two-player zero-sum game, so CFR provably converges to a
Nash equilibrium. Stacks are equal (100 bb) every hand, which means all-ins
always resolve to equal contributions — no side pots to model.

We solve with external-sampling MCCFR (sample chance + the opponent's actions,
enumerate the traverser's), CFR+ regret flooring. The average strategy is
exported to bots/fullhouse_cfr/data/blueprint.npz as pickle-free arrays.

Run:  python training/holdem_train.py [iterations]
"""

import os
import sys
import random
import numpy as np
import eval7

import holdem_abstraction as A

SB, BB, STACK = 50, 100, 10_000
RAISE_CAP = 4                       # max aggressive actions per street
BUCKET_ITERS = 40                   # MC rollouts for bucketing during training
BLUEPRINT_PATH = os.path.join(A.DATA_DIR, "blueprint.npz")
MAX_ACTIONS = 4        # max legal actions at a node: x,h,b,a (no-bet) / f,c,r,a


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

def new_state(holes, board, buckets):
    return {
        "holes": holes, "board": board, "buckets": buckets,
        "street": 0,
        "contrib": [SB, BB],
        "street_contrib": [SB, BB],
        "stacks": [STACK - SB, STACK - BB],
        "current_bet": BB,
        "to_act": 0,                 # SB (button) acts first preflop
        "acted": [False, False],
        "num_raises": 0,
        "history": [],
        "terminal": False, "folded": False, "winner": None,
    }


def legal_actions(s):
    """Return list of (code, chips_added)."""
    p = s["to_act"]
    pot = s["contrib"][0] + s["contrib"][1]
    to_call = s["current_bet"] - s["street_contrib"][p]
    stack = s["stacks"][p]
    acts = []
    if to_call <= 0:
        acts.append(("x", 0))
        if stack > 0:
            half = min(max(BB, int(0.5 * pot)), stack)   # 'h' = half-pot bet
            full = min(max(BB, int(1.0 * pot)), stack)    # 'b' = pot bet
            if 0 < half < stack:
                acts.append(("h", half))
            if half < full < stack:
                acts.append(("b", full))
            acts.append(("a", stack))
    else:
        acts.append(("f", 0))
        call = min(to_call, stack)
        acts.append(("c", call))
        if stack > call:
            if s["num_raises"] < RAISE_CAP:
                radd = min(to_call + max(BB, pot + to_call), stack)
                if call < radd < stack:
                    acts.append(("r", radd))
            acts.append(("a", stack))
    return acts


def apply_action(s, code, amount):
    n = dict(s)
    n["contrib"] = s["contrib"][:]
    n["street_contrib"] = s["street_contrib"][:]
    n["stacks"] = s["stacks"][:]
    n["acted"] = s["acted"][:]
    n["history"] = s["history"] + [code]
    p = s["to_act"]
    opp = 1 - p

    if code == "f":
        n["terminal"] = True
        n["folded"] = True
        n["winner"] = opp
        return n

    n["contrib"][p] += amount
    n["street_contrib"][p] += amount
    n["stacks"][p] -= amount
    n["acted"][p] = True
    prev_bet = n["current_bet"]
    n["current_bet"] = max(n["street_contrib"])
    if n["street_contrib"][p] > prev_bet:
        n["num_raises"] = s["num_raises"] + 1

    matched = n["street_contrib"][0] == n["street_contrib"][1]
    both_acted = n["acted"][0] and n["acted"][1]
    either_all_in = n["stacks"][0] == 0 or n["stacks"][1] == 0

    if matched and both_acted:
        if s["street"] == 3 or either_all_in:
            return _showdown(n)
        return _advance_street(n)
    n["to_act"] = opp
    return n


def _advance_street(n):
    n["street"] += 1
    n["street_contrib"] = [0, 0]
    n["current_bet"] = 0
    n["acted"] = [False, False]
    n["num_raises"] = 0
    n["to_act"] = 1                  # BB acts first postflop (heads-up)
    n["history"] = n["history"] + ["/"]
    return n


def _showdown(n):
    n["terminal"] = True
    n["folded"] = False
    hole0 = [A.CARD_STR[c] for c in n["holes"][0]]
    hole1 = [A.CARD_STR[c] for c in n["holes"][1]]
    board = [A.CARD_STR[c] for c in n["board"]]
    s0 = eval7.evaluate(hole0 + board)
    s1 = eval7.evaluate(hole1 + board)
    n["winner"] = 0 if s0 > s1 else (1 if s1 > s0 else -1)
    return n


def util_to_p0(s):
    c0, c1 = s["contrib"]
    if s["folded"]:
        return c1 if s["winner"] == 0 else -c0
    if s["winner"] == 0:
        return c1
    if s["winner"] == 1:
        return -c0
    return 0


def infoset_key(s):
    p = s["to_act"]
    bucket = s["buckets"][p][s["street"]]
    return "%d|%d|%s" % (s["street"], bucket, "".join(s["history"]))


# ---------------------------------------------------------------------------
# MCCFR
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("codes", "na", "regret", "strat_sum", "visits")

    def __init__(self, codes):
        self.codes = codes               # legal action codes (deterministic per key)
        self.na = len(codes)
        self.regret = [0.0] * self.na
        self.strat_sum = [0.0] * self.na
        self.visits = 0.0                # plain visit count (for the runtime gate)

    def strategy(self):
        r = [x if x > 0 else 0.0 for x in self.regret]
        tot = sum(r)
        if tot > 0:
            return [x / tot for x in r]
        return [1.0 / self.na] * self.na

    def average(self):
        tot = sum(self.strat_sum)
        if tot > 0:
            return [x / tot for x in self.strat_sum]
        return [1.0 / self.na] * self.na


NODES = {}


def _node(key, codes):
    nd = NODES.get(key)
    if nd is None:
        nd = Node(codes)
        NODES[key] = nd
    return nd


def mccfr(s, traverser, t):
    if s["terminal"]:
        u0 = util_to_p0(s)
        return u0 if traverser == 0 else -u0

    acts = legal_actions(s)
    node = _node(infoset_key(s), [c for c, _ in acts])
    sigma = node.strategy()

    if s["to_act"] == traverser:
        util = [0.0] * len(acts)
        node_util = 0.0
        for i, (code, amt) in enumerate(acts):
            util[i] = mccfr(apply_action(s, code, amt), traverser, t)
            node_util += sigma[i] * util[i]
        for i in range(len(acts)):
            node.regret[i] += util[i] - node_util
            if node.regret[i] < 0.0:
                node.regret[i] = 0.0          # CFR+ flooring
        return node_util

    # opponent node: accumulate the average strategy (Linear CFR: weight by t),
    # bump the plain visit counter, and sample one action.
    node.visits += 1.0
    for i in range(len(acts)):
        node.strat_sum[i] += t * sigma[i]
    r = random.random()
    cum = 0.0
    chosen = len(acts) - 1
    for i in range(len(acts)):
        cum += sigma[i]
        if r <= cum:
            chosen = i
            break
    code, amt = acts[chosen]
    return mccfr(apply_action(s, code, amt), traverser, t)


# ---------------------------------------------------------------------------
# Dealing + bucketing
# ---------------------------------------------------------------------------

def _deal(rng):
    cards = rng.sample(range(52), 9)
    names = [str(A.FULL_DECK[i]) for i in cards]
    holes = [[names[0], names[1]], [names[2], names[3]]]
    board = names[4:9]
    return holes, board


def _buckets(holes, board, centroids, rng):
    out = []
    for h in holes:
        row = [A.preflop_index(h)]
        for street, blen in (("flop", 3), ("turn", 4), ("river", 5)):
            f = A.postflop_features(h, board[:blen], iters=BUCKET_ITERS, rng=rng)
            row.append(A.assign_bucket(f, centroids[street]))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Train + export
# ---------------------------------------------------------------------------

def train(iters, centroids, seed=0, out_path=BLUEPRINT_PATH):
    rng = random.Random(seed)
    ckpt = max(1, iters // 8)
    for t in range(iters):
        holes, board = _deal(rng)
        buckets = _buckets(holes, board, centroids, rng)
        for traverser in (0, 1):
            mccfr(new_state(holes, board, buckets), traverser, t + 1)
        if (t + 1) % ckpt == 0:
            n = export_blueprint(path=out_path, min_visits=20.0)   # checkpoint
            print("  [seed %d] %d/%d iters, infosets=%d, exported=%d"
                  % (seed, t + 1, iters, len(NODES), n))


def export_blueprint(path=BLUEPRINT_PATH, min_visits=1.0):
    keys, codes, probs, visits = [], [], [], []
    for key, node in NODES.items():
        if node.visits < min_visits:
            continue
        avg = node.average()
        keys.append(key)
        codes.append("".join(node.codes))
        visits.append(node.visits)
        row = np.zeros(MAX_ACTIONS, dtype=np.float16)
        for i in range(min(node.na, MAX_ACTIONS)):
            row[i] = avg[i]
        probs.append(row)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        keys=np.array(keys),
        codes=np.array(codes),
        probs=np.array(probs, dtype=np.float16),
        visits=np.array(visits, dtype=np.float32),
    )
    return len(keys)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_preflop_open(centroids):
    """Learning evidence: SB's first-action strategy by hand strength."""
    examples = [("AA", ["As", "Ad"]), ("KK", ["Ks", "Kd"]), ("AKs", ["As", "Ks"]),
                ("QJs", ["Qs", "Js"]), ("T9o", ["Ts", "9d"]), ("72o", ["7s", "2d"])]
    print("\nPreflop SB open strategy (street 0, no prior action):")
    print("  hand   bucket   fold   call   raise  allin")
    for name, hole in examples:
        b = A.preflop_index(hole)
        node = NODES.get("0|%d|" % b)
        if node is None:
            print("  %-5s  %-6d   (unvisited)" % (name, b))
            continue
        avg = node.average()
        pad = avg + [0.0] * (4 - len(avg))
        print("  %-5s  %-6d   %.2f   %.2f   %.2f   %.2f" % (name, b, pad[0], pad[1], pad[2], pad[3]))


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out_path = sys.argv[3] if len(sys.argv) > 3 else BLUEPRINT_PATH
    centroids = A.load_centroids()
    print("Heads-up NLHE MCCFR — %d iters, seed %d -> %s" % (iters, seed, out_path))
    train(iters, centroids, seed=seed, out_path=out_path)
    n = export_blueprint(path=out_path, min_visits=20.0)
    print("\nexported %d infosets (>=20 visits) -> %s" % (n, os.path.normpath(out_path)))
    if out_path == BLUEPRINT_PATH:
        _print_preflop_open(centroids)


if __name__ == "__main__":
    main()
