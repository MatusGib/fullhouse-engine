"""
CFR on Leduc Hold'em — CFR on a TWO-street game with a public board card, i.e.
Hold'em in miniature. It reuses the exact regret-matching machinery proven on
Kuhn (see cfr_kuhn.py); the new ingredients are a second betting round, a
revealed community card, capped raises, and a real showdown rule. These are the
same structural pieces the full Hold'em solver will need.

LEDUC RULES (standard)
  Deck: 6 cards = ranks {J,Q,K} x 2 suits (suit is irrelevant — no flushes, so
  we key infosets on rank only). Each player antes 1 and gets one private card.
  Round 0 bets are size 2; then one community card is revealed; round 1 bets are
  size 4. At most 2 raises per round. Showdown: pairing the board wins; otherwise
  high card wins; equal ranks split.

VALIDATION (both computed EXACTLY by enumerating all 120 deals)
  * game value of the average strategy vs. itself settles to a stable constant;
  * exploitability — how much a best response beats the average strategy,
    averaged over both seats — shrinks toward 0 as iterations grow. That is the
    rigorous signal that CFR is approaching a Nash equilibrium.
  * spot-check: pairing the board on round 1 should bet/raise almost always.

Run:  python training/cfr_leduc.py     (exit code 0 if all checks pass)
"""

import random
import sys

BET = [2, 4]            # round 0 / round 1 bet sizes
RAISE_CAP = 2           # max 'r' actions per round
DECK = list(range(6))   # cards 0..5; rank(c) = c // 2 -> 0=J, 1=Q, 2=K


def rank(c):
    return c // 2


# ---------------------------------------------------------------------------
# Betting-round mechanics (shared by CFR, the exact EV walk, and best response)
# ---------------------------------------------------------------------------

def legal_actions(rh):
    """Legal actions at a non-terminal round history rh (list of 'c'/'r')."""
    facing_bet = bool(rh) and rh[-1] == "r"
    if facing_bet:
        acts = ["f", "c"]                 # fold, call
        if rh.count("r") < RAISE_CAP:
            acts.append("r")              # reraise
        return acts
    return ["c", "r"]                     # check, bet


def round_closed(rh):
    """True if betting ended by check-check or a call (NOT by a fold)."""
    if not rh or rh[-1] != "c":
        return False
    if "r" in rh:                         # a call closed the betting
        return True
    return len(rh) >= 2                   # check-check


def is_fold(rh):
    return bool(rh) and rh[-1] == "f"


def contributions(r0, r1):
    """Chips each player has committed, given both rounds' action lists."""
    contrib = [1, 1]                      # antes
    for rnd, rh in ((0, r0), (1, r1)):
        S = BET[rnd]
        invested = [0, 0]
        to_match = 0
        for i, a in enumerate(rh):
            pl = i % 2                    # player 0 acts first each round
            if a == "r":
                to_match += S
                pay = to_match - invested[pl]
                invested[pl] += pay
                contrib[pl] += pay
            elif a == "c":
                pay = to_match - invested[pl]
                invested[pl] += pay
                contrib[pl] += pay
    return contrib


def showdown_winner(c0, c1, board):
    r0, r1, rb = rank(c0), rank(c1), rank(board)
    p0_pair, p1_pair = (r0 == rb), (r1 == rb)
    if p0_pair != p1_pair:
        return 0 if p0_pair else 1
    if r0 != r1:
        return 0 if r0 > r1 else 1
    return -1                             # tie (same rank, neither/both pair)


def util_to_p0(cards, r0, r1):
    """Terminal chips to player 0."""
    c0, c1, board = cards
    contrib = contributions(r0, r1)
    for rh in (r0, r1):
        if is_fold(rh):
            folder = (len(rh) - 1) % 2
            return contrib[1] if folder == 1 else -contrib[0]
    w = showdown_winner(c0, c1, board)
    if w == 0:
        return contrib[1]
    if w == 1:
        return -contrib[0]
    return 0


def infoset_key(card, board, r0, r1):
    """Acting player's view: own rank, board rank (round 1 only), full history."""
    b = "-" if board is None else str(rank(board))
    return "%d|%s|%s|%s" % (rank(card), b, "".join(r0), "".join(r1))


# ---------------------------------------------------------------------------
# CFR (chance-sampled: one random deal per iteration, full walk of the betting)
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("legal", "regret_sum", "strategy_sum")

    def __init__(self, legal):
        self.legal = legal
        self.regret_sum = [0.0] * len(legal)
        self.strategy_sum = [0.0] * len(legal)

    def strategy(self, realization_weight):
        n = len(self.legal)
        strat = [r if r > 0 else 0.0 for r in self.regret_sum]
        total = sum(strat)
        if total > 0:
            strat = [s / total for s in strat]
        else:
            strat = [1.0 / n] * n
        for i in range(n):
            self.strategy_sum[i] += realization_weight * strat[i]
        return strat

    def average(self):
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / len(self.legal)] * len(self.legal)


NODES = {}


def _node(key, legal):
    nd = NODES.get(key)
    if nd is None:
        nd = Node(legal)
        NODES[key] = nd
    return nd


def cfr(cards, rnd, r0, r1, p0, p1):
    """Returns utility to PLAYER 0 (we use a fixed reference player and flip the
    sign for player 1's regrets — avoids perspective bugs across round resets)."""
    rh = r0 if rnd == 0 else r1
    if is_fold(rh):
        return util_to_p0(cards, r0, r1)
    if round_closed(rh):
        if rnd == 0:
            return cfr(cards, 1, r0, [], p0, p1)     # deal board, start round 1
        return util_to_p0(cards, r0, r1)             # showdown

    player = len(rh) % 2
    board = None if rnd == 0 else cards[2]
    key = infoset_key(cards[player], board, r0, r1)
    legal = legal_actions(rh)
    node = _node(key, legal)

    own_reach = p0 if player == 0 else p1
    strat = node.strategy(own_reach)

    util0 = [0.0] * len(legal)
    node_util0 = 0.0
    for i, a in enumerate(legal):
        nr0 = r0 + [a] if rnd == 0 else r0
        nr1 = r1 if rnd == 0 else r1 + [a]
        if player == 0:
            util0[i] = cfr(cards, rnd, nr0, nr1, p0 * strat[i], p1)
        else:
            util0[i] = cfr(cards, rnd, nr0, nr1, p0, p1 * strat[i])
        node_util0 += strat[i] * util0[i]

    sign = 1.0 if player == 0 else -1.0      # convert player-0 utility to actor's
    cf_reach = p1 if player == 0 else p0      # opponent's reach probability
    for i in range(len(legal)):
        node.regret_sum[i] += cf_reach * sign * (util0[i] - node_util0)
    return node_util0


def train(iterations):
    for _ in range(iterations):
        c0, c1, board = random.sample(DECK, 3)
        cfr((c0, c1, board), 0, [], [], 1.0, 1.0)


# ---------------------------------------------------------------------------
# Exact evaluation by full enumeration of all 120 deals
# ---------------------------------------------------------------------------

ALL_DEALS = [(a, b, c) for a in DECK for b in DECK for c in DECK
             if len({a, b, c}) == 3]


def _ev_sigma(cards, rnd, r0, r1):
    """Expected value to player 0 with BOTH players using the average strategy."""
    rh = r0 if rnd == 0 else r1
    if is_fold(rh):
        return util_to_p0(cards, r0, r1)
    if round_closed(rh):
        if rnd == 0:
            return _ev_sigma(cards, 1, r0, [])
        return util_to_p0(cards, r0, r1)
    player = len(rh) % 2
    board = None if rnd == 0 else cards[2]
    legal = legal_actions(rh)
    node = NODES.get(infoset_key(cards[player], board, r0, r1))
    avg = node.average() if node else [1.0 / len(legal)] * len(legal)
    v = 0.0
    for i, a in enumerate(legal):
        nr0 = r0 + [a] if rnd == 0 else r0
        nr1 = r1 if rnd == 0 else r1 + [a]
        v += avg[i] * _ev_sigma(cards, rnd, nr0, nr1)
    return v


def game_value():
    return sum(_ev_sigma(d, 0, [], []) for d in ALL_DEALS) / len(ALL_DEALS)


# ---------------------------------------------------------------------------
# Exact best response (vector / belief method) -> exploitability
# ---------------------------------------------------------------------------

def _traverse(me, me_card, rnd, r0, r1, particles, me_mode):
    """Value to `me` vs. the opponent playing the average strategy.
    `particles` = list of [opp_card, board, weight] — the chance x opponent-reach
    probability of each hidden opponent card consistent with me's public view.
    me_mode='max'   -> `me` best-responds (the value we want for exploitability).
    me_mode='sigma' -> `me` also follows the average strategy (used only to
                       cross-check this walk against the exact game value)."""
    rh = r0 if rnd == 0 else r1

    # terminal: a fold, or a closed round 1 (showdown)
    if is_fold(rh) or (rnd == 1 and round_closed(rh)):
        ev = 0.0
        for opp_card, board, w in particles:
            cards = (me_card, opp_card, board) if me == 0 else (opp_card, me_card, board)
            u0 = util_to_p0(cards, r0, r1)
            ev += w * (u0 if me == 0 else -u0)
        return ev

    # Round 0 closed -> the board is revealed. The board is PUBLIC, so we BRANCH
    # on it (each board is a separate public continuation that `me` observes) and
    # recurse once per board; within each board the belief is over the opponent's
    # hidden card only. (Blending boards into one belief would force a single
    # action across different boards and understate the best response.)
    if rnd == 0 and round_closed(rh):
        ev = 0.0
        for b in DECK:
            if b == me_card:
                continue
            child = [[oc, b, w * 0.25] for oc, _, w in particles if oc != b]
            if child:
                ev += _traverse(me, me_card, 1, r0, [], child, me_mode)
        return ev

    player = len(rh) % 2
    board = None if rnd == 0 else particles[0][1]   # post-branch: one board here
    legal = legal_actions(rh)

    if player == me:
        if me_mode == "max":
            best = None
            for a in legal:
                nr0 = r0 + [a] if rnd == 0 else r0
                nr1 = r1 if rnd == 0 else r1 + [a]
                v = _traverse(me, me_card, rnd, nr0, nr1, particles, me_mode)
                if best is None or v > best:
                    best = v
            return best
        # me_mode == 'sigma': me also follows its average strategy
        node = NODES.get(infoset_key(me_card, board, r0, r1))
        avg = node.average() if node else [1.0 / len(legal)] * len(legal)
        ev = 0.0
        for i, a in enumerate(legal):
            nr0 = r0 + [a] if rnd == 0 else r0
            nr1 = r1 if rnd == 0 else r1 + [a]
            ev += avg[i] * _traverse(me, me_card, rnd, nr0, nr1, particles, me_mode)
        return ev

    # opponent node: split particles by the opponent's average strategy
    ev = 0.0
    for i, a in enumerate(legal):
        child = []
        for opp_card, bd, w in particles:
            node = NODES.get(infoset_key(opp_card, bd, r0, r1))
            avg = node.average() if node else [1.0 / len(legal)] * len(legal)
            wa = w * avg[i]
            if wa > 0:
                child.append([opp_card, bd, wa])
        if child:
            nr0 = r0 + [a] if rnd == 0 else r0
            nr1 = r1 if rnd == 0 else r1 + [a]
            ev += _traverse(me, me_card, rnd, nr0, nr1, child, me_mode)
    return ev


def _br(me, me_card, rnd, r0, r1, particles):
    return _traverse(me, me_card, rnd, r0, r1, particles, "max")


def exploitability():
    """(BR0 value to P0 + BR1 value to P1) / 2.  -> 0 at equilibrium."""
    vals = []
    for me in (0, 1):
        ev = 0.0
        for me_card in DECK:
            particles = [[oc, None, 1.0 / 5.0] for oc in DECK if oc != me_card]
            ev += (1.0 / 6.0) * _br(me, me_card, 0, [], [], particles)
        vals.append(ev)
    return (vals[0] + vals[1]) / 2.0


# ---------------------------------------------------------------------------
# Main: train to checkpoints, show exploitability shrinking, spot-check
# ---------------------------------------------------------------------------

def _paired_vs_unpaired_open():
    """Mean P(bet) at round-1 OPENING infosets, split by whether our card pairs
    the board. A sane strategy bets the pair (a monster in Leduc) far more often
    than an unpaired hand."""
    paired, unpaired = [], []
    for key, node in NODES.items():
        card_r, board_r, _, r1s = key.split("|")
        if r1s == "" and board_r != "-" and node.legal == ["c", "r"]:
            bucket = paired if card_r == board_r else unpaired
            bucket.append(node.average()[1])     # P(bet)
    mp = sum(paired) / len(paired) if paired else 0.0
    mu = sum(unpaired) / len(unpaired) if unpaired else 0.0
    return mp, mu


def main():
    random.seed(7)
    checkpoints = [20_000, 100_000, 400_000]
    done = 0
    history = []
    print("Leduc Hold'em — chance-sampled CFR")
    print("%10s  %12s  %14s" % ("iters", "game value", "exploitability"))
    for cp in checkpoints:
        train(cp - done)
        done = cp
        gv = game_value()
        expl = exploitability()
        history.append(expl)
        print("%10d  %12.4f  %14.4f" % (cp, gv, expl))

    paired, unpaired = _paired_vs_unpaired_open()
    print("\nspot-check: round-1 open P(bet) — paired board %.3f vs unpaired %.3f"
          % (paired, unpaired))

    # ---- verification ----
    ok_shrink = history[-1] < history[0]          # exploitability decreased
    ok_small = history[-1] < 0.05                 # near equilibrium (chips)
    ok_nonneg = min(history) >= -1e-9             # exploitability is never negative
    ok_paired = paired > unpaired + 0.15 and paired > 0.55   # pair -> bet far more
    print("\nchecks")
    print("  exploitability decreased         :", ok_shrink,
          "(%.4f -> %.4f)" % (history[0], history[-1]))
    print("  final exploitability < 0.05      :", ok_small)
    print("  exploitability non-negative      :", ok_nonneg)
    print("  bets pair >> unpaired (round-1)  :", ok_paired)

    all_ok = ok_shrink and ok_small and ok_nonneg and ok_paired
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
