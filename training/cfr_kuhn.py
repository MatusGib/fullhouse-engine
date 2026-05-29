"""
Vanilla CFR on Kuhn poker — the smallest game that proves the machinery works.

KUHN POKER
  Deck of 3 cards: J < Q < K (encoded 0 < 1 < 2). Two players each ante 1 and
  receive one private card. One betting round with two actions:
      'p' = pass  (check, or fold if facing a bet)
      'b' = bet   (bet 1, or call if facing a bet)
  Showdowns and folds pay net chips (+1 ante, or +2 when a bet was called).

WHAT TO WATCH
  CFR runs self-play, accumulating per-infoset regrets and an average strategy.
  The AVERAGE strategy converges to a Nash equilibrium. Two known facts let us
  check correctness:
    * the game value to the first player is exactly -1/18 ≈ -0.0556, and
    * the first player bluffs the Jack with probability alpha in [0, 1/3]
      (and bets the King at 3*alpha — value-betting 3x as often as bluffing).

Run:  python training/cfr_kuhn.py     (exit code 0 if all checks pass)
"""

import random
import sys

ACTIONS = ["p", "b"]
NUM_ACTIONS = 2
KUHN_GAME_VALUE = -1.0 / 18.0


class Node:
    """One information set: cumulative regrets + average-strategy accumulator."""
    __slots__ = ("regret_sum", "strategy_sum")

    def __init__(self):
        self.regret_sum = [0.0] * NUM_ACTIONS
        self.strategy_sum = [0.0] * NUM_ACTIONS

    def strategy(self, realization_weight):
        # Regret matching: play in proportion to POSITIVE regret. If no action
        # has positive regret yet, play uniformly.
        strat = [r if r > 0 else 0.0 for r in self.regret_sum]
        total = sum(strat)
        if total > 0:
            strat = [s / total for s in strat]
        else:
            strat = [1.0 / NUM_ACTIONS] * NUM_ACTIONS
        # Accumulate the average strategy, weighted by how often we actually
        # reach this infoset (our own reach probability).
        for a in range(NUM_ACTIONS):
            self.strategy_sum[a] += realization_weight * strat[a]
        return strat

    def average_strategy(self):
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / NUM_ACTIONS] * NUM_ACTIONS


NODE_MAP = {}


def cfr(cards, history, p0, p1):
    """Recursive CFR. Returns expected value to the player about to act.
    p0, p1 = reach probabilities contributed so far by each player's strategy."""
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    # ---- terminal payoffs ----
    if plays >= 2:
        terminal_pass = history[-1] == "p"
        double_bet = history[-2:] == "bb"
        player_wins_showdown = cards[player] > cards[opponent]
        if terminal_pass:
            if history == "pp":                 # check-check: showdown for the antes
                return 1 if player_wins_showdown else -1
            return 1                            # opponent folded to a bet
        if double_bet:                          # bet-call: showdown for 2
            return 2 if player_wins_showdown else -2

    # ---- decision node ----
    infoset = str(cards[player]) + history
    node = NODE_MAP.get(infoset)
    if node is None:
        node = Node()
        NODE_MAP[infoset] = node

    strategy = node.strategy(p0 if player == 0 else p1)
    util = [0.0] * NUM_ACTIONS
    node_util = 0.0
    for a in range(NUM_ACTIONS):
        nxt = history + ACTIONS[a]
        # value is negated because the recursive call returns it from the
        # OTHER player's perspective (zero-sum).
        if player == 0:
            util[a] = -cfr(cards, nxt, p0 * strategy[a], p1)
        else:
            util[a] = -cfr(cards, nxt, p0, p1 * strategy[a])
        node_util += strategy[a] * util[a]

    # ---- counterfactual regret update ----
    # Each regret is weighted by the OPPONENT's reach probability (how likely
    # the opponent's play is to bring us to this infoset at all).
    cf_reach = p1 if player == 0 else p0
    for a in range(NUM_ACTIONS):
        NODE_MAP[infoset].regret_sum[a] += cf_reach * (util[a] - node_util)
    return node_util


def train(iterations):
    cards = [0, 1, 2]
    total = 0.0
    for _ in range(iterations):
        random.shuffle(cards)
        total += cfr(cards, "", 1.0, 1.0)
    return total / iterations


def main():
    random.seed(42)
    iterations = 300_000
    value = train(iterations)

    print("Kuhn poker — vanilla CFR")
    print("iterations      :", iterations)
    print("game value (P0) : %.5f   (known %.5f)" % (value, KUHN_GAME_VALUE))

    card_name = {"0": "J", "1": "Q", "2": "K"}
    print("\ninfoset   P(pass)  P(bet)   meaning")
    for infoset in sorted(NODE_MAP):
        avg = NODE_MAP[infoset].average_strategy()
        hist = infoset[1:] or "open"
        print("  %-6s  %6.3f  %6.3f   [%s, %s]"
              % (infoset, avg[0], avg[1], card_name[infoset[0]], hist))

    # ---- verification against known equilibrium facts ----
    alpha = NODE_MAP["0"].average_strategy()[1]       # bluff-bet the Jack first
    bet_king = NODE_MAP["2"].average_strategy()[1]    # value-bet the King first
    bet_queen = NODE_MAP["1"].average_strategy()[1]   # Queen should not open-bet

    ok_value = abs(value - KUHN_GAME_VALUE) < 0.01
    ok_alpha = 0.0 <= alpha <= 0.34
    ok_queen = bet_queen < 0.05
    ok_king = bet_king >= alpha                       # King bets at least as often

    print("\nchecks")
    print("  value within 0.01 of -1/18      :", ok_value)
    print("  P(bet J first) in [0, 1/3]      :", ok_alpha, "(alpha = %.3f)" % alpha)
    print("  P(bet Q first) ~ 0              :", ok_queen, "(%.3f)" % bet_queen)
    print("  P(bet K first) >= P(bet J first):", ok_king,
          "(K=%.3f vs J=%.3f)" % (bet_king, alpha))

    all_ok = ok_value and ok_alpha and ok_queen and ok_king
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
