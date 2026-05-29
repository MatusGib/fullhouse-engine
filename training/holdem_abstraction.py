"""
Card abstraction for the heads-up Hold'em blueprint.

Hold'em has ~10^9 distinct (hole, board) situations — far too many to give each
its own CFR infoset. We bucket them:

  * Preflop: the 169 strategically-distinct starting hands (13 pairs + 78 suited
    + 78 offsuit). Exact, no clustering needed.
  * Flop / turn / river: a 2-D equity feature per hand, clustered with KMeans:
        f1 = equity-to-river vs. a random hand (how often we win by showdown)
        f2 = made-strength-now  (how often our CURRENT best-5 beats a random
                                 current best-5 — ignores future cards)
    f1 alone conflates a made hand with a draw of equal equity; the (f1, f2)
    pair separates them (a draw has high f1 but low f2), which is exactly the
    distinction a good strategy must act on.

We fit the KMeans centroids offline and ship them as a plain numpy array (no
pickled model — `pickle` is banned in the sandbox). At runtime the bot computes
the same feature and assigns the nearest centroid with pure numpy.

Run:  python training/holdem_abstraction.py        (builds + saves + sanity check)
"""

import os
import random
import numpy as np
import eval7

RANKS = "23456789TJQKA"
SUITS = "shdc"
FULL_DECK = [eval7.Card(r + s) for r in RANKS for s in SUITS]
CARD_STR = {str(c): c for c in FULL_DECK}
RANK_IDX = {r: i for i, r in enumerate(RANKS)}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "bots", "fullhouse_cfr", "data")
ABSTRACTION_PATH = os.path.join(DATA_DIR, "abstraction.npz")

STREET_BOARD_LEN = {"flop": 3, "turn": 4, "river": 5}
N_BUCKETS = 8                      # postflop buckets per street (coarse v1)


# ---------------------------------------------------------------------------
# Preflop: 169 canonical hands -> compact index 0..168
# ---------------------------------------------------------------------------

def preflop_index(hole):
    i1, i2 = RANK_IDX[hole[0][0]], RANK_IDX[hole[1][0]]
    suited = hole[0][1] == hole[1][1]
    if i1 == i2:
        return i1                                  # pair: 0..12
    hi, lo = max(i1, i2), min(i1, i2)
    pair_id = hi * (hi - 1) // 2 + lo              # 0..77
    return 13 + pair_id if suited else 13 + 78 + pair_id


# ---------------------------------------------------------------------------
# Postflop: 2-D equity feature via Monte-Carlo
# ---------------------------------------------------------------------------

def postflop_features(hole, board, iters=120, rng=random):
    """(equity_to_river, made_strength_now) vs. a random opponent hand."""
    hole_c = [CARD_STR[c] for c in hole]
    board_c = [CARD_STR[c] for c in board]
    known = set(hole) | set(board)
    deck = [c for c in FULL_DECK if str(c) not in known]
    need = 5 - len(board)

    my_now = eval7.evaluate(hole_c + board_c)
    w_river = w_now = 0.0
    for _ in range(iters):
        s = rng.sample(deck, 2 + need)
        opp, run = s[:2], s[2:]
        opp_now = eval7.evaluate(opp + board_c)
        if my_now > opp_now:
            w_now += 1
        elif my_now == opp_now:
            w_now += 0.5
        comm = board_c + run
        my_r = eval7.evaluate(hole_c + comm)
        opp_r = eval7.evaluate(opp + comm)
        if my_r > opp_r:
            w_river += 1
        elif my_r == opp_r:
            w_river += 0.5
    return (w_river / iters, w_now / iters)


def assign_bucket(feature, centroids):
    """Nearest-centroid index (pure numpy — what the bot uses at runtime)."""
    d = ((centroids - np.asarray(feature, dtype=np.float64)) ** 2).sum(axis=1)
    return int(np.argmin(d))


# ---------------------------------------------------------------------------
# Build + persist the centroids
# ---------------------------------------------------------------------------

def build_centroids(n_samples=2500, iters=100, k=N_BUCKETS, seed=0):
    from sklearn.cluster import KMeans
    rng = random.Random(seed)
    out = {}
    for street, blen in STREET_BOARD_LEN.items():
        feats = []
        for _ in range(n_samples):
            cards = rng.sample(range(52), 2 + blen)
            hole = [str(FULL_DECK[i]) for i in cards[:2]]
            board = [str(FULL_DECK[i]) for i in cards[2:]]
            feats.append(postflop_features(hole, board, iters=iters, rng=rng))
        X = np.array(feats, dtype=np.float64)
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        # sort centroids by equity-to-river so bucket 0 = weakest .. k-1 = strongest
        order = np.argsort(km.cluster_centers_[:, 0])
        out[street] = km.cluster_centers_[order].astype(np.float32)
    return out


def save_centroids(centroids, path=ABSTRACTION_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **centroids)


def load_centroids(path=ABSTRACTION_PATH):
    z = np.load(path)
    return {s: z[s] for s in z.files}


# ---------------------------------------------------------------------------
# Build + sanity check
# ---------------------------------------------------------------------------

def main():
    print("Building card abstraction (%d buckets/street)..." % N_BUCKETS)
    centroids = build_centroids()
    save_centroids(centroids)
    print("saved ->", os.path.normpath(ABSTRACTION_PATH))
    for s in STREET_BOARD_LEN:
        print("\n%s centroids [equity_to_river, made_now] (weak->strong):" % s)
        for i, c in enumerate(centroids[s]):
            print("  bucket %d: [%.3f, %.3f]" % (i, c[0], c[1]))

    # Sanity: distinct hand types should land in sensibly different buckets.
    print("\nsanity (flop):")
    cases = [
        ("nut flush draw + overs", ["Ah", "Kh"], ["Qh", "7h", "2c"]),
        ("top set",                [" As".replace(" ", ""), "Ad"], ["As", "Kd", "2c"]),
        ("top pair top kicker",    ["Ah", "Kd"], ["Ac", "8d", "2s"]),
        ("middle pair",            ["8h", "9d"], ["Ac", "8d", "2s"]),
        ("total air",              ["7c", "2d"], ["Ah", "Ks", "Qd"]),
    ]
    rng = random.Random(1)
    for name, hole, board in cases:
        # fix the AsAd/As board collision in the demo set
        if name == "top set":
            hole, board = ["Ah", "Ad"], ["As", "Kd", "2c"]
        f = postflop_features(hole, board, iters=400, rng=rng)
        b = assign_bucket(f, centroids["flop"])
        print("  %-22s feat=[%.3f, %.3f] -> bucket %d" % (name, f[0], f[1], b))


if __name__ == "__main__":
    main()
