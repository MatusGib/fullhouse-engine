# training/ — offline CFR solver workspace

**Not part of the competition submission.** The portal only receives a zip of
`bots/fullhouse_cfr/` (`bot.py` + optional `data/`). Everything in this folder
runs *offline* on your machine to produce a strategy **blueprint** that the bot
loads at runtime.

## What CFR is (the 60-second version)

Counterfactual Regret Minimization is a self-play algorithm that provably
converges to a Nash equilibrium in two-player zero-sum games. The pieces:

- **Information set (infoset):** everything one player knows at a decision point
  — their cards + the public betting history. Two deals that look identical to
  the acting player share one infoset.
- **Regret matching:** at each infoset we keep a *cumulative regret* per action
  (how much better we'd have done had we always taken that action). Next
  iteration's strategy is proportional to the *positive* regrets.
- **Counterfactual value:** each regret update is weighted by the probability the
  *opponent* (and chance) would let us reach this infoset in the first place.
- **The average strategy converges — not the current one.** We accumulate the
  strategy played on every iteration; that running average approaches
  equilibrium. That average is the blueprint we ship.

## Why start on toy games

Hold'em's game tree is astronomically large — you can't eyeball whether the
solver is right. Kuhn and Leduc are small enough to have *known* answers, so they
prove the machinery (regret matching, tree traversal, chance nodes, averaging)
is correct before we scale it up.

- `cfr_kuhn.py` — Kuhn poker (3 cards, one betting round). Vanilla CFR. Verifies
  the game value converges to the known **-1/18 ≈ -0.0556** and that the bluff
  frequency on the Jack lands in the known **[0, 1/3]** band.
- `cfr_leduc.py` — Leduc Hold'em (6 cards, a community card, **two** betting
  rounds). Same shape as Hold'em in miniature. Verifies via **exploitability**
  (best-response value) dropping toward zero as iterations grow.

Run them (from the repo root, in the WSL venv):

```bash
python training/cfr_kuhn.py
python training/cfr_leduc.py
```

## The Hold'em blueprint pipeline (implemented)

Scope: a **heads-up** (2-player) NLHE blueprint. Heads-up is two-player
zero-sum, so CFR provably approaches Nash; full 6-max is not. The bot uses the
blueprint only at true heads-up tables and falls back to the heuristic for
multiway pots — which is where most matches *end* (every match plays down to a
heads-up endgame), so it's high-value and theoretically sound.

Files (run in order, from the repo root in the WSL venv):

1. `holdem_abstraction.py` — **card abstraction**.
   - Preflop: the 169 canonical hands.
   - Flop/turn/river: a 2-D equity feature `[equity-to-river, made-strength-now]`
     clustered into 8 buckets/street with sklearn KMeans. Ships **centroids** as
     numpy (no pickled model). `python training/holdem_abstraction.py` builds
     `bots/fullhouse_cfr/data/abstraction.npz`.
2. `holdem_train.py` — **heads-up NLHE game + external-sampling MCCFR** (CFR+
   regret flooring). Action abstraction: `check / bet-0.75pot / all-in` with no
   bet, `fold / call / pot-raise / all-in` facing one (raises capped). Exports
   the average strategy to `bots/fullhouse_cfr/data/blueprint.npz` as
   pickle-free arrays (`keys`, `codes`, `probs:float16`).
   `python training/holdem_train.py [iterations]`  (≈230 iters/sec).
3. **Runtime** (`bots/fullhouse_cfr/bot.py`): loads both `.npz` at the warmup
   call; in heads-up spots it replays the action log into the same abstract
   infoset key (translating real bet sizes to the abstract bet menu), looks up
   the strategy, samples, and translates back to a legal action — falling back
   to the equity heuristic for any off-blueprint or inconsistent spot.

### Running a long offline training

Strength scales with iterations. A few thousand iters is noisy (it will *fold
AKo*); preflop ranges sharpen by ~10^5 iters, common postflop spots by ~10^6.
For the real bot, run something like:

```bash
python training/holdem_train.py 5000000      # ~6 hours; re-exports the blueprint
```

Only infosets visited ≥ 20 times are exported (each with its visit count); at
runtime the bot trusts an infoset only above `BP_MIN_VISITS` and falls back to
the heuristic otherwise.

### Status of the v1 blueprint (important)

`eval_blueprint.py` plays the blueprint vs. a blueprint-disabled (heuristic-only)
clone in **mirrored** heads-up matches — each deal is played both ways so card +
position variance cancels (the identical-bots control reads ~0 bb/100, i.e. the
harness is unbiased). At 1M iterations the v1 abstraction trains to a clean
near-Nash but measures **~break-even-to-slightly-worse than the heuristic**
(≈ −2 to −5 bb/100 at full usage). That's the *abstraction ceiling*, not
undertraining — the whole abstract tree is only ~14k infosets and is
well-converged. The heuristic plays full-resolution NLHE (exact equities, draws,
opponent reads) and exploits the coarse abstraction.

So the bot ships with **`BP_ENABLED = False`** and plays the stronger heuristic.
To make the blueprint surpass it and turn it on:

1. **Refine the abstraction** — more postflop buckets (`N_BUCKETS` in
   `holdem_abstraction.py`) and a finer bet menu (add half-pot / overbet sizes in
   `holdem_train.py`'s `legal_actions`).
2. **Retrain** (longer) and re-run `python training/eval_blueprint.py`.
3. If it shows a **positive** bb/100, set `BP_ENABLED = True` (and tune
   `BP_MIN_VISITS`) in `bots/fullhouse_cfr/bot.py`, and include
   `data/abstraction.npz` + `data/blueprint.npz` in the submission zip.

## Constraints that shape all of the above

- `pickle` / `joblib` / `threading` / `importlib` are **banned imports** → the
  blueprint and any model artifacts must be plain numpy `.npz` / `.npy`.
- `data/` ≤ 200 MB, no `.py` files inside; loaded once at module import (the
  30 s warmup call absorbs the load). Every live decision must still return in
  **2 s** on 0.5 CPU.
