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
   regret flooring, Linear-CFR averaging). Action abstraction:
   `check / half-pot / pot / all-in` with no bet, `fold / call / pot-raise /
   all-in` facing one (raises capped). Exports the average strategy to
   `bots/fullhouse_cfr/data/blueprint.npz` as pickle-free arrays (`keys`,
   `codes`, `probs:float16`, `visits:float32`).
   `python training/holdem_train.py [iterations]`.
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

### Status: blueprint ENABLED (v1.5)

`eval_blueprint.py` plays the blueprint vs. a blueprint-disabled (heuristic-only)
clone in **mirrored** heads-up matches (each deal played both ways so card +
position variance cancels). `robustness_eval.py` checks it across opponent styles
(nit / LAG / station / maniac / random) so we don't overfit to the reference
bots. The journey:

- **v1** (8 buckets, single 0.75-pot bet, 1M iters): clean near-Nash but
  **−2.5 bb/100** vs. the heuristic — one bet size let the heuristic exploit it.
- **v2** (16 buckets + half-pot, 1.5M iters): the 7× bigger tree (104k infosets)
  was *undertrained* at that budget and did worse (−7.6). Finer abstraction needs
  far more iterations than a coarse one — convergence beats resolution here.
- **v1.5 — shipped** (8 buckets + half-pot bet, ~2M iters, Linear CFR): adding
  the half-pot bet (sizing flexibility + the ability to distinguish small vs.
  large bets it faces) flipped it to a small **consistent net win, ~+1.5 bb/100**
  over the heuristic. Robustness holds across every style (worst case +33 bb/100,
  matching the heuristic), so enabling it costs no exploitation value vs. a weak
  field while adding GTO robustness vs. strong opponents.

So the bot ships with **`BP_ENABLED = True`**, `BP_MIN_VISITS = 150`. The margin
is small (heads-up only, near the harness noise floor) but consistently positive
and GTO-robust. To push further: more iterations + a finer bet/bucket menu, then
re-confirm with both `eval_blueprint.py` and `robustness_eval.py`. Set
`BP_ENABLED = False` to fall back to the pure heuristic at any time.

## Constraints that shape all of the above

- `pickle` / `joblib` / `threading` / `importlib` are **banned imports** → the
  blueprint and any model artifacts must be plain numpy `.npz` / `.npy`.
- `data/` ≤ 200 MB, no `.py` files inside; loaded once at module import (the
  30 s warmup call absorbs the load). Every live decision must still return in
  **2 s** on 0.5 CPU.
