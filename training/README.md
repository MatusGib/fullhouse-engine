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

## Roadmap to the Hold'em blueprint

1. ✅ Vanilla CFR proven on Kuhn + Leduc.
2. **Abstraction** (what makes Hold'em tractable):
   - *Card abstraction:* bucket hands by eval7 equity features with sklearn
     KMeans. Preflop = the 169 canonical starting hands; postflop = a few
     thousand buckets per street. Ship cluster **centroids** as numpy — never a
     pickled model (`pickle` is a banned import in the sandbox).
   - *Action abstraction:* a small bet menu (fold / call / ~½-pot / pot / all-in).
3. **External-Sampling MCCFR** (with CFR+ / linear discounting) over the
   abstracted game; train as long as time allows.
4. **Export** the average strategy to `bots/fullhouse_cfr/data/blueprint.npz`
   (int-keyed `float16` tables — compact and `pickle`-free).
5. **Runtime:** `bot.py` loads the blueprint at the warmup call, maps the live
   state into the abstraction (including translating opponents' real bet sizes
   to the nearest abstract action), looks up the strategy and samples — falling
   back to the current equity heuristic for any off-blueprint spot.

## Constraints that shape all of the above

- `pickle` / `joblib` / `threading` / `importlib` are **banned imports** → the
  blueprint and any model artifacts must be plain numpy `.npz` / `.npy`.
- `data/` ≤ 200 MB, no `.py` files inside; loaded once at module import (the
  30 s warmup call absorbs the load). Every live decision must still return in
  **2 s** on 0.5 CPU.
