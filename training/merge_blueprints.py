"""
Merge independently-trained blueprints into one.

Parallel training runs N independent MCCFR workers (different seeds). Each worker's
average strategy approaches the same Nash equilibrium, so combining them by a
visit-weighted average of the per-infoset strategies behaves like one much longer
run (with lower variance) — the practical way to get "more simulations" across
cores. Visit counts are summed (and feed the runtime trust gate).

Run:  python training/merge_blueprints.py <bp1.npz> <bp2.npz> ... <out.npz>
"""

import sys
import numpy as np


def main():
    paths, out = sys.argv[1:-1], sys.argv[-1]
    acc = {}                       # key -> [codes, weighted_prob_sum, visit_sum]
    width = 0
    used = 0
    for p in paths:
        try:
            z = np.load(p)
        except Exception as e:
            print("skip %s (%s)" % (p, e))
            continue
        used += 1
        keys, codes, probs, visits = z["keys"], z["codes"], z["probs"], z["visits"]
        width = probs.shape[1]
        for i in range(len(keys)):
            v = float(visits[i])
            if v <= 0:
                continue
            k = str(keys[i])
            e = acc.get(k)
            if e is None:
                e = [str(codes[i]), np.zeros(width, dtype=np.float64), 0.0]
                acc[k] = e
            e[1] += v * probs[i].astype(np.float64)   # visit-weighted strategy
            e[2] += v

    mkeys, mcodes, mprobs, mvisits = [], [], [], []
    for k, (cd, psum, vsum) in acc.items():
        mkeys.append(k)
        mcodes.append(cd)
        mprobs.append((psum / vsum).astype(np.float16))
        mvisits.append(vsum)

    np.savez(out,
             keys=np.array(mkeys),
             codes=np.array(mcodes),
             probs=np.array(mprobs, dtype=np.float16),
             visits=np.array(mvisits, dtype=np.float32))
    print("merged %d/%d blueprints -> %d infosets -> %s" % (used, len(paths), len(mkeys), out))


if __name__ == "__main__":
    main()
