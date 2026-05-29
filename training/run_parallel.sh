#!/bin/bash
# Parallel MCCFR training: N independent workers (distinct seeds), then merge.
# Each worker's average strategy approaches the same Nash equilibrium; the
# visit-weighted merge behaves like one much longer run. Used to get many more
# effective simulations for the richer (6 bet-size) abstraction across cores.
set -u
WT=/mnt/c/Users/mateusz/Poker_bot/fullhouse-engine/.claude/worktrees/focused-jang-7c84f8
PY=/mnt/c/Users/mateusz/Poker_bot/fullhouse-engine/.venv/bin/python
ITERS=${1:-1500000}
WORKERS=${2:-10}
OUT=${3:-$WT/bots/fullhouse_cfr/data/blueprint.npz}
cd "$WT/training"

FILES=""
s=0
while [ "$s" -lt "$WORKERS" ]; do
  "$PY" holdem_train.py "$ITERS" "$s" "/tmp/bp_$s.npz" >"/tmp/w_$s.log" 2>&1 &
  FILES="$FILES /tmp/bp_$s.npz"
  s=$((s + 1))
done
echo "launched $WORKERS workers x $ITERS iters; waiting..."
wait
echo "all workers done; merging..."
"$PY" merge_blueprints.py $FILES "$OUT"
echo ALL_DONE
