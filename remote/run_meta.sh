#!/usr/bin/env bash
# Resilient meta-algorithm sweep over the drone H=1 pool: one short retryable process
# per (beta,seed) -> survives the flaky box. Resumable (skips finished chains).
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=${PY:-python3}
OUT=${OUT:-results_meta_drone}
EPOCHS=${EPOCHS:-300000}
BETAS=${BETAS:-"0.5 0.3 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001"}
SEEDS=${SEEDS:-"0 1 2 3 4 5"}
MAX_RETRIES=${MAX_RETRIES:-50}
mkdir -p "$OUT"
for b in $BETAS; do
  for s in $SEEDS; do
    tries=0
    until $PY -u remote/meta_chain.py --dir results_remote --beta "$b" --seed "$s" --epochs "$EPOCHS" --out "$OUT"; do
      tries=$((tries+1))
      [ "$tries" -ge "$MAX_RETRIES" ] && { echo "GAVE UP beta=$b seed=$s"; break; }
    done
  done
done
echo "=== aggregate ==="
t=0; until $PY -u remote/meta_aggregate.py --out "$OUT"; do t=$((t+1)); [ "$t" -ge "$MAX_RETRIES" ] && break; done
