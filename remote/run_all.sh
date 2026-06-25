#!/usr/bin/env bash
# One-click remote run for the n=4 DRONE experiment.
# Run from the MARL repo root:  bash remote/run_all.sh
#
# Produces (in results_remote/):
#   - part_H1_sched_*.pkl  : large scheduled-batch H=1 pool (the main "is p>0?" test)
#   - part_H0_sched_*.pkl  : H=0 memoryless pool (representation-gap ceiling)
#   - fig_drone_strategies.pdf, SUMMARY.txt
#
# Tunables (env or edit below):
#   NCORES   : parallel workers (default: cores-2). KEEP <= physical cores to avoid thrash.
#   POOL_Q   : total H=1 scheduled candidates (default 24000).
#   H0_Q     : total H=0 candidates (default 4000).
#   EPISODES : PSGA episodes per candidate (default 6000; plateau is ~8000, 6000 is fine).
set -e
cd "$(dirname "$0")/.."                       # -> MARL repo root

# ---- CRITICAL: pin BLAS/OpenMP threads so each worker is truly single-threaded.
# Without this, BLAS spawns ~7 threads/worker -> load 80+ on a 12-core box -> 3x slowdown.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

NCORES=${NCORES:-$(python3 -c "import os;print(max(1,(os.cpu_count() or 4)-2))")}
POOL_Q=${POOL_Q:-24000}
H0_Q=${H0_Q:-4000}
EPISODES=${EPISODES:-6000}
OUT=${OUT:-results_remote}
mkdir -p "$OUT"
echo "NCORES=$NCORES  POOL_Q=$POOL_Q  H0_Q=$H0_Q  EPISODES=$EPISODES  OUT=$OUT"

# ---- Phase 1: large scheduled-batch H=1 pool (split across workers) ----
per=$(( (POOL_Q + NCORES - 1) / NCORES ))
echo "=== Phase 1: H=1 scheduled pool, $NCORES workers x $per candidates ==="
pids=()
for ((w=0; w<NCORES; w++)); do
  off=$(( w * per ))
  python3 -u remote/pool_worker.py --offset $off --count $per --episodes $EPISODES \
      --chunk 250 --H 1 --batch sched --out "$OUT" > "$OUT/log_H1_$off.log" 2>&1 &
  pids+=($!)
done

# ---- Phase 2: H=0 memoryless ceiling (cheap; run on a couple of the cores) ----
echo "=== Phase 2: H=0 memoryless pool (single worker) ==="
python3 -u remote/pool_worker.py --offset 0 --count $H0_Q --episodes 4000 \
    --chunk 1000 --H 0 --batch sched --out "$OUT" > "$OUT/log_H0.log" 2>&1 &
pids+=($!)

echo "waiting for ${#pids[@]} workers..."
for p in "${pids[@]}"; do wait "$p"; done

# ---- Phase 3: analyze + figure ----
echo "=== Phase 3: meta-algorithm analysis + figure ==="
python3 -u remote/analyze.py --dir "$OUT"
echo "=== DONE. See $OUT/SUMMARY.txt and $OUT/fig_drone_strategies.pdf ==="
