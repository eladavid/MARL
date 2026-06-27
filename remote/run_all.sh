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
#   CHUNK    : candidates per checkpoint (default 50). Smaller = finer live progress + a
#              crash loses <= CHUNK candidates. H0_CHUNK overrides it for the H=0 pool.
set -e
cd "$(dirname "$0")/.."                       # -> MARL repo root

# ---- CRITICAL: pin BLAS/OpenMP threads so each worker is truly single-threaded.
# Without this, BLAS spawns ~7 threads/worker -> load 80+ on a 12-core box -> 3x slowdown.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Python interpreter: override with PY=... (e.g. a conda env that has torch). Default python3.
PY=${PY:-python3}

NCORES=${NCORES:-$($PY -c "import os;print(max(1,(os.cpu_count() or 4)-2))")}
POOL_Q=${POOL_Q:-24000}
H0_Q=${H0_Q:-4000}
EPISODES=${EPISODES:-6000}
CHUNK=${CHUNK:-50}            # candidates per checkpoint (H=1); smaller = finer live progress + cheaper resume
H0_CHUNK=${H0_CHUNK:-$CHUNK}  # checkpoint size for the H=0 pool
BATCH=${BATCH:-sched}         # sched (4->128 schedule) | <int> fixed batch, e.g. 32 or 64 (much cheaper)
OUT=${OUT:-results_remote}
mkdir -p "$OUT"
if [ "$BATCH" = "sched" ]; then BARGS="--batch sched"; else BARGS="--batch fixed --fixed-batch $BATCH"; fi
echo "NCORES=$NCORES  POOL_Q=$POOL_Q  H0_Q=$H0_Q  EPISODES=$EPISODES  CHUNK=$CHUNK  BATCH=$BATCH  OUT=$OUT"

# ---- supervisor: relaunch a worker that dies, resuming from its checkpoint.
# CRITICAL: run_worker can't tell a crash from an intentional kill, so on script
# termination we MUST reap the worker subshells + their python, or they orphan and
# resurrect (orphan stacking -> box overload). Trap handles graceful stops; for a
# hard kill use remote/stop_grind.ps1 (kills supervisors before workers).
MAX_RETRIES=${MAX_RETRIES:-20}
cleanup() {
  trap - EXIT INT TERM
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null; done
  pkill -f 'remote/pool_worker.py' 2>/dev/null || true
  pkill -f 'remote/random_search.py' 2>/dev/null || true
}
trap cleanup EXIT INT TERM
run_worker() {                 # $1=logfile, rest = command (with args)
  local logf="$1"; shift
  : > "$logf"
  local tries=0 rc=0
  while true; do
    rc=0; "$@" >> "$logf" 2>&1 || rc=$?   # || keeps set -e from aborting; captures real exit
    [ "$rc" -eq 0 ] && return 0
    tries=$((tries+1))
    echo "[supervisor] worker exit=$rc -> retry $tries/$MAX_RETRIES (resuming from checkpoint)" >> "$logf"
    [ "$tries" -ge "$MAX_RETRIES" ] && { echo "[supervisor] GAVE UP after $MAX_RETRIES retries" >> "$logf"; return "$rc"; }
    sleep 3 2>/dev/null || true
  done
}

# ---- Phase 1: large scheduled-batch H=1 pool (split across workers) ----
per=$(( (POOL_Q + NCORES - 1) / NCORES ))
echo "=== Phase 1: H=1 scheduled pool, $NCORES workers x $per candidates ==="
pids=()
for ((w=0; w<NCORES; w++)); do
  off=$(( w * per ))
  run_worker "$OUT/log_H1_$off.log" \
    $PY -u remote/pool_worker.py --offset $off --count $per --episodes $EPISODES \
        --chunk $CHUNK --H 1 $BARGS --out "$OUT" &
  pids+=($!)
done

# ---- Phase 2: H=0 memoryless ceiling (cheap; run on a couple of the cores) ----
echo "=== Phase 2: H=0 memoryless pool (single worker) ==="
run_worker "$OUT/log_H0.log" \
  $PY -u remote/pool_worker.py --offset 0 --count $H0_Q --episodes 4000 \
      --chunk $H0_CHUNK --H 0 $BARGS --out "$OUT" &
pids+=($!)

# ---- Phase 2b: random-deterministic baseline (training-free; reproduces "0 of 1e6") ----
echo "=== Phase 2b: random-deterministic search (${RAND_TOTAL:-1000000} policies) ==="
run_worker "$OUT/log_random.log" \
  $PY -u remote/random_search.py --total ${RAND_TOTAL:-1000000} --H 1 --out "$OUT" &
pids+=($!)

echo "waiting for ${#pids[@]} workers..."
for p in "${pids[@]}"; do wait "$p" || true; done   # tolerate a worker that exhausted its retries

# ---- Phase 3: analyze + figure ----
echo "=== Phase 3: meta-algorithm analysis + figure ==="
$PY -u remote/analyze.py --dir "$OUT"
echo "=== DONE. See $OUT/SUMMARY.txt and $OUT/fig_drone_strategies.pdf ==="
