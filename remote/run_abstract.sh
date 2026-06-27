#!/usr/bin/env bash
# Abstract-game SCALING run: does Nash-hopping still concentrate on the global optimum
# as the number of agents grows? Builds graded-game pools at k=5,6,7,8 (closed-form Phi*,
# so optimality is known at any k), runs the validated meta-algorithm, emits the scaling
# figure nu^beta(k).  Run from the MARL repo root:  bash remote/run_abstract.sh
#
# HARDENED for the flaky i9-14900K box: PY override, small CHUNK (resume), run_worker
# supervisor (relaunch dead workers), cleanup trap. Workers resume from checkpoint, so a
# crash -- even a full BSOD (just re-run the same command) -- loses at most CHUNK candidates.
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

PY=${PY:-python3}                 # override with a conda env that has torch
KS=${KS:-"5 6 7 8"}
Q=${Q:-3000}                      # candidates per k. Higher k -> smaller optimum-rate p -> needs larger Q.
EPISODES=${EPISODES:-5000}
CHUNK=${CHUNK:-50}                # candidates per checkpoint; small so a crash loses <= CHUNK
NCORES=${NCORES:-$($PY -c "import os;print(max(1,(os.cpu_count() or 4)-2))")}
OUT=${OUT:-results_abstract}
MAX_RETRIES=${MAX_RETRIES:-100}
mkdir -p "$OUT"
echo "KS='$KS'  Q=$Q  EPISODES=$EPISODES  CHUNK=$CHUNK  NCORES=$NCORES  OUT=$OUT"

run_worker() {                    # $1=logfile, rest = command
  local logf="$1"; shift
  : > "$logf"
  local tries=0 rc=0
  while true; do
    rc=0; "$@" >> "$logf" 2>&1 || rc=$?
    [ "$rc" -eq 0 ] && return 0
    tries=$((tries+1))
    echo "[supervisor] worker exit=$rc -> retry $tries/$MAX_RETRIES (resuming from checkpoint)" >> "$logf"
    [ "$tries" -ge "$MAX_RETRIES" ] && { echo "[supervisor] GAVE UP after $MAX_RETRIES" >> "$logf"; return "$rc"; }
    sleep 3 2>/dev/null || true
  done
}
cleanup() { trap - EXIT INT TERM; for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null; done; pkill -f 'remote/abstract_worker.py' 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# One k at a time; split that k's Q across NCORES workers (chunked, resumable, supervised).
for k in $KS; do
  echo "=== k=$k : $NCORES workers x ~$(( (Q + NCORES - 1) / NCORES )) candidates ==="
  per=$(( (Q + NCORES - 1) / NCORES ))
  pids=()
  for ((w=0; w<NCORES; w++)); do
    off=$(( w * per ))
    [ $off -ge $Q ] && break
    cnt=$per; [ $((off+cnt)) -gt $Q ] && cnt=$((Q-off))
    run_worker "$OUT/log_k${k}_${off}.log" \
      $PY -u remote/abstract_worker.py --k $k --offset $off --count $cnt \
          --episodes $EPISODES --chunk $CHUNK --rho 0.7 --out "$OUT" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
done

echo "=== analysis + scaling figure ==="
t=0; until $PY -u remote/abstract_analyze.py --dir "$OUT" --ks "$(echo $KS | tr ' ' ',')"; do
  t=$((t+1)); [ "$t" -ge "$MAX_RETRIES" ] && break; sleep 2 2>/dev/null || true; done
echo "=== DONE. See $OUT/fig_abstract_scaling.pdf and $OUT/SUMMARY.txt ==="
