#!/usr/bin/env bash
# Abstract-game SCALING run: does Nash-hopping still concentrate on the global optimum
# as the number of agents grows? Builds graded-game pools at k=5,6,7,8 (closed-form Phi*,
# so optimality is known at any k), runs the validated meta-algorithm, emits the scaling
# figure nu^beta(k).  Run from the MARL repo root:  bash remote/run_abstract.sh
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

KS=${KS:-"5 6 7 8"}
Q=${Q:-3000}              # candidates per k. Higher k -> smaller optimum-rate p -> needs larger Q.
EPISODES=${EPISODES:-5000}
NCORES=${NCORES:-$(python3 -c "import os;print(max(1,(os.cpu_count() or 4)-2))")}
OUT=${OUT:-results_abstract}
mkdir -p "$OUT"
echo "KS='$KS'  Q=$Q  EPISODES=$EPISODES  NCORES=$NCORES  OUT=$OUT"

# One k at a time; split that k's Q across NCORES workers (chunked, resumable).
for k in $KS; do
  echo "=== k=$k : $NCORES workers ==="
  per=$(( (Q + NCORES - 1) / NCORES ))
  pids=()
  for ((w=0; w<NCORES; w++)); do
    off=$(( w * per ))
    [ $off -ge $Q ] && break
    cnt=$per; [ $((off+cnt)) -gt $Q ] && cnt=$((Q-off))
    python3 -u remote/abstract_worker.py --k $k --offset $off --count $cnt \
        --episodes $EPISODES --chunk 250 --rho 0.7 --out "$OUT" \
        > "$OUT/log_k${k}_${off}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
done

echo "=== analysis + scaling figure ==="
python3 -u remote/abstract_analyze.py --dir "$OUT" --ks "$(echo $KS | tr ' ' ',')"
echo "=== DONE. See $OUT/fig_abstract_scaling.pdf and $OUT/SUMMARY.txt ==="
