#!/usr/bin/env bash
#
# Steady-state training throughput of FlexFlow's Realm backend on YOLOv10x.
#
# Reports the time for 500 training iterations at batch size 6, averaged over 3
# runs.  Startup -- process launch, graph construction, weight initialization,
# cuDNN algorithm selection -- is excluded by timing two runs of different
# lengths and subtracting, so nothing needs to be instrumented inside FlexFlow.
#
# Takes about 15 minutes after the build. No arguments, no setup.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
WORK="$HERE/work/flexflow"

ITERATIONS=500
WARMUP=100
REPEATS=3

# Realm allocates one pool up front; below ~26000 MB it cannot fit the model,
# and above ~28000 there is too little left for CUDA's own allocations.
FSIZE=27000

source "$HERE/build_env.sh"

mkdir -p "$WORK"

echo "=== building FlexFlow ($FF_BUILD_KIND) ==="
ffbuild

echo "=== compiling the YOLOv10x graph ==="
ffrun "$BIN/export-model-arch/export-model-arch yolov10x > '$WORK/cg.json'"
ffrun "$BIN/compile-model/compile-model '$WORK/cg.json' '$WORK/mpcg.json' passthrough"

echo "=== generating an input batch and a label ==="
python3 "$HERE/write_tensors.py" --input "$WORK/input.bin" --label "$WORK/label.bin"

run_iterations() {
  local n="$1"
  local start end
  start=$(date +%s%N)
  ffrun "REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize $FSIZE -cuda:dynfb 0' \
    FF_LOAD_TENSORS='$WORK/input.bin,$WORK/label.bin' \
    FF_LOSS=mean_squared_error_avg FF_LOSS_LOGIT=model.23.boxes \
    FF_ITERATIONS=$n \
    $GL $BIN/run-model/run-model '$WORK/mpcg.json'" \
    > "$WORK/run.log" 2>&1 || { cat "$WORK/run.log"; exit 1; }
  end=$(date +%s%N)
  echo $(( (end - start) / 1000000 ))
}

echo "=== timing ==="
: > "$WORK/timings.txt"
for repeat in $(seq 1 $REPEATS); do
  warm_ms=$(run_iterations "$WARMUP")
  full_ms=$(run_iterations $(( WARMUP + ITERATIONS )))
  echo "$(( full_ms - warm_ms ))" >> "$WORK/timings.txt"
  printf 'run %d:  %d iterations %.2f s  |  %d iterations %.2f s  |  difference (%d iterations) %.2f s\n' \
    "$repeat" \
    "$WARMUP" "$(echo "$warm_ms" | awk '{print $1/1000}')" \
    $(( WARMUP + ITERATIONS )) "$(echo "$full_ms" | awk '{print $1/1000}')" \
    "$ITERATIONS" "$(echo "$(( full_ms - warm_ms ))" | awk '{print $1/1000}')"
done

python3 "$HERE/report.py" --label "FlexFlow (Realm backend)" \
  --iterations "$ITERATIONS" --timings "$WORK/timings.txt"
