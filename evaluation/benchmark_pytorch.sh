#!/usr/bin/env bash
#
# Steady-state training throughput of YOLOv10x under PyTorch, for comparison
# with benchmark_flexflow.sh.
#
# Same measurement: 500 training iterations at batch size 6, averaged over 3
# runs, with startup excluded by timing two runs of different lengths and
# subtracting.
#
# Reported twice, because the setting matters and upstream ships it off:
# ultralytics leaves torch.backends.cudnn.benchmark unset (see init_seeds in
# ultralytics/utils/torch_utils.py, where it is commented out because it hurts
# their AutoBatch path), while FlexFlow selects convolution algorithms by
# measurement.
#
# Takes about 25 minutes. No arguments, no setup.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
WORK="$HERE/work/pytorch"

ITERATIONS=500
WARMUP=100
REPEATS=3

VENV="$REPO/yolov10-validation/.venv"
PYTHON="$VENV/bin/python"
ULTRALYTICS="${ULTRALYTICS:-/home/eslaught/flexflow/ultralytics}"
ULTRALYTICS_REVISION=94e9819c536e2f8a543a5f9b5c5629cdae9b9658

mkdir -p "$WORK"

if [ ! -d "$ULTRALYTICS" ]; then
  echo "=== fetching ultralytics ==="
  git clone https://github.com/ultralytics/ultralytics.git "$ULTRALYTICS"
  git -C "$ULTRALYTICS" checkout --quiet "$ULTRALYTICS_REVISION"
fi

if [ ! -x "$PYTHON" ]; then
  echo "=== creating the Python environment ==="
  "$REPO/yolov10-validation/setup_venv.sh"
fi

echo "using ultralytics $(git -C "$ULTRALYTICS" describe --tags --always)"

run_iterations() {
  local n="$1" flag="$2"
  local start end
  start=$(date +%s%N)
  PYTHONPATH="$ULTRALYTICS:$REPO/yolov10-validation" \
    "$PYTHON" "$HERE/train_pytorch.py" --iterations "$n" $flag \
    > "$WORK/run.log" 2>&1 || { cat "$WORK/run.log"; exit 1; }
  end=$(date +%s%N)
  echo $(( (end - start) / 1000000 ))
}

for config in shipped benchmark; do
  if [ "$config" = shipped ]; then
    flag=""
    title="PyTorch (as ultralytics ships it)"
  else
    flag="--cudnn-benchmark"
    title="PyTorch (cudnn.benchmark=True)"
  fi

  echo
  echo "=== timing: $title ==="
  : > "$WORK/timings-$config.txt"
  for repeat in $(seq 1 $REPEATS); do
    warm_ms=$(run_iterations "$WARMUP" "$flag")
    full_ms=$(run_iterations $(( WARMUP + ITERATIONS )) "$flag")
    echo "$(( full_ms - warm_ms ))" >> "$WORK/timings-$config.txt"
    printf 'run %d:  %d iterations %.2f s  |  %d iterations %.2f s  |  difference (%d iterations) %.2f s\n' \
      "$repeat" \
      "$WARMUP" "$(echo "$warm_ms" | awk '{print $1/1000}')" \
      $(( WARMUP + ITERATIONS )) "$(echo "$full_ms" | awk '{print $1/1000}')" \
      "$ITERATIONS" "$(echo "$(( full_ms - warm_ms ))" | awk '{print $1/1000}')"
  done

  python3 "$HERE/report.py" --label "$title" \
    --iterations "$ITERATIONS" --timings "$WORK/timings-$config.txt"
done
