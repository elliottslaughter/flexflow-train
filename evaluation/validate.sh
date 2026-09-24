#!/usr/bin/env bash
#
# Check FlexFlow's YOLOv10 numerics against ultralytics, in four stages:
#
#   forward   every layer's output, each fed FlexFlow's own intermediates
#   backward  every layer's output gradient and every weight gradient
#   update    five SGD steps replayed through torch.optim.SGD
#   overall   the network's final output, end to end
#
# Each stage prints its own PASSED or FAILED line and the worst tensor it saw.
# The script exits non-zero if any stage fails. The model and batch size come
# from config.sh (YOLOv10x at batch size 6 unless overridden).
#
# Takes about 30 minutes after the build. No arguments, no setup.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
HARNESS="$REPO/yolov10-validation"

source "$HERE/config.sh"
WORK="$HERE/work/validation/$CONFIG"

VENV="$HARNESS/.venv"
PYTHON="$VENV/bin/python"
ULTRALYTICS="${ULTRALYTICS:-/home/eslaught/flexflow/ultralytics}"
ULTRALYTICS_REVISION=94e9819c536e2f8a543a5f9b5c5629cdae9b9658

UPDATE_STEPS=5

# Convolution algorithms chosen by cuDNN's heuristics rather than by measuring
# them. Measured selection picks kernels that accumulate with atomics, so two
# runs of the same program disagree in the low bits and the update comparison
# below fails about one run in five. Throughput is measured separately, with
# measurement left on; see benchmark_flexflow.sh.
export FF_CUDNN_BENCHMARK=0

source "$HERE/build_env.sh"

pyrun() {
  PYTHONPATH="$ULTRALYTICS:$HARNESS" "$PYTHON" "$@"
}

mkdir -p "$WORK"
FAILURES=0

if [ ! -d "$ULTRALYTICS" ]; then
  echo "=== fetching ultralytics ==="
  git clone https://github.com/ultralytics/ultralytics.git "$ULTRALYTICS"
  git -C "$ULTRALYTICS" checkout --quiet "$ULTRALYTICS_REVISION"
fi

if [ ! -x "$PYTHON" ]; then
  echo "=== creating the Python environment ==="
  "$HARNESS/setup_venv.sh"
fi

echo "=== building FlexFlow ($FF_BUILD_KIND) ==="
ffbuild

echo "=== compiling the $MODEL graph at batch size $BATCH ==="
ffrun "$BIN/export-model-arch/export-model-arch --batch-size $BATCH $MODEL > '$WORK/cg.json'"
ffrun "$BIN/compile-model/compile-model '$WORK/cg.json' '$WORK/mpcg.json' passthrough"

echo "=== exporting weights, an input batch and ultralytics' reference output ==="
pyrun "$HARNESS/export_reference.py" \
  --cg-json "$WORK/cg.json" \
  --batch-size "$BATCH" \
  --inputs "$WORK/inputs.bin" \
  --reference "$WORK/reference.bin"

for logit in boxes scores; do
  pyrun "$HARNESS/make_label.py" --batch-size "$BATCH" \
    --out "$WORK/label_$logit.bin" --logit "model.23.$logit"
done

names() {
  pyrun - "$WORK/cg.json"
}

SPINE=$(names <<'PY'
import reference_model
model = reference_model.build_model()
names = list(reference_model.get_backbone_layer_outputs(model).values())
print(",".join(names + ["model.23.boxes", "model.23.scores"]))
PY
)

WEIGHTS=$(names <<'PY'
import sys, reference_model
from export_reference import get_flexflow_weight_names
model = reference_model.build_model()
present = set(get_flexflow_weight_names(sys.argv[1]))
print(",".join(w for w in reference_model.get_flexflow_weight_names(model).values()
                if w in present))
PY
)

stage() {
  echo
  echo "############################################################"
  echo "# $1"
  echo "############################################################"
}

# ---------------------------------------------------------------------------
stage "forward: every layer"

ffrun "REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize $FSIZE -cuda:dynfb 0' \
  FF_LOAD_TENSORS='$WORK/inputs.bin' \
  FF_DUMP_TENSORS='$WORK/forward.bin' FF_DUMP_NAMES='$SPINE' \
  FF_FORWARD_ONLY=1 FF_ITERATIONS=1 \
  $GL $BIN/run-model/run-model '$WORK/mpcg.json'"

pyrun "$HARNESS/compare_layerwise.py" \
  --ff-tensors "$WORK/forward.bin" --inputs "$WORK/inputs.bin" || FAILURES=$((FAILURES+1))

# ---------------------------------------------------------------------------
stage "overall: the network end to end"

# Looser than the per-layer check on purpose: this network amplifies small
# differences enough that reduced-precision convolutions alone move the final
# output by a few percent.
pyrun "$HARNESS/compare.py" \
  --reference "$WORK/reference.bin" --actual "$WORK/forward.bin" \
  --rms-rel-tolerance 0.05 || FAILURES=$((FAILURES+1))

# ---------------------------------------------------------------------------
stage "backward: every layer's gradient"

BWD=$(names <<'PY'
import sys, reference_model
from export_reference import get_flexflow_weight_names
model = reference_model.build_model()
present = set(get_flexflow_weight_names(sys.argv[1]))
spine = list(reference_model.get_backbone_layer_outputs(model).values())
names = spine + ["grad:" + s for s in spine]
names += ["model.23.boxes", "model.23.scores",
          "grad:model.23.boxes", "grad:model.23.scores"]
names += ["grad:" + w for w in reference_model.get_flexflow_weight_names(model).values()
          if w in present]
print(",".join(names))
PY
)

# Two runs: a loss attaches to one tensor at a time, and the box and class heads
# only receive a gradient when it is attached to theirs.
for logit in boxes scores; do
  echo
  echo "--- loss against model.23.$logit ---"
  ffrun "REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize $FSIZE -cuda:dynfb 0' \
      FF_LOAD_TENSORS='$WORK/inputs.bin,$WORK/label_$logit.bin' \
    FF_DUMP_TENSORS='$WORK/backward_$logit.bin' FF_DUMP_NAMES='$BWD' \
    FF_LOSS=mean_squared_error_avg FF_LOSS_LOGIT='model.23.$logit' \
    FF_ITERATIONS=1 \
    $GL $BIN/run-model/run-model '$WORK/mpcg.json'"

  pyrun "$HARNESS/compare_layerwise_backward.py" \
    --ff-tensors "$WORK/backward_$logit.bin" --inputs "$WORK/inputs.bin" \
    --label "$WORK/label_$logit.bin" --logit "model.23.$logit" \
    || FAILURES=$((FAILURES+1))
done

# ---------------------------------------------------------------------------
stage "update: $UPDATE_STEPS SGD steps"

WEIGHT_GRADS=$(echo "$WEIGHTS" | sed 's/\([^,]*\)/grad:\1/g')

ffrun "REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize $FSIZE -cuda:dynfb 0' \
  FF_LOAD_TENSORS='$WORK/inputs.bin,$WORK/label_boxes.bin' \
  FF_DUMP_TENSORS='$WORK/update_%.bin' \
  FF_DUMP_NAMES='$WEIGHTS,$WEIGHT_GRADS' \
  FF_LOSS=mean_squared_error_avg FF_LOSS_LOGIT=model.23.boxes \
  FF_ITERATIONS=$UPDATE_STEPS \
  $GL $BIN/run-model/run-model '$WORK/mpcg.json'"

STEP_ARGS=()
for step in $(seq 0 $((UPDATE_STEPS - 1))); do
  STEP_ARGS+=(--iteration "$WORK/update_$step.bin")
done

echo
pyrun "$HARNESS/compare_optimizer.py" \
  --initial "$WORK/inputs.bin" "${STEP_ARGS[@]}" || FAILURES=$((FAILURES+1))

# ---------------------------------------------------------------------------
echo
echo "############################################################"
if [ "$FAILURES" -eq 0 ]; then
  echo "# all stages PASSED"
else
  echo "# $FAILURES stage(s) FAILED"
fi
echo "############################################################"
exit "$FAILURES"
