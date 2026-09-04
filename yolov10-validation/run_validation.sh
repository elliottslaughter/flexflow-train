#!/usr/bin/env bash
#
# Validate FlexFlow's YOLOv10x forward pass against upstream ultralytics.
#
# Usage (from anywhere):
#   yolov10-validation/run_validation.sh [workdir]
#
# The script builds FlexFlow, exports and compiles the YOLOv10x graph, generates
# a random input batch plus a full set of weights with ultralytics, runs both
# frameworks on that same data, and reports the difference both per-layer and
# end-to-end.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
WORKDIR="${1:-$HERE/work}"
ULTRALYTICS="${ULTRALYTICS:-/home/eslaught/flexflow/ultralytics}"
PYTHON="${PYTHON:-$HERE/.venv/bin/python}"
NIX="${NIX:-nix}"
# Realm aborts allocating instances below ~26000, and above ~28000 there is not
# enough left outside its pool for CUDA's own resources once the update pass is
# running. See the README.
FSIZE="${FSIZE:-27000}"

mkdir -p "$WORKDIR"

# Run a command inside the project's gpu nix devshell.
ffdev() {
  ( cd "$REPO" && NIXPKGS_ALLOW_UNFREE=1 "$NIX" develop .#gpu --accept-flake-config --impure --command bash -c "$*" )
}

echo "=== building FlexFlow ==="
ffdev 'proj build --release'

echo "=== exporting and compiling the YOLOv10x graph ==="
ffdev "./build/release/bin/export-model-arch/export-model-arch yolov10x > '$WORKDIR/yolov10x_cg.json'"
ffdev "./build/release/bin/compile-model/compile-model '$WORKDIR/yolov10x_cg.json' '$WORKDIR/yolov10x_mpcg.json' passthrough"

echo "=== generating inputs and the ultralytics reference ==="
PYTHONPATH="$ULTRALYTICS:$HERE" "$PYTHON" "$HERE/export_reference.py" \
  --cg-json "$WORKDIR/yolov10x_cg.json" \
  --inputs "$WORKDIR/ff_inputs.bin" \
  --reference "$WORKDIR/reference.bin"

for logit in boxes scores; do
  PYTHONPATH="$HERE" "$PYTHON" "$HERE/make_label.py" \
    --out "$WORKDIR/label_$logit.bin" --logit "model.23.$logit"
done

# The output of every backbone layer, plus the two detection head outputs.  See
# reference_model.get_backbone_layer_outputs() for where these names come from.
NAMES=$(PYTHONPATH="$ULTRALYTICS:$HERE" "$PYTHON" - <<'PY'
import reference_model
model = reference_model.build_model()
names = list(reference_model.get_backbone_layer_outputs(model).values())
names += ["model.23.boxes", "model.23.scores"]
print(",".join(names))
PY
)

echo "=== running FlexFlow on the same input and weights ==="
ffdev "REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize $FSIZE -cuda:dynfb 0' \
  FF_LOAD_TENSORS='$WORKDIR/ff_inputs.bin' \
  FF_DUMP_TENSORS='$WORKDIR/ff_tensors.bin' \
  FF_DUMP_NAMES='$NAMES' \
  FF_FORWARD_ONLY=1 FF_ITERATIONS=1 \
  nixGL -- ./build/release/bin/run-model/run-model '$WORKDIR/yolov10x_mpcg.json'"

# ---------------------------------------------------------------------------
# Weight initialization
#
# Everything else here hands FlexFlow weights exported from PyTorch, so this is
# the one stage that exercises the weights FlexFlow starts with on its own.
# ---------------------------------------------------------------------------

echo
echo "=== dumping the weights FlexFlow initializes itself with ==="
WEIGHT_NAMES=$(PYTHONPATH="$ULTRALYTICS:$HERE" "$PYTHON" - "$WORKDIR/yolov10x_cg.json" <<'PY'
import sys
import reference_model
from export_reference import get_flexflow_weight_names

model = reference_model.build_model()
present = set(get_flexflow_weight_names(sys.argv[1]))
print(",".join(w for w in reference_model.get_flexflow_weight_names(model).values()
                if w in present))
PY
)

# Two identical runs, so that initialization can be checked for reproducibility.
for run in a b; do
  ffdev "REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize $FSIZE -cuda:dynfb 0' \
    FF_DUMP_TENSORS='$WORKDIR/ff_init_$run.bin' \
    FF_DUMP_NAMES='$WEIGHT_NAMES' \
    FF_ITERATIONS=0 \
    nixGL -- ./build/release/bin/run-model/run-model '$WORKDIR/yolov10x_mpcg.json'"
done

echo
echo "=== initialization comparison against PyTorch's reset_parameters ==="
PYTHONPATH="$ULTRALYTICS:$HERE" "$PYTHON" "$HERE/check_initialization.py" \
  --weights "$WORKDIR/ff_init_a.bin" \
  --repeat "$WORKDIR/ff_init_b.bin"

# Train from that initialization rather than from PyTorch's weights.  Only the
# input and the label are loaded; the weights are FlexFlow's own.
PYTHONPATH="$HERE" "$PYTHON" - "$WORKDIR/ff_inputs.bin" "$WORKDIR/input_only.bin" <<'PY'
import sys
from ff_tensor_file import read_tensor_file, write_tensor_file

write_tensor_file(sys.argv[2], {"input": read_tensor_file(sys.argv[1])["input"]})
PY

# These all have to lie upstream of the logit the loss is taken against, or they
# would legitimately have no gradient.  See check_training.py.
TRACKED='model.0.conv.FILTER,model.10.attn.pe.conv.FILTER,model.10.attn.qkv.conv.FILTER,model.0.bn.GAMMA,model.0.bn.BETA,model.22.cv1.conv.FILTER,model.23.cv2.0.2.FILTER,model.23.cv2.0.2.BIAS,model.23.cv2.2.0.conv.FILTER'
TRACKED_GRADS=$(echo "$TRACKED" | sed 's/\([^,]*\)/grad:\1/g')

TRAIN_STEPS=5
echo
echo "=== training $TRAIN_STEPS iterations from FlexFlow's own initialization ==="
ffdev "REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize $FSIZE -cuda:dynfb 0' \
  FF_LOAD_TENSORS='$WORKDIR/input_only.bin,$WORKDIR/label_boxes.bin' \
  FF_DUMP_TENSORS='$WORKDIR/ff_train_%.bin' \
  FF_DUMP_NAMES='model.23.boxes,label,$TRACKED,$TRACKED_GRADS' \
  FF_LOSS=mean_squared_error_avg FF_LOSS_LOGIT=model.23.boxes \
  FF_ITERATIONS=$TRAIN_STEPS \
  nixGL -- ./build/release/bin/run-model/run-model '$WORKDIR/yolov10x_mpcg.json'"

TRAIN_ARGS=()
for step in $(seq 0 $((TRAIN_STEPS - 1))); do
  TRAIN_ARGS+=(--iteration "$WORKDIR/ff_train_$step.bin")
done

echo
echo "=== training smoke test ==="
PYTHONPATH="$HERE" "$PYTHON" "$HERE/check_training.py" \
  --logit model.23.boxes "${TRAIN_ARGS[@]}"

echo
echo "=== per-layer comparison (each layer fed FlexFlow's own intermediates) ==="
PYTHONPATH="$ULTRALYTICS:$HERE" "$PYTHON" "$HERE/compare_layerwise.py" \
  --ff-tensors "$WORKDIR/ff_tensors.bin" \
  --inputs "$WORKDIR/ff_inputs.bin"

echo
echo "=== end-to-end comparison ==="
echo "Note: this network is chaotic enough that reduced-precision convolutions"
echo "alone move the final output by a few percent, so the tolerance here is"
echo "necessarily loose.  See the per-layer table above for the tight check."
PYTHONPATH="$HERE" "$PYTHON" "$HERE/compare.py" \
  --reference "$WORKDIR/reference.bin" \
  --actual "$WORKDIR/ff_tensors.bin" \
  --rms-rel-tolerance 0.05

# ---------------------------------------------------------------------------
# Backward pass
#
# The loss can only be attached to one tensor at a time, so the head is covered
# in two runs: the box head's weights only receive a gradient when the loss is
# taken against `boxes`, and the class head's only when it is taken against
# `scores`.
# ---------------------------------------------------------------------------

BWD_NAMES=$(PYTHONPATH="$ULTRALYTICS:$HERE" "$PYTHON" - "$WORKDIR/yolov10x_cg.json" <<'PY'
import sys
import reference_model
from export_reference import get_flexflow_weight_names

model = reference_model.build_model()
present = set(get_flexflow_weight_names(sys.argv[1]))
spine = list(reference_model.get_backbone_layer_outputs(model).values())
names = list(spine) + ["grad:" + s for s in spine]
names += ["model.23.boxes", "model.23.scores",
          "grad:model.23.boxes", "grad:model.23.scores"]
names += ["grad:" + w
          for w in reference_model.get_flexflow_weight_names(model).values()
          if w in present]
print(",".join(names))
PY
)

for logit in boxes scores; do
  echo
  echo "=== running FlexFlow's backward pass (loss against model.23.$logit) ==="
  ffdev "REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize $FSIZE -cuda:dynfb 0' \
    FF_LOAD_TENSORS='$WORKDIR/ff_inputs.bin,$WORKDIR/label_$logit.bin' \
    FF_DUMP_TENSORS='$WORKDIR/ff_bwd_$logit.bin' \
    FF_DUMP_NAMES='$BWD_NAMES' \
    FF_LOSS=mean_squared_error_avg FF_LOSS_LOGIT='model.23.$logit' \
    FF_ITERATIONS=1 \
    nixGL -- ./build/release/bin/run-model/run-model '$WORKDIR/yolov10x_mpcg.json'"

  echo
  echo "=== per-layer backward comparison (loss against model.23.$logit) ==="
  PYTHONPATH="$ULTRALYTICS:$HERE" "$PYTHON" "$HERE/compare_layerwise_backward.py" \
    --ff-tensors "$WORKDIR/ff_bwd_$logit.bin" \
    --inputs "$WORKDIR/ff_inputs.bin" \
    --label "$WORKDIR/label_$logit.bin" \
    --logit "model.23.$logit" || true
done

# ---------------------------------------------------------------------------
# Update pass
#
# Run several training iterations, dumping every weight and weight gradient at
# the end of each, then replay FlexFlow's own gradients through torch.optim.SGD
# and check it arrives at the same weights.
# ---------------------------------------------------------------------------

OPT_NAMES=$(PYTHONPATH="$ULTRALYTICS:$HERE" "$PYTHON" - "$WORKDIR/yolov10x_cg.json" <<'PY'
import sys
import reference_model
from export_reference import get_flexflow_weight_names

model = reference_model.build_model()
present = set(get_flexflow_weight_names(sys.argv[1]))
weights = [w for w in reference_model.get_flexflow_weight_names(model).values()
           if w in present]
print(",".join(weights + ["grad:" + w for w in weights]))
PY
)

STEPS=5
echo
echo "=== running FlexFlow for $STEPS training iterations ==="
ffdev "REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize $FSIZE -cuda:dynfb 0' \
  FF_LOAD_TENSORS='$WORKDIR/ff_inputs.bin,$WORKDIR/label_boxes.bin' \
  FF_DUMP_TENSORS='$WORKDIR/ff_opt_%.bin' \
  FF_DUMP_NAMES='$OPT_NAMES' \
  FF_LOSS=mean_squared_error_avg FF_LOSS_LOGIT=model.23.boxes \
  FF_ITERATIONS=$STEPS \
  nixGL -- ./build/release/bin/run-model/run-model '$WORKDIR/yolov10x_mpcg.json'"

ITER_ARGS=()
for step in $(seq 0 $((STEPS - 1))); do
  ITER_ARGS+=(--iteration "$WORKDIR/ff_opt_$step.bin")
done

echo
echo "=== optimizer comparison against torch.optim.SGD ==="
PYTHONPATH="$ULTRALYTICS:$HERE" "$PYTHON" "$HERE/compare_optimizer.py" \
  --initial "$WORKDIR/ff_inputs.bin" "${ITER_ARGS[@]}"
