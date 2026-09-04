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
ffdev "REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize 28500 -cuda:dynfb 0' \
  FF_LOAD_TENSORS='$WORKDIR/ff_inputs.bin' \
  FF_DUMP_TENSORS='$WORKDIR/ff_tensors.bin' \
  FF_DUMP_NAMES='$NAMES' \
  FF_FORWARD_ONLY=1 FF_ITERATIONS=1 \
  nixGL -- ./build/release/bin/run-model/run-model '$WORKDIR/yolov10x_mpcg.json'"

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
