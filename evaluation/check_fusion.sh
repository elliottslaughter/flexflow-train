#!/usr/bin/env bash
#
# Check that the compiler finds the operator fusions FlexFlow otherwise makes
# at execution time.
#
# run-model folds every batch norm into the activation that follows it
# (perform_operation_fusion, lib/task-spec/.../operation_fusion.cc). This runs
# compile-model's unity search over the model instead, with operator costs
# measured on this machine's GPU by LocalCostEstimator, and checks that it
# chooses every one of those fusions on its own. YOLOv10x has 159, all batch
# norm + SiLU.
#
# The model and batch size come from config.sh (YOLOv10x at batch size 6 unless
# overridden). The 159 is only checked for YOLOv10x; other models are checked
# against however many pairs they have.
#
# Takes about a minute after the build. No arguments, no setup.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

source "$HERE/config.sh"
WORK="$HERE/work/fusion/$CONFIG"

source "$HERE/build_env.sh"

mkdir -p "$WORK"

EXPECTED=()
if [ "$MODEL" = yolov10x ]; then
  EXPECTED=(--expected-silu 159)
fi

echo "=== building FlexFlow ($FF_BUILD_KIND) ==="
ffbuild

echo "=== exporting the $MODEL graph at batch size $BATCH ==="
ffrun "$BIN/export-model-arch/export-model-arch --batch-size $BATCH $MODEL > '$WORK/cg.json'"

# One GPU, which is all unity searches over. The bandwidths are never used,
# since nothing moves between devices, but the format requires them.
cat > "$WORK/machine.json" <<'EOF'
{
  "num_nodes": 1,
  "num_cpus_per_node": 1,
  "num_gpus_per_node": 1,
  "inter_node_bandwidth_bytes_per_second": 1e10,
  "intra_node_bandwidth_bytes_per_second": 1e11
}
EOF

echo "=== compiling with unity, measuring operator costs on the GPU ==="
ffrun "$GL $BIN/compile-model/compile-model '$WORK/cg.json' '$WORK/mpcg.json' unity \
  --machine-spec-json '$WORK/machine.json'"

echo
python3 "$HERE/check_fusion.py" \
  --cg "$WORK/cg.json" --mpcg "$WORK/mpcg.json" "${EXPECTED[@]}"
