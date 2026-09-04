"""Per-layer comparison of FlexFlow's YOLOv10x against ultralytics.

An end-to-end comparison of the two frameworks is not very discriminating: this
network is chaotic enough that simply toggling TF32 in PyTorch changes the final
output by a few percent, so a few percent of end-to-end disagreement says
nothing about whether FlexFlow computes the right function.

This script instead compares one layer at a time *without accumulating error*:
each ultralytics layer is fed FlexFlow's own intermediate tensors and its output
is compared against FlexFlow's output for that same layer.  Each comparison is
then only one layer deep, so agreement should be at the level of floating-point
noise (~1e-3 with reduced-precision convolutions), and a structural difference
shows up as an unmistakable O(1) error at a specific layer.
"""

import argparse
import sys

import torch

from ff_tensor_file import read_tensor_file
import reference_model
from compare import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ff-tensors", required=True, help="tensor file dumped by run-model")
    parser.add_argument("--inputs", required=True, help="tensor file fed to run-model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rms-rel-tolerance", type=float, default=5e-3)
    parser.add_argument("--tf32", default="1")
    args = parser.parse_args()

    allow_tf32 = args.tf32 == "1"
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    ff = read_tensor_file(args.ff_tensors)
    model_input = read_tensor_file(args.inputs)["input"]

    model = reference_model.build_model(seed=args.seed).to(args.device)
    model.train()

    layer_outputs = reference_model.get_backbone_layer_outputs(model)
    layer_inputs = reference_model.get_layer_inputs(model)
    num_layers = len(model.model)

    # FlexFlow's value for the output of each layer (index -1 is the model input).
    ff_by_index = {-1: model_input}
    for index, name in layer_outputs.items():
        if name in ff:
            ff_by_index[index] = ff[name]

    rows = []
    skipped = []
    with torch.no_grad():
        for i in range(num_layers):
            is_head = i == num_layers - 1
            sources = layer_inputs[i]
            sources = [i - 1 if s == -1 else s for s in sources]

            if not all(s in ff_by_index for s in sources):
                skipped.append((i, "missing FlexFlow input"))
                continue
            if not is_head and i not in ff_by_index:
                skipped.append((i, "missing FlexFlow output"))
                continue

            args_in = [ff_by_index[s].to(args.device) for s in sources]
            module = model.model[i]

            if is_head:
                preds = module(args_in)
                one2many = preds["one2many"] if "one2many" in preds else preds
                produced = {
                    "model.23.boxes": one2many["boxes"],
                    "model.23.scores": one2many["scores"],
                }
            else:
                # Layers with a single source take a tensor; the rest (Concat)
                # take the list, exactly as ultralytics' _predict_once does.
                out = module(args_in[0] if len(args_in) == 1 else args_in)
                produced = {layer_outputs[i]: out}

            for name, out in produced.items():
                if name not in ff:
                    skipped.append((i, f"{name} not dumped by FlexFlow"))
                    continue
                got = ff[name]
                ref = out.detach().to("cpu", torch.float32)
                if ref.shape != got.shape:
                    print(
                        f"layer {i:>2} {name}: SHAPE MISMATCH "
                        f"ultralytics {tuple(ref.shape)} vs FlexFlow {tuple(got.shape)}"
                    )
                    return 1
                row = summarize(name, ref, got)
                row["layer"] = i
                row["type"] = type(module).__name__
                rows.append(row)

    header = (
        f"{'layer':>5} {'module':<9} {'tensor':<24} {'ref rms':>10} "
        f"{'ff rms':>10} {'max abs':>10} {'rel rms':>10} {'cosine':>11}"
    )
    print(header)
    print("-" * len(header))
    failed = []
    for row in rows:
        print(
            f"{row['layer']:>5} {row['type']:<9} {row['name']:<24} "
            f"{row['ref_rms']:>10.4g} {row['got_rms']:>10.4g} {row['max_abs']:>10.3g} "
            f"{row['rms_rel']:>10.3g} {row['cosine']:>11.8f}"
        )
        if row["nans"] or row["infs"]:
            failed.append(f"layer {row['layer']} {row['name']}: {row['nans']} NaNs, {row['infs']} infs")
        elif row["rms_rel"] > args.rms_rel_tolerance:
            failed.append(
                f"layer {row['layer']} {row['name']}: relative RMS error "
                f"{row['rms_rel']:.3g} exceeds tolerance {args.rms_rel_tolerance:g}"
            )

    if skipped:
        print()
        for index, reason in skipped:
            print(f"skipped layer {index}: {reason}")

    print()
    if failed:
        print("FAILED")
        for message in failed:
            print(f"  {message}")
        return 1
    print(f"PASSED ({len(rows)} layers within {args.rms_rel_tolerance:g} relative RMS error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
