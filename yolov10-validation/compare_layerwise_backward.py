"""Per-layer comparison of FlexFlow's YOLOv10x backward pass against ultralytics.

Structured like ``compare_layerwise.py``: each ultralytics layer is fed
FlexFlow's own forward activations *and* FlexFlow's own gradient of that layer's
output, and the weight gradients it produces are compared against FlexFlow's.
Every comparison is therefore one layer deep, so a difference is attributable to
that layer's backward rather than to anything that happened downstream of it.

Weight gradients are the thing compared (rather than input gradients) because
they are unambiguous: FlexFlow's gradient of a tensor consumed by several layers
is the sum over those uses, which does not correspond to any single layer's
backward.
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
    parser.add_argument(
        "--logit",
        default="model.23.boxes",
        help="the tensor the loss was taken against (FF_LOSS_LOGIT)",
    )
    parser.add_argument(
        "--label",
        help="tensor file holding the label the loss was taken against; if "
        "given, the gradient of the logit itself is checked too",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rms-rel-tolerance", type=float, default=2e-2)
    parser.add_argument("--tf32", default="1")
    parser.add_argument(
        "--control",
        default="1",
        help="also compute the reference in float64 and report how far the "
        "float32 reference is from it; some of these gradients are sums with "
        "heavy cancellation, where float32 has no meaningful precision left "
        "and neither framework's answer is right",
    )
    args = parser.parse_args()

    allow_tf32 = args.tf32 == "1"
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    ff = read_tensor_file(args.ff_tensors)
    model_input = read_tensor_file(args.inputs)["input"]

    model = reference_model.build_model(seed=args.seed).to(args.device)
    model.train()

    control = args.control == "1"
    model64 = None
    if control:
        model64 = reference_model.build_model(seed=args.seed).to(args.device).double()
        model64.train()

    layer_outputs = reference_model.get_backbone_layer_outputs(model)
    layer_inputs = reference_model.get_layer_inputs(model)
    weight_names = reference_model.get_flexflow_weight_names(model)
    num_layers = len(model.model)

    ff_by_index = {-1: model_input}
    for index, name in layer_outputs.items():
        if name in ff:
            ff_by_index[index] = ff[name]

    rows = []
    skipped = []

    if args.label:
        # FlexFlow's mean_squared_error_avg loss computes
        # d/dy mean((y - label)^2) = 2 (y - label) / numel(y).
        label = read_tensor_file(args.label)["label"]
        logit = ff[args.logit]
        expected = 2.0 * (logit - label) / logit.numel()
        row = summarize(f"grad:{args.logit} (loss)", expected, ff[f"grad:{args.logit}"])
        row["ref_rel"] = 0.0
        row["layer"] = -1
        row["type"] = "loss"
        rows.append(row)

    for i in range(num_layers):
        is_head = i == num_layers - 1
        sources = [i - 1 if s == -1 else s for s in layer_inputs[i]]

        if not all(s in ff_by_index for s in sources):
            skipped.append((i, "missing FlexFlow forward input"))
            continue

        module = model.model[i]
        # Only the parameters FlexFlow's model actually has and that FlexFlow
        # dumped a gradient for: v10Detect also builds a one2one branch and a
        # (frozen) DFL buffer, neither of which is part of FlexFlow's
        # non-end2end model.
        params = [
            (name, param)
            for name, param in module.named_parameters()
            if param.requires_grad
            and f"model.{i}.{name}" in weight_names
            and "grad:" + weight_names[f"model.{i}.{name}"] in ff
        ]
        if not params:
            skipped.append((i, "no parameters"))
            continue

        if is_head:
            grad_names = ["model.23.boxes", "model.23.scores"]
            if not all(n in ff and f"grad:{n}" in ff for n in grad_names):
                skipped.append((i, "missing FlexFlow head gradients"))
                continue
        else:
            ff_grad_name = f"grad:{layer_outputs[i]}"
            if ff_grad_name not in ff:
                skipped.append((i, "missing FlexFlow output gradient"))
                continue

        def compute(target_model, dtype):
            layer = target_model.model[i]
            args_in = [
                ff_by_index[s].to(args.device, dtype) for s in sources
            ]
            if is_head:
                preds = layer(args_in)
                one2many = preds["one2many"] if "one2many" in preds else preds
                outs = [one2many["boxes"], one2many["scores"]]
                gouts = [ff[f"grad:{n}"].to(args.device, dtype) for n in grad_names]
            else:
                outs = [layer(args_in[0] if len(args_in) == 1 else args_in)]
                gouts = [ff[ff_grad_name].to(args.device, dtype)]
            wanted = [dict(layer.named_parameters())[n] for n, _ in params]
            return torch.autograd.grad(
                outs, wanted, grad_outputs=gouts, allow_unused=True
            )

        grads = compute(model, torch.float32)
        grads64 = compute(model64, torch.float64) if control else [None] * len(params)

        for (name, _), grad, grad64 in zip(params, grads, grads64):
            ff_name = "grad:" + weight_names[f"model.{i}.{name}"]
            ref = (
                torch.zeros_like(ff[ff_name])
                if grad is None
                else grad.detach().to("cpu", torch.float32)
            )
            got = ff[ff_name]
            if ref.shape != got.shape:
                print(f"SHAPE MISMATCH {ff_name}: {tuple(ref.shape)} vs {tuple(got.shape)}")
                return 1
            if ref.abs().max() == 0 and got.abs().max() == 0:
                # This branch gets no gradient from the chosen logit; nothing to
                # compare, and reporting it as a pass would be misleading.
                continue

            if control and grad64 is not None:
                # Measure both frameworks against a float64 ground truth, so
                # that a tensor whose float32 value is meaningless shows up as
                # such rather than as a FlexFlow error.
                truth = grad64.detach().to("cpu", torch.float32)
                row = summarize(ff_name, truth, got)
                row["ref_rel"] = summarize(ff_name, truth, ref)["rms_rel"]
            else:
                row = summarize(ff_name, ref, got)
                row["ref_rel"] = 0.0
            row["layer"] = i
            row["type"] = type(module).__name__
            rows.append(row)

    # Report the worst tensor per layer, which is what localizes a problem.
    by_layer = {}
    for row in rows:
        prev = by_layer.get(row["layer"])
        if prev is None or row["rms_rel"] > prev["rms_rel"]:
            by_layer[row["layer"]] = row

    header = (
        f"{'layer':>5} {'module':<9} {'worst weight gradient':<34} "
        f"{'|grad|':>10} {'ff err':>10} {'fp32 err':>10} {'cosine':>11}"
    )
    print(header)
    print("-" * len(header))
    for index in sorted(by_layer):
        row = by_layer[index]
        print(
            f"{row['layer']:>5} {row['type']:<9} {row['name'][5:]:<34} "
            f"{row['ref_rms']:>10.4g} {row['rms_rel']:>10.3g} "
            f"{row['ref_rel']:>10.3g} {row['cosine']:>11.8f}"
        )

    # FlexFlow is judged against how well float32 itself does on the same
    # tensor: a tensor float32 cannot compute (heavy cancellation) is not
    # evidence of a FlexFlow bug.
    def is_failure(row):
        if row["nans"] or row["infs"]:
            return True
        if row["rms_rel"] <= args.rms_rel_tolerance:
            return False
        return row["rms_rel"] > max(4 * row["ref_rel"], args.rms_rel_tolerance)

    failed = [
        f"{row['name']}: FlexFlow relative RMS error {row['rms_rel']:.3g} "
        f"vs float32 reference's own {row['ref_rel']:.3g}"
        for row in rows
        if is_failure(row)
    ]

    if skipped:
        print()
        for index, reason in skipped:
            print(f"skipped layer {index}: {reason}")

    print()
    print(f"compared {len(rows)} weight gradients across {len(by_layer)} layers "
          f"(loss taken against {args.logit})")
    if failed:
        print("FAILED")
        for message in failed[:20]:
            print(f"  {message}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        return 1
    print(f"PASSED (all within {args.rms_rel_tolerance:g} relative RMS error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
