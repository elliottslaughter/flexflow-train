"""Validate FlexFlow's optimizer against torch.optim.SGD.

FlexFlow is run for several iterations, dumping every weight and every weight
gradient at the end of each one. This script then replays FlexFlow's *own*
gradients through ``torch.optim.SGD`` and checks that the weights it arrives at
are FlexFlow's, step by step. Feeding PyTorch FlexFlow's gradients is what makes
this a test of the optimizer alone: it does not depend on the two frameworks
agreeing on the gradients, which ``compare_layerwise_backward.py`` covers
separately.

What is compared is the *step* (the change in the weight), not the weight
itself: with lr=0.001 the weights barely move, so comparing them directly would
mostly be comparing a number to itself.
"""

import argparse
import sys

import torch

from ff_tensor_file import read_tensor_file
import reference_model
from compare import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True, help="tensor file with the starting weights")
    parser.add_argument(
        "--iteration",
        action="append",
        required=True,
        help="per-iteration dump (weights after the step, gradients used by it); "
        "repeat in order",
    )
    # These have to match the OptimizerAttrs run-model builds.
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--rms-rel-tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    initial = read_tensor_file(args.initial)
    iterations = [read_tensor_file(path) for path in args.iteration]

    model = reference_model.build_model()
    weight_names = reference_model.get_flexflow_weight_names(model)

    # Every weight FlexFlow dumped a gradient for, in a fixed order.
    names = sorted(
        ff_name
        for ff_name in weight_names.values()
        if ff_name in initial and f"grad:{ff_name}" in iterations[0]
    )
    if not names:
        print("error: no weights in common", file=sys.stderr)
        return 1

    params = [initial[name].clone().requires_grad_(True) for name in names]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        dampening=0.0,
    )

    rows = []
    for step, dump in enumerate(iterations):
        previous = [p.detach().clone() for p in params]
        for param, name in zip(params, names):
            param.grad = dump[f"grad:{name}"].clone()
        optimizer.step()

        for param, before, name in zip(params, previous, names):
            ff_step = dump[name] - before
            ref_step = param.detach() - before
            if ref_step.abs().max() == 0 and ff_step.abs().max() == 0:
                continue
            row = summarize(name, ref_step, ff_step)
            row["step"] = step
            rows.append(row)

    by_step = {}
    for row in rows:
        prev = by_step.get(row["step"])
        if prev is None or row["rms_rel"] > prev["rms_rel"]:
            by_step[row["step"]] = row

    header = (
        f"{'step':>5} {'worst weight update':<34} {'ref rms':>11} "
        f"{'ff rms':>11} {'rel rms':>10} {'cosine':>11}"
    )
    print(header)
    print("-" * len(header))
    for step in sorted(by_step):
        row = by_step[step]
        print(
            f"{row['step']:>5} {row['name']:<34} {row['ref_rms']:>11.4g} "
            f"{row['got_rms']:>11.4g} {row['rms_rel']:>10.3g} {row['cosine']:>11.8f}"
        )

    failed = [
        f"step {row['step']} {row['name']}: relative RMS error {row['rms_rel']:.3g}"
        for row in rows
        if row["nans"] or row["infs"] or row["rms_rel"] > args.rms_rel_tolerance
    ]

    print()
    print(
        f"compared {len(rows)} weight updates over {len(iterations)} steps "
        f"(lr={args.lr}, momentum={args.momentum}, weight_decay={args.weight_decay}, "
        f"nesterov={args.nesterov})"
    )
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
