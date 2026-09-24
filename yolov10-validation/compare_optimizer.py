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

Even the step has a precision floor, though. It is the difference of two
float32 weights, so it can be no more precise than a rounding step at the
weight's magnitude: a batch norm scale near 1.0 moving by 2e-6 per step is
resolved only to about 3%. Two correct float32 optimizers can then disagree by
one rounding in one element, which is enough to exceed the tolerance. So the
same steps are also replayed in float64, and a tensor whose FlexFlow step is
outside the tolerance only fails if FlexFlow is further from the float64 answer
than float32 PyTorch is -- the same rule compare_layerwise_backward.py uses.
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

    def make_optimizer(dtype):
        params = [
            initial[name].clone().to(dtype).requires_grad_(True) for name in names
        ]
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            dampening=0.0,
        )
        return params, optimizer

    params, optimizer = make_optimizer(torch.float32)
    params64, optimizer64 = make_optimizer(torch.float64)

    rows = []
    for step, dump in enumerate(iterations):
        previous = [p.detach().clone() for p in params]
        previous64 = [p.detach().clone() for p in params64]
        for param, param64, name in zip(params, params64, names):
            param.grad = dump[f"grad:{name}"].clone()
            param64.grad = dump[f"grad:{name}"].to(torch.float64)
        optimizer.step()
        optimizer64.step()

        ff_previous = initial if step == 0 else iterations[step - 1]
        for param, before, param64, before64, name in zip(
            params, previous, params64, previous64, names
        ):
            ff_step = dump[name] - before
            ref_step = param.detach() - before
            if ref_step.abs().max() == 0 and ff_step.abs().max() == 0:
                continue
            row = summarize(name, ref_step, ff_step)
            row["step"] = step
            # Both float32 steps against the float64 one, each measured along
            # its own trajectory.
            exact = param64.detach() - before64
            row["ff_exact_rel"] = summarize(
                name, exact, dump[name] - ff_previous[name]
            )["rms_rel"]
            row["ref_exact_rel"] = summarize(name, exact, ref_step)["rms_rel"]
            rows.append(row)

    by_step = {}
    for row in rows:
        prev = by_step.get(row["step"])
        if prev is None or row["rms_rel"] > prev["rms_rel"]:
            by_step[row["step"]] = row

    header = (
        f"{'step':>5} {'worst weight update':<34} {'ref rms':>11} "
        f"{'ff rms':>11} {'rel rms':>10} {'fp32 err':>10} {'cosine':>11}"
    )
    print(header)
    print("-" * len(header))
    for step in sorted(by_step):
        row = by_step[step]
        print(
            f"{row['step']:>5} {row['name']:<34} {row['ref_rms']:>11.4g} "
            f"{row['got_rms']:>11.4g} {row['rms_rel']:>10.3g} "
            f"{row['ref_exact_rel']:>10.3g} {row['cosine']:>11.8f}"
        )

    def is_failure(row):
        if row["nans"] or row["infs"]:
            return True
        if row["rms_rel"] <= args.rms_rel_tolerance:
            return False
        return row["ff_exact_rel"] > max(
            4 * row["ref_exact_rel"], args.rms_rel_tolerance
        )

    failed = [
        f"step {row['step']} {row['name']}: relative RMS error {row['rms_rel']:.3g} "
        f"(float64 error: FlexFlow {row['ff_exact_rel']:.3g}, "
        f"float32 PyTorch {row['ref_exact_rel']:.3g})"
        for row in rows
        if is_failure(row)
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
    print(
        f"PASSED (all within {args.rms_rel_tolerance:g} relative RMS error, or "
        f"within 4x float32 PyTorch's own error where float32 cannot resolve "
        f"the step)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
