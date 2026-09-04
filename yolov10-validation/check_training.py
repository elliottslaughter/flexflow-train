"""Check that a model trains from FlexFlow's own weight initialization.

Every other script here hands FlexFlow a set of weights exported from PyTorch,
so none of them exercise the weights FlexFlow starts with on its own. This one
runs the model from its own initialization and checks the things that go wrong
when an initialization is bad:

* the forward pass stays finite -- badly scaled weights blow a network this deep
  up long before the output;
* the output neither collapses towards zero nor grows without bound as training
  proceeds;
* every weight gradient is finite and non-zero, i.e. the network is not born
  dead or saturated (which is why the tracked weights all have to lie upstream
  of the chosen logit -- the class head gets no gradient at all when the loss is
  taken against the box head, and vice versa);
* every weight stays finite and actually moves, i.e. the training loop is doing
  something and not destroying the model as it goes.

Note that this dumps once per iteration, which waits on the context's
outstanding events and so puts a barrier between iterations. That hides races
*between* iterations; ``compare_training_run.py`` is the one that looks for
those, by dumping only at the end.

The loss is reported but *not* required to fall. At the learning rate run-model
uses (1e-3, momentum 0.9) against a random label, neither framework moves it
measurably in the handful of iterations available: an identical PyTorch run of
the upstream model goes 5.156 -> 5.145 over six steps, a 0.2% change, while
FlexFlow's wobbles by well under a percent in the same way. A loss criterion
here would be testing the learning rate, not the initialization.
"""

import argparse
import sys

import torch

from ff_tensor_file import read_tensor_file

# How far the output scale is allowed to drift over the run before it counts as
# diverging or collapsing rather than training.
MAX_LOGIT_RMS_DRIFT = 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iteration",
        action="append",
        required=True,
        help="per-iteration dump holding the logit, the label, and some weights "
        "with their gradients; repeat in order",
    )
    parser.add_argument("--logit", required=True, help="name of the logit tensor")
    parser.add_argument("--label", default="label")
    args = parser.parse_args()

    dumps = [read_tensor_file(path) for path in args.iteration]

    weight_names = sorted(
        name
        for name in dumps[0]
        if f"grad:{name}" in dumps[0] and name != args.logit
    )

    header = (
        f"{'iteration':>9} {'loss':>13} {'logit rms':>12} {'logit max':>11} "
        f"{'min |grad| rms':>15} {'weights moved':>14}"
    )
    print(header)
    print("-" * len(header))

    failed = []
    logit_rmss = []
    for step, dump in enumerate(dumps):
        logit = dump[args.logit].double()
        label = dump[args.label].double()

        if not torch.isfinite(logit).all():
            failed.append(
                f"iteration {step}: the logit holds "
                f"{(~torch.isfinite(logit)).sum().item()} non-finite values"
            )
            print(f"{step:>9} {'nan/inf':>13}")
            continue

        loss = ((logit - label) ** 2).mean().item()
        logit_rms = logit.pow(2).mean().sqrt().item()
        logit_rmss.append(logit_rms)

        grad_rmss = {}
        for name in weight_names:
            grad = dump[f"grad:{name}"].double()
            if not torch.isfinite(grad).all():
                failed.append(f"iteration {step}: grad:{name} is not finite")
                continue
            grad_rms = grad.pow(2).mean().sqrt().item()
            if grad_rms == 0.0:
                failed.append(
                    f"iteration {step}: grad:{name} is exactly zero, so nothing "
                    f"is training that weight"
                )
            grad_rmss[name] = grad_rms

        for name in weight_names:
            if not torch.isfinite(dump[name]).all():
                failed.append(f"iteration {step}: the weight {name} is not finite")

        moved = 0
        if step > 0:
            for name in weight_names:
                if not torch.equal(dump[name], dumps[step - 1][name]):
                    moved += 1
            if moved != len(weight_names):
                failed.append(
                    f"iteration {step}: {len(weight_names) - moved} of "
                    f"{len(weight_names)} weights did not change"
                )

        print(
            f"{step:>9} {loss:>13.6g} {logit_rms:>12.6g} "
            f"{logit.abs().max():>11.6g} "
            f"{min(grad_rmss.values()) if grad_rmss else float('nan'):>15.4g} "
            f"{(str(moved) + '/' + str(len(weight_names))) if step else '-':>14}"
        )

    if len(logit_rmss) >= 2:
        drift = max(logit_rmss) / min(logit_rmss)
        if drift > MAX_LOGIT_RMS_DRIFT:
            failed.append(
                f"the output scale drifted by a factor of {drift:.2f} over the "
                f"run (limit {MAX_LOGIT_RMS_DRIFT})"
            )

    print()
    print(
        f"ran {len(dumps)} iterations from FlexFlow's own initialization, "
        f"tracking {len(weight_names)} weights and their gradients"
    )
    if failed:
        print("FAILED")
        for message in failed[:20]:
            print(f"  {message}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        return 1
    print(
        "PASSED (forward pass finite, gradients finite and non-zero, every "
        "tracked weight updated, output scale stable)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
