"""Validate a whole training run: initialize, forward, backward, update, repeat.

Every other comparison here tests one stage in isolation, feeding it FlexFlow's
own tensors so no error accumulates. This one runs the four stages *composed*:
FlexFlow initializes its own weights and trains for several iterations with
nothing re-synchronized in between, and PyTorch is handed those same initial
weights and runs the identical loop.

Two things make this different from the per-stage comparisons, and both matter.

**What it can check.** Because nothing is re-synchronized, FlexFlow's iterations
are free to overlap, and a missing dependency between them shows up here and
nowhere else. Dumping only at the end of the run is deliberate: a per-iteration
dump waits on the context's outstanding events, which inserts a barrier between
iterations and hides exactly the class of bug this is here to catch. The
finiteness check below is the load-bearing one.

**What it cannot check.** The end-to-end gradient of this network is chaotic.
Toggling TF32 *within PyTorch*, with identical weights and input, changes the
whole-network gradients by around 100% relative RMS -- not because either answer
is wrong, but because a 3.6% difference in a 24-layer forward pass compounds
through 24 Jacobians on the way back. So there is no useful comparison of the
*direction* of a weight update after a full backward pass, at any number of
iterations, and this is precisely why the rest of the harness compares per layer
rather than end to end.

What survives the chaos is *magnitude*: the size of each weight's change, and
the loss. Those are compared here, against the PyTorch TF32-vs-fp32 pair as the
noise floor rather than against a fixed tolerance.
"""

import argparse
import sys

import torch

from ff_tensor_file import read_tensor_file
import reference_model

# How far outside the TF32-vs-fp32 control FlexFlow may sit before the
# difference stops looking like precision.
CONTROL_FACTOR = 4.0

# Floors, for when the control is so tight that the factor above would make the
# check hostage to run-to-run noise.
MAGNITUDE_FLOOR = 0.3
LOSS_FLOOR = 0.02

# Weight changes smaller than this are float32 quantization of a nearly
# unchanged weight, not something the run produced.
MIN_CHANGE = 1e-9


def train_reference(ff_initial, inputs, label, steps, logit_slot,
                    lr, momentum, weight_decay, allow_tf32, device):
    """Run `steps` training iterations from `ff_initial`, returning the weights
    it ends at and the loss at each step.

    A fresh model is built per call so the two precision settings start from
    identical state, batch norm's running statistics included.
    """
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    model = reference_model.build_model().to(device)
    model.train()

    params = dict(model.named_parameters())
    trained = {}
    for ff_name, value in ff_initial.items():
        param_name = reference_model.flexflow_weight_name_to_param_name(ff_name)
        with torch.no_grad():
            params[param_name].copy_(value.to(device))
        trained[ff_name] = params[param_name]

    # Everything else (the one2one heads, the DFL buffer) lies outside
    # FlexFlow's graph and cannot reach the logit, so freeze it rather than let
    # the optimizer wander it around.
    ff_param_names = set(
        reference_model.flexflow_weight_name_to_param_name(name)
        for name in ff_initial)
    for param_name, param in params.items():
        param.requires_grad_(param_name in ff_param_names)

    optimizer = torch.optim.SGD(
        list(trained.values()),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=False,
        dampening=0.0,
    )

    losses = []
    for _step in range(steps):
        preds = model(inputs)
        one2many = preds["one2many"] if "one2many" in preds else preds
        out = one2many[logit_slot]

        loss = ((out - label) ** 2).mean()
        losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    result = {name: param.detach().to("cpu", torch.float32).clone()
              for name, param in trained.items()}

    del optimizer, trained, params, model
    torch.cuda.empty_cache()

    return result, losses


def rms(tensor):
    return float(tensor.double().pow(2).mean().sqrt())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True,
                        help="FlexFlow's weights before any training")
    parser.add_argument("--final", required=True,
                        help="FlexFlow's weights and logit after the run")
    parser.add_argument("--input", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--logit", required=True)
    parser.add_argument("--steps", type=int, required=True)
    # These have to match the OptimizerAttrs run-model builds.
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ff_initial = read_tensor_file(args.initial)
    ff_final = read_tensor_file(args.final)
    inputs = read_tensor_file(args.input)["input"].to(args.device)
    label = read_tensor_file(args.label)["label"].to(args.device)

    failed = []

    # The load-bearing check: a run whose iterations were never separated by a
    # barrier has to come out finite.
    non_finite = sorted(name for name, tensor in ff_final.items()
                        if not torch.isfinite(tensor).all())
    if non_finite:
        failed.append(
            f"{len(non_finite)} of {len(ff_final)} tensors are non-finite after "
            f"{args.steps} iterations (e.g. {non_finite[0]}); a training run "
            f"with no barrier between its iterations did not stay finite")

    logit_slot = args.logit.rsplit(".", 1)[-1]

    print(f"running {args.steps} reference iterations (TF32 enabled) ...",
          file=sys.stderr)
    ref, ref_losses = train_reference(
        ff_initial, inputs, label, args.steps, logit_slot,
        args.lr, args.momentum, args.weight_decay,
        allow_tf32=True, device=args.device)

    print(f"running {args.steps} reference iterations (TF32 disabled) ...",
          file=sys.stderr)
    control, control_losses = train_reference(
        ff_initial, inputs, label, args.steps, logit_slot,
        args.lr, args.momentum, args.weight_decay,
        allow_tf32=False, device=args.device)

    # How big was each weight's change, relative to the reference's?
    ff_ratios, control_ratios = [], []
    for name in ff_initial:
        if name not in ff_final:
            continue
        reference_change = rms(ref[name] - ff_initial[name])
        if reference_change < MIN_CHANGE:
            continue
        ff_ratios.append(rms(ff_final[name] - ff_initial[name]) / reference_change)
        control_ratios.append(rms(control[name] - ff_initial[name]) / reference_change)
    ff_ratios.sort()
    control_ratios.sort()

    def band(values):
        return (values[int(len(values) * 0.1)],
                values[len(values) // 2],
                values[int(len(values) * 0.9)])

    ff_low, ff_median, ff_high = band(ff_ratios)
    ctl_low, ctl_median, ctl_high = band(control_ratios)

    header = (f"{'size of the weight change, vs the reference':<46} "
              f"{'10th':>9} {'median':>9} {'90th':>9}")
    print(header)
    print("-" * len(header))
    print(f"{'FlexFlow / PyTorch TF32':<46} {ff_low:>9.4f} {ff_median:>9.4f} "
          f"{ff_high:>9.4f}")
    print(f"{'PyTorch fp32 / PyTorch TF32  (noise floor)':<46} "
          f"{ctl_low:>9.4f} {ctl_median:>9.4f} {ctl_high:>9.4f}")

    ff_loss = float(((ff_final[args.logit].double() - label.cpu().double()) ** 2).mean())
    print()
    print(f"loss after {args.steps} iterations: FlexFlow {ff_loss:.6g}, "
          f"PyTorch TF32 {ref_losses[-1]:.6g}, PyTorch fp32 {control_losses[-1]:.6g}")
    print(f"  PyTorch TF32 trajectory: " + ", ".join(f"{x:.5f}" for x in ref_losses))
    print(f"  PyTorch fp32 trajectory: " + ", ".join(f"{x:.5f}" for x in control_losses))

    magnitude_limit = max(CONTROL_FACTOR * abs(ctl_median - 1.0), MAGNITUDE_FLOOR)
    if abs(ff_median - 1.0) > magnitude_limit:
        failed.append(
            f"the median weight change is {ff_median:.4f}x the reference's, "
            f"outside 1 +/- {magnitude_limit:.3f} (the control sits at "
            f"{ctl_median:.4f})")

    control_loss_gap = abs(ref_losses[-1] - control_losses[-1]) / abs(ref_losses[-1])
    ff_loss_gap = abs(ref_losses[-1] - ff_loss) / abs(ref_losses[-1])
    loss_limit = max(CONTROL_FACTOR * control_loss_gap, LOSS_FLOOR)
    if ff_loss_gap > loss_limit:
        failed.append(
            f"the final loss differs from the reference by {ff_loss_gap:.3g} "
            f"relative, over a limit of {loss_limit:.3g} (the control differs "
            f"by {control_loss_gap:.3g})")

    print()
    print(f"composed {args.steps} full training iterations over "
          f"{len(ff_ratios)} weights, with no barrier between them")
    if failed:
        print("FAILED")
        for message in failed:
            print(f"  {message}")
        return 1
    print("PASSED (run stayed finite, and both the size of the weight change "
          "and the final loss sit within the TF32-vs-fp32 noise floor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
