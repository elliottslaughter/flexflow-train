"""Validate the weights FlexFlow starts a run with.

FlexFlow fills each weight from the ``InitializerAttrs`` its layer was built
with (see ``lib/realm-execution/src/realm-execution/weight_initialization.cc``).
Those initializers are the ones ``op-attrs`` picks by default, which are
transcriptions of what PyTorch's ``reset_parameters`` does, so what the weights
should look like is known analytically:

* a conv2d kernel is drawn with the standard deviation of
  ``kaiming_uniform_(w, a=sqrt(5))``, i.e. ``sqrt(2/(1+5)) / sqrt(fan_in)``;
* a conv2d bias is drawn uniformly from ``[-1/sqrt(fan_in), 1/sqrt(fan_in)]``;
* a batch norm scale is exactly 1 and its shift exactly 0.

The values themselves cannot be compared against PyTorch's -- the two use
different random number generators -- so what is checked here is the
distribution: every tensor's sample mean and standard deviation have to agree
with the analytic ones to within the sampling error of a tensor that size, and
the uniformly-drawn ones additionally have to stay inside their bounds, which is
an exact check.

This is what catches a wrong fan calculation: a depthwise convolution's fan_in
differs from its fan_out by the channel count, so getting the two the wrong way
round changes that layer's standard deviation by more than an order of
magnitude.

Passing a second dump with ``--repeat`` additionally checks that initialization
is reproducible, i.e. that two runs produce bit-identical weights.
"""

import argparse
import math
import sys

import torch
from torch import nn

from ff_tensor_file import read_tensor_file
import reference_model

# The number of standard errors a tensor's sample statistics are allowed to sit
# away from the analytic ones. Some of these tensors have only 64 elements, so
# the sampling error is not small; 6 sigma keeps false failures away while still
# being far tighter than any plausible bug.
SIGMA_TOLERANCE = 6.0

# nn.Conv2d.reset_parameters uses kaiming_uniform_(weight, a=math.sqrt(5)).
CONV_KAIMING_A = math.sqrt(5.0)


def expected_distribution(module, param_name, tensor):
    """The (mean, stddev, bound) PyTorch's reset_parameters would draw with.

    ``bound`` is the hard limit on a uniformly-drawn tensor, or None for one
    drawn from an unbounded distribution.
    """
    if isinstance(module, nn.BatchNorm2d):
        # nn.BatchNorm2d.reset_parameters: ones_(weight), zeros_(bias).
        return (1.0, 0.0, None) if param_name == "weight" else (0.0, 0.0, None)

    assert isinstance(module, nn.Conv2d), module
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)

    if param_name == "bias":
        bound = 1.0 / math.sqrt(fan_in)
        return 0.0, bound / math.sqrt(3.0), bound

    # kaiming_uniform_(w, a) draws uniformly from +/- sqrt(3) * gain / sqrt(fan),
    # which has standard deviation gain / sqrt(fan). FlexFlow's
    # KaimingNormalAttrs draws from the normal distribution with that same
    # standard deviation, so the bound does not apply.
    gain = nn.init.calculate_gain("leaky_relu", CONV_KAIMING_A)
    return 0.0, gain / math.sqrt(fan_in), None


def check_tensor(name, values, mean, stddev, bound):
    """Return a list of complaints about ``values``."""
    problems = []
    n = values.numel()
    values = values.double()

    if not torch.isfinite(values).all():
        problems.append("contains non-finite values")
        return problems

    if stddev == 0.0:
        # A constant initializer; require it exactly.
        if not (values == mean).all():
            problems.append(
                f"expected every element to be {mean:g}, got "
                f"[{values.min():.6g}, {values.max():.6g}]"
            )
        return problems

    if bound is not None and values.abs().max() > bound:
        problems.append(
            f"exceeds its uniform bound {bound:.6g} "
            f"(max |value| {values.abs().max():.6g})"
        )

    # The sample mean has standard error stddev/sqrt(n).
    mean_z = (values.mean().item() - mean) / (stddev / math.sqrt(n))
    if abs(mean_z) > SIGMA_TOLERANCE:
        problems.append(
            f"mean {values.mean():.6g} is {abs(mean_z):.1f} standard errors "
            f"from {mean:g}"
        )

    # The sample standard deviation has relative standard error 1/sqrt(2n).
    sample_stddev = values.std(unbiased=True).item()
    stddev_z = (sample_stddev / stddev - 1.0) * math.sqrt(2 * n)
    if abs(stddev_z) > SIGMA_TOLERANCE:
        problems.append(
            f"standard deviation {sample_stddev:.6g} is {abs(stddev_z):.1f} "
            f"standard errors from {stddev:.6g} "
            f"(ratio {sample_stddev / stddev:.4f})"
        )

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="FlexFlow's initial weights")
    parser.add_argument(
        "--repeat",
        help="a second dump from an identical run, to check reproducibility",
    )
    args = parser.parse_args()

    weights = read_tensor_file(args.weights)

    model = reference_model.build_model()
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())

    rows = []
    failed = []
    overridden = []
    for ff_name in sorted(weights):
        param_name = reference_model.flexflow_weight_name_to_param_name(ff_name)
        module_path, _, leaf = param_name.rpartition(".")
        module = modules[module_path]

        mean, stddev, bound = expected_distribution(module, leaf, weights[ff_name])
        problems = check_tensor(ff_name, weights[ff_name], mean, stddev, bound)
        for problem in problems:
            failed.append(f"{ff_name}: {problem}")

        values = weights[ff_name].double()
        rows.append(
            {
                "name": ff_name,
                "n": values.numel(),
                "expected": stddev,
                "actual": values.std(unbiased=True).item() if values.numel() > 1 else 0.0,
            }
        )

        # Whether upstream leaves the parameter at whatever reset_parameters
        # produced, or overwrites it afterwards.
        reference = params[param_name].detach().double()
        if stddev == 0.0:
            if not (reference == mean).all():
                overridden.append(ff_name)
        elif bound is not None and reference.abs().max() > bound:
            overridden.append(ff_name)

    header = (
        f"{'weight':<40} {'elements':>9} {'expected std':>13} "
        f"{'actual std':>12} {'ratio':>8}"
    )
    print(header)
    print("-" * len(header))
    # Show the tensors whose sample standard deviation sits furthest from the
    # analytic one, which is where a systematic error would show up first.
    interesting = sorted(
        (r for r in rows if r["expected"] > 0),
        key=lambda r: abs(r["actual"] / r["expected"] - 1.0),
        reverse=True,
    )
    for row in interesting[:10]:
        print(
            f"{row['name']:<40} {row['n']:>9} {row['expected']:>13.6g} "
            f"{row['actual']:>12.6g} {row['actual'] / row['expected']:>8.4f}"
        )

    print()
    if args.repeat:
        repeat = read_tensor_file(args.repeat)
        differing = [
            name
            for name in weights
            if name not in repeat or not torch.equal(weights[name], repeat[name])
        ]
        if differing:
            failed.append(
                f"{len(differing)} weights differ between two runs, so "
                f"initialization is not reproducible (e.g. {differing[0]})"
            )
        else:
            print(f"reproducible: all {len(weights)} weights identical across two runs")

    if overridden:
        print(
            f"note: upstream overwrites {len(overridden)} of these after "
            f"construction, so FlexFlow starts them differently by design or by "
            f"omission (e.g. {overridden[0]})"
        )

    print(f"checked {len(rows)} initialized weights")
    if failed:
        print("FAILED")
        for message in failed[:20]:
            print(f"  {message}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        return 1
    print(
        f"PASSED (every weight's mean and standard deviation within "
        f"{SIGMA_TOLERANCE:g} standard errors of the analytic value)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
