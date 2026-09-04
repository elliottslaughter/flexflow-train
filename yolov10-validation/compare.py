"""Compare tensors produced by FlexFlow against the ultralytics reference."""

import argparse
import sys

import torch

from ff_tensor_file import read_tensor_file


def summarize(name, ref, got):
    ref = ref.to(torch.float64).flatten()
    got = got.to(torch.float64).flatten()

    diff = (got - ref).abs()
    scale = ref.abs().max().clamp(min=1e-30)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    # Relative to the magnitude of the tensor as a whole, which is the right
    # measure here: an element-wise relative error blows up on elements that
    # happen to be near zero.
    max_rel = max_abs / scale.item()
    rms_rel = (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    cosine = torch.nn.functional.cosine_similarity(ref, got, dim=0).item()
    nans = int(torch.isnan(got).sum().item())
    infs = int(torch.isinf(got).sum().item())

    return {
        "name": name,
        "shape": tuple(ref.shape),
        "ref_absmax": ref.abs().max().item(),
        "ref_rms": ref.pow(2).mean().sqrt().item(),
        "got_rms": got.pow(2).mean().sqrt().item(),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "rms_rel": rms_rel,
        "cosine": cosine,
        "nans": nans,
        "infs": infs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument(
        "--rms-rel-tolerance",
        type=float,
        default=2e-2,
        help="maximum acceptable RMS relative error; the default is loose "
        "because FlexFlow runs convolutions with cuDNN tensor-op math "
        "(reduced-precision accumulation) enabled",
    )
    args = parser.parse_args()

    reference = read_tensor_file(args.reference)
    actual = read_tensor_file(args.actual)

    names = [name for name in reference if name in actual]
    missing = sorted(set(reference) - set(actual))
    if missing:
        print(f"note: not produced by FlexFlow: {missing}", file=sys.stderr)
    if not names:
        print("error: no tensors in common", file=sys.stderr)
        return 1

    rows = []
    for name in names:
        ref = reference[name]
        got = actual[name]
        if ref.shape != got.shape:
            print(f"SHAPE MISMATCH {name}: ref {tuple(ref.shape)} vs ff {tuple(got.shape)}")
            return 1
        rows.append(summarize(name, ref, got))

    header = f"{'tensor':<28} {'ref rms':>11} {'ff rms':>11} {'max abs':>11} {'rel rms':>10} {'cosine':>10}"
    print(header)
    print("-" * len(header))
    failed = []
    for row in rows:
        print(
            f"{row['name']:<28} {row['ref_rms']:>11.4g} {row['got_rms']:>11.4g} "
            f"{row['max_abs']:>11.4g} {row['rms_rel']:>10.3g} {row['cosine']:>10.7f}"
        )
        if row["nans"] or row["infs"]:
            failed.append(f"{row['name']}: {row['nans']} NaNs, {row['infs']} infs")
        elif row["rms_rel"] > args.rms_rel_tolerance:
            failed.append(
                f"{row['name']}: relative RMS error {row['rms_rel']:.3g} "
                f"exceeds tolerance {args.rms_rel_tolerance:g}"
            )

    print()
    if failed:
        print("FAILED")
        for message in failed:
            print(f"  {message}")
        return 1

    print(f"PASSED ({len(rows)} tensors within {args.rms_rel_tolerance:g} relative RMS error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
