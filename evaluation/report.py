"""Print the mean and spread of a set of timing runs."""

import argparse
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--timings", required=True,
                        help="file of milliseconds, one run per line")
    args = parser.parse_args()

    with open(args.timings) as f:
        runs = [int(line) for line in f if line.strip()]

    mean = statistics.mean(runs)

    def row(label, ms):
        print(f"  {label:<22}{ms / 1000:8.2f} s{ms / args.iterations:9.2f} ms/iteration")

    print()
    print(f"{args.label}: {args.iterations} training iterations, "
          f"batch size 6, 640x640")
    for i, ms in enumerate(runs, start=1):
        row(f"run {i}", ms)
    row(f"mean of {len(runs)} runs", mean)
    if len(runs) > 1:
        print(f"  {'spread (max-min)/mean':<22}{(max(runs) - min(runs)) / mean * 100:7.1f} %")


if __name__ == "__main__":
    main()
