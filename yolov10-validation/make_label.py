"""Write the label tensor that the loss is taken against.

``run-model``'s loss needs a label with the same shape as the logit; the label
lives outside the computation graph, so it is loaded by the name ``label``.
"""

import argparse

import torch

from ff_tensor_file import write_tensor_file

SHAPES = {
    "model.23.boxes": (64, 8400),
    "model.23.scores": (80, 8400),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--logit", default="model.23.boxes", choices=sorted(SHAPES))
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    label = torch.randn(
        (args.batch_size,) + SHAPES[args.logit], generator=generator
    )
    write_tensor_file(args.out, {"label": label})
    print(f"wrote label {tuple(label.shape)} for {args.logit} to {args.out}")


if __name__ == "__main__":
    main()
