"""Run N training iterations of YOLOv10x under PyTorch and exit.

Matches what the FlexFlow benchmark runs: batch size 6 at 640x640, mean-squared
error against the one2many box head, SGD with the same hyperparameters, and the
same set of trained weights.

What remains gives PyTorch slightly more work than FlexFlow: v10Detect is an
end-to-end head, so in training mode its forward pass also computes the one2one
branch that FlexFlow does not build. Its backward is skipped below, but the
forward cannot be without changing the model.
"""

import argparse
import sys

import torch

import reference_model

BATCH = 6
IMAGE = 640
BOX_DIMS = (64, 8400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    args = parser.parse_args()

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    device = "cuda"
    model = reference_model.build_model().to(device)
    model.train()

    generator = torch.Generator(device=device).manual_seed(1)
    images = torch.randn((BATCH, 3, IMAGE, IMAGE), generator=generator, device=device)
    label = torch.randn((BATCH,) + BOX_DIMS, generator=generator, device=device)

    for name, parameter in model.named_parameters():
        if "one2one" in name:
            parameter.requires_grad_(False)
    trained = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(trained, lr=0.001, momentum=0.9,
                                weight_decay=0.001)

    for _ in range(args.iterations):
        predictions = model(images)
        # The non-end2end model FlexFlow builds corresponds to this branch.
        head = predictions["one2many"] if "one2many" in predictions else predictions
        loss = ((head["boxes"] - label) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(main())
