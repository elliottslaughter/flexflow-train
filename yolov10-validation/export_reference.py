"""Export the inputs FlexFlow needs and the reference outputs to compare against.

Writes two tensor files (see ``ff_tensor_file.py``):

  * ``--inputs``: the input batch plus every weight FlexFlow's YOLOv10x
    computation graph expects, keyed by FlexFlow tensor name.  Feed this to
    ``run-model`` via ``FF_LOAD_TENSORS``.
  * ``--reference``: the outputs (and any requested intermediates) produced by
    ultralytics for that same input and those same weights.
"""

import argparse
import json
import sys

import torch

from ff_tensor_file import write_tensor_file
import reference_model


def get_flexflow_weight_names(cg_json_path):
    """Read the names and shapes of every weight in FlexFlow's YOLOv10x graph."""
    with open(cg_json_path) as f:
        cg = json.load(f)

    weights = {}
    for _node_id, label in cg["raw_graph"]["node_labels"]:
        attrs = label["op_attrs"]
        if attrs["type"] != "weight":
            continue
        name = label["name"]
        if name is None:
            raise ValueError(
                "found an unnamed weight in the computation graph; the model "
                "must be built with names for the tensor mapping to work"
            )
        # Generated format_as() functions quote enum values, so "FILTER" comes
        # through as '"FILTER"'.
        name = name.replace('"', "")
        dims = attrs["value"]["tensor_shape"]["dims"]["ff_ordered"]
        weights[name] = tuple(dims)
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cg-json", required=True, help="FlexFlow computation graph JSON")
    parser.add_argument("--inputs", required=True, help="tensor file to write inputs/weights to")
    parser.add_argument("--reference", required=True, help="tensor file to write reference outputs to")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--capture",
        default="",
        help="comma-separated FlexFlow tensor names to additionally record",
    )
    parser.add_argument(
        "--tf32",
        default="1",
        help="run the reference with TF32 enabled (1) or disabled (0); "
        "FlexFlow enables cuDNN tensor-op math unconditionally, so 1 is the "
        "apples-to-apples setting",
    )
    args = parser.parse_args()

    allow_tf32 = args.tf32 == "1"
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    ff_weights = get_flexflow_weight_names(args.cg_json)
    print(f"FlexFlow graph has {len(ff_weights)} weights", file=sys.stderr)

    model = reference_model.build_model(seed=args.seed)
    weights = reference_model.get_weight_tensors(model, ff_weights.keys())

    for name, tensor in weights.items():
        if tuple(tensor.shape) != ff_weights[name]:
            raise ValueError(
                f"shape mismatch for {name}: "
                f"FlexFlow {ff_weights[name]} vs PyTorch {tuple(tensor.shape)}"
            )
    print("all weight shapes agree between FlexFlow and ultralytics", file=sys.stderr)

    generator = torch.Generator().manual_seed(args.seed + 1)
    inputs = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        generator=generator,
    )

    tensors = {"input": inputs}
    tensors.update(weights)
    write_tensor_file(args.inputs, tensors)
    print(f"wrote {len(tensors)} tensors to {args.inputs}", file=sys.stderr)

    # FlexFlow's batch norm always runs cudnnBatchNormalizationForwardTraining,
    # i.e. it normalizes using the statistics of the current batch, so the
    # reference has to run in training mode for the comparison to be meaningful.
    model = model.to(args.device)
    model.train()

    capture = [name for name in args.capture.split(",") if name]
    with torch.no_grad():
        outputs, captured = reference_model.run_reference(
            model, inputs.to(args.device), capture=capture
        )

    missing = sorted(set(capture) - set(captured) - set(outputs))
    if missing:
        raise ValueError(f"could not capture these tensors: {missing}")

    reference = dict(outputs)
    reference.update(captured)
    write_tensor_file(args.reference, reference)
    print(f"wrote {len(reference)} reference tensors to {args.reference}", file=sys.stderr)
    for name, tensor in reference.items():
        print(
            f"  {name} {tuple(tensor.shape)} "
            f"mean={tensor.mean():.6g} std={tensor.std():.6g}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
