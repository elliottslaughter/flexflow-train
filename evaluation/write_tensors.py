"""Write the input batch and label that the throughput benchmark feeds FlexFlow.

The container format is the one ``bin/run-model/src/run-model/main.cc`` reads:
a magic string, an entry count, then per entry a name, dimensions and
float32 data, all little-endian.
"""

import argparse
import array
import random
import struct

MAGIC = b"FFTENSR1"

INPUT_DIMS = (6, 3, 640, 640)
LABEL_DIMS = (6, 64, 8400)


def write_tensor_file(path, tensors):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<q", len(tensors)))
        for name, (dims, data) in tensors.items():
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<i", len(dims)))
            for dim in dims:
                f.write(struct.pack("<q", dim))
            raw = data.tobytes()
            f.write(struct.pack("<q", len(raw)))
            f.write(raw)


def filled(dims, seed):
    count = 1
    for dim in dims:
        count *= dim
    # A tile of random values repeated to length, because generating ten million
    # of them one at a time in Python takes longer than the benchmark does. What
    # the numbers are cannot change the timing: these kernels do the same work
    # whatever the data, and the values stay in a range that avoids denormals.
    rng = random.Random(seed)
    tile = array.array("f", [rng.gauss(0.0, 1.0) for _ in range(65536)])
    data = array.array("f")
    while len(data) < count:
        data.extend(tile[: count - len(data)])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    write_tensor_file(args.input, {"input": (INPUT_DIMS, filled(INPUT_DIMS, 1))})
    write_tensor_file(args.label, {"label": (LABEL_DIMS, filled(LABEL_DIMS, 2))})
    print(f"wrote input {INPUT_DIMS} and label {LABEL_DIMS}")


if __name__ == "__main__":
    main()
