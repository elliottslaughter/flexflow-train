"""Reader/writer for the simple binary tensor container used to move tensors
between PyTorch and FlexFlow's ``run-model``.

The format (see ``bin/run-model/src/run-model/main.cc``) is::

    magic:       8 bytes, "FFTENSR1"
    num_entries: int64
    entries:
      name_len:  int32
      name:      name_len bytes of UTF-8
      num_dims:  int32
      dims:      num_dims * int64  (FlexFlow ff_ordered, i.e. row-major NCHW)
      num_bytes: int64
      data:      num_bytes of float32, row-major/C-contiguous

All integers are little-endian.
"""

import struct

import torch

MAGIC = b"FFTENSR1"


def write_tensor_file(path, tensors):
    """Write ``tensors`` (a dict of name -> float32 tensor) to ``path``."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<q", len(tensors)))
        for name, tensor in tensors.items():
            tensor = tensor.detach().to("cpu", torch.float32).contiguous()
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<i", tensor.dim()))
            for dim in tensor.shape:
                f.write(struct.pack("<q", dim))
            data = tensor.numpy().tobytes() if hasattr(tensor, "numpy") else None
            if data is None:
                data = bytes(tensor.flatten().untyped_storage())
            f.write(struct.pack("<q", len(data)))
            f.write(data)


def read_tensor_file(path):
    """Read ``path`` and return a dict of name -> float32 tensor."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        (num_entries,) = struct.unpack("<q", f.read(8))
        result = {}
        for _ in range(num_entries):
            (name_len,) = struct.unpack("<i", f.read(4))
            name = f.read(name_len).decode("utf-8")
            (num_dims,) = struct.unpack("<i", f.read(4))
            dims = [struct.unpack("<q", f.read(8))[0] for _ in range(num_dims)]
            (num_bytes,) = struct.unpack("<q", f.read(8))
            data = f.read(num_bytes)
            if len(data) != num_bytes:
                raise ValueError(f"{path}: truncated entry {name}")
            tensor = torch.frombuffer(bytearray(data), dtype=torch.float32)
            result[name] = tensor.reshape(dims)
        return result
