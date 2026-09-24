# Reproducing the evaluation

Three scripts. Each takes no arguments, needs no setup, and builds or fetches
whatever it needs on first run.

| script | what it answers | runtime |
|---|---|---|
| `./validate.sh` | Does FlexFlow compute the same thing as ultralytics? | ~30 min |
| `./benchmark_flexflow.sh` | How fast is FlexFlow? | ~15 min |
| `./benchmark_pytorch.sh` | How fast is PyTorch on the same work? | ~25 min |

All three train YOLOv10 at 640x640 on one GPU: YOLOv10x at batch size 6 unless
told otherwise (see [Choosing the model and batch size](#choosing-the-model-and-batch-size)).

`validate.sh` checks the forward pass layer by layer, the backward pass layer by
layer, five SGD steps against `torch.optim.SGD`, and the network end to end. It
prints a PASSED or FAILED line per stage and exits non-zero if any stage fails.

The two benchmarks report the time for 500 training iterations, averaged over 3
runs. Startup is excluded by timing a short run and a long one and subtracting,
which costs an extra run per repeat but needs no instrumentation inside either
framework. `benchmark_pytorch.sh` reports two numbers, because ultralytics ships
with `cudnn.benchmark` off while FlexFlow selects convolution algorithms by
measurement; the first number is the fair comparison against upstream as
released, the second against PyTorch tuned the way FlexFlow is.

Intermediate files and per-run logs are left under `work/`.

## Choosing the model and batch size

`config.sh` reads three environment variables, which all three scripts honor:

```bash
MODEL=yolov10l BATCH=11 FSIZE=26500 ./validate.sh
```

| variable | meaning | default |
|---|---|---|
| `MODEL` | `yolov10n`, `yolov10s`, `yolov10m`, `yolov10b`, `yolov10l` or `yolov10x` | `yolov10x` |
| `BATCH` | batch size | `6` |
| `FSIZE` | MiB of GPU memory Realm takes up front for its pool (`-ll:fsize`) | `27000` |

Each configuration gets its own directory under `work/`, named after the model
and batch size, so results for different configurations do not overwrite each
other.

`FSIZE` has to be chosen with the batch size, and it can be wrong in both
directions. The pool holds every tensor of the model, so too small and Realm
cannot allocate them. What the pool leaves free is all that CUDA, cuDNN and the
convolution workspaces get, so too large and a `cudaMalloc` fails part way
through training.

That second budget is bigger than it looks: 4.3 to 6.3 GiB here, depending on
the configuration. About 1.5 GiB is the CUDA context and cuDNN's handle
workspace. The rest is convolution scratch (`lib/kernels/src/kernels/device_scratch.cc`),
which is kept per CUDA stream so that tasks on different streams never share a
buffer. Realm runs tasks on 12 streams, and each stream's buffer grows to the
largest workspace a convolution on it asks for, up to the 512 MiB budget. It
takes a few iterations for every stream to have seen a large convolution, so a
configuration that survives three iterations can still run out of memory in the
benchmark. Startup briefly uses more again, because cuDNN's algorithm
measurement tries candidates in whatever memory is free, but that adapts to
what it finds.

These have been checked on the RTX PRO 4500 (31.9 GiB) by running the
benchmark, 600 iterations per run:

| `MODEL` | `BATCH` | `FSIZE` (MiB) | smallest pool that works | steady-state use | for |
|---|---|---|---|---|---|
| `yolov10x` | 6 | 27000 | 19350 | 30.7 GiB | 32 GB GPU |
| `yolov10l` | 10 | 24000 | 23650 | 28.9 GiB | 32 GB GPU |
| `yolov10b` | 12 | 23750 | 23350 | 29.5 GiB | 32 GB GPU |
| `yolov10x` | 5 | 16500 | 16150 | 20.3 GiB | 24 GB GPU |
| `yolov10l` | 7 | 17100 | 16750 | 20.7 GiB | 24 GB GPU |
| `yolov10b` | 8 | 16100 | 15700 | 20.2 GiB | 24 GB GPU |

`yolov10l` at 10 and `yolov10b` at 12 are the largest batches that fit this GPU
with room to spare. `yolov10l` at 11 runs out of memory in the benchmark, and
`yolov10b` at 13 would need about 31.2 GiB of the 31.9, too little margin to be
dependable.

The last three are for GPUs with 24 GB, and were checked by running them with
another process holding everything but 22000 MiB (21.5 GiB) of this one, which
leaves at least 2 GiB to spare on a 24 GiB card. One more (`yolov10x` at 6,
`yolov10l` at 8, `yolov10b` at 9) does not fit in 21.5 GiB.

## Which build they use

The two FlexFlow scripts pick up whichever build is present. If
`deploy/sapling.sh` has configured a build under `build/`, they use it directly.
Otherwise they fall back to `proj` and `build/release/` inside `nix develop`,
which is also where `nixGL` comes from. Each script prints which one it chose.
The PyTorch script never needs either; it only wants the venv.

## What these produced here

On an RTX PRO 4500 Blackwell (82 SMs), CUDA 12.8, cuDNN 9.7.1, at commit
`0858b025`, `validate.sh` passed all four stages and the benchmarks reported:

| | ms/iteration | spread over 3 runs |
|---|---|---|
| FlexFlow (Realm backend) | 147.06 | 0.6% |
| PyTorch, as ultralytics ships it | 165.85 | 0.1% |
| PyTorch, `cudnn.benchmark=True` | 148.24 | 1.1% |

FlexFlow is 11.3% faster than upstream as released, and 0.8% faster than PyTorch
with algorithm selection turned on to match it.

### YOLOv10l and YOLOv10b, and YOLOv10x for 24 GB

On the same GPU, with the Nix build (CUDA 12.9, cuDNN 9.10.2) at commit
`a8f7b5d7a` plus the changes that added these configurations, against PyTorch
2.14.0 (CUDA 13.0, cuDNN 9.24). `validate.sh` passed all four stages for every
configuration below. The 24 GB configurations were timed with another
process holding all but 21.5 GiB of the GPU, both frameworks alike.

| configuration | FlexFlow | PyTorch as shipped | PyTorch, `cudnn.benchmark=True` |
|---|---|---|---|
| `yolov10l`, batch 10 (32 GB) | **164.19** | 188.35 (FlexFlow 12.8% faster) | 179.71 (8.6% faster) |
| `yolov10b`, batch 12 (32 GB) | **157.91** | 183.15 (13.8% faster) | 178.12 (11.3% faster) |
| `yolov10l`, batch 7 (24 GB) | **113.45** | 129.47 (12.4% faster) | 124.56 (8.9% faster) |
| `yolov10b`, batch 8 (24 GB) | **103.78** | 117.67 (11.8% faster) | 113.35 (8.4% faster) |
| `yolov10x`, batch 5 (24 GB) | **117.34** | 138.28 (15.1% faster) | 122.63 (4.3% faster) |

All in ms/iteration, mean of 3 runs of 500 iterations. Every FlexFlow spread was
under 2%. PyTorch with `cudnn.benchmark=True` on `yolov10l` at batch 7 is the
exception on the PyTorch side: it picks its algorithms afresh in every process,
and nine runs of it ranged from 117.7 to 129.0 ms/iteration; the table gives
the mean of the three taken when the machine was quietest.

At the same batch size FlexFlow needs about twice the GPU memory PyTorch does:
on `yolov10l` at batch 7, PyTorch's allocator peaks at 10.6 GiB, where FlexFlow
uses 20.7 GiB in all. That, not speed, is what limits the batch size on a 24 GB
GPU.
