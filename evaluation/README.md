# Reproducing the evaluation

Three scripts. Each takes no arguments, needs no setup, and builds or fetches
whatever it needs on first run.

| script | what it answers | runtime |
|---|---|---|
| `./validate.sh` | Does FlexFlow compute the same thing as ultralytics? | ~30 min |
| `./benchmark_flexflow.sh` | How fast is FlexFlow? | ~15 min |
| `./benchmark_pytorch.sh` | How fast is PyTorch on the same work? | ~25 min |

All three train YOLOv10x at batch size 6 and 640x640 on one GPU.

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
