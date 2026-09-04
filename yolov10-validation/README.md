# YOLOv10x validation against upstream ultralytics

These scripts check that FlexFlow's YOLOv10x
(`lib/models/src/models/yolov10/yolov10.cc`) computes the same function as the
upstream ultralytics implementation, by running both on the same random input
with the same weights and comparing the results.

## What is compared

FlexFlow's YOLOv10x is built with `end2end = false`, which corresponds to
ultralytics' **one2many** head (`model.23.cv2` / `model.23.cv3`); the one2one
branch and the DFL buffer that `v10Detect` also creates are deliberately not
part of FlexFlow's graph. The comparison therefore targets ultralytics'
`Detect.forward_head` outputs, `boxes` (6x64x8400) and `scores` (6x80x8400).

The comparison is done two ways:

* **Per-layer** (`compare_layerwise.py`, the discriminating check). Each
  ultralytics layer is fed *FlexFlow's own* intermediate tensors, so no error
  accumulates and every comparison is exactly one layer deep. Agreement should
  be at the level of floating-point noise, and any structural difference shows
  up as an unmistakable O(1) error at a specific layer.
* **End-to-end** (`compare.py`). Useful as a smoke test, but not
  discriminating on its own: this network is chaotic enough that merely
  toggling TF32 *within PyTorch* moves `boxes` by ~3.6% relative RMS, and
  FlexFlow enables cuDNN tensor-op math for all convolutions unconditionally
  (`lib/kernels/src/cuda/ops/conv_2d_kernels.cu`). Run
  `export_reference.py --tf32 0` to reproduce that control experiment.

Only the **forward** pass is covered. The backward and update passes are not
validated here.

## How the two frameworks' tensors are matched up

FlexFlow's model builder names each layer after the corresponding ultralytics
module path (`model.0.conv`, `model.23.cv3.2.1.1.bn`, ...), and
`ComputationGraphBuilder::add_layer` names each weight after the layer that
consumes it plus its slot (`model.0.conv.FILTER`, `model.0.bn.GAMMA`, ...).
That makes the mapping mechanical:

| FlexFlow slot | ultralytics parameter |
| ------------- | --------------------- |
| `FILTER`      | `weight` (conv2d)     |
| `BIAS`        | `bias` (conv2d)       |
| `GAMMA`       | `weight` (batch norm) |
| `BETA`        | `bias` (batch norm)   |

`export_reference.py` checks that all 513 of FlexFlow's weights resolve to an
ultralytics parameter of exactly the same shape, which is itself a useful
structural check on the model definition.

An ultralytics `Conv` is conv2d + batch norm + SiLU, which FlexFlow builds as
three layers, so a `Conv`'s output is FlexFlow's `<path>.act` (or `<path>.bn`
when the activation is disabled). `reference_model.py` encodes this.

## Running it

```bash
./setup_venv.sh          # once: creates .venv with PyTorch and ultralytics' deps
./run_validation.sh      # builds FlexFlow, runs both frameworks, prints the tables
```

`ULTRALYTICS` (default `/home/eslaught/flexflow/ultralytics`) selects the
ultralytics checkout, `NIX` the `nix` binary, and the first positional argument
the work directory (default `./work`, ~1.2 GB of tensor dumps).

## Feeding data into FlexFlow

`bin/run-model` gained the ability to read and write named tensors, driven by
environment variables (`lib/utils/cli` only supports boolean flags and
positional arguments, so there was nowhere to hang real options):

| Variable           | Meaning                                                              |
| ------------------ | -------------------------------------------------------------------- |
| `FF_LOAD_TENSORS`  | tensor file to copy into the correspondingly-named forward tensors    |
| `FF_DUMP_TENSORS`  | tensor file to write after running                                    |
| `FF_DUMP_NAMES`    | comma-separated tensor names to write to `FF_DUMP_TENSORS`            |
| `FF_FORWARD_ONLY`  | `1` to run only the forward pass (leaves the loaded weights alone)    |
| `FF_ITERATIONS`    | number of iterations (default 5)                                      |
| `FF_LIST_TENSORS`  | `1` to print the name and shape of every nameable tensor, then exit   |
| `FF_LIST_TASKS`    | `1` to print a count of the FWD/BWD/UPD/LOSS tasks, then exit         |

Gradients are addressed by the name of the tensor they are the gradient of,
with a `grad:` prefix (`grad:model.0.conv.FILTER`). Note that these are all
zero today: `run-model` builds its `PCGInstance` with no loss, so nothing ever
seeds the gradient of the output. See "Backward pass" below.

With none of them set, `run-model` behaves exactly as before. The file format
is documented in `ff_tensor_file.py` and implemented on both sides.

Listing what is available is often the quickest way to find a name to dump:

```bash
REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize 28500 -cuda:dynfb 0' FF_LIST_TENSORS=1 \
  nixGL -- ./build/release/bin/run-model/run-model yolov10x_mpcg.json
```

## Files

| File                   | Purpose                                                        |
| ---------------------- | -------------------------------------------------------------- |
| `run_validation.sh`    | end-to-end driver                                              |
| `setup_venv.sh`        | creates the Python environment                                 |
| `ff_tensor_file.py`    | reader/writer for the binary tensor container                  |
| `reference_model.py`   | builds the ultralytics model; FlexFlow <-> ultralytics naming  |
| `export_reference.py`  | writes FlexFlow's inputs and the ultralytics reference outputs |
| `compare_layerwise.py` | per-layer comparison                                           |
| `compare.py`           | end-to-end comparison and the error metrics                    |

## Backward pass

Not validated. Three things measured on this model stand in the way, none of
which are addressed here:

* `run-model` passes `loss = std::nullopt` to `create_pcg_instance`, so no
  `LOSS` task is inserted and the gradient of the output is never seeded. The
  618 `BWD` tasks do run, but every gradient comes out exactly zero (verified by
  dumping `grad:model.23.boxes` and several weight gradients after a full
  iteration). Seeding it requires plumbing a `ParallelLossConfig` through
  `run-model`, which needs a `MappedOperatorTaskGroup` for the loss node.
* `ParallelLossConfig` carries a single `logit_tensor`, but this head produces
  two outputs (`boxes` and `scores`), so they would have to be validated one at
  a time (which is fine: it corresponds exactly to `boxes.backward(g)` in
  PyTorch).
* The backward kernels accumulate into gradient buffers (`beta = 1.0`, so that
  a tensor consumed more than once sums its subgradients), but nothing zeros
  those buffers between iterations. Only the first iteration of a fresh process
  would be meaningful.

Separately, after one and after five full training iterations every weight is
bit-identical to what was loaded, even though 513 `UPD` tasks are scheduled and
`weight_decay` is non-zero (so the update should move the weights by ~1e-6
relative even with zero gradients). Cause not determined.
