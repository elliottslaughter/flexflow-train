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

The **backward** and **update** passes are covered the same way — see their
sections below.

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
| `FF_LOAD_TENSORS`  | comma-separated tensor files to copy into the correspondingly-named tensors |
| `FF_DUMP_TENSORS`  | tensor file to write after running; a `%` in the path is replaced by the iteration number and dumps every iteration |
| `FF_DUMP_NAMES`    | comma-separated tensor names to write to `FF_DUMP_TENSORS`            |
| `FF_FORWARD_ONLY`  | `1` to run only the forward pass (leaves the loaded weights alone)    |
| `FF_LOSS`          | loss to attach: `mean_squared_error_avg`, `mean_squared_error_sum`, `categorical_crossentropy`, `identity` |
| `FF_LOSS_LOGIT`    | name of the tensor the loss is taken against (set with `FF_LOSS`)     |
| `FF_ITERATIONS`    | number of iterations (default 5)                                      |
| `FF_LIST_TENSORS`  | `1` to print the name and shape of every nameable tensor, then exit   |
| `FF_LIST_TASKS`    | `1` to print a count of the FWD/BWD/UPD/LOSS tasks, then exit         |

Gradients are addressed by the name of the tensor they are the gradient of,
with a `grad:` prefix (`grad:model.0.conv.FILTER`). The label tensor that loss
insertion creates is named `label`; it lives outside the computation graph, so
unlike everything else it is not named after a layer.

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
| `compare_layerwise.py` | per-layer comparison of the forward pass                       |
| `compare_layerwise_backward.py` | per-layer comparison of the backward pass             |
| `compare_optimizer.py` | replays FlexFlow's gradients through `torch.optim.SGD`         |
| `make_label.py`        | writes the label tensor the loss is taken against              |
| `compare.py`           | end-to-end comparison and the error metrics                    |

## Backward pass

`compare_layerwise_backward.py` feeds each ultralytics layer FlexFlow's own
forward activations *and* FlexFlow's own gradient of that layer's output, and
compares the **weight gradients** it produces. Weight gradients are what is
compared (rather than input gradients) because they are unambiguous: FlexFlow's
gradient of a tensor consumed by several layers is the sum over those uses,
which does not correspond to any single layer's backward.

Some of these gradients are sums with heavy cancellation, where float32 has
little precision left and neither framework's answer is meaningful (`model.9`
and `model.10` have batch-norm shift gradients whose true value is ~1e-19). So
the reference is also computed in float64, and FlexFlow is judged against how
well the float32 reference itself does on the same tensor rather than against a
fixed tolerance.

The loss can only be attached to one tensor at a time, so the driver runs it
twice: the box head's weights only receive a gradient from `model.23.boxes` and
the class head's only from `model.23.scores`. Between them the two runs cover
all 513 weight gradients. Each run also checks FlexFlow's gradient of the logit
itself against the analytic `2 (y - label) / numel(y)`.

## Update pass

`compare_optimizer.py` replays FlexFlow's *own* weight gradients through
`torch.optim.SGD` and checks that it arrives at FlexFlow's weights, step by
step. Feeding PyTorch FlexFlow's gradients is what makes this a test of the
optimizer alone rather than of the gradients, which the backward comparison
covers separately. What is compared is the *step* (the change in the weight):
at lr=0.001 the weights barely move, so comparing them directly would mostly be
comparing a number to itself.

Five steps are enough to exercise the momentum buffer (step 0 has an empty
buffer, later steps accumulate) as well as the weight decay term.

## Memory

Running a full training iteration is tight on a 32 GB card at batch size 6.
Realm aborts while allocating instances below about `-ll:fsize 26000`, and above
about `28000` there is not enough memory left *outside* Realm's pool for CUDA's
own resources once the update pass is issuing its 513 extra tasks per iteration
— it fails partway through with `CUDA failure: out of memory` from
`device_stream_t.cc`. The driver uses `27000`; override with `FSIZE=`.

Note that `-ll:fsize 28500` (which appears in some older notes) worked only
because the update tasks were never actually being spawned.


## Known issues

Four bugs had to be fixed before the backward and update passes could be
validated at all; they are described in the session notes. What remains:

* **Nine batch-norm scale/shift gradients** are 3-100x worse than the float32
  reference's own error (they are the only failures the driver reports; cosine
  similarity is still >= 0.993). The cause is
  `CUDNN_BATCHNORM_SPATIAL_PERSISTENT`, which
  `lib/kernels/src/cuda/ops/batch_norm_kernels.cu` selects unconditionally on
  cuDNN >= 7: building with `mode = CUDNN_BATCHNORM_SPATIAL` instead makes all
  471/490 gradients pass. Whether the accuracy is worth the speed is a
  judgement call, so the mode is left as it is.

* **Weights are never initialized.** Nothing in the execution path applies the
  `InitializerAttrs` recorded in the computation graph, so without
  `FF_LOAD_TENSORS` every weight is zero and the model computes zeros.
