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
ultralytics checkout, `NIX` the `nix` binary, `FSIZE` Realm's frame-buffer size,
and the first positional argument the work directory (default `./work`, ~9 GB of
tensor dumps).

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
REALM_DEFAULT_ARGS='-ll:gpu 1 -ll:fsize 27000 -cuda:dynfb 0' FF_LIST_TENSORS=1 \
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
| `check_initialization.py` | checks the weights FlexFlow starts with against PyTorch's rules |
| `check_training.py`    | smoke test of training from FlexFlow's own initialization       |
| `compare_training_run.py` | whole training run, iterations left free to overlap         |

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

Nine batch-norm scale/shift gradients used to be 3-100x worse than the float32
reference's own error, because `lib/kernels/src/cuda/ops/batch_norm_kernels.cu`
always selected `CUDNN_BATCHNORM_SPATIAL_PERSISTENT`. Now that the mode is part
of `BatchNormAttrs` and YOLOv10 asks for `BatchNormMode::SPATIAL`, all of them
pass.

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

## Weight initialization

Unlike everything else here, this stage does *not* hand FlexFlow weights
exported from PyTorch: it checks the weights FlexFlow fills in for itself from
the `InitializerAttrs` recorded in the computation graph. The values cannot be
compared against PyTorch's, since the two use different random number
generators, so `check_initialization.py` compares the *distribution* against the
one PyTorch's `reset_parameters` draws from, which is known analytically:

| parameter        | distribution                                            |
| ---------------- | ------------------------------------------------------- |
| conv2d kernel    | standard deviation of `kaiming_uniform_(w, a=sqrt(5))`, i.e. `sqrt(2/6)/sqrt(fan_in)` |
| conv2d bias      | uniform on `[-1/sqrt(fan_in), +1/sqrt(fan_in)]`          |
| batch norm scale | exactly 1                                                |
| batch norm shift | exactly 0                                                |

Every tensor's sample mean and standard deviation have to land within six
standard errors of the analytic ones, and the uniformly-drawn ones additionally
have to stay inside their bounds, which is an exact check. This is what catches
a wrong fan calculation: a depthwise convolution's `fan_in` and `fan_out` differ
by the channel count, so swapping them changes that layer's standard deviation
by more than an order of magnitude.

The check also runs FlexFlow twice and requires the weights to come out
bit-identical, and `check_training.py` then trains for a few iterations from
that initialization (loading only the input and the label) and checks that the
forward pass stays finite, the output scale neither collapses nor diverges,
every weight gradient is finite and non-zero, and every weight moves.

`check_training.py` deliberately does *not* require the loss to fall. At
run-model's learning rate (1e-3, momentum 0.9) against a random label, six steps
move it by well under a percent in either framework — an identical PyTorch run
of the upstream model goes 5.156 -> 5.145. A loss criterion here would be
testing the learning rate, not the initialization.

FlexFlow's conv2d default is `KaimingNormalAttrs`, i.e. the *normal*
distribution with the standard deviation of the *uniform* one PyTorch uses. The
two agree on scale and differ only in the shape of the distribution.

Under the Realm backend the filling is done by a task per shard
(`WEIGHT_INIT_TASK_ID`), launched on the device that owns the shard, so a weight
is written by the node that holds it and no weight data crosses the network:
each task generates its own values from the initializer. The check above passes
identically whether the values are produced that way or by the controller, which
is what says the task plumbing is faithful.

## Whole training run

`compare_training_run.py` is the only stage that runs the four stages
*composed*: FlexFlow initializes its own weights and trains for several
iterations with nothing re-synchronized in between, and PyTorch is handed those
same initial weights and runs the identical loop.

It dumps only at the end of the run, deliberately. A per-iteration dump waits on
the context's outstanding events, which puts a barrier between iterations and
hides any missing dependency between them -- which is exactly the class of bug
this stage exists to catch, and did (see below). The finiteness check is the
load-bearing one.

What it cannot check is the *direction* of a weight update. The end-to-end
gradient of this network is chaotic: toggling TF32 within PyTorch, with
identical weights and input, changes the whole-network gradients by about 100%
relative RMS -- a 3.6% difference in a 24-layer forward pass compounds through
24 Jacobians on the way back. That is not a property of any one iteration count;
it is already true after a single step. It is also the reason the rest of this
harness compares per layer instead of end to end: per-layer comparison is not a
convenience, it is the only way to get a discriminating number out of this
model.

What does survive is *magnitude*. The size of each weight's change and the final
loss are compared against the PyTorch TF32-vs-fp32 pair as a noise floor rather
than a fixed tolerance. Typical numbers: FlexFlow's weight changes come out at a
median 0.86x the reference's where the fp32-vs-TF32 control sits at 0.93x, and
the final loss agrees to a few parts in a thousand against a control spread of
about the same.

## Memory

A full training iteration is comfortable on a 32 GB card at batch size 6:
`-ll:fsize 27000` is what the driver uses, and `30000` also works. 300
iterations run in about 4m20s (~0.86 s/iteration).

This was not always true. `get_legion_stream` called `cudaStreamCreate` on every
kernel launch and never destroyed the result -- roughly 2200 streams per
training iteration, each costing on the order of a megabyte of device memory
outside Realm's pool. That capped runs at about 5 full iterations (or 18
forward-only) and forced `fsize` into a narrow 26000-28000 window: too high and
CUDA had no room left for its own resources, too low and Realm could not
allocate its instances. `kernels` now takes the stream from a provider that
`realm-execution` installs (Realm's per-task stream, `Cuda::get_task_cuda_stream`),
falling back to one stream per thread when no runtime has installed one.

## Long runs

1400 iterations run in 19m43s (0.85 s/iteration) and stay numerically sound.
From FlexFlow's own initialization, with nothing re-synchronized at any point:

| after | FlexFlow loss | PyTorch loss, identical loop |
| ----- | ------------- | ---------------------------- |
| 100 iterations  | 1.03493 | 1.03488 |
| 1400 iterations | 1.00526 | 1.00422 |

The gap at 1400 iterations is 1.0e-3 relative, which is *smaller than PyTorch's
own run-to-run spread*: two PyTorch runs of the same 100 steps came out at
1.03488 and 1.03773, a difference of 2.8e-3, because cuDNN's backward
convolution algorithms are not deterministic. So over 1400 composed iterations
FlexFlow stays within the reference's own noise.

Host memory grows during the first few hundred iterations and then levels off,
peaking at 23.3 GB and staying flat from roughly iteration 900 onwards. It is
therefore not an unbounded leak, and wall-clock time rather than memory is what
limits how long a run can be. FlexFlow takes 0.85 s/iteration where PyTorch
takes 0.16 s on the same GPU and model.

Getting there took fixing three things, none of which anything could reach while
the stream leak capped runs at five iterations:

* **The transpose kernel read its own uninitialized output.**
  `transpose_simple_kernel` computes `out = out * beta + in`, and the forward
  pass passes `beta = 0`. Multiplying by zero does not clear a stale bit
  pattern: `0 * NaN` is `NaN`, so whenever a freshly allocated output buffer
  happened to contain a NaN or an infinity, the transpose produced one.
  `compute-sanitizer --tool initcheck` reported 5,184,000 uninitialized reads in
  a single iteration; zero after the fix.

* **The SiLU backward overflowed.** It evaluated the derivative as
  `e^bx (bx + e^bx + 1) / (e^bx + 1)^2`. `expf` overflows to infinity once `bx`
  is much above 88, and the expression then evaluates `inf/inf = NaN`. The
  activations here grow steadily during training (21 -> 35 over the first 14
  iterations in the class head), so crossing that threshold was a matter of
  time. Writing the same derivative as `sigmoid(bx) * (1 + bx * (1 -
  sigmoid(bx)))` is algebraically identical and stays finite. The forward was
  already written in the stable form; only the backward was affected, on both
  the GPU and CPU paths.

  What made this one confusing is that `0 * NaN` is `NaN`: the class head
  receives *exactly zero* gradient when the loss is taken against the box head,
  and it still produced a NaN, which then poisoned the gradient sum for every
  tensor upstream of it.

* **A race between iterations**, described under Known issues in earlier
  revisions and fixed in `zero_gradients_for_pcg_instance`.

`FF_POISON_INSTANCES=1` fills every freshly allocated instance with NaN, which
turns any read-before-write into an immediate, deterministic NaN instead of a
silently plausible number. It is what confirmed no tensor is read before it is
written.

## Known issues

Several bugs had to be fixed before the backward, update and initialization
paths could be validated at all; they are described in the session notes. What
remains:

* **A race between iterations, now fixed.** `zero_gradients_for_pcg_instance`
  issued its fills waiting only on each instance's *allocation* event, which
  triggers once at startup. The barrier in
  `execute_distributed_dynamic_node_invocation_set` makes the tasks it spawns
  depend on the fills, but does nothing to stop the fills running ahead of what
  came before -- so from the second iteration onwards the fills raced the
  previous iteration's backward and update tasks. Three runs in six ended with
  339 of 514 weights NaN; after giving the fills the outstanding events as a
  precondition, zero in six, and the run-to-run spread halved. Only
  `compare_training_run.py` sees this, because every other stage dumps per
  iteration and so inserts the barrier that hides it.

* Nothing outstanding from long runs. Two NaN bugs that only appeared once runs
  could exceed a handful of iterations are described under **Long runs** above;
  both are fixed.

* **`Detect.bias_init()` is not applied.** After building the model, ultralytics
  overwrites the detection head's final biases (`cv2[-1].bias = 1.0`,
  `cv3[-1].bias = log(5/nc/(640/s)^2)`); FlexFlow leaves them at the conv2d
  default. `check_initialization.py` reports these six tensors. It moves the box
  head's output scale noticeably: FlexFlow's `boxes` starts at RMS 0.375 where
  upstream's starts at 2.04.

* **`InitializerAttrs` carries a fixed seed of 0** in every default op-attrs
  picks, so the seed alone cannot distinguish one layer from another. The
  backends therefore salt it with the weight's tensor guid (see
  `lib/realm-execution/src/realm-execution/weight_initialization.cc`). Without
  that, all 513 weights of this model collapse onto 35 distinct values -- every
  convolution of a given kernel shape gets bit-identical weights, including the
  three parallel detection-head branches. A seed that is genuinely part of the
  model would be a better fix than a salt applied at execution time.
