## FlexFlow Overview

FlexFlow is a compiler and runtime system for training machine learning models.
The vast majority of FlexFlow is factored into libraries located under `lib`.
For example, `lib/compiler` is the FlexFlow compiler.
Here are some of the major components of FlexFlow and their roles:

* `lib/compiler`: the FlexFlow compiler. Takes a `ComputationGraph` and optimizes it to produce a `MappedParallelComputationGraph`
* `lib/kernels`: specific (CPU and GPU) kernels for various operators used in machine learning.
* `lib/local-execution`: a single-device (CPU or GPU) execution engine for `ComputationGraph`s.
* `lib/models`: specific machine-learning models implemented in the FlexFlow API.
* `lib/op-attrs`: the list of operators supported by FlexFlow, and their "attrs" or attributes that determine how to run them.
* `lib/pcg`: the `ParallelComputationGraph` and variant `MappedParallelComputationGraph`, along with utilities to work with them.
* `lib/task-spec`: a middleware layer between the compiler and runtime system. Defines a "dynamic graph" data structure that is more explicit than the `MappedParallelComputationGraph` and is progressively lowered to contain information necessary for execution.
* `lib/utils`: a wide variety of utilities: data structures, algorithms, adapters for various libraries (formatting, JSON-encoding, hashing), etc.

Specific binaries that use `lib` to implement specific functions are located in `bin`.
These are usually thin wrappers on the corresponding `lib` functionality.

## `proj` Overview

Everything in the project is managed via a command named `proj`. E.g.:

* Build with `proj build`
* Test with `proj test --skip-gpu-tests` (on a machine without GPUs). This also builds FlexFlow, if it hasn't been built already
* Format with `proj format`

If a tests fails it will include the command to rerun that specific test inside a debugger.

IMPORTANT: Any file that contains `.dtg` in the filename (e.g., `.dtg.h` or `.dtg.cc`) is a generated file.
These files are produced automatically from the corresponding `.dtg.toml` when running `proj build` or `proj test`.
Generated files should never be modified directly; instead modify the corresponding `.dtg.toml` file.
