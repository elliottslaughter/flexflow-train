#include "kernels/accessor.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "op-attrs/ops/loss_functions/nonconfigurable_loss_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/file_format/v1/v1_mapped_parallel_computation_graph.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "realm-execution/distributed_ff_handle.h"
#include "realm-execution/dynamic_tensor_accessor_from_instance.h"
#include "realm-execution/parallel_loss_config.dtg.h"
#include "realm-execution/pcg_instance.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/realm_manager.h"
#include "task-spec/dynamic_graph/dynamic_loss_tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_task_type.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.dtg.h"
#include "task-spec/fwb_tensor_type.dtg.h"
#include "task-spec/permissions.h"
#include "utils/containers/contains_key.h"
#include "utils/cli/cli_get_help_message.h"
#include "utils/cli/cli_parse.h"
#include "utils/cli/cli_parse_result.h"
#include "utils/cli/cli_spec.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/positive_int/positive_int.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string_view>

using namespace FlexFlow;

static char *leak_string_contents(std::string_view str) {
  // Realm command-line arguments require char* so intentionally leak the
  // allocated string contents here
  std::vector<char> *content = new std::vector<char>{str.begin(), str.end()};
  content->push_back(0); // NUL byte
  return content->data();
}

static std::vector<char *> make_realm_args(std::string_view executable_name) {
  std::vector<char *> result;
  result.push_back(leak_string_contents(executable_name));
  return result;
}

/**
 * \name Named-tensor I/O
 *
 * The options below make it possible to feed externally-generated data into a
 * model (e.g., an input batch and a set of weights exported from PyTorch) and
 * to read individual tensors back out, which is what makes it possible to
 * compare FlexFlow's numerics against another framework's. They are driven by
 * environment variables rather than command-line flags because
 * \ref utils/cli only supports boolean flags and positional arguments.
 *
 * - <tt>FF_LOAD_TENSORS</tt>: comma-separated paths to tensor files (see \ref
 *   TensorFileEntry) whose contents are copied into the correspondingly-named
 *   forward tensors before running.
 * - <tt>FF_DUMP_TENSORS</tt>: path to write a tensor file to after running.
 * - <tt>FF_DUMP_NAMES</tt>: comma-separated list of tensor names to write to
 *   <tt>FF_DUMP_TENSORS</tt>.
 * - <tt>FF_FORWARD_ONLY</tt>: if set to 1, run only the forward pass (so that
 *   the loaded weights are not modified by the optimizer).
 * - <tt>FF_LOSS</tt>: the loss function to attach
 *   (<tt>mean_squared_error_avg</tt>, <tt>mean_squared_error_sum</tt>,
 *   <tt>categorical_crossentropy</tt> or <tt>identity</tt>). Without a loss
 *   nothing seeds the gradient of the output and the backward pass computes
 *   only zeros.
 * - <tt>FF_LOSS_LOGIT</tt>: the name of the tensor the loss is taken against.
 *   Set together with <tt>FF_LOSS</tt>.
 * - <tt>FF_ITERATIONS</tt>: number of iterations to run (default 5).
 * - <tt>FF_LIST_TENSORS</tt>: if set to 1, print the name and shape of every
 *   nameable forward tensor and exit.
 *
 * A tensor is named after the layer that produces it, so, e.g., the weights of
 * the layer named <tt>model.0.conv</tt> are named
 * <tt>model.0.conv.FILTER</tt>, and the output of that layer is named
 * <tt>model.0.conv</tt>. A tensor's gradient takes the same name with a
 * <tt>grad:</tt> prefix, and the label tensor created by loss insertion is
 * named <tt>label</tt>.
 * \{
 */

static constexpr char TENSOR_FILE_MAGIC[8] = {
    'F', 'F', 'T', 'E', 'N', 'S', 'R', '1'};

struct TensorFileEntry {
  std::string name;
  std::vector<int64_t> dims;
  std::vector<char> data;
};

template <typename T>
static void read_raw(std::istream &s, T &value) {
  s.read(reinterpret_cast<char *>(&value), sizeof(T));
  ASSERT(!s.fail(), "unexpected end of tensor file");
}

template <typename T>
static void write_raw(std::ostream &s, T const &value) {
  s.write(reinterpret_cast<char const *>(&value), sizeof(T));
}

static std::vector<TensorFileEntry> read_tensor_file(std::string const &path) {
  std::ifstream f(path, std::ios::binary);
  ASSERT(f.good(), "could not open tensor file", path);

  char magic[8];
  f.read(magic, sizeof(magic));
  ASSERT(std::memcmp(magic, TENSOR_FILE_MAGIC, sizeof(magic)) == 0,
         "tensor file has an unexpected magic number",
         path);

  int64_t num_entries;
  read_raw(f, num_entries);

  std::vector<TensorFileEntry> result;
  for (int64_t i = 0; i < num_entries; i++) {
    TensorFileEntry entry;

    int32_t name_len;
    read_raw(f, name_len);
    entry.name.resize(name_len);
    f.read(entry.name.data(), name_len);

    int32_t num_dims;
    read_raw(f, num_dims);
    entry.dims.resize(num_dims);
    for (int32_t d = 0; d < num_dims; d++) {
      read_raw(f, entry.dims.at(d));
    }

    int64_t num_bytes;
    read_raw(f, num_bytes);
    entry.data.resize(num_bytes);
    f.read(entry.data.data(), num_bytes);
    ASSERT(!f.fail(), "unexpected end of tensor file", path, entry.name);

    result.push_back(entry);
  }

  return result;
}

static void write_tensor_file(std::string const &path,
                              std::vector<TensorFileEntry> const &entries) {
  std::ofstream f(path, std::ios::binary);
  ASSERT(f.good(), "could not open tensor file for writing", path);

  f.write(TENSOR_FILE_MAGIC, sizeof(TENSOR_FILE_MAGIC));
  write_raw(f, static_cast<int64_t>(entries.size()));

  for (TensorFileEntry const &entry : entries) {
    write_raw(f, static_cast<int32_t>(entry.name.size()));
    f.write(entry.name.data(), entry.name.size());

    write_raw(f, static_cast<int32_t>(entry.dims.size()));
    for (int64_t dim : entry.dims) {
      write_raw(f, dim);
    }

    write_raw(f, static_cast<int64_t>(entry.data.size()));
    f.write(entry.data.data(), entry.data.size());
  }

  ASSERT(f.good(), "error while writing tensor file", path);
}

static std::optional<std::string> get_env(char const *name) {
  char const *value = std::getenv(name);
  if (value == nullptr || std::string{value}.empty()) {
    return std::nullopt;
  }
  return std::string{value};
}

static bool get_env_flag(char const *name) {
  return get_env(name) == std::optional<std::string>{"1"};
}

static LossAttrs parse_loss_attrs(std::string const &name) {
  if (name == "mean_squared_error_avg") {
    return LossAttrs{NonconfigurableLossAttrs{
        LossFunction::MEAN_SQUARED_ERROR_AVG_REDUCE}};
  } else if (name == "mean_squared_error_sum") {
    return LossAttrs{NonconfigurableLossAttrs{
        LossFunction::MEAN_SQUARED_ERROR_SUM_REDUCE}};
  } else if (name == "categorical_crossentropy") {
    return LossAttrs{
        NonconfigurableLossAttrs{LossFunction::CATEGORICAL_CROSSENTROPY}};
  } else if (name == "identity") {
    return LossAttrs{NonconfigurableLossAttrs{LossFunction::IDENTITY}};
  } else {
    PANIC("Unknown loss function", name);
  }
}

static std::vector<std::string> split_on_commas(std::string const &s) {
  std::vector<std::string> result;
  std::string current;
  for (char c : s) {
    if (c == ',') {
      if (!current.empty()) {
        result.push_back(current);
      }
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) {
    result.push_back(current);
  }
  return result;
}

/**
 * \brief Strip the quotes that FlexFlow's generated <tt>format_as</tt>
 * functions put around enum values.
 */
static std::string strip_quotes(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '"'), s.end());
  return s;
}

/**
 * \brief Map each named forward tensor of \p mpcg to the corresponding value
 * in \p pcg_instance's \ref TensorInstanceBacking.
 *
 * A tensor takes the name of the layer that produces it (see
 * \ref ComputationGraphBuilder::add_layer for how weights get their names).
 */
static std::map<parallel_tensor_guid_t, std::string>
    get_parallel_tensor_names(MappedParallelComputationGraph const &mpcg) {
  std::map<parallel_tensor_guid_t, std::string> tensor_names;
  for (parallel_layer_guid_t layer : mpcg_get_parallel_layers(mpcg)) {
    MappedParallelLayerAttrs attrs = mpcg.raw_graph.at(layer.raw_graph_node);
    if (!attrs.name.has_value()) {
      continue;
    }
    std::string layer_name = attrs.name.value();
    for (auto const &[slot, tensor] : mpcg_get_outgoing_tensors(mpcg, layer)) {
      // Layers with a single output (the common case) name their output after
      // themselves; layers with multiple outputs (e.g., split) additionally
      // qualify each output with its slot name.
      if (slot == TensorSlotName::OUTPUT) {
        tensor_names.insert({tensor, layer_name});
      } else {
        tensor_names.insert(
            {tensor, layer_name + "." + strip_quotes(format_as(slot))});
      }
    }
  }
  return tensor_names;
}

/**
 * \brief The name given to the label tensor that loss insertion creates.
 *
 * The label lies outside the computation graph, so unlike every other tensor it
 * is not named after a layer.
 */
static constexpr char LABEL_TENSOR_NAME[] = "label";

static std::map<std::string, DynamicValueAttrs>
    get_named_forward_values(MappedParallelComputationGraph const &mpcg,
                             PCGInstance const &pcg_instance) {
  std::map<parallel_tensor_guid_t, std::string> tensor_names =
      get_parallel_tensor_names(mpcg);

  std::map<std::string, std::vector<DynamicValueAttrs>> candidates;
  for (auto const &[value, instance] :
       pcg_instance.get_tensor_instance_backing().backing) {
    if (value.tensor_guid.has<dynamic_loss_tensor_guid_t>()) {
      candidates[LABEL_TENSOR_NAME].push_back(value);
      continue;
    }
    if (!value.tensor_guid.has<parallel_tensor_guid_t>()) {
      continue;
    }
    parallel_tensor_guid_t tensor =
        value.tensor_guid.get<parallel_tensor_guid_t>();
    if (!contains_key(tensor_names, tensor)) {
      continue;
    }

    // Gradients get the same name as the tensor they are the gradient of, with
    // a "grad:" prefix. Values carrying a subgradient_id are the per-use
    // partial gradients that pass expansion creates for a tensor that is
    // consumed more than once; only their sum (the value without a
    // subgradient_id) is the gradient of the tensor.
    std::string prefix;
    if (value.role == DynamicTensorRole{FwbTensorType::FORWARD}) {
      prefix = "";
    } else if (value.role == DynamicTensorRole{FwbTensorType::GRADIENT} &&
               value.subgradient_id == std::nullopt) {
      prefix = "grad:";
    } else {
      continue;
    }
    candidates[prefix + tensor_names.at(tensor)].push_back(value);
  }

  // Layers that were not explicitly named fall back to a default name derived
  // from their operator type (e.g., "EW_ADD"), so a name is only usable if
  // exactly one tensor has it. Ambiguous names are simply dropped: referring
  // to one later on then fails with a "not present in the model" error.
  std::map<std::string, DynamicValueAttrs> result;
  for (auto const &[name, values] : candidates) {
    if (values.size() == 1) {
      result.insert({name, values.at(0)});
    }
  }

  return result;
}

/**
 * \brief Build the loss configuration that seeds the gradient of the tensor
 * named \p logit_name.
 *
 * Without a loss there is nothing to start the backward pass from, so every
 * gradient comes out zero.
 */
static ParallelLossConfig
    make_loss_config(MappedParallelComputationGraph const &mpcg,
                     RealmContext &ctx,
                     LossAttrs const &loss_attrs,
                     std::string const &logit_name) {
  std::map<parallel_tensor_guid_t, std::string> tensor_names =
      get_parallel_tensor_names(mpcg);

  std::optional<parallel_tensor_guid_t> logit_tensor = std::nullopt;
  for (auto const &[tensor, name] : tensor_names) {
    if (name == logit_name) {
      ASSERT(logit_tensor == std::nullopt,
             "more than one tensor has the requested logit name",
             logit_name);
      logit_tensor = tensor;
    }
  }
  ASSERT(logit_tensor.has_value(),
         "no tensor in the model has the requested logit name",
         logit_name);

  parallel_layer_guid_t logit_layer =
      mpcg_get_source_layer(mpcg, logit_tensor.value());

  TensorSlotName logit_slot = [&] {
    for (auto const &[slot, tensor] :
         mpcg_get_outgoing_tensors(mpcg, logit_layer)) {
      if (tensor == logit_tensor.value()) {
        return slot;
      }
    }
    PANIC("logit tensor is not produced by its own source layer", logit_name);
  }();

  // The loss node consumes the label (in slot INPUT) and the logit (in slot
  // LOGIT), both of which have the same shape and sharding as the logit. So its
  // task group is the task group of the layer producing the logit, with that
  // layer's coordinate for the logit bound to both of the loss node's slots.
  // Held in a local because get_shard_bindings() returns a reference into the
  // task group, which a range-for does not keep alive if it is a temporary.
  MappedOperatorTaskGroup logit_layer_group =
      mpcg_get_mapping_for_layer(mpcg, logit_layer);

  bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> loss_bindings;
  for (auto const &[machine_coord, binding] :
       logit_layer_group.get_shard_bindings()) {
    ParallelTensorSpaceCoordinate logit_coord =
        binding.tensor_coords.at(logit_slot);

    loss_bindings.equate(machine_coord,
                         OperatorAtomicTaskShardBinding{{
                             {TensorSlotName::INPUT, logit_coord},
                             {TensorSlotName::LOGIT, logit_coord},
                         }});
  }

  ParallelTensorShape logit_shape =
      mpcg_get_parallel_tensor_attrs(mpcg, logit_tensor.value()).shape;

  // create_pcg_instance wants a label accessor, but it keys it on a
  // pre-shard-expansion value that no longer matches anything by the time
  // instance allocation runs, so the accessor is dropped and an instance is
  // allocated for the label like for any other tensor. Fill that instance by
  // name (see LABEL_TENSOR_NAME) instead of relying on this accessor.
  GenericTensorAccessorW label_tensor =
      ctx.get_current_device_allocator().allocate_tensor(
          get_piece_shape(logit_shape));

  return ParallelLossConfig{
      /*loss_attrs=*/loss_attrs,
      /*label_tensor=*/read_only_accessor_from_write_accessor(label_tensor),
      /*logit_tensor=*/logit_tensor.value(),
      /*loss_mapping=*/MappedOperatorTaskGroup{loss_bindings},
  };
}

/**
 * \brief Get a \ref GenericTensorAccessorW pointing at the (possibly
 * device-resident) storage backing \p value.
 */
static GenericTensorAccessorW
    get_accessor_for_value(PCGInstance &pcg_instance,
                           RealmContext &ctx,
                           DynamicValueAttrs const &value) {
  auto const &instance =
      pcg_instance.get_tensor_instance_backing().backing.at(value);

  return dynamic_tensor_accessor_from_instance(
             /*inst=*/instance.first,
             /*ready=*/instance.second,
             /*parallel_tensor_shape=*/value.parallel_tensor_shape.value(),
             /*permissions=*/Permissions::RW,
             /*for_processor=*/ctx.get_current_processor())
      .require_write();
}

static std::vector<int64_t> get_dims(TensorShape const &shape) {
  std::vector<int64_t> result;
  for (positive_int dim : shape.dims.ff_ordered) {
    result.push_back(dim.int_from_positive_int());
  }
  return result;
}

///\}

int main(int argc, char **argv) {
  CLISpec cli = empty_cli_spec();

  CLIArgumentKey arg_key_help = cli_add_help_flag(cli);

  CLIArgumentKey key_mapped_pcg_json = cli_add_positional_argument(
      cli,
      CLIPositionalArgumentSpec{
          "mapped_pcg_json",
          std::nullopt,
          "path to a file containing mappped PCG encoded as JSON"});

  ASSERT(argc >= 1);
  std::string prog_name = argv[0];

  CLIParseResult parsed = ({
    tl::expected<CLIParseResult, std::string> result =
        cli_parse(cli, argc, argv);
    if (!result.has_value()) {
      std::string error_msg = result.error();
      std::cerr << cli_get_help_message(prog_name, cli);
      std::cerr << std::endl;
      std::cerr << "error: " << error_msg << std::endl;
      return 1;
    }

    result.value();
  });

  bool help = cli_get_flag(parsed, arg_key_help);
  if (help) {
    std::cerr << cli_get_help_message(prog_name, cli);
    return 1;
  }

  std::string mapped_pcg_json = cli_get_argument(parsed, key_mapped_pcg_json);

  std::optional<std::string> load_tensors_paths = get_env("FF_LOAD_TENSORS");
  std::optional<std::string> dump_tensors_path = get_env("FF_DUMP_TENSORS");
  std::vector<std::string> dump_tensor_names =
      split_on_commas(get_env("FF_DUMP_NAMES").value_or(""));
  bool forward_only = get_env_flag("FF_FORWARD_ONLY");
  std::optional<std::string> loss_name = get_env("FF_LOSS");
  std::optional<std::string> loss_logit_name = get_env("FF_LOSS_LOGIT");
  ASSERT(loss_name.has_value() == loss_logit_name.has_value(),
         "FF_LOSS and FF_LOSS_LOGIT must be set together");
  bool list_tensors = get_env_flag("FF_LIST_TENSORS");
  int num_iterations =
      std::stoi(get_env("FF_ITERATIONS").value_or(std::string{"5"}));

  std::vector<char *> realm_args = make_realm_args(prog_name);
  int realm_argc = realm_args.size();
  char **realm_argv = realm_args.data();
  RealmManager manager(&realm_argc, &realm_argv);

  ControllerTaskResult result =
      manager.start_controller([&](RealmContext &ctx) {
        MappedParallelComputationGraph mpcg = [&]() {
          std::ifstream f(mapped_pcg_json);
          nlohmann::json mpcg_json = nlohmann::json::parse(f);
          return from_v1(mpcg_json.get<V1MappedParallelComputationGraph>());
        }();

        // instantiate computation graph
        OptimizerAttrs optimizer_attrs =
            OptimizerAttrs{SGDOptimizerAttrs{/*lr=*/0.001,
                                             /*momentum=*/0.9,
                                             /*nesterov=*/false,
                                             /*weight_decay=*/0.001}};

        std::map<DynamicValueAttrs, DynamicTensorAccessor> input_tensors;

        DistributedFfHandle device_handle =
            create_distributed_ff_handle(ctx,
                                         /*workSpaceSize=*/1024 * 1024,
                                         /*allowTensorOpMathConversion=*/true);

        bool has_gpus = []() {
          FlexFlow::Realm::Machine::ProcessorQuery pq(
              FlexFlow::Realm::Machine::get_machine());
          pq.only_kind(FlexFlow::Realm::Processor::Kind::TOC_PROC);
          return pq.count() > 0;
        }();

        std::optional<ParallelLossConfig> loss = std::nullopt;
        if (loss_name.has_value()) {
          loss = make_loss_config(/*mpcg=*/mpcg,
                                  /*ctx=*/ctx,
                                  /*loss_attrs=*/
                                  parse_loss_attrs(loss_name.value()),
                                  /*logit_name=*/loss_logit_name.value());
        }

        PCGInstance pcg_instance = create_pcg_instance(
            /*ctx=*/ctx,
            /*mpcg=*/mpcg,
            /*optimizer=*/optimizer_attrs,
            /*loss=*/loss,
            /*input_tensors=*/input_tensors,
            /*profiling_settings=*/ProfilingSettings{0, 1},
            /*device_handle=*/device_handle,
            /*device_type=*/has_gpus ? DeviceType::GPU : DeviceType::CPU);

        std::map<std::string, DynamicValueAttrs> named_values =
            get_named_forward_values(mpcg, pcg_instance);

        if (get_env_flag("FF_LIST_TASKS")) {
          std::map<std::string, int> counts;
          for (DynamicNodeInvocation const &invocation :
               pcg_instance.get_execution_order()) {
            counts[fmt::to_string(
                assert_unwrap(invocation.node_attrs.task_type))] += 1;
          }
          for (auto const &[task_type, count] : counts) {
            std::cout << task_type << " " << count << std::endl;
          }
          return;
        }

        if (list_tensors) {
          for (auto const &[name, value] : named_values) {
            std::cout << name << " "
                      << get_piece_shape(value.parallel_tensor_shape.value())
                      << std::endl;
          }
          return;
        }

        for (std::string const &path :
             split_on_commas(load_tensors_paths.value_or(""))) {
          std::vector<TensorFileEntry> entries = read_tensor_file(path);
          for (TensorFileEntry const &entry : entries) {
            ASSERT(contains_key(named_values, entry.name),
                   "tensor file contains a tensor that is not present in the "
                   "model",
                   entry.name);
            DynamicValueAttrs value = named_values.at(entry.name);
            TensorShape shape =
                get_piece_shape(value.parallel_tensor_shape.value());
            ASSERT(get_dims(shape) == entry.dims,
                   "tensor file and model disagree about a tensor's shape",
                   entry.name,
                   shape);

            GenericTensorAccessorW dst =
                get_accessor_for_value(pcg_instance, ctx, value);
            GenericTensorAccessorR src = GenericTensorAccessorR{
                shape,
                const_cast<char *>(entry.data.data()),
                DeviceType::CPU,
            };
            copy_accessor_data_to_l_from_r(dst, src);
          }
          std::cerr << "run-model: loaded " << entries.size()
                    << " tensors from " << path << std::endl;
        }

        // begin training loop
        for (int i = 0; i < num_iterations; i++) {
          if (forward_only) {
            perform_forward_pass_for_pcg_instance(
                /*instance=*/pcg_instance,
                /*profiling_settings=*/ProfilingSettings{0, 1},
                /*device_handle=*/device_handle);
          } else {
            perform_all_passes_for_pcg_instance(
                /*instance=*/pcg_instance,
                /*profiling_settings=*/ProfilingSettings{0, 1},
                /*device_handle=*/device_handle);
          }
        }

        if (dump_tensors_path.has_value()) {
          ctx.get_outstanding_events().wait();

          std::vector<TensorFileEntry> entries;
          for (std::string const &name : dump_tensor_names) {
            ASSERT(contains_key(named_values, name),
                   "requested a dump of a tensor that is not present in the "
                   "model",
                   name);
            DynamicValueAttrs value = named_values.at(name);
            TensorShape shape =
                get_piece_shape(value.parallel_tensor_shape.value());

            GenericTensorAccessorW src =
                get_accessor_for_value(pcg_instance, ctx, value);

            TensorFileEntry entry;
            entry.name = name;
            entry.dims = get_dims(shape);
            entry.data.resize(get_size_in_bytes(shape)
                                  .unwrap_num_bytes()
                                  .unwrap_nonnegative());

            GenericTensorAccessorW dst = GenericTensorAccessorW{
                shape,
                entry.data.data(),
                DeviceType::CPU,
            };
            copy_accessor_data_to_l_from_r(
                dst, read_only_accessor_from_write_accessor(src));

            entries.push_back(entry);
          }
          write_tensor_file(dump_tensors_path.value(), entries);
          std::cerr << "run-model: dumped " << entries.size() << " tensors to "
                    << dump_tensors_path.value() << std::endl;
        }
      });
  result.wait();

  return 0;
}
