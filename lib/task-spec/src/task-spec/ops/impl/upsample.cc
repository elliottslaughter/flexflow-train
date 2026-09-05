#include "task-spec/ops/impl/upsample.h"
#include "kernels/upsample_kernels.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {

  UpsampleAttrs attrs = acc.get_op_attrs().require_upsample();
  ProfilingSettings profiling = acc.get_profiling_settings();
  device_stream_t stream = acc.get_device_stream();
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(upsample_forward_kernel,
                 profiling,
                 stream,
                 "[Upsample] forward_time = {:.2lf}ms\n",
                 attrs,
                 input,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  device_stream_t stream = acc.get_device_stream();

  UpsampleAttrs attrs = acc.get_op_attrs().require_upsample();
  auto input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  auto output = acc.get_tensor<Permissions::RO>(TensorSlotName::OUTPUT);
  auto output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  auto input_grad = acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);

  return profile(upsample_backward_kernel,
                 profiling,
                 stream,
                 "[Upsample] backward_time = {:.2lf}ms\n",
                 attrs,
                 output,
                 output_grad,
                 input,
                 input_grad);
}

TaskImplFunction get_upsample_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_upsample_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow
