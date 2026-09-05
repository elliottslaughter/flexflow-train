#include "kernels/device_stream_t.h"

namespace FlexFlow {

device_stream_t get_gpu_device_stream(ffStream_t stream) {
  return device_stream_t{stream};
}

device_stream_t get_cpu_device_stream() {
  return device_stream_t{std::monostate{}};
}

} // namespace FlexFlow
