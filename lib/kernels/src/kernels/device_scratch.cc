#include "kernels/device_scratch.h"
#include <map>
#include <mutex>

namespace FlexFlow {

namespace {

struct Scratch {
  void *ptr = nullptr;
  size_t size = 0;
};

std::mutex scratch_mutex;
std::map<ffStream_t, Scratch> scratch_by_stream;

} // namespace

void *get_device_scratch_for_stream(ffStream_t stream, size_t size) {
  std::lock_guard<std::mutex> lock{scratch_mutex};

  Scratch &scratch = scratch_by_stream[stream];
  if (scratch.size < size) {
    if (scratch.ptr != nullptr) {
      // Work already submitted to this stream may still be reading the old
      // buffer, and nothing else can be: a buffer belongs to one stream, and
      // work on a single stream runs in order. So waiting on the stream is
      // enough to make freeing it safe. This costs a synchronization, but a
      // stream's scratch only ever grows to the largest a kernel on it asks
      // for, so it happens a handful of times and then never again.
      checkCUDA(cudaStreamSynchronize(stream));
      checkCUDA(cudaFree(scratch.ptr));
      scratch.ptr = nullptr;
      scratch.size = 0;
    }
    checkCUDA(cudaMalloc(&scratch.ptr, size));
    scratch.size = size;
  }
  return scratch.ptr;
}

} // namespace FlexFlow
