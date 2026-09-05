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
    // Deliberately not freeing the old buffer: kernels already launched on this
    // stream may still be using it, and there is no point at which that is
    // known to be over. Growing happens at most a handful of times.
    checkCUDA(cudaMalloc(&scratch.ptr, size));
    scratch.size = size;
  }
  return scratch.ptr;
}

} // namespace FlexFlow
