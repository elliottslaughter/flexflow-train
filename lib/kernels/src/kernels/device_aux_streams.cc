#include "kernels/device_aux_streams.h"
#include "internal/device.h"
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace FlexFlow {

namespace {

struct AuxStream {
  ffStream_t stream = nullptr;
  ffHandle_t dnn = nullptr;
  // Reused round-robin. A wait is bound to the recording that preceded it, so
  // an event may be recorded again once the wait naming it has been submitted;
  // the pool is only here so that does not have to be reasoned about.
  std::vector<ffEvent_t> events;
  size_t next_event = 0;
  ffEvent_t join_event = nullptr;
  bool used = false;
};

struct AuxStreams {
  std::vector<AuxStream> streams;
};

std::mutex aux_mutex;
std::map<ffStream_t, AuxStreams> aux_by_task_stream;

AuxStream &get_or_create(ffStream_t task_stream, int index) {
  AuxStreams &streams = aux_by_task_stream[task_stream];
  while (streams.streams.size() <= static_cast<size_t>(index)) {
    AuxStream created;
    checkCUDA(
        cudaStreamCreateWithFlags(&created.stream, cudaStreamNonBlocking));
    checkCUDNN(cudnnCreate(&created.dnn));
    checkCUDA(
        cudaEventCreateWithFlags(&created.join_event, cudaEventDisableTiming));
    streams.streams.push_back(created);
  }
  return streams.streams.at(index);
}

} // namespace

int num_aux_streams() {
  static int const count = []() {
    char const *value = std::getenv("FF_AUX_STREAMS");
    return value == nullptr ? 0 : std::stoi(value);
  }();
  return count;
}

ffStream_t get_aux_stream(ffStream_t task_stream, int index) {
  std::lock_guard<std::mutex> lock{aux_mutex};
  return get_or_create(task_stream, index).stream;
}

ffHandle_t get_aux_dnn_handle(ffStream_t task_stream, int index) {
  std::lock_guard<std::mutex> lock{aux_mutex};
  return get_or_create(task_stream, index).dnn;
}

void fork_to_aux_stream(ffStream_t task_stream, ffStream_t aux) {
  std::lock_guard<std::mutex> lock{aux_mutex};
  AuxStreams &streams = aux_by_task_stream.at(task_stream);
  for (AuxStream &candidate : streams.streams) {
    if (candidate.stream != aux) {
      continue;
    }
    if (candidate.events.size() < 256) {
      ffEvent_t created;
      checkCUDA(cudaEventCreateWithFlags(&created, cudaEventDisableTiming));
      candidate.events.push_back(created);
    }
    ffEvent_t event =
        candidate.events.at(candidate.next_event % candidate.events.size());
    candidate.next_event += 1;
    checkCUDA(cudaEventRecord(event, task_stream));
    checkCUDA(cudaStreamWaitEvent(aux, event, 0));
    candidate.used = true;
    return;
  }
}

void join_aux_streams(ffStream_t task_stream) {
  std::lock_guard<std::mutex> lock{aux_mutex};
  auto found = aux_by_task_stream.find(task_stream);
  if (found == aux_by_task_stream.end()) {
    return;
  }
  for (AuxStream &aux : found->second.streams) {
    if (!aux.used) {
      continue;
    }
    checkCUDA(cudaEventRecord(aux.join_event, aux.stream));
    checkCUDA(cudaStreamWaitEvent(task_stream, aux.join_event, 0));
    aux.used = false;
  }
}

} // namespace FlexFlow
