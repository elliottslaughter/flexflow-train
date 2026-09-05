#ifndef _FLEXFLOW_KERNELS_PROFILING_H
#define _FLEXFLOW_KERNELS_PROFILING_H

#include "kernels/device.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/profiling_settings.dtg.h"
#include "pcg/device_type.dtg.h"
#include "utils/units/milliseconds_t.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename F, typename... Ts>
std::optional<milliseconds_t> profiling_wrapper(F const &f,
                                                bool enable_profiling,
                                                device_stream_t const &stream,
                                                Ts &&...ts) {
  if (enable_profiling) {
    ProfilingSettings settings = ProfilingSettings{
        /*warmup_iters=*/0,
        /*measure_iters=*/1,
    };
    return profiling_wrapper<F, Ts...>(
        f, settings, stream, std::forward<Ts>(ts)...);
  } else {
    f(stream, std::forward<Ts>(ts)...);
    return std::nullopt;
  }
}

template <typename F, typename... Ts>
std::optional<milliseconds_t>
    profiling_wrapper(F const &f,
                      ProfilingSettings const &settings,
                      device_stream_t const &stream,
                      Ts &&...ts) {
  if (settings.measure_iters <= 0) {
    // Not measuring. Run the kernel and leave it at that -- note that this
    // still has to *run* it, which is why the check cannot simply return.
    //
    // Measuring is not free: it synchronizes on a CUDA event after every
    // kernel, which stops the host from queueing the next one and leaves the
    // device idle in between. It is worth doing when the per-kernel timings
    // are actually wanted, and worth avoiding otherwise.
    f(stream, std::forward<Ts>(ts)...);
    return std::nullopt;
  }

  if (stream.is_gpu()) {
    return gpu_profiling_wrapper(f, settings, stream, std::forward<Ts>(ts)...);
  } else {
    ASSERT(stream.is_cpu());
    return cpu_profiling_wrapper(f, settings, stream, std::forward<Ts>(ts)...);
  }
}

template <typename F, typename... Ts>
milliseconds_t cpu_profiling_wrapper(F const &f,
                                     ProfilingSettings const &settings,
                                     device_stream_t const &stream,
                                     Ts &&...ts) {
  ASSERT(settings.measure_iters > 0);

  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

  std::optional<TimePoint> start = std::nullopt;
  std::optional<TimePoint> end = std::nullopt;

  for (int i = 0; i < settings.warmup_iters + settings.measure_iters; i++) {
    if (i == settings.warmup_iters) {
      start = std::chrono::steady_clock::now();
    }
    f(stream, std::forward<Ts>(ts)...);
  }
  end = std::chrono::steady_clock::now();

  std::chrono::duration<double, std::milli> avg_duration =
      (end.value() - start.value()) / settings.measure_iters;

  return milliseconds_t{
      static_cast<float>(avg_duration.count()),
  };
}

template <typename F, typename... Ts>
milliseconds_t gpu_profiling_wrapper(F const &f,
                                     ProfilingSettings const &settings,
                                     device_stream_t const &stream,
                                     Ts &&...ts) {
  ASSERT(settings.measure_iters > 0);

  ffEvent_t t_start, t_end;
  checkCUDA(ffEventCreate(&t_start));
  checkCUDA(ffEventCreate(&t_end));

  for (int i = 0; i < settings.warmup_iters + settings.measure_iters; i++) {
    if (i == settings.warmup_iters) {
      checkCUDA(ffEventRecord(t_start, stream.require_gpu()));
    }
    f(stream, std::forward<Ts>(ts)...);
  }

  float elapsed = 0;
  checkCUDA(ffEventRecord(t_end, stream.require_gpu()));
  checkCUDA(ffEventSynchronize(t_end));
  checkCUDA(ffEventElapsedTime(&elapsed, t_start, t_end));
  checkCUDA(ffEventDestroy(t_start));
  checkCUDA(ffEventDestroy(t_end));
  return milliseconds_t{
      elapsed / settings.measure_iters,
  };
}

} // namespace FlexFlow

#endif
