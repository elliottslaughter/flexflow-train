#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PROFILING_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PROFILING_H

#include "kernels/profiling.h"
#include "utils/optional.h"
#include <spdlog/spdlog.h>

namespace FlexFlow {

template <typename F, typename... Ts, typename Str>
std::optional<milliseconds_t>
    profile(F const &f,
            std::optional<ProfilingSettings> const &profiling,
            device_stream_t const &stream,
            Str s,
            Ts &&...ts) {
  if (!profiling.has_value()) {
    f(stream, std::forward<Ts>(ts)...);
    return std::nullopt;
  } else {
    ProfilingSettings settings = assert_unwrap(profiling);
    milliseconds_t elapsed = profiling_wrapper<F, Ts...>(
        f, profiling.value(), stream, std::forward<Ts>(ts)...);
    spdlog::debug(s, elapsed);
    return elapsed;
  }
}

} // namespace FlexFlow

#endif
