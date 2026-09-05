#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_DEVICE_SCRATCH_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_DEVICE_SCRATCH_H

#include "kernels/device.h"
#include <cstddef>

namespace FlexFlow {

/**
 * @brief A scratch buffer of at least \p size bytes that only work on \p stream
 * may use.
 *
 * cuDNN scratch has to belong to the one piece of work using it. \ref
 * PerDeviceFFHandle carries a single buffer per device, which was fine when a
 * device ran one kernel at a time, but kernels from different tasks are handed
 * different streams and do run at once -- so several convolutions would take
 * that one buffer as their workspace simultaneously and write over each other.
 * That corrupts results rather than crashing, and only when the timing lines
 * up.
 *
 * Work submitted to a single stream is ordered, so one buffer per stream is
 * enough to make the use exclusive.
 *
 * \note Buffers are created on first use for a given stream and never freed:
 * a stream belongs to the runtime and outlives any one task, and there is no
 * later point at which the last kernel using a buffer is known to have
 * finished.
 */
void *get_device_scratch_for_stream(ffStream_t stream, size_t size);

} // namespace FlexFlow

#endif
