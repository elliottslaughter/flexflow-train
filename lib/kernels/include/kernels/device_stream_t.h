#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_DEVICE_STREAM_T_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_DEVICE_STREAM_T_H

#include "kernels/device_stream_t.dtg.h"

namespace FlexFlow {

/**
 * \brief Wrap \p stream, the stream a kernel should place its device work on.
 *
 * Note that this takes the stream rather than producing one. A GPU stream is a
 * resource owned by whoever is running the task: Realm hands each task on a GPU
 * processor a stream of its own and detects the task's device work through it
 * (see \ref RealmContext::get_current_device_stream), and code that is not
 * running under a runtime owns one explicitly via \ref ManagedFFStream. Work
 * left on a stream the runtime did not hand out is not ordered against the
 * task's dependencies, and a stream created per call is a resource leak, so
 * there is deliberately nothing here that will conjure one on demand.
 */
device_stream_t get_gpu_device_stream(ffStream_t stream);

/**
 * \brief The stream value standing for "this kernel runs on the host".
 *
 * \relates get_gpu_device_stream
 */
device_stream_t get_cpu_device_stream();

} // namespace FlexFlow

#endif
