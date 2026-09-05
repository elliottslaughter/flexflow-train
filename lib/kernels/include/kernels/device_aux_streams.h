#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_DEVICE_AUX_STREAMS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_DEVICE_AUX_STREAMS_H

#include "kernels/device.h"

namespace FlexFlow {

/**
 * @brief Streams a task may put work on besides the one it was given, so that
 * work with nothing waiting on it can run alongside the work that does.
 *
 * A task's own stream is the runtime's: everything on it is what the runtime
 * takes the task to have done. These are ours, hung off that stream so that two
 * tasks running at once never share one, and created on first use.
 *
 * Every auxiliary stream carries a cuDNN handle of its own. cuDNN handles are
 * not meant to be used from two streams at once, and the point of these is to
 * have work in flight on two streams at once.
 *
 * \warning Work put on an auxiliary stream is not ordered against the task
 * stream by itself. Use \ref fork_to_aux_stream before it and \ref
 * join_aux_streams before the task returns, or the runtime will take the task
 * to be finished while its work is still running.
 */
int num_aux_streams();

ffStream_t get_aux_stream(ffStream_t task_stream, int index);

ffHandle_t get_aux_dnn_handle(ffStream_t task_stream, int index);

/**
 * @brief Make \p aux wait for everything submitted to \p task_stream so far.
 */
void fork_to_aux_stream(ffStream_t task_stream, ffStream_t aux);

/**
 * @brief Make \p task_stream wait for every auxiliary stream hung off it.
 *
 * Must be called before a task that used one returns.
 */
void join_aux_streams(ffStream_t task_stream);

} // namespace FlexFlow

#endif
