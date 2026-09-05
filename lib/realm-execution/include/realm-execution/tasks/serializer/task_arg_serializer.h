#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_TASK_ARG_SERIALIZER_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_TASK_ARG_SERIALIZER_H

#include <cstdint>
#include <nlohmann/json.hpp>
#include <vector>

namespace FlexFlow {

/**
 * \brief Encode \p args into the buffer sent along with a task launch.
 *
 * MessagePack rather than JSON text: the arguments are a few kilobytes of
 * structure per task and there are thousands of tasks per training iteration,
 * so the encoding shows up directly in how fast the runtime can issue work and
 * how long a task takes to start. Text encoding and decoding accounted for
 * roughly two thirds of the time spent in serialization; the binary form is a
 * drop-in for it, since nlohmann builds the same DOM either way.
 */
template <typename T>
std::vector<std::uint8_t> serialize_task_args(T const &args) {
  nlohmann::json j = args;
  return nlohmann::json::to_msgpack(j);
}

/**
 * \brief Decode what \ref serialize_task_args produced.
 */
template <typename T>
T deserialize_task_args(void const *args, size_t arglen) {
  std::uint8_t const *bytes = static_cast<std::uint8_t const *>(args);
  return nlohmann::json::from_msgpack(bytes, bytes + arglen).template get<T>();
}

} // namespace FlexFlow

#endif
