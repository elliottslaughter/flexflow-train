#include "realm-execution/op_task_arg_registry.h"
#include "utils/containers/contains_key.h"
#include <libassert/assert.hpp>
#include <map>
#include <mutex>

namespace FlexFlow {

namespace {

// Written by the registration tasks during initialization and read by every
// task afterwards, from whichever threads the runtime happens to use.
std::mutex registry_mutex;
std::map<dynamic_invocation_id_t, OpTaskArgs> registry;

} // namespace

void register_op_task_args(dynamic_invocation_id_t const &invocation_id,
                           OpTaskArgs const &args) {
  std::lock_guard<std::mutex> lock{registry_mutex};
  registry.insert_or_assign(invocation_id, args);
}

OpTaskArgs const &
    get_registered_op_task_args(dynamic_invocation_id_t const &invocation_id) {
  std::lock_guard<std::mutex> lock{registry_mutex};
  ASSERT(contains_key(registry, invocation_id),
         "no task arguments were registered for this invocation",
         invocation_id);
  return registry.at(invocation_id);
}

} // namespace FlexFlow
