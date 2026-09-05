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
std::map<dynamic_invocation_id_t, std::vector<OpTaskArgs const *>>
    group_registry;

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

void register_op_task_group(
    dynamic_invocation_id_t const &group_id,
    std::vector<dynamic_invocation_id_t> const &member_ids) {
  std::lock_guard<std::mutex> lock{registry_mutex};
  std::vector<OpTaskArgs const *> members;
  for (dynamic_invocation_id_t const &member_id : member_ids) {
    ASSERT(contains_key(registry, member_id),
           "no task arguments were registered for this member of a fused "
           "group",
           member_id);
    // std::map does not move its elements, so these stay valid for as long as
    // the entries do, which is for the life of the process.
    members.push_back(&registry.at(member_id));
  }
  group_registry.insert_or_assign(group_id, members);
}

std::vector<OpTaskArgs const *> const &
    get_registered_op_task_group(dynamic_invocation_id_t const &group_id) {
  std::lock_guard<std::mutex> lock{registry_mutex};
  ASSERT(contains_key(group_registry, group_id),
         "no fused group was registered under this id",
         group_id);
  return group_registry.at(group_id);
}

} // namespace FlexFlow
