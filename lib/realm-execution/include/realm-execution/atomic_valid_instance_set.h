#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_ATOMIC_VALID_INSTANCE_SET_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_ATOMIC_VALID_INSTANCE_SET_H

#include "realm-execution/copy_requirement.dtg.h"
#include "realm-execution/realm.h"
#include <optional>
#include <unordered_set>

namespace FlexFlow {

struct AtomicValidInstanceSet {
public:
  AtomicValidInstanceSet() = default;

  std::optional<CopyRequirement> add_write(Realm::RegionInstance inst);
  std::optional<CopyRequirement> add_read(Realm::RegionInstance inst);

private:
  std::optional<CopyRequirement>
      compute_required_copy(Realm::RegionInstance inst, bool is_write);

private:
  std::unordered_set<Realm::RegionInstance> valid_instances;
  std::unordered_set<Realm::RegionInstance> invalid_instances;
};

} // namespace FlexFlow

#endif
