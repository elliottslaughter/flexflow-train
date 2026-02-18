#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_VALID_INSTANCE_SET_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_VALID_INSTANCE_SET_H

#include "realm-execution/atomic_valid_instance_set.h"
#include "realm-execution/copy_requirement.dtg.h"
#include "realm-execution/realm.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include <optional>
#include <unordered_map>

namespace FlexFlow {

struct ValidInstanceSet {
public:
  ValidInstanceSet() = default;

  std::optional<CopyRequirement> add_write(DynamicValueAttrs const &value,
                                           Realm::RegionInstance inst);
  std::optional<CopyRequirement> add_read(DynamicValueAttrs const &value,
                                          Realm::RegionInstance inst);

private:
  AtomicValidInstanceSet &
      get_atomic_valid_instance_set(DynamicValueAttrs const &value);

private:
  std::unordered_map<DynamicValueAttrs, AtomicValidInstanceSet>
      atomic_valid_instances;
};

} // namespace FlexFlow

#endif
