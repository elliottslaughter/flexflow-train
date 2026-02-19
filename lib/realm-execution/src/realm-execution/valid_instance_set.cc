#include "realm-execution/valid_instance_set.h"
#include "utils/containers/contains_key.h"

namespace FlexFlow {

std::optional<CopyRequirement>
    ValidInstanceSet::get_copy_for_write(DynamicValueAttrs const &value,
                                         Realm::RegionInstance inst) {
  return this->get_atomic_valid_instance_set(value).get_copy_for_write(inst);
}

std::optional<CopyRequirement>
    ValidInstanceSet::get_copy_for_read(DynamicValueAttrs const &value,
                                        Realm::RegionInstance inst) {
  return this->get_atomic_valid_instance_set(value).get_copy_for_read(inst);
}

AtomicValidInstanceSet &ValidInstanceSet::get_atomic_valid_instance_set(
    DynamicValueAttrs const &value) {
  if (!contains_key(this->atomic_valid_instances, value)) {
    this->atomic_valid_instances.insert({value, AtomicValidInstanceSet{}});
  }
  return this->atomic_valid_instances.at(value);
}

} // namespace FlexFlow
