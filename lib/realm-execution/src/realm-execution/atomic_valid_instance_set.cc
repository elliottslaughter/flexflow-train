#include "realm-execution/atomic_valid_instance_set.h"
#include "realm-execution/copy_requirement.dtg.h"
#include "utils/containers/contains.h"
#include "utils/exception.h"
#include <optional>

namespace FlexFlow {

std::optional<CopyRequirement>
    AtomicValidInstanceSet::add_write(Realm::RegionInstance inst) {
  // We assume that writes may read the existing data, so a valid copy must be
  // provided
  std::optional<CopyRequirement> required_copy =
      this->compute_required_copy(inst, true);
  this->invalid_instances.insert(this->valid_instances.begin(),
                                 this->valid_instances.end());
  this->valid_instances.clear();
  this->valid_instances.insert(inst);
  this->invalid_instances.erase(inst);
  return required_copy;
}

std::optional<CopyRequirement>
    AtomicValidInstanceSet::add_read(Realm::RegionInstance inst) {
  std::optional<CopyRequirement> required_copy =
      this->compute_required_copy(inst, true);
  this->valid_instances.insert(inst);
  this->invalid_instances.erase(inst);
  return required_copy;
}

std::optional<CopyRequirement>
    AtomicValidInstanceSet::compute_required_copy(Realm::RegionInstance inst,
                                                  bool is_write) {
  if (contains(this->valid_instances, inst)) {
    return std::nullopt;
  }

  // FIXME: issue copy from the closest valid instance, not just a random one
  if (this->valid_instances.begin() != this->valid_instances.end()) {
    return CopyRequirement{/*copy_from=*/*this->valid_instances.begin(),
                           /*copy_to=*/inst};
  }

  // We can skip copies if we're writing and there is no existing valid data
  ASSERT(is_write);
  return std::nullopt;
}

} // namespace FlexFlow
