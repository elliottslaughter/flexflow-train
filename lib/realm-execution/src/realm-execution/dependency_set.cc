#include "realm-execution/dependency_set.h"
#include "realm-execution/atomic_dependency_set.h"
#include "utils/containers/contains_key.h"

namespace FlexFlow {

DependencySet::DependencySet(Realm::Event precondition)
    : precondition(precondition) {}

void DependencySet::add_writer(dynamic_value_id_t const &value,
                               Realm::Event writer) {
  AtomicDependencySet &atomic_dependence_set =
      this->get_atomic_dependency_set(value);
  atomic_dependence_set.add_writer(writer);
}

void DependencySet::add_reader(dynamic_value_id_t const &value,
                               Realm::Event reader) {
  AtomicDependencySet &atomic_dependence_set =
      this->get_atomic_dependency_set(value);
  atomic_dependence_set.add_reader(reader);
}

Realm::Event DependencySet::get_dependency_for_writer(
    dynamic_value_id_t const &value) const {
  if (contains_key(this->atomic_dependencies, value)) {
    return this->atomic_dependencies.at(value).get_dependency_for_writer();
  }
  return this->precondition;
}

Realm::Event DependencySet::get_dependency_for_reader(
    dynamic_value_id_t const &value) const {
  if (contains_key(this->atomic_dependencies, value)) {
    return this->atomic_dependencies.at(value).get_dependency_for_reader();
  }
  return this->precondition;
}

AtomicDependencySet &
    DependencySet::get_atomic_dependency_set(dynamic_value_id_t const &value) {
  if (!contains_key(this->atomic_dependencies, value)) {
    this->atomic_dependencies.insert(
        {value, AtomicDependencySet{this->precondition}});
  }
  return this->atomic_dependencies.at(value);
}

} // namespace FlexFlow
