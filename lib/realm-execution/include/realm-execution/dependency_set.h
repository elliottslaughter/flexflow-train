#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DEPENDENCY_SET_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DEPENDENCY_SET_H

#include "realm-execution/atomic_dependency_set.h"
#include "realm-execution/realm.h"
#include "task-spec/dynamic_graph/dynamic_value_id_t.dtg.h"
#include <map>

namespace FlexFlow {

/**
 * @brief Tracks dependencies on values during the execution of tasks, using the
 * SWMR (single-writer, multiple-reader) algorithm.
 *
 * Values are identified by \ref dynamic_value_id_t rather than by
 * \ref DynamicValueAttrs. This is on the path that issues every task of every
 * iteration, and an id is a pair of small integers where the value it stands
 * for is a structure whose comparison walks a parallel tensor shape, a
 * mapping and a bidict.
 *
 * \warning Ids are only meaningful for the graph they were computed from. See
 * \ref PreparedInvocation.
 */
struct DependencySet {
public:
  DependencySet() = delete;
  explicit DependencySet(Realm::Event precondition);

  void add_writer(dynamic_value_id_t const &value, Realm::Event writer);
  void add_reader(dynamic_value_id_t const &value, Realm::Event reader);

  Realm::Event get_dependency_for_writer(dynamic_value_id_t const &value) const;
  Realm::Event get_dependency_for_reader(dynamic_value_id_t const &value) const;

private:
  AtomicDependencySet &
      get_atomic_dependency_set(dynamic_value_id_t const &value);

private:
  Realm::Event precondition;
  std::map<dynamic_value_id_t, AtomicDependencySet> atomic_dependencies;
};

} // namespace FlexFlow

#endif
