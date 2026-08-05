#include "pcg/file_format/v1/v1_machine_specification.h"

namespace FlexFlow {

V1MachineSpecification to_v1(MachineSpecification const &m) {
  return V1MachineSpecification{
      /*num_nodes=*/m.compute_specification.num_nodes,
      /*num_cpus_per_node=*/m.compute_specification.num_cpus_per_node,
      /*num_gpus_per_node=*/m.compute_specification.num_gpus_per_node,
      /*inter_node_bandwidth_bytes_per_second=*/
      m.interconnect_specification.inter_node_bandwidth
          .unwrap_bytes_per_second(),
      /*intra_node_bandwidth_bytes_per_second=*/
      m.interconnect_specification.intra_node_bandwidth
          .unwrap_bytes_per_second(),
  };
}

MachineSpecification from_v1(V1MachineSpecification const &m) {
  return MachineSpecification{
      /*compute_specification=*/MachineComputeSpecification{
          /*num_nodes=*/m.num_nodes,
          /*num_cpus_per_node=*/m.num_cpus_per_node,
          /*num_gpus_per_node=*/m.num_gpus_per_node,
      },
      /*interconnect_specification=*/
      MachineInterconnectSpecification{
          /*inter_node_bandwith=*/bytes_per_second_t{
              m.inter_node_bandwidth_bytes_per_second},
          /*intra_node_bandwith=*/
          bytes_per_second_t{m.intra_node_bandwidth_bytes_per_second},
      },
  };
}

} // namespace FlexFlow
