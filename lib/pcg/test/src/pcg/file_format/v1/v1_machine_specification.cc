#include "pcg/file_format/v1/v1_machine_specification.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("to_v1(MachineSpecification const &)") {
    MachineSpecification input = MachineSpecification{
        /*compute_specification=*/MachineComputeSpecification{
            /*num_nodes=*/8_p,
            /*num_cpus_per_node=*/32_p,
            /*num_gpus_per_node=*/4_p,
        },
        /*interconnect_specification=*/
        MachineInterconnectSpecification{
            /*inter_node_bandwith=*/bytes_per_second_t{58.0},
            /*intra_node_bandwith=*/bytes_per_second_t{32.0},
        },
    };

    V1MachineSpecification result = to_v1(input);
    V1MachineSpecification correct = V1MachineSpecification{
        /*num_nodes=*/8_p,
        /*num_cpus_per_node=*/32_p,
        /*num_gpus_per_node=*/4_p,
        /*inter_node_bandwidth_bytes_per_second=*/58.0,
        /*intra_node_bandwidth_bytes_per_second=*/32.0,
    };

    CHECK(result == correct);
  }

  TEST_CASE("from_v1(V1MachineSpecification const &)") {
    V1MachineSpecification input = V1MachineSpecification{
        /*num_nodes=*/8_p,
        /*num_cpus_per_node=*/32_p,
        /*num_gpus_per_node=*/4_p,
        /*inter_node_bandwidth_bytes_per_second=*/58.0,
        /*intra_node_bandwidth_bytes_per_second=*/32.0,
    };

    MachineSpecification result = from_v1(input);
    MachineSpecification correct = MachineSpecification{
        /*compute_specification=*/MachineComputeSpecification{
            /*num_nodes=*/8_p,
            /*num_cpus_per_node=*/32_p,
            /*num_gpus_per_node=*/4_p,
        },
        /*interconnect_specification=*/
        MachineInterconnectSpecification{
            /*inter_node_bandwith=*/bytes_per_second_t{58.0},
            /*intra_node_bandwith=*/bytes_per_second_t{32.0},
        },
    };

    CHECK(result == correct);
  }
}
