#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_MACHINE_SPECIFICATION_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_MACHINE_SPECIFICATION_H

#include "pcg/file_format/v1/v1_machine_specification.dtg.h"
#include "pcg/machine_specification.dtg.h"

namespace FlexFlow {

V1MachineSpecification to_v1(MachineSpecification const &);
MachineSpecification from_v1(V1MachineSpecification const &);

} // namespace FlexFlow

#endif
