#ifndef TOTALLY_CORRECTIVE_BOOSTING_ORACLES_FACTORY_HPP
#define TOTALLY_CORRECTIVE_BOOSTING_ORACLES_FACTORY_HPP

#include "ConfigFile.hpp"
#include "math/sparse_vector.hpp"

#include <vector>
#include <iosfwd>

namespace totally_corrective_boosting
{

class AbstractOracle; // forward declaration

AbstractOracle *new_oracle_instance(const ConfigFile &config,
                                    const std::vector<SparseVector> &data,
                                    const std::vector<int> &labels,
                                    const bool transposed,
                                    std::ostream &log_stream = std::cout);

} // end of namespace totally_corrective_boosting

#endif // TOTALLY_CORRECTIVE_BOOSTING_ORACLES_FACTORY_HPP
