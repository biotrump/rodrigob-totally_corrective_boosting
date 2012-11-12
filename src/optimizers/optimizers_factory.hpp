#ifndef TOTALLY_CORRECTIVE_BOOSTING_OPTIMIZERS_FACTORY_HPP
#define TOTALLY_CORRECTIVE_BOOSTING_OPTIMIZERS_FACTORY_HPP

#include "ConfigFile.hpp"

namespace totally_corrective_boosting
{

class AbstractOptimizer; // forward declaration

AbstractOptimizer* new_optimizer_instance(
        const ConfigFile &config,
        const size_t labels_size,
        const bool transposed,
        const double eta, const double nu, const double epsilon,
        const bool binary,
        std::ostream &log_stream = std::cout);

} // namespace totally_corrective_boosting

#endif // TOTALLY_CORRECTIVE_BOOSTING_OPTIMIZERS_FACTORY_HPP
