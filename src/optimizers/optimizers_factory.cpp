#include "optimizers_factory.hpp"

#include "ProjectedGradientOptimizer.hpp"
#include "ZhangdAndHagerOptimizer.hpp"
#include "LbfgsbOptimizer.hpp"
#include "CoordinateDescentOptimizer.hpp"

#if defined(USE_TAO)
#include "TaoOptimizer.hpp"
#endif

#include <string>
#include <stdexcept>

namespace totally_corrective_boosting
{

/// Optimizers factory
AbstractOptimizer* new_optimizer_instance(
        const ConfigFile &config,
        const size_t labels_size,
        const bool transposed,
        const double eta, const double nu, const double epsilon,
        const bool binary,
        std::ostream &log_stream)
{


    std::string optimizer_type;
    config.readInto(optimizer_type, "optimizer_type", std::string("lbfgsb"));

    log_stream << "Using optimizer_type == " << optimizer_type << std::endl;

    AbstractOptimizer* optimizer = NULL;

    if(optimizer_type == "tao")
    {
#if defined(USE_TAO)
        solver = new TaoOptimizer(labels_size, transposed, eta, nu, eps, binary, argc, argv);
#else
        std::stringstream os;
        os << "You must compile with TAO support enabled to use TAO"
           << std::endl << "See readme.text for details" << std::endl;
        throw std::invalid_argument(os.str());
#endif
    } else if(optimizer_type == "pg")
    {
        optimizer = new ProjectedGradientOptimizer(labels_size, transposed, eta, nu, epsilon, binary);
    }
    else if(optimizer_type == "hz")
    {
        optimizer = new ZhangdAndHagerOptimizer(labels_size, transposed, eta, nu, epsilon, binary);
    }
    else if(optimizer_type == "lbfgsb")
    {
        optimizer = new LbfgsbOptimizer(labels_size, transposed, eta, nu, epsilon, binary);
    }
    else if(optimizer_type == "cd")
    {
        optimizer = new CoordinateDescentOptimizer(labels_size, transposed, eta, nu, epsilon, binary);
    }
    else
    {
        throw std::runtime_error("Received an unknown value for optimizer_type");
    }

    return optimizer;
}


} // end of namespace totally_corrective_boosting
