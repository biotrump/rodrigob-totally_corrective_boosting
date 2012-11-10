#include "tKlBoost.hpp"

#include <cmath>

namespace totally_corrective_boosting
{

tKlBoost::tKlBoost(AbstractOracle *oracle,
                   const int num_data_points,
                   const int max_iterations,
                   const double epsilon,
                   const double capital_dee,
                   const bool binary,
                   AbstractOptimizer *solver)
    : ErlpBoost(oracle, num_data_points, max_iterations, epsilon,
                compute_eta(num_data_points, epsilon, capital_dee),
                compute_nu(capital_dee),
                binary, solver),
      D(capital_dee)
{
    std::cout << "Using lambda == " << 1/eta << std::endl;
    return;
}


tKlBoost::~tKlBoost()
{
    // nothing to do here
    return;
}


double tKlBoost::compute_nu(const double capital_dee)
{
    const double
            D = capital_dee,
            nu = 1/D;

    return nu;
}


double tKlBoost::compute_eta(const int num_data_points, const double epsilon, const double capital_dee)
{
    const double
            D = capital_dee,
            N = num_data_points,
            lambda = epsilon * std::sqrt(1 + std::pow(std::log(N) -1, 2)) / (2*std::log(N*D)),
            eta = 1/lambda;

    return eta;
}



void tKlBoost::_update_examples_distribution(const AbstractWeakLearner &wl)
{
    return update_examples_distribution(wl);
}


void tKlBoost::_update_linear_ensemble(const AbstractWeakLearner &wl)
{
    return update_linear_ensemble(wl);
}


bool tKlBoost::_stopping_criterion(std::ostream &log_stream)
{
    return stopping_criterion(log_stream);
}


void tKlBoost::_update_stopping_criterion(const AbstractWeakLearner &wl)
{
    return update_stopping_criterion(wl);
}




} // end of namespace totally_corrective_boosting
