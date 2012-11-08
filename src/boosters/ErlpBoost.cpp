
#include "ErlpBoost.hpp"
#include "math/vector_operations.hpp"

#include <iostream>
#include <cmath>


namespace totally_corrective_boosting
{

ErlpBoost::ErlpBoost(AbstractOracle *oracle_,
                     const int num_data_points,
                     const int max_iterations,
                     const double eps_,
                     const double nu_,
                     const bool binary,
                     AbstractOptimizer *solver_)
    : AbstractBooster(oracle_, num_data_points, max_iterations),
      found(false),
      binary(binary),
      minPt1dt1(-1.0), minPqdq1(1.0),
      eps(eps_), nu(nu_), solver(solver_)
{

    if(binary)
    {
        eta = 2.0*(1.0+log(num_data_points/nu_))/eps_;
    }
    else
    {
        eta = 2.0*log(num_data_points/nu_)/eps_;
    }

    assert(solver);
    // Set reference to the dist array in the solver
    solver->set_distribution(examples_distribution);
    return;
}

ErlpBoost::ErlpBoost(AbstractOracle *oracle,
                     const int num_data_points,
                     const int max_iterations,
                     const double eps_,
                     const double eta_,
                     const double nu_,
                     const bool binary,
                     AbstractOptimizer *solver_)
    : AbstractBooster(oracle, num_data_points, max_iterations),
      found(false),
      binary(binary), minPt1dt1(-1.0), minPqdq1(1.0), eps(eps_),
      nu(nu_), eta(eta_), solver(solver_)
{
    assert(solver);
    // Set reference to the dist array in the solver
    solver->set_distribution(examples_distribution);
    return;
}


ErlpBoost::~ErlpBoost()
{
    // nothing to do here
    return;
}


void ErlpBoost::update_linear_ensemble(const AbstractWeakLearner& wl)
{

    WeightedWeakLearner wwl(&wl, 0.0);
    found = model.add(wwl);

    return;
}


bool ErlpBoost::stopping_criterion(std::ostream& os)
{
    std::cout << "min of Obj Values : " << minPqdq1 << std::endl;
    std::cout << "min Lower Bound : " << minPt1dt1 << std::endl;
    std::cout << "epsilon gap: " <<  minPqdq1 - minPt1dt1<< std::endl;
    os << "epsilon gap: " <<  minPqdq1 - minPt1dt1<< std::endl;
    return(minPqdq1 <=  minPt1dt1 + eps/2.0);
}


void ErlpBoost::update_stopping_criterion(const AbstractWeakLearner& wl)
{

    double gamma = wl.get_edge();
    if(binary){
        gamma += (binary_relative_entropy(examples_distribution, nu)/eta);
    }else{
        gamma += (relative_entropy(examples_distribution)/eta);
    }

    if(gamma < minPqdq1) minPqdq1 = gamma;
    return;
}


void ErlpBoost::update_weights(const AbstractWeakLearner& wl)
{

    // The predictions are already pre-multiplied with the labels already
    // in the weak learner

    if(!found){
        // need to push into the solver
        SparseVector prediction = wl.get_prediction();
        solver->push_back(prediction);
    }

    // Call the solver
    const int info = solver->solve();
    assert(!info);

    // We get back the distribution and max edge for free.
    // Only need to read wts back
    // BEWARE:
    // Only the relevant entries of solver->x are copied over.

    model.set_weights(solver->x);

    minPt1dt1 = -solver->dual_obj;
    return;
}

} // end of namespace totally_corrective_boosting
