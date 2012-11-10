
#include "ErlpBoost.hpp"
#include "math/vector_operations.hpp"


#include <stdexcept>
#include <iostream>
#include <cmath>


namespace totally_corrective_boosting
{

ErlpBoost::ErlpBoost(AbstractOracle * const oracle_,
                     const int num_data_points,
                     const int max_iterations,
                     const double epsilon_,
                     const double nu_,
                     const bool binary_,
                     AbstractOptimizer * const solver_)
    : AbstractBooster(oracle_, num_data_points, max_iterations),
      new_weak_learner_was_already_in_model(false),
      binary(binary_),
      minPt1dt1(-1.0), minPqdq1(1.0),
      epsilon(epsilon_), nu(nu_), solver(solver_)
{

    if(binary)
    {
        eta = 2.0*(1.0+log(num_data_points/nu))/epsilon;
    }
    else
    {
        eta = 2.0*log(num_data_points/nu)/epsilon;
    }

    assert(solver);
    // Set reference to the dist array in the solver
    solver->set_distribution(examples_distribution);
    return;
}


ErlpBoost::ErlpBoost(AbstractOracle * const oracle,
                     const int num_data_points,
                     const int max_iterations,
                     const double eps_,
                     const double eta_,
                     const double nu_,
                     const bool binary_,
                     AbstractOptimizer * const solver_)
    : AbstractBooster(oracle, num_data_points, max_iterations),
      new_weak_learner_was_already_in_model(false),
      binary(binary_), minPt1dt1(-1.0), minPqdq1(1.0), epsilon(eps_),
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


void ErlpBoost::update_linear_ensemble(const AbstractWeakLearner& weak_learner)
{
    WeightedWeakLearner weighted_weak_learner(&weak_learner, 0.0);
    new_weak_learner_was_already_in_model = model.add(weighted_weak_learner);
    return;
}


bool ErlpBoost::stopping_criterion(std::ostream& log_stream)
{
    log_stream << "min of Obj Values : " << minPqdq1 << std::endl;
    log_stream << "min Lower Bound : " << minPt1dt1 << std::endl;
    log_stream << "epsilon gap: " <<  minPqdq1 - minPt1dt1 << std::endl;

    return (minPqdq1 <=  minPt1dt1 + epsilon/2.0);
}


void ErlpBoost::update_stopping_criterion(const AbstractWeakLearner& wl)
{

    double gamma = wl.get_edge();
    if(binary)
    {
        gamma += (binary_relative_entropy(examples_distribution, nu)/eta);
    }
    else
    {
        gamma += (relative_entropy(examples_distribution)/eta);
    }

    if(gamma < minPqdq1)
    {
        minPqdq1 = gamma;
    }
    return;
}


void ErlpBoost::update_examples_distribution(const AbstractWeakLearner& weak_learner)
{

    // The predictions are already pre-multiplied with the labels already
    // in the weak learner

    if(not new_weak_learner_was_already_in_model)
    {
        // need to push into the solver
        const SparseVector &prediction = weak_learner.get_prediction();
        solver->push_back(prediction);
    }

    // Call the solver
    const int info = solver->solve();
    if(info != 0)
    {
        throw std::runtime_error("Something went wrong inside the solver... sorry.");
    }

    // We get back the distribution and max edge for free.
    // Only need to read wts back
    // BEWARE:
    // Only the relevant entries of solver->x are copied over.

    model.set_weights(solver->x);

    minPt1dt1 = -solver->dual_obj;
    return;
}

} // end of namespace totally_corrective_boosting
