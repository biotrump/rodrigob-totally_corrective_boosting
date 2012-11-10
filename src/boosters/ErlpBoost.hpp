#ifndef _ERLPBOOST_HPP_
#define _ERLPBOOST_HPP_

#include "AbstractBooster.hpp"
#include "weak_learners/AbstractWeakLearner.hpp"
#include "optimizers/AbstractOptimizer.hpp"

namespace totally_corrective_boosting
{


/// Derived class. Implements ERLPBoost
class ErlpBoost: public AbstractBooster
{

protected:


    bool new_weak_learner_was_already_in_model;

    /// Are we going to use Binary relative entropy
    bool binary;

    /// minPt1dt1 contains the minimum of the piecewise linear lower bound:
    /// P^{t-1}(d^{t-1})
    double minPt1dt1;

    /// minpqdq1 contains the minimum function value seen so far:
    /// min_{t} P^{q}(d^{q-1})
    double minPqdq1;

    /// Variable used as stopping criterion
    const double epsilon;

    /// nu for softboost (looks like a 'v')
    double nu;

    /// Regularization constant (looks like an 'n')
    double eta;
    
    /// solver
    AbstractOptimizer* solver;

protected:

    void update_examples_distribution(const AbstractWeakLearner& wl);

    void update_linear_ensemble(const AbstractWeakLearner& wl);

    bool stopping_criterion(std::ostream& log_stream);

    void update_stopping_criterion(const AbstractWeakLearner& wl);

public:

    ErlpBoost(AbstractOracle * const oracle,
              const int num_data_points,
              const int max_iterations,
              const double epsilon,
              const double nu,
              const bool binary,
              AbstractOptimizer * const solver);

    ErlpBoost(AbstractOracle* const oracle,
              const int num_data_points,
              const int max_iterations,
              const double epsilon,
              const double eta,
              const double nu,
              const bool binary,
              AbstractOptimizer* const solver_);

    virtual ~ErlpBoost();

};

} // end of namespace totally_corrective_boosting

#endif
