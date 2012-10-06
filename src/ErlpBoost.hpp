#ifndef _ERLPBOOST_HPP_
#define _ERLPBOOST_HPP_

#include "AbstractBooster.hpp"
#include "WeakLearner.hpp"
#include "AbstractOptimizer.hpp"

namespace totally_corrective_boosting
{


/// Derived class. Implements ERLPBoost
class ErlpBoost: public AbstractBooster
{

protected:
    bool found;

    /// Are we going to use Binary relative entropy
    bool binary;

    /// minPt1dt1 contains the minimum of the piecewise linear lower bound:
    /// P^{t-1}(d^{t-1})
    double minPt1dt1;

    /// minpqdq1 contains the minimum function value seen so far:
    /// min_{t} P^{q}(d^{q-1})
    double minPqdq1;

    double eps;

    /// nu for softboost
    double nu;

    /// Regularization constant
    double eta;
    
    /// solver
    AbstractOptimizer* solver;

protected:

    void update_weights(const WeakLearner& wl);

    void update_linear_ensemble(const WeakLearner& wl);

    bool stopping_criterion(std::ostream& os);

    void update_stopping_criterion(const WeakLearner& wl);

public:

    ErlpBoost(AbstractOracle* &oracle,
              const int& num_data_points,
              const int& max_iterations,
              const double& eps,
              const double& nu,
              const bool& binary,
              AbstractOptimizer* &solver);

    ErlpBoost(AbstractOracle* &oracle,
              const int& num_data_points,
              const int& max_iterations,
              const double& eps,
              const double& eta,
              const double& nu,
              const bool& binary,
              AbstractOptimizer* &solver);

    ~ErlpBoost();

};

} // end of namespace totally_corrective_boosting

#endif
