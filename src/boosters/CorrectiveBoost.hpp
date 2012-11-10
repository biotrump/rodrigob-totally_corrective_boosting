#ifndef _CORRECTIVE_HPP_
#define _CORRECTIVE_HPP_

#include "AbstractBooster.hpp"

namespace totally_corrective_boosting
{

/// Derived class. Implements Corrective
class CorrectiveBoost: public AbstractBooster
{

private:

    bool linesearch;

    // minPt1dt1 contains the minimum of the piecewise linear lower bound:
    // P^{t-1}(d^{t-1})
    double minPt1dt1;

    // minpqdq1 contains the minimum function value seen so far:
    // min_{t} P^{q}(d^{q-1})
    double minPqdq1;

    double eps;

    // nu for softboost
    double nu;

    // Regularization constant
    double eta;

    // holds current value of U*w
    DenseVector UW;


protected:

    void update_examples_distribution(const AbstractWeakLearner& wl);

    void update_linear_ensemble(const AbstractWeakLearner& wl);

    bool stopping_criterion(std::ostream& os);

    void update_stopping_criterion(const AbstractWeakLearner& wl);

    /// @returns the dual objective function after projection
    double proj_simplex(DenseVector& examples_distribution, const double& exp_max);

    double line_search(DenseVector ut);

    DenseVector tmp_update_dist(DenseVector ut, double alpha);

public:

    CorrectiveBoost(AbstractOracle* oracle,
                    const int num_data_points,
                    const int max_iterations,
                    const double eps,
                    const double nu,
                    const bool linesearch,
                    const int display_frequency);

    CorrectiveBoost(AbstractOracle* oracle,
                    const int num_data_points,
                    const int max_iterations,
                    const double eps,
                    const double eta,
                    const double nu,
                    const bool linesearch,
                    const int display_frequency);

    ~CorrectiveBoost();

};

} // end of namespace totally_corrective_boosting

#endif
