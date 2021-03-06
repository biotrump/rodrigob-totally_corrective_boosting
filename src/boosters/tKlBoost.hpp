#ifndef TOTALLY_CORRECTIVE_BOOSTING_TKLBOOST_HPP
#define TOTALLY_CORRECTIVE_BOOSTING_TKLBOOST_HPP

#include "ErlpBoost.hpp"

namespace totally_corrective_boosting {

/// Implementation of Meizhu Liu's totally corrective boosting (see CVPR2011 paper and her thesis).
/// This implementation uses total Kullback–Leibler divergence for regularization
class tKlBoost: public ErlpBoost
{
protected:

    const double D;

public:

    /// The term D (capital_dee) dictates the regularization
    /// (D=1/7, 1/5, 1/3) was used ni Liu's experiments
    tKlBoost(const boost::shared_ptr<AbstractOracle> &oracle,
             const int num_data_points,
             const int max_iterations,
             const double epsilon,
             const double capital_dee,
             const bool binary,
             const boost::shared_ptr<AbstractOptimizer> &solver);
    ~tKlBoost();

    /// we define compute nu as static since it will be needed outside the class
    static double compute_nu(const double capital_dee);

    /// we define compute eta as static since it will be needed outside the class
    static double compute_eta(const int num_data_points,
                              const double epsilon,
                              const double capital_dee);

public:

    /// Python style "back door" access, for easier experimental integration in foreign software
    /// @{
    void _update_examples_distribution(const AbstractWeakLearner& wl);
    void _update_linear_ensemble(const AbstractWeakLearner& wl);
    bool _stopping_criterion(std::ostream& log_stream);
    void _update_stopping_criterion(const AbstractWeakLearner& wl);
    /// @}

};

} // end of namespace totally_corrective_boosting

#endif // TOTALLY_CORRECTIVE_BOOSTING_TKLBOOST_HPP
