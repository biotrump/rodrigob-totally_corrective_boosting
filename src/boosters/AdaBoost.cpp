
#include "AdaBoost.hpp"

#include <iostream>
#include <cmath>

namespace totally_corrective_boosting
{

AdaBoost::AdaBoost(AbstractOracle *oracle,
                   const int num_pt,
                   const int max_iter,
                   const int disp_freq)
    : AbstractBooster(oracle, num_pt, max_iter,disp_freq)
{
    alpha = 0.0;
    return;
}


AdaBoost::~AdaBoost()
{
    // nothing to do here
    return;
}


void AdaBoost::update_weights(const AbstractWeakLearner& wl)
{

    SparseVector pred = wl.get_prediction();

    // Since the prediction is a sparse vector
    // We only need to update those components of dist for which hx.val is
    // non-zero.
    for(size_t i = 0; i < pred.nnz; i++){
        // Predictions are premultiplied with the labels already in the weak
        // learner
        int index = pred.index[i];
        examples_distribution.val[index] = examples_distribution.val[index]*exp(-alpha*pred.val[i]);
    }
    normalize(examples_distribution);
    return;
}


void AdaBoost::update_linear_ensemble(const AbstractWeakLearner &wl)
{

    double gamma = wl.get_edge();
    double eps = 0.5*(1.0 - gamma);
    alpha = 0.5*log((1-eps-1e-4)/(eps+1e-4));
    WeightedWeakLearner wwl(&wl, alpha);
    model.add(wwl);

    return;
}


bool AdaBoost::stopping_criterion(std::ostream& os)
{
    return false; // AdaBoost will progress until max number of iterations is reached
}


void AdaBoost::update_stopping_criterion(const AbstractWeakLearner& wl)
{
    // nothing to update
    return;
}

} // end of namespace totally_corrective_boosting
