
#include <iostream>
#include <cmath>
#include "AdaBoost.hpp"

namespace totally_corrective_boosting
{

AdaBoost::AdaBoost(AbstractOracle* &oracle,
                     const int& num_pt, 
                     const int& max_iter,
		     const int& disp_freq):
  AbstractBooster(oracle, num_pt, max_iter,disp_freq){
  alpha = 0.0;
  return;
}

AdaBoost::~AdaBoost()
{
    // nothing to do here
    return;
}

void AdaBoost::update_weights(const WeakLearner& wl){
  
  SparseVector pred = wl.get_prediction();
  
  // Since the prediction is a sparse vector
  // We only need to update those components of dist for which hx.val is
  // non-zero. 
  for(size_t i = 0; i < pred.nnz; i++){
    // Predictions are premultiplied with the labels already in the weak
    // learner
    int idx = pred.idx[i]; 
    dist.val[idx] = dist.val[idx]*exp(-alpha*pred.val[i]);
  }
  normalize(dist);
  return;
}


void AdaBoost::update_linear_ensemble(const WeakLearner& wl){
  
  double gamma = wl.get_edge();
  double eps = 0.5*(1.0 - gamma);
  alpha = 0.5*log((1-eps-1e-4)/(eps+1e-4));
  WeightedWeakLearner wwl(&wl, alpha);
  model.add(wwl);
  
  return;
}

bool AdaBoost::stopping_criterion(std::ostream& os){
  return false;
}


void AdaBoost::update_stopping_criterion(const WeakLearner& wl)
{
    // nothing to update
    return;
}

} // end of namespace totally_corrective_boosting
