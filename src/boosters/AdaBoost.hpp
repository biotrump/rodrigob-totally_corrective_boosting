#ifndef _ADABOOST_HPP_
#define _ADABOOST_HPP_

#include "AbstractBooster.hpp"
#include "weak_learners/WeakLearner.hpp"

namespace totally_corrective_boosting
{


/// Derived class. Implements AdaBoost
class AdaBoost: public AbstractBooster{

private:
  double alpha;
  
protected:
  
  void update_weights(const WeakLearner& wl);
  
  void update_linear_ensemble(const WeakLearner& wl);

  bool stopping_criterion(std::ostream& os);

  void update_stopping_criterion(const WeakLearner& wl);
  
public:

  AdaBoost(AbstractOracle* &oracle,
            const int& num_data_points,
            const int& max_iterations,
            const int& display_frequency);
  ~AdaBoost();
  
};


} // end of namespace totally_corrective_boosting

#endif
