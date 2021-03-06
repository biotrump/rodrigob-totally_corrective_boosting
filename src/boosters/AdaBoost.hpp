#ifndef _ADABOOST_HPP_
#define _ADABOOST_HPP_

#include "AbstractBooster.hpp"
#include "weak_learners/AbstractWeakLearner.hpp"

namespace totally_corrective_boosting
{


/// Derived class. Implements AdaBoost
class AdaBoost: public AbstractBooster
{

private:
  double alpha;
  
protected:
  
  void update_examples_distribution(const AbstractWeakLearner& weak_learner);
  
  void update_linear_ensemble(const AbstractWeakLearner& wl);

  bool stopping_criterion(std::ostream& os);

  void update_stopping_criterion(const AbstractWeakLearner &wl);
  
public:

  AdaBoost(const boost::shared_ptr<AbstractOracle> &oracle,
            const int num_data_points,
            const int max_iterations,
            const int display_frequency);
  ~AdaBoost();
  
};


} // end of namespace totally_corrective_boosting

#endif
