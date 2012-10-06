#ifndef _ADABOOST_HPP_
#define _ADABOOST_HPP_

#include "AbstractBooster.hpp"
#include "WeakLearner.hpp"

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
            const int& num_pt, 
            const int& max_iter,
	    const int& disp_freq);
  ~AdaBoost();
  
};


} // end of namespace totally_corrective_boosting

#endif
