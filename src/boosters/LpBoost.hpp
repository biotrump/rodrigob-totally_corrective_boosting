#ifndef _LPBOOST_HPP_
#define _LPBOOST_HPP_

#if not defined(USE_CLP)
#error This file should not be included if the COIN Linear Programming library is not included.
#endif

#include "AbstractBooster.hpp"

#include <coin/ClpSimplex.hpp>

namespace totally_corrective_boosting
{

/// Derived class. Implements LPBoost
class LpBoost: public AbstractBooster
{
  
protected:
  
  /// minPt1dt1 contains the minimum of the piecewise linear lower bound:
  /// P^{t-1}(d^{t-1})
  double minPt1dt1;
  
  /// minpqdq1 contains the minimum function value seen so far:
  /// min_{t} P^{q}(d^{q-1})
  double minPqdq1;

  /// Stopping criterion
  const double epsilon;
  
  /// nu for softboost
  const double nu;
  
  // solver
  ClpSimplex solver;
  
protected:
  
  void update_weights(const AbstractWeakLearner& wl);
  
  void update_linear_ensemble(const AbstractWeakLearner& wl);

  bool stopping_criterion(std::ostream& os);

  void update_stopping_criterion(const AbstractWeakLearner& wl);
  
public:

  LpBoost(AbstractOracle* oracle,
           const int num_data_points,
           const int max_iterations,
           const double epsilon,
           const double nu);
  ~LpBoost(void);
  
};

} // end of namespace totally_corrective_boosting

#endif
