#ifndef _CORRECTIVE_HPP_
#define _CORRECTIVE_HPP_

#include "AbstractBooster.hpp"

namespace totally_corrective_boosting
{

/// Derived class. Implements Corrective
class CorrectiveBoost: public AbstractBooster{

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

  void update_weights(const WeakLearner& wl);
  
  void update_linear_ensemble(const WeakLearner& wl);

  bool stopping_criterion(std::ostream& os);

  void update_stopping_criterion(const WeakLearner& wl);

  double proj_simplex(DenseVector& dist, const double& exp_max);

  double line_search(DenseVector ut);

  DenseVector tmp_update_dist(DenseVector ut, double alpha);
  
  public:

  CorrectiveBoost(AbstractOracle* &oracle,
             const int& num_pt, 
             const int& max_iter,
             const double& eps, 
	      const double& nu,
	      const bool& linesearch,
	      const int& disp_freq);

  CorrectiveBoost(AbstractOracle* &oracle,
             const int& num_pt, 
             const int& max_iter,
             const double& eps, 
	      const double& eta,
	      const double& nu,
	      const bool& linesearch,
	      const int& disp_freq);

  ~CorrectiveBoost();
  
};

} // end of namespace totally_corrective_boosting

#endif
