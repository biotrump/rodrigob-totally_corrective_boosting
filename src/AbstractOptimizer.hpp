
#ifndef _OPTIMIZER_HPP_
#define _OPTIMIZER_HPP_

// Implement the projected gradient algorithm in the w domain. 

#include <vector>

#include "vec.hpp"
#include "Timer.hpp"

namespace totally_corrective_boosting
{

// Class to encapsulate the optimization problem we are solving
// Different implementations of the optimizer simply overload the
// solve function 

namespace Optimizer{
  const double INFTY = 1e30;
  const double kkt_gap_tol = 1e-3; // KKT gap violation tolerance
  const double pgnorm_tol = 1e-3;  // Max norm of projected gradient 
  const double wt_sum_tol = 1e-3;  // How much tolerance for the sum of wt - 1 
}

class AbstractOptimizer{

private:

  // ERLPBoost function value and gradient
  double erlp_function(void);
  
  DenseVector erlp_gradient(void);

  // Binary ERLPBoost function value and gradient
  double binary_function(void);
  
  DenseVector binary_gradient(void);
  
  // duality gap
  double gap;
    
protected:
  // Columns of U 
  size_t num_weak_learners;
  
  // Rows of U
  size_t dim;        
  
  // Weak learners 
  std::vector<DenseVector> U; 
  
  // Do we store U or U transpose?
  bool transposed; 
  
  // Regularization parameter
  double eta;          
  
  // nu for softboost
  double nu;           
  
  // epsilon tolerance of outer boosting loop
  double epsilon;
  
  // Are we solving binary boost problem?
  bool binary;

  // max edge 
  double edge; 
  
  // KKT gap for w < kkt_gap_tol?
  bool kkt_gap_met(const DenseVector& gradk);
  
  // projected gradient norm for psi < pgnorm_tol
  bool pgnorm_met(const DenseVector& gradk);

  // Keep track of time spent in function and gradient evaluation 
  Timer fun_timer;
  
  Timer grad_timer;

  void report_stats(void);
  
public:
  
  // Below are return values. 
  
  /// Vector with current solution
  DenseVector x;
  
  /// Vector to store the distribution
  DenseVector distribution;
  
  // minimum primal value seen so far
  double min_primal;

  // current dual objective
  double dual_obj;
  
  
  AbstractOptimizer(const size_t& dim,
             const bool& transposed, 
             const double& eta, 
             const double& nu,
             const double& epsilon,
             const bool& binary = false); 
  
  virtual ~AbstractOptimizer(void);
  
  void set_distribution(const DenseVector& _distribution);
  
  void push_back(const SparseVector& u);
  
  /// ERLPBoost function
  double function(void);
  
  /// ERLPBoost gradient
  DenseVector gradient(void);

  // ERLPBoost primal function
  double primal(void);
  
  bool converged(const DenseVector& gradk);

  bool duality_gap_met(void);
  
  
  /// Derived classes will implement this method
  virtual int solve(void) = 0;
  
}; 

double log_one_plus_x(const double& x);

} // end of namespace totally_corrective_boosting

# endif
