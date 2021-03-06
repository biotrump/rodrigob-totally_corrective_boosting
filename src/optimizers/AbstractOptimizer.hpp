
#ifndef _OPTIMIZER_HPP_
#define _OPTIMIZER_HPP_

// Implement the projected gradient algorithm in the w domain. 

#include "math/dense_vector.hpp"
#include "math/sparse_vector.hpp"

#include "Timer.hpp"

#include <vector>


namespace totally_corrective_boosting
{

// Class to encapsulate the optimization problem we are solving
// Different implementations of the optimizer simply overload the
// solve function 

namespace Optimizer
{
  const double infinity = 1e30;
  const double kkt_gap_tol = 1e-3; // KKT gap violation tolerance
  const double pgnorm_tol = 1e-3;  // Max norm of projected gradient 
  const double wt_sum_tol = 1e-3;  // How much tolerance for the sum of wt - 1 
}


/// These are optimizers for the ERLPBoost problem (only)
class AbstractOptimizer
{

private:

  /// ERLPBoost function value and gradient
  double erlp_function();
  
  DenseVector erlp_gradient();

  /// Binary ERLPBoost function value and gradient
  double binary_function();
  
  DenseVector binary_gradient();
  
  /// duality gap
  double gap;
    
protected:
  /// Columns of U
  size_t num_weak_learners;
  
  /// Rows of U
  size_t dim;        
  
  /// Weak learners
  std::vector<DenseVector> U; 
  
  /// Do we store U or U transpose?
  bool transposed; 
  
  /// Regularization parameter
  double eta;          
  
  /// nu for softboost
  double nu;           
  
  /// epsilon tolerance of outer boosting loop
  double epsilon;
  
  /// Are we solving binary boost problem?
  bool binary;

  /// max edge
  double edge; 
  
  /// KKT gap for w < kkt_gap_tol?
  bool kkt_gap_met(const DenseVector& gradk);
  
  /// projected gradient norm for psi < pgnorm_tol
  bool pgnorm_met(const DenseVector& gradk);

  /// Keep track of time spent in function and gradient evaluation
  Timer function_timer, gradient_timer;

  void report_statistics();
  
public:
  
  // Below are return values. 
  
  /// Vector with current solution
  DenseVector x;
  
  /// Vector to store the distribution
  DenseVector distribution;
  
  /// minimum primal value seen so far
  double min_primal;

  /// current dual objective
  double dual_obj;
  
  
  AbstractOptimizer(const size_t& dim,
             const bool& transposed, 
             const double& eta, 
             const double& nu,
             const double& epsilon,
             const bool& binary = false); 
  
  virtual ~AbstractOptimizer();
  
  void set_distribution(const DenseVector& _distribution);
  
  void push_back(const SparseVector& u);
  
  /// ERLPBoost function
  double function();
  
  /// ERLPBoost gradient
  DenseVector gradient();

  /// ERLPBoost primal function
  double primal();
  
  bool converged(const DenseVector& gradk);

  bool duality_gap_met();
  
  /// Derived classes will implement this method
  virtual int solve() = 0;
  
}; 

double log_one_plus_x(const double& x);

} // end of namespace totally_corrective_boosting

# endif
