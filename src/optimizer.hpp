/* Copyright (c) 2009, S V N Vishwanathan
 * All rights reserved. 
 * 
 * The contents of this file are subject to the Mozilla Public License 
 * Version 1.1 (the "License"); you may not use this file except in 
 * compliance with the License. You may obtain a copy of the License at 
 * http://www.mozilla.org/MPL/ 
 * 
 * Software distributed under the License is distributed on an "AS IS" 
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the 
 * License for the specific language governing rights and limitations 
 * under the License. 
 * 
 * Authors: S V N Vishwanathan 
 *
 * Created: (20/04/2009) 
 *
 * Last Updated: (20/04/2008)   
 */

#ifndef _OPTIMIZER_HPP_
#define _OPTIMIZER_HPP_

// Implement the projected gradient algorithm in the w domain. 

#include <vector>

#include "vec.hpp"
#include "timer.hpp"

// Class to encapsulate the optimization problem we are solving
// Different implementations of the optimizer simply overload the
// solve function 

namespace Optimizer{
  const double INFTY = 1e30;
  const double kkt_gap_tol = 1e-3; // KKT gap violation tolerance
  const double pgnorm_tol = 1e-3;  // Max norm of projected gradient 
  const double wt_sum_tol = 1e-3;  // How much tolerance for the sum of wt - 1 
}

class COptimizer{  

private:

  // ERLPBoost function value and gradient
  double fun_erlp(void);
  
  dvec grad_erlp(void);  

  // Binary ERLPBoost function value and gradient
  double fun_binary(void);
  
  dvec grad_binary(void);  
  
  // duality gap
  double gap;
    
protected:
  // Columns of U 
  size_t num_wl; 
  
  // Rows of U
  size_t dim;        
  
  // Weak learners 
  std::vector<dvec> U; 
  
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
  bool kkt_gap_met(const dvec& gradk);
  
  // projected gradient norm for psi < pgnorm_tol
  bool pgnorm_met(const dvec& gradk);

  // Keep track of time spent in function and gradient evaluation 
  CTimer fun_timer;
  
  CTimer grad_timer;

  void report_stats(void);
  
public:
  
  // Below are return values. 
  
  // Vector with current solution
  dvec x;
  
  // Vector to store the distribution
  dvec dist;          
  
  // minimum primal value seen so far
  double min_primal;

  // current dual objective
  double dual_obj;
  
  
  COptimizer(const size_t& dim, 
             const bool& transposed, 
             const double& eta, 
             const double& nu,
             const double& epsilon,
             const bool& binary = false); 
  
  virtual ~COptimizer(void); 
  
  void set_dist(const dvec& _dist){
    // dist is just a reference to the true array
    dist.val = _dist.val;
    dist.dim = _dist.dim;
    return;
  }
  
  void push_back(const svec& u);
  
  // ERLPBoost function 
  double fun(void);
  
  // ERLPBoost gradient 
  dvec grad(void);

  // ERLPBoost primal function
  double primal(void);
  
  bool converged(const dvec& gradk);

  bool duality_gap_met(void);
  
  
  // Derived classes will implement these methods 
  virtual int solve(void) = 0;
  
}; 

double log_one_plus_x(const double& x);

# endif
