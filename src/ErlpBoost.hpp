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
 * Created: (28/03/2009) 
 *
 * Last Updated: (28/03/2008)   
 */
#ifndef _ERLPBOOST_HPP_
#define _ERLPBOOST_HPP_

#include "AbstractBooster.hpp"
#include "WeakLearner.hpp"
#include "AbstractOptimizer.hpp"

namespace totally_corrective_boosting
{


/// Derived class. Implements ERLPBoost
class ErlpBoost: public AbstractBooster{

private:
  
  bool found;
  
private: 
  // Are we going to use Binary relative entropy
  bool binary;
  
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
    
  // solver
  AbstractOptimizer* solver;
  
protected:
  
  void update_weights(const WeakLearner& wl);
  
  void update_linear_ensemble(const WeakLearner& wl);

  bool stopping_criterion(std::ostream& os);

  void update_stopping_criterion(const WeakLearner& wl);
  
public:

  ErlpBoost(AbstractOracle* &oracle,
             const int& num_pt, 
             const int& max_iter,
             const double& eps, 
             const double& nu,
             const bool& binary,
             AbstractOptimizer* &solver);

  ErlpBoost(AbstractOracle* &oracle,
             const int& num_pt, 
             const int& max_iter,
             const double& eps, 
	     const double& eta,
             const double& nu,
             const bool& binary,
             AbstractOptimizer* &solver);

  ~ErlpBoost();
  
};

} // end of namespace totally_corrective_boosting

#endif
