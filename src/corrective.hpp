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
 * Authors: Karen Glocer
 *
 * Created: (30/04/2009) 
 *
 * Last Updated: (02/05/2009)   
 */
#ifndef _CORRECTIVE_HPP_
#define _CORRECTIVE_HPP_

#include "booster.hpp"
#include "weak_learner.hpp"


/** Derived class. Implements Corrective 
 */

class CCorrective: public CBooster{

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
  dvec UW;
 

  protected:

  void update_weights(const CWeakLearner& wl);
  
  void update_linear_ensemble(const CWeakLearner& wl);

  bool stopping_criterion(std::ostream& os);

  void update_stopping_criterion(const CWeakLearner& wl);

  double proj_simplex(dvec& dist, const double& exp_max);

  double line_search(dvec ut);

  dvec tmp_update_dist(dvec ut, double alpha);
  
  public:

  CCorrective(COracle* &oracle, 
             const int& num_pt, 
             const int& max_iter,
             const double& eps, 
	      const double& nu,
	      const bool& linesearch,
	      const int& disp_freq);

  CCorrective(COracle* &oracle, 
             const int& num_pt, 
             const int& max_iter,
             const double& eps, 
	      const double& eta,
	      const double& nu,
	      const bool& linesearch,
	      const int& disp_freq);

  ~CCorrective(){};
  
};


#endif
