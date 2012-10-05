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

#include <iostream>
#include <cmath>

#include "erlpboost.hpp"
#include "vec.hpp"

CERLPBoost::CERLPBoost(COracle* &oracle, 
                       const int& num_pt, 
                       const int& max_iter,
                       const double& eps, 
                       const double& nu,
                       const bool& binary,
                       COptimizer* &solver):
  CBooster(oracle, num_pt, max_iter), found(false), 
  binary(binary), minPt1dt1(-1.0), minPqdq1(1.0), 
  eps(eps), nu(nu), solver(solver){
  
  if(binary){
    eta = 2.0*(1.0+log(num_pt/nu))/eps;
  }else{
    eta = 2.0*log(num_pt/nu)/eps;
  }
  assert(solver);
  // Set reference to the dist array in the solver 
  solver->set_dist(dist);
  return;
}

CERLPBoost::CERLPBoost(COracle* &oracle, 
                       const int& num_pt, 
                       const int& max_iter,
                       const double& eps, 
                       const double& eta,
                       const double& nu,
                       const bool& binary,
                       COptimizer* &solver):
  CBooster(oracle, num_pt, max_iter), found(false), 
  binary(binary), minPt1dt1(-1.0), minPqdq1(1.0), eps(eps),  
  nu(nu), eta(eta), solver(solver){
  
  assert(solver);
  // Set reference to the dist array in the solver 
  solver->set_dist(dist);
  return;
}


CERLPBoost::~CERLPBoost(void){
  
}

void CERLPBoost::update_linear_ensemble(const CWeakLearner& wl){
  
  weighted_wl wwl(&wl, 0.0);
  found = model.add(wwl);
  
  return;
}

bool CERLPBoost::stopping_criterion(std::ostream& os){
  std::cout << "min of Obj Values : " << minPqdq1 << std::endl;
  std::cout << "min Lower Bound : " << minPt1dt1 << std::endl;
  std::cout << "epsilon gap: " <<  minPqdq1 - minPt1dt1<< std::endl;
  os << "epsilon gap: " <<  minPqdq1 - minPt1dt1<< std::endl;
  return(minPqdq1 <=  minPt1dt1 + eps/2.0);
}


void CERLPBoost::update_stopping_criterion(const CWeakLearner& wl){

  double gamma = wl.get_edge();
  if(binary){
    gamma += (binary_relent(dist, nu)/eta);
  }else{
    gamma += (relent(dist)/eta);
  }
  
  if(gamma < minPqdq1) minPqdq1 = gamma;
  return;
}

void CERLPBoost::update_weights(const CWeakLearner& wl){
  
  // The predictions are already pre-multiplied with the labels already
  // in the weak learner

  if(!found){
    // need to push into the solver 
    svec pred = wl.get_prediction();
    solver->push_back(pred);
  }
  
  // Call the solver
  int info = solver->solve(); assert(!info);
  
  // We get back dist and max edge for free.
  // Only need to read wts back
  // BEWARE:
  // Only the relevant entries of solver->x are copied over.
  
  model.set_wts(solver->x);

  minPt1dt1 = -solver->dual_obj; 
  
  return;
}
