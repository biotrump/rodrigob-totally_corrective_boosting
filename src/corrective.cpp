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
 * Last Updated: (2/05/2009)   
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include "corrective.hpp"


CorrectiveBoost::CorrectiveBoost(AbstractOracle* &oracle, 
			 const int& num_pt, 
			 const int& max_iter,
			 const double& eps, 
			 const double& nu,
			 const bool& linesearch,
			 const int& disp_freq):
  AbstractBooster(oracle, num_pt, max_iter,disp_freq), 
  linesearch(linesearch), 
  minPt1dt1(-1.0), 
  minPqdq1(1.0), 
  eps(eps), 
  nu(nu){
  eta = 2.0*log(num_pt/nu)/eps;
  return;
}

CorrectiveBoost::CorrectiveBoost(AbstractOracle* &oracle, 
                         const int& num_pt, 
                         const int& max_iter,
                         const double& eps, 
                         const double& eta,
                         const double& nu,
                         const bool& linesearch,
			 const int& disp_freq):
  AbstractBooster(oracle, num_pt, max_iter,disp_freq), 
  linesearch(linesearch), 
  minPt1dt1(-1.0), 
  minPqdq1(1.0), 
  eps(eps), 
  nu(nu), 
  eta(eta){
    return;
}

CorrectiveBoost::~CorrectiveBoost()
{
    // nothing to do here
    return;
}

void CorrectiveBoost::update_weights(const WeakLearner& wl){
  
  double exp_max = 0.0;
  
  for(size_t i = 0; i < dist.dim; i++){
    dist.val[i] = -eta * UW.val[i];
    exp_max = std::max(exp_max,dist.val[i]);
  }
  
  double dual_obj = 0.0;
  for(size_t i = 0; i < dist.dim; i++){
    dist.val[i] = exp(dist.val[i] - exp_max)/dist.dim;
    dual_obj += dist.val[i];
  }
  
  minPt1dt1 = proj_simplex(dist, exp_max);
  
  return;
}


void CorrectiveBoost::update_linear_ensemble(const WeakLearner& wl){

  std::cout << "Num of wl: " << model.size() << std::endl;
  
  SparseVector ut = wl.get_prediction();
  
  DenseVector x(ut.dim);
  for(size_t i = 0; i < ut.nnz; i++){
    x.val[ut.idx[i]] = ut.val[i];
  }
  
  DenseVector ut_dense(x);
  
  if(UW.dim>0){
    DenseVector w = model.get_wts();
    // subtract Uw from x and store in x
    for(size_t i = 0; i < x.dim; i++){
      x.val[i] -=  UW.val[i];
    }
  }

  // set alpha
  double dx = dot(dist,x);

  // find the maximum value of x
  double max_x = max(x); 
  double denom = pow(max_x,2);
  double alpha;
  
  // svnvish: BUGBUG
  // silently ignore linesearch request 
  alpha = std::max(0.0, std::min(1.0, dx/eta/denom));
  
  // if(linesearch){
  //   alpha = line_search(ut_dense);
  // }else{
  //   alpha = std::max(0.0, std::min(1.0, dx/eta/denom));
  // }

  // update the other weights
  model.scale_wts(1.0-alpha);

  WeightedWeakLearner wwl(&wl, alpha);
  model.add(wwl);

  if(UW.dim ==0){
    UW.dim = x.dim;
    UW.val = new double[x.dim];
    for(size_t i = 0; i < UW.dim; i++){
      UW.val[i] = alpha *ut_dense.val[i];
    }
  }else{
    for(size_t i = 0; i < UW.dim; i++){
      UW.val[i] = (1.0 - alpha)*UW.val[i] + alpha * ut_dense.val[i];
    }
  }
  
  return;
}

bool CorrectiveBoost::stopping_criterion(std::ostream& os){
  std::cout << "min of Obj Values : " << minPqdq1 << std::endl;
  std::cout << "min Lower Bound : " << minPt1dt1 << std::endl;
  std::cout << "epsilon gap: " <<  minPqdq1 - minPt1dt1<< std::endl;
  os << "epsilon gap: " <<  minPqdq1 - minPt1dt1<< std::endl;
  return(minPqdq1 <=  minPt1dt1 + eps/2.0);
}

void CorrectiveBoost::update_stopping_criterion(const WeakLearner& wl){
  
  minPqdq1 = std::min(wl.get_edge()+(relent(dist)/eta), minPqdq1);
  return;
}


// Return the dual objective function after projection
double CorrectiveBoost::proj_simplex(DenseVector& dist, const double& exp_max){

  double cap = 1.0/nu;
  double theta = 0.0;
  size_t dim = dist.dim;
  int N = dist.dim - 1;
  DenseVector tmpdist(dist);
  // sort dist from smallest to largest
  std::sort(tmpdist.val, tmpdist.val+tmpdist.dim);
  // store the sum of the dist in Z			 
  double Z = sum(tmpdist);
  // find theta
  for(size_t i = 0; i < tmpdist.dim; i++){
    theta = (1.0 - cap * i)/Z;
    if(theta * tmpdist.val[N-i] <= cap){break;}
    else{Z -= tmpdist.val[N-i];}
  }

  double ubndsum = 0.0;
  double psisum = 0.0;
  size_t bnd = 0;
  
  for(size_t i = 0; i < dist.dim; i++){
    if(theta*dist.val[i] > cap){
      bnd++;
      psisum -= log(dist.val[i]*dim)/eta; 
      dist.val[i] = cap;
    }else{
      ubndsum += dist.val[i];
      dist.val[i] = theta*dist.val[i];
    }
  }
  
  double obj = exp_max + log(ubndsum/(1.0 - cap*bnd));
  obj *= ((bnd*cap - 1.0)/eta);
  obj -= ((bnd*cap/eta)*log(nu/dim));
  obj += psisum;
  
  return obj; 
}
