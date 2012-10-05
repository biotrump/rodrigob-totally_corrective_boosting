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
 * Authors: S V N Vishwanathan (vishy@stat.purdue.edu)
 * 
 * 
 * Created: (29/05/2009) 
 *
 * Last Updated: (29/05/2009)   
 */

#include <cmath>
#include <algorithm>
#include <limits>

#include "optimizer_cd.hpp"

COptimizer_CD::COptimizer_CD(const size_t& dim, 
                             const bool& transposed, 
                             const double& eta, 
                             const double& nu,
                             const double& epsilon,
                             const bool& binary):
  COptimizer(dim, transposed, eta, nu, epsilon, binary){ }

int COptimizer_CD::solve(void){
  
  // svnvish: BUGBUG
  // only hard margin case 
  assert(nu == 1);
  
  // svnvish: BUGBUG
  // no binary ERLPboost yet
  assert(!binary);
  
  dvec W;
  W.val = x.val;
  W.dim = num_wl;

  dvec UW;
  
  // since the booster stores U transpose do transpose dot
  if(transposed)
    transpose_dot(U, W, UW);
  else
    dot(U, W, UW);

  double min_primal = std::numeric_limits<double>::max(); 
  
  // Loop through many times 
  for(size_t j = 0; j < CD::max_iter; j++){
    
    // Find max element for safe exponentiation 
    double exp_max = -std::numeric_limits<double>::max(); 
    
    for(size_t i = 0; i < dist.dim; i++){
      dist.val[i] = - eta*(UW.val[i]);
      if(dist.val[i] > exp_max) exp_max = dist.val[i];
    }
    
    // Safe exponentiation
    double obj = 0.0;
    for(size_t i = 0; i < dist.dim; i++) {
      dist.val[i] = exp(dist.val[i] - exp_max)/dim;
      obj += dist.val[i];
    }
    
    proj_simplex(dist, exp_max);
    
    obj += 1e-10;
    obj = (log(obj)+ exp_max)/eta;
    
    dvec grad_w;

    // since the booster stores U transpose do normal dot and not
    // transpose dot
    if(transposed)
      dot(U, dist, grad_w);
    else
      transpose_dot(U, dist, grad_w);
    
    edge = max(grad_w);
    
    // This is the lowest primal objective we have seen so far
    min_primal = std::min(min_primal, primal());
    double gap = min_primal + obj;
    
    if(gap < 0.05*epsilon){
      std::cout << "Converged in " << j << " iterations" << std::endl;
      report_stats(); 
      // This is not a memory leak!
      W.val = NULL;
      W.dim = 0;
      return 0;
    }
    
    size_t idx = argmax(grad_w);
    assert(grad_w.val[idx] > 0); 
    
    dvec store(UW);
    
    if(transposed){
      dvec tmp(num_wl);
      tmp.val[idx] = 1.0;
      dvec U_idx;
      transpose_dot(U, tmp, U_idx);
      axpy(-1.0, UW, U_idx, UW);
    }else
      axpy(-1.0, UW, U[idx], UW);

    double eta_t = line_search(W, grad_w, idx); 
    // double denom = abs_max(UW);
    // denom *= denom;
    // double eta_t = std::max(0.0, 
    //                         std::min(1.0, dot(dist, UW)/eta/denom));

    scale(W, (1.0-eta_t));
    W.val[idx] += eta_t;    
    axpy(eta_t, UW, store, UW);
  }
  
  // This is not a memory leak!
  W.val = NULL;
  W.dim = 0;
  
  return 1;
}

double COptimizer_CD::line_search(dvec& W, 
                                  const dvec& grad_w, 
                                  const size_t& idx){
  
  double lower = 0.0;
  double upper = 1.0;

  dvec orig_W(W);
  
  for(size_t i = 0; i < CD::ls_max_iter; i++){

    double mid = lower + ((upper - lower)/2.0);
    
    if (std::abs(mid-lower) < 1e-8)
      break;

    axpy(mid, grad_w, orig_W, W);
    fun();
    dvec G = grad();
    
    if(G.val[idx] > 0)
      lower = mid;
    else
      upper = mid;

  }
  
  for(size_t i = 0; i < num_wl; i++)
    W.val[i] = orig_W.val[i];
  
  return lower + ((upper - lower)/2.0);
}

// Return the dual objective function after projection
double COptimizer_CD::proj_simplex(dvec& dist, const double& exp_max){

  double cap = 1.0/nu;
  double theta = 0.0;
  size_t dim = dist.dim;
  int N = dist.dim - 1;
  dvec tmpdist(dist);
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
