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
 * Created: (14/04/2009) 
 *
 * Last Updated: (20/04/2008)   
 */

#ifndef _OPTIMIZER_PG_HPP_
#define _OPTIMIZER_PG_HPP_

#include "optimizer.hpp"

// Implement the projected gradient algorithm in w domain

/* Credits:
 *
 * Dai-Fletcher Projected Gradient method for SVM [1].
 * Modified from the GPDT software [2] for inequality constraints.
 * Projected gradient method [3] for simplex constraints. 
 * 
 * 
 * References:
 *
 *   [1] Y. H. Dai and R. Fletcher,
 *       New algorithms for singly linearly constrained quadratic programs
 *       subject to lower and upper bounds, 
 *       Math. Program., 2006.
 *
 *   [2] L. Zanni, T. Serafini, and G. Zanghirati,
 *       Parallel Software for Training Large Scale Support Vector Machines
 *       on Multiprocessor Systems,
 *       JMLR 7, 2006.
 *
 *   [3] E. G. Birgin, J. M. Martinez, and M. Raydan
 *       Nonmonotone Spectral Projected Gradient Methods
 *       On Convex Sets,
 *       SIAM J. Optim., Vol. 10, No. 4, pp. 1196-1211
 *
 */



namespace DaiFletcher{
  const double tol_r = 1e-16;
  const double tol_lam = 1e-15;
  const size_t max_iter = 10000;
}

// Magic parameters of the Projected Gradient algorithm
// Do not mess!
// svnvish: BUGBUG
// All constants are arbitrary

namespace ProjGrad{
  const double alpha_min = 1e-30;
  const double alpha_max = 10;
  const double gamma = 1e-4;
  const double sigma1 = 0.1;
  const double sigma2 = 0.9;
  const size_t M = 10; 
  const size_t max_iter = 10000;
}

class COptimizer_PG : public COptimizer {

private:
  double phi(dvec& x, 
             const dvec& a, 
             const double& b, 
             const dvec& z, 
             const dvec& l, 
             const dvec& u,
             const double& lambda);
  
  size_t project(dvec& x,
                 const dvec& a, 
                 const double& b, 
                 const dvec& z, 
                 const dvec& l, 
                 const dvec& u, 
                 const size_t& max_iter);
  
protected:
  void project_erlp(dvec& z);
  
  void project_binary(dvec& z);
  
public:
  
  COptimizer_PG(const size_t& dim, 
                const bool& transposed, 
                const double& eta, 
                const double& nu,
                const double& epsilon,
                const bool& binary);
  ~COptimizer_PG(void){ } 
  
  virtual int solve(void);
}; 

# endif
