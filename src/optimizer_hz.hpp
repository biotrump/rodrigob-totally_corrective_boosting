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
 * Created: (26/05/2009) 
 *
 * Last Updated: (26/05/2009)   
 */

#ifndef _OPTIMIZER_HZ_HPP_
#define _OPTIMIZER_HZ_HPP_

#include "optimizer_pg.hpp"


// Magic parameters of the algorithm
// Do not mess!
// svnvish: BUGBUG
// All constants are from Paul J S Silva's ipg code

namespace ProjGrad_HZ{
  const double alpha_min = 1.0e-30;
  const double alpha_max = 10; // 1.0e+30;
  const double gamma = 1e-4;
  const double sigma1 = 0.1;
  const double sigma2 = 0.9;
  
  const double etadown = 0.75;
  const double etaup = 0.999;
  // assert(0.0 < etadown && etadown <= 1.0);
  // assert(0.0 < etadown && etadown <= etaup);
  const double min_step = 1e-64;
  const double decrease = 0.5;
  
  const size_t max_iter = 100000;
}


/// Implements the  Zhang and Hager projected gradient algorithm
///
/// H. Zhang and W. W. Hager
/// A Nonmonotone line search technique and its application to unconstrained optimization
/// SIAM J. Optim., Vol. 14, No. 4, pp. 1043-1056
class ZhangdAndHagerOptimizer:public ProjectedGradientOptimizer {

public:
  
  ZhangdAndHagerOptimizer(const size_t& dim,
                const bool& transposed, 
                const double& eta, 
                const double& nu,
                const double& epsilon, 
                const bool& binary);
  ~ZhangdAndHagerOptimizer(void){ }
  
  int solve(void);
  
}; 

# endif
