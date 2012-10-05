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
 * Created: (29/05/2009) 
 *
 * Last Updated: (29/05/2009)   
 */

#ifndef _OPTIMIZER_CD_HPP_
#define _OPTIMIZER_CD_HPP_

#include "optimizer.hpp"

// Implement coordinate descent. Basically this is nothing but the
// corrective algorithm which is run in a loop. 

namespace CD{
  const size_t max_iter = 50000;
  const size_t ls_max_iter = 20;
}


class COptimizer_CD : public AbstractOptimizer {
  
private:
  double proj_simplex(dvec& dist, const double& exp_max);

  double line_search(dvec& W, 
                     const dvec& grad_w, 
                     const size_t& idx);
    
public:
  
  COptimizer_CD(const size_t& dim, 
                const bool& transposed, 
                const double& eta, 
                const double& nu,
                const double& epsilon,
                const bool& binary);
  ~COptimizer_CD(void){ } 
  
  int solve(void);
  
}; 

# endif
