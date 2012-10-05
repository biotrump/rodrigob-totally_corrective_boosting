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
 * Created: (25/04/2009) 
 *
 * Last Updated: (25/04/2008)   
 */

#ifndef _OPTIMIZER_LBFGSB_HPP_
#define _OPTIMIZER_LBFGSB_HPP_

#include "optimizer.hpp"
#include "lbfgsb.h"

// Augmented Lagrangian solver in the w domain. 

namespace LBFGSB{
  const size_t max_iter = 10000;
  const size_t lbfgsb_max_iter = 1000;
  const size_t lbfgsb_m = 5; // Past gradients stored in lbfgsb
}

class COptimizer_LBFGSB : public COptimizer {
  
private:
  // Dual variable value we are adjusting
  double lambda;    
  
  // Regularizer for the Lagrangian
  double mu;        
  
public:

  COptimizer_LBFGSB(const size_t& dim, 
                    const bool& transposed, 
                    const double& eta, 
                    const double& nu,
                    const double& epsilon, 
                    const bool& binary);
  ~COptimizer_LBFGSB(void);
  
  int solve(void);
  
  void bounds(ap::integer_1d_array& nbd,
              ap::real_1d_array& l,
              ap::real_1d_array& u);
  
  void bounds_binary(ap::integer_1d_array& nbd,
                     ap::real_1d_array& l,
                     ap::real_1d_array& u);
  
  // Take as input an array to return the gradient
  // Return as output the objective value of augmented
  // lagrangian 
  double aug_lag_fg(const ap::real_1d_array& x0,
                    ap::real_1d_array& g);
  
}; 

# endif
