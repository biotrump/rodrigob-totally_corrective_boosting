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
 * Last Updated: (15/04/2008)   
 */

#ifndef _OPTIMIZER_TAO_HPP_
#define _OPTIMIZER_TAO_HPP_

#if not defined(USE_TAO)
#error This file should only be included when the TAO library is used.
#endif

#include "AbstractOptimizer.hpp"

#include <tao.h>
#include <petscvec.h>


namespace totally_corrective_boosting
{


namespace TAO{
  const size_t max_iter = 10000;
}

/// Augmented Lagrangian solver in the w domain.
class TaoOptimizer : public AbstractOptimizer {
  
private:
  // Dual variable value we are adjusting
  double lambda;    
  
  // Regularizer for the Lagrangian
  double mu;        
  
public:

  TaoOptimizer(const size_t& dim,
                 const bool& transposed, 
                 const double& eta, 
                 const double& nu,
                 const double& epsilon, 
                 const bool& binary,
                 int& argc, 
                 char** argv);
  
  ~TaoOptimizer(void);
  
  int solve(void);
  
  int bounds(Vec& xl, Vec& xu);
  
  int bounds_binary(Vec& xl, Vec& xu);
  
  // Take as input an array to return the gradient
  // Return as output the objective value of augmented
  // lagrangian 
  double aug_lag_fg(const Vec& X, Vec& G);
  
}; 

int tao_fun_grad(TAO_APPLICATION taoapp,
                 Vec X,
                 double *obj,
                 Vec G,
                 void *ctx);

int tao_bounds(TAO_APPLICATION taoapp, 
               Vec xl, 
               Vec xu, 
               void *ctx);

int tao_bounds_binary(TAO_APPLICATION taoapp, 
                      Vec xl, 
                      Vec xu, 
                      void *ctx);

} // end of namespace totally_corrective_boosting

# endif
