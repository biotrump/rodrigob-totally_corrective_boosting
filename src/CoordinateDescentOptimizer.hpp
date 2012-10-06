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

#include "AbstractOptimizer.hpp"

namespace totally_corrective_boosting
{


namespace CD{
  const size_t max_iter = 50000;
  const size_t ls_max_iter = 20;
}

/// Implement coordinate descent. Basically this is nothing but the
/// corrective algorithm which is run in a loop.
class CoordinateDescentOptimizer : public AbstractOptimizer {
  
private:
  double proj_simplex(DenseVector& dist, const double& exp_max);

  double line_search(DenseVector& W, 
                     const DenseVector& grad_w, 
                     const size_t& idx);
    
public:
  
  CoordinateDescentOptimizer(const size_t& dim,
                const bool& transposed, 
                const double& eta, 
                const double& nu,
                const double& epsilon,
                const bool& binary);
  ~CoordinateDescentOptimizer(void){ }
  
  int solve(void);
  
}; 

} // end of namespace totally_corrective_boosting

# endif
