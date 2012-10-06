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
 * Last Updated: (09/06/2009)   
 */
#ifndef _SVM_HPP_
#define _SVM_HPP_

// SVM oracle. Hypothesis are the data themselves. 
// Prediction of an oracle x on data x' is simply 
// <x, x'>

#include "AbstractOracle.hpp"

#include "vec.hpp"

#include <vector>

namespace totally_corrective_boosting
{


class Svm: public AbstractOracle {
  
private:
  bool reflexive; // if true then training set is [data, -data]
  
public:
  Svm(std::vector<SparseVector>& data,
       std::vector<int>& labels,
       const bool& transposed, 
       const bool& reflexive);
  ~Svm();
  
  // given distribution return weak learner with maximum edge
  WeakLearner* max_edge_wl(const DenseVector& dist); 
  
};

} // end of namespace totally_corrective_boosting

#endif