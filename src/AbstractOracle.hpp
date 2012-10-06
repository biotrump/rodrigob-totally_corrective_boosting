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
 * Last Updated: (23/04/2008)   
 */
#ifndef _ORACLE_HPP_
#define _ORACLE_HPP_


#include "vec.hpp"
#include "WeakLearner.hpp"

#include <vector>

namespace totally_corrective_boosting
{

/// Base class to encapsulate a oracle. It essentially represents a set
/// of weak learners. Given a distribution over data it picks out the
/// weak learner with the maximum edge.
/// Every new oracle has to implement the max_edge_wl function.
class AbstractOracle{

protected:
  std::vector<SparseVector> data;
  std::vector<int> labels;
  
  bool transposed;
  
public:
  AbstractOracle(std::vector<SparseVector>& data,
          std::vector<int>& labels,
          const bool& transposed = false): 
    data(data), labels(labels), transposed(transposed){}
  virtual ~AbstractOracle(){}
  
  // given distribution return weak learner with maximum edge
  virtual WeakLearner* max_edge_wl(const DenseVector& dist) = 0;
};

} // end of namespace totally_corrective_boosting


#endif
