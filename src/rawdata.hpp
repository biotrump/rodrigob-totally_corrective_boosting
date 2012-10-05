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
#ifndef _RAWDATA_HPP_
#define _RAWDATA_HPP_

/** Base class to encapsulate a oracle. It essentially represents a set
    of weak learners. Given a distribution over data it picks out the
    weak learner with the maximum edge. */ 

#include <vector>
#include "vec.hpp"
#include "oracle.hpp"

class CRawData: public COracle {
  
private:
  bool reflexive; // if true then training set is [data, -data]
  
public:
  CRawData(std::vector<svec>& data, 
           std::vector<int>& labels,
           const bool& transposed, 
           const bool& reflexive);
  ~CRawData();
  
  // given distribution return weak learner with maximum edge
  CWeakLearner* max_edge_wl(const dvec& dist); 
  
};

#endif
