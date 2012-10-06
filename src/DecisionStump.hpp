/* Copyright (c) 2009
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
 * Authors: Karen Glocer
 *
 * Created: (20/04/2009) 
 *
 * Last Updated: (20/04/2008)   
 */

#ifndef _DECISIONSTUMP_HPP_
#define _DECISIONSTUMP_HPP_

#include "AbstractOracle.hpp"
#include "WeakLearner.hpp"
#include "DecisionStumpWeakLearner.hpp"
#include "Timer.hpp"

#include "dvec.hpp"
#include "svec.hpp"
#include "ivec.hpp"


#include <vector>
#include <iostream>

namespace totally_corrective_boosting
{

// using namespace std; 


/// Decision stump oracle
class DecisionStump: public AbstractOracle {

private:

  // if less_than == true, the we consider x <= thresh
  // as well as x >= thresh
  // sort of correponds to reflexive
  bool less_than; 

  std::vector<DenseIntegerVector> sorted_data;
  
  // Keep track of time spent in max_edge_wl
  Timer timer;
  
public:
  DecisionStump(std::vector<SparseVector>& data,
                 std::vector<int>& labels, 
                 bool less_than);

  ~DecisionStump();

  // given distribution return weak learner with maximum edge
  DecisionStumpWeakLearner* max_edge_wl(const DenseVector& dist); 
  //CWeakLearner max_edge_wl(const dvec& dist); 

  // given a hypothesis and distribution, return the best threshold
  // the best edge, and the direction of the best threshold
  // if ge==true, then x >= thresh else x <= thresh
  void fbthresh(const size_t& idx, 
                const DenseVector& dist, 
                const double& init_edge,
                double& best_threshold,
                double& best_edge, 
                bool& ge);
  
  // given a sorted vector of (hyp,label,dist) triplets,
  // return the best threshold and edge for hyp <= thresh
  void fbthresh_le(const size_t& idx, 
                   const double& dist_diff, 
                   const DenseVector& dist, 
                   const double& init_edge,
                   double& best_threshold, 
                   double& best_edge);
  
  // given a sorted vector of (hyp,label,dist) triplets,
  // return the best threshold and edge for hyp >= thresh
  void fbthresh_ge(const size_t& idx, 
                   const double& dist_diff, 
                   const DenseVector& dist, 
                   const double& init_edge,
                   double& best_threshold, 
                   double& best_edge);
  
  DenseIntegerVector argsort(SparseVector unsorted);
  
};


} // end of namespace totally_corrective_boosting



#endif