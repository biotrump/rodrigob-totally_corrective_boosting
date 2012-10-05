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
 * Authors: Karen Glocer, S V N Vishwanathan
 *
 * Created: (20/04/2009) 
 *
 * Last Updated: (23/04/2009)   
 */
#ifndef _WEAKLEARNERDSTUMP_HPP_
#define _WEAKLEARNERDSTUMP_HPP_

/** Class to encapsulate a weak learner. For now our weak learner is a
    linear predictor. 
*/ 

#include "vec.hpp"
#include "weak_learner.hpp"
#include "weak_learner_dstump.hpp"

class CWeakLearnerDstump: public WeakLearner{
  //class CWeakLearnerDstump{
private:


  // threshold
  double thresh;

  // direction of threshold: 
  // if dir == true, then x >= thresh
  // else x <= thresh
  bool direction;
  
  // index of hypothesis for best decision stump
  size_t idx;
  
public:
  
  CWeakLearnerDstump();

  CWeakLearnerDstump(const svec& wt, const double& edge, const svec& prediction, 
		     const double& thresh, const bool& direction, const int& idx);
  
  CWeakLearnerDstump(const CWeakLearnerDstump& wl);
  
  ~CWeakLearnerDstump(){ }

  std::string get_type() const;
  
  // Predict on examples
  double predict(const dvec& x) const;
  double predict(const svec& x) const;
  // predict on a data matrix
  // assumes it's read in using readlibSVM_transpose
  // i.e. Data must be a vector of hypotheses
  dvec   predict(const std::vector<svec>& Data) const;

  // methods to dump and load data
  void dump(std::ostream& os) const;
  void load(std::istream& in);
  bool equal(const WeakLearner *wl) const;

  // accessor methods
  bool get_direction(void) const {return direction; }
  double get_thresh(void) const {return thresh; }
  size_t get_idx(void) const {return idx;}
  
  friend 
  std::ostream& operator << (std::ostream& os, const CWeakLearnerDstump& wl);  

  friend
  std::istream& operator >> (std::istream& in, CWeakLearnerDstump& wl);

  friend 
  bool operator == (const CWeakLearnerDstump& wl1, const CWeakLearnerDstump& wl2);

};

#endif
