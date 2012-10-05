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
 * Last Updated: (28/03/2008)   
 */
#ifndef _WEAKLEARNER_HPP_
#define _WEAKLEARNER_HPP_


#include "vec.hpp"

/// Class to encapsulate a weak learner. For now our weak learner is a linear predictor.
// FIXME should separate the abstract weak learned from the linear predictor implementation
class WeakLearner{

protected:
  // svnvish: BUGBUG
  // Perhaps these should be const members?
  
  // The weak learner predicts with <wt, x> 
  svec wt;
  
  // Edge on the training dataset
  double edge;
  
  // Vector of predictions on the training dataset.
  svec prediction;
  
public:

  WeakLearner();
  
  WeakLearner(const svec& wt, const double& edge, const svec& prediction);
  
  WeakLearner(const WeakLearner& wl);
  
  virtual ~WeakLearner(){ }
  
  // Predict on single example
  virtual double predict(const dvec& x) const;
  virtual double predict(const svec& x) const;
  // predict on a data matrix
  // assumes it's read in using readlibSVM_transpose
  // i.e. Data must be a vector of hypotheses
  virtual dvec   predict(const std::vector<svec>& Data) const;

  // functions to get around the fact that friends can't be virtual
  virtual void dump(std::ostream& os) const;
  virtual void load(std::istream& in);
  virtual bool equal(const WeakLearner *wl) const;
  virtual std::string get_type() const;
  
  double get_edge(void) const { return edge; }
  svec get_wt(void) const { return wt;}
  svec get_prediction(void) const { return prediction; }

  // ugly hack. Need to figure out how to avoid.
  virtual bool get_direction(void) const {return false; }
  virtual double get_thresh(void) const {return 0.0; }
  virtual size_t get_idx(void) const {return 0;}

  friend 
  bool operator == (const WeakLearner& wl1, const WeakLearner& wl2);

  friend 
  std::ostream& operator << (std::ostream& os, const WeakLearner& wl);

  friend
  std::istream& operator >> (std::istream& in, WeakLearner& wl);
  
};

#endif
