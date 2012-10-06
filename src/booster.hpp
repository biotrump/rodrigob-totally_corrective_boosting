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
#ifndef _BOOSTER_HPP_
#define _BOOSTER_HPP_

#include "ensemble.hpp"
#include "oracle.hpp"
#include "timer.hpp"

#include <vector>
#include <iostream>


/// Base Class to encapsulate a boosting algorithm. Different
/// implementations have to implement the virtual methods in this class
/// to specify a full fledged boosting algorithm.
class AbstractBooster {

protected:
    
  /// Oracle to supply hypothesis with max edge
  AbstractOracle* oracle;

  /// Number of data points
  int num_pt;
  
  /// Maximum number of iterations
  int max_iter;

  /// Affects how often we dump the model to file
  int disp_freq;
  
  /// The model is just an ensemble of weak hypothesis seen so far
  Ensemble model;
  
  /// Distribution on the examples
  DenseVector dist;

  /// Keep track of time per iteration
  Timer timer;

    
  virtual void update_weights(const WeakLearner& wl)=0;
  
  virtual void update_linear_ensemble(const WeakLearner& wl)=0;
  
  virtual bool stopping_criterion(std::ostream& os)=0;
  
  virtual void update_stopping_criterion(const WeakLearner& wl)=0;
  
public:
    
  AbstractBooster(AbstractOracle* &oracle,
           const int& num_pt, 
           const int& max_iter);


  AbstractBooster(AbstractOracle* &oracle,
           const int& num_pt, 
           const int& max_iter,
	   const int& disp_freq);

  virtual ~AbstractBooster();

  /// Boost and save intermediate results
  size_t boost(std::ostream& os = std::cout);
  
  Ensemble get_ensemble(void){ return model; }

};

#endif


