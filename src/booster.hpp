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

#include <vector>
#include <iostream>

#include "ensemble.hpp"
#include "oracle.hpp"
#include "timer.hpp"



/** Base Class to encapsulate a boosting algorithm. Different
    implementations have to implement the virtual methods in this class
    to specify a full fledged boosting algorithm.
*/ 


class CBooster{

protected:
    
  // Oracle to supply hypothesis with max edge
  COracle* oracle;

  // Number of data points 
  int num_pt;
  
  // Maximum number of iterations
  int max_iter;

  // Affects how often we dump the model to file
  int disp_freq;
  
  // The model is just an ensemble of weak hypothesis seen so far
  CEnsemble model;
  
  // Distribution on the examples
  dvec dist;

  // Keep track of time per iteration 
  CTimer timer;

    
  virtual void update_weights(const CWeakLearner& wl)=0;
  
  virtual void update_linear_ensemble(const CWeakLearner& wl)=0;
  
  virtual bool stopping_criterion(std::ostream& os)=0;
  
  virtual void update_stopping_criterion(const CWeakLearner& wl)=0;
  
public:
    
  CBooster(COracle* &oracle, 
           const int& num_pt, 
           const int& max_iter);


  CBooster(COracle* &oracle, 
           const int& num_pt, 
           const int& max_iter,
	   const int& disp_freq);

  virtual ~CBooster();
  // Boost and save intermediate results
  size_t boost(std::ostream& os = std::cout);
  
  CEnsemble get_ensemble(void){ return model; }

};

#endif


