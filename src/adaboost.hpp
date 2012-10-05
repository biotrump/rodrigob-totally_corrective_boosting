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
#ifndef _ADABOOST_HPP_
#define _ADABOOST_HPP_

#include "booster.hpp"
#include "weak_learner.hpp"


/** Derived class. Implements AdaBoost 
 */

class CAdaBoost: public CBooster{

private:
  double alpha;
  
protected:
  
  void update_weights(const CWeakLearner& wl);
  
  void update_linear_ensemble(const CWeakLearner& wl);

  bool stopping_criterion(std::ostream& os);

  void update_stopping_criterion(const CWeakLearner& wl);
  
public:

  CAdaBoost(COracle* &oracle, 
            const int& num_pt, 
            const int& max_iter,
	    const int& disp_freq);
  ~CAdaBoost(){};
  
};


#endif
