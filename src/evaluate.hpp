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
 * Authors: Karen Glocer 
 *
 * Created: (28/04/2009) 
 *
 * Last Updated: (28/04/2008)   
 */

#ifndef _EVALUATE_HPP_
#define _EVALUATE_HPP_

#include <vector>
#include <iostream>
#include "vec.hpp"

// evaluates binary loss
// 
class CEvaluate{
  
  // private:
  // if tie_breaking == true, break ties in favor of +1 class
  // else break ties in favor of -1 class
  bool tie_breaking;
  
public:

  CEvaluate();

  CEvaluate(bool tie_breaking);

  ~CEvaluate(){ }

  // evaluate loss on single example
  // returns:
  //     0 if prediction has same sign as label
  //     1 otherwise
  int binary_loss(const double pred, const int label) const;

  // evaluate binary loss on all examples
  // side effect: 
  // total_loss is the total binary loss:
  //     0 if prediction has same sign as label
  //     1 otherwise
  // percent_err is the error rate
  void binary_loss(const dvec pred, const std::vector<int>& labels,
		   int& total_loss, double& percent_err) const;
  
};

# endif
