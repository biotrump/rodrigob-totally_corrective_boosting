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
 * Last Updated: (20/04/2009)   
 */

#include "DecisionStumpWeakLearner.hpp"

#include "parse.hpp"

#include <cassert>
#include <iostream>

namespace totally_corrective_boosting
{

DecisionStumpWeakLearner::DecisionStumpWeakLearner():
  WeakLearner(),
  thresh(0), direction(true), idx(0){}

//BUG: wt is completely unnecessary. 
DecisionStumpWeakLearner::DecisionStumpWeakLearner(const SparseVector& wt, 
				       const double& edge, 
				       const SparseVector& prediction,
				       const double& thresh,
				       const bool& direction,
				       const int& idx):
  WeakLearner(wt,edge,prediction), 
  thresh(thresh), direction(direction), idx(idx){}



DecisionStumpWeakLearner::DecisionStumpWeakLearner(const DecisionStumpWeakLearner& wl): 
WeakLearner(wt,edge,prediction), thresh(thresh), direction(direction){}

std::string DecisionStumpWeakLearner::get_type() const {
  return "DSTUMP";
}

// these are the wrong thing to do
double DecisionStumpWeakLearner::predict(const DenseVector& x) const{

  double tmp = dot(wt,x);
  double result;

  if(direction){
    if( tmp >= thresh){result = 1.0;}
    else {result = -1.0;}
  }
  else{
    if( tmp <= thresh){result = 1.0;}
    else {result = -1.0;}
  }
  
    return result;
}

double DecisionStumpWeakLearner::predict(const SparseVector& x) const{

  double tmp = dot(wt,x);
  double result;

  if(direction){
    if( tmp >= thresh){result = 1.0;}
    else {result = -1.0;}
  }
  else{
    if( tmp <= thresh){result = 1.0;}
    else {result = -1.0;}
  }
  
    return result;
}


DenseVector DecisionStumpWeakLearner::predict(const std::vector<SparseVector>& Data) const{

  DenseVector result(Data[idx].dim);

  for(size_t i = 0; i < Data[idx].nnz; i++){
    int tmpidx = Data[idx].idx[i];
    result.val[tmpidx] = Data[idx].val[i];
  }

  for(size_t i = 0; i < Data[idx].dim; i++){

    if(direction){
      if( result.val[i] >= thresh){result.val[i] = 1.0;}
      else {result.val[i] = -1.0;}
    }
    else{
      if( result.val[i] <= thresh){result.val[i] = 1.0;}
      else {result.val[i] = -1.0;}
    }

  }
  
    return result;
}

void DecisionStumpWeakLearner::dump(std::ostream& os) const{
  os  << wt;
  os << "edge: " << edge << std::endl;
  os << "thresh: " << thresh << std::endl;
  os << "dir: " << direction << std::endl;
  os << "idx: " << idx << std::endl;
  return;
}

void DecisionStumpWeakLearner::load(std::istream& in){
  try {
    in >> wt;
    expect_keyword(in,"edge:");
    in >> edge;
    expect_keyword(in,"thresh:");
    in >> thresh;
    expect_keyword(in,"dir:");
    in >> direction;
    expect_keyword(in,"idx:");
    in >> idx;
  }
  catch (std::string s) {
    std::cerr << "Error when reading ensemble: " << s << std::endl;
    exit(1);
  }
  return;
}

bool DecisionStumpWeakLearner::equal(const WeakLearner *wl) const{
  return ( this->thresh == wl->get_thresh() 
	   && this->direction == wl->get_direction()
	   && this->idx == wl->get_idx()
	   && this->wt == wl->get_wt());
}

std::ostream& operator << (std::ostream& os, const DecisionStumpWeakLearner& wl){
  wl.dump(os);
  return os;
  
}

std::istream& operator >> (std::istream& in, DecisionStumpWeakLearner& wl){
  wl.load(in);
  return in;
}

bool operator == (const DecisionStumpWeakLearner& wl1, const DecisionStumpWeakLearner& wl2){

  return wl1.equal(&wl2);
}

} // end of namespace totally_corrective_boosting
