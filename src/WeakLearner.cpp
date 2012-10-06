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

#include "WeakLearner.hpp"

#include "parse.hpp"

#include <cassert>
#include <iostream>

namespace totally_corrective_boosting
{

WeakLearner::WeakLearner():
  wt(), edge(0), prediction(){}

WeakLearner::WeakLearner(const SparseVector& wt,
                           const double& edge, 
                           const SparseVector& prediction):
  wt(wt), edge(edge), prediction(prediction){}

WeakLearner::WeakLearner(const WeakLearner& wl):
  wt(wl.wt), edge(wl.edge), prediction(wl.prediction){ }

double WeakLearner::predict(const DenseVector& x) const{
  return dot(wt, x);
}

double WeakLearner::predict(const SparseVector& x) const{
  return dot(wt, x);
}

DenseVector WeakLearner::predict(const std::vector<SparseVector>& Data) const{
  

  DenseVector result(Data[0].dim);
  
  if(wt.nnz == 1){
    int idx = wt.idx[0];
    for(size_t i = 0; i < Data[idx].nnz; i++){
      int tmpidx = Data[idx].idx[i];
      result.val[tmpidx] = wt.val[0]*Data[idx].val[i];
    }
  }
  else{
    SparseVector tmp;
    
    transpose_dot(Data,wt,tmp);
    for(size_t i = 0; i < tmp.nnz; i++){
      result.val[i] = tmp.val[i];
    }
  }
  return result;
}

std::string WeakLearner::get_type() const {
  return "RAWDATA";
}

void WeakLearner::dump(std::ostream& os) const{

  os << wt;
  os << "Edge: " << edge << std::endl;
  return;
}

void WeakLearner::load(std::istream& in){
  try {
    in >> wt;
    expect_keyword(in,"Edge:");
    in >> edge;
  }
  catch (std::string s) {
    std::cerr << "Error when reading raw data weak learner: " << s << std::endl;
    exit(1);
  }
  return;
}


bool WeakLearner::equal(const WeakLearner *wl) const{
  return ( this->wt == wl->get_wt());
}


std::ostream& operator << (std::ostream& os, const WeakLearner& wl){
  wl.dump(os);
  return os;
}

std::istream& operator >> (std::istream& in, WeakLearner& wl){
  wl.load(in);
  return in;
}


bool operator == (const WeakLearner& wl1, const WeakLearner& wl2){

  return wl1.equal(&wl2);
}

} // end of namespace totally_corrective_boosting

