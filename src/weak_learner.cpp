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

#include <cassert>
#include <iostream>
#include "weak_learner.hpp"
#include "parse.hpp"

CWeakLearner::CWeakLearner():
  wt(), edge(0), prediction(){}

CWeakLearner::CWeakLearner(const svec& wt, 
                           const double& edge, 
                           const svec& prediction):
  wt(wt), edge(edge), prediction(prediction){}

CWeakLearner::CWeakLearner(const CWeakLearner& wl): 
  wt(wl.wt), edge(wl.edge), prediction(wl.prediction){ }

double CWeakLearner::predict(const dvec& x) const{
  return dot(wt, x);
}

double CWeakLearner::predict(const svec& x) const{
  return dot(wt, x);
}

dvec CWeakLearner::predict(const std::vector<svec>& Data) const{
  

  dvec result(Data[0].dim);
  
  if(wt.nnz == 1){
    int idx = wt.idx[0];
    for(size_t i = 0; i < Data[idx].nnz; i++){
      int tmpidx = Data[idx].idx[i];
      result.val[tmpidx] = wt.val[0]*Data[idx].val[i];
    }
  }
  else{
    svec tmp;
    
    transpose_dot(Data,wt,tmp);
    for(size_t i = 0; i < tmp.nnz; i++){
      result.val[i] = tmp.val[i];
    }
  }
  return result;
}

std::string CWeakLearner::get_type() const {
  return "RAWDATA";
}

void CWeakLearner::dump(std::ostream& os) const{

  os << wt;
  os << "Edge: " << edge << std::endl;
  return;
}

void CWeakLearner::load(std::istream& in){
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


bool CWeakLearner::equal(const CWeakLearner *wl) const{
  return ( this->wt == wl->get_wt());
}


std::ostream& operator << (std::ostream& os, const CWeakLearner& wl){
  wl.dump(os);
  return os;
}

std::istream& operator >> (std::istream& in, CWeakLearner& wl){
  wl.load(in);
  return in;
}


bool operator == (const CWeakLearner& wl1, const CWeakLearner& wl2){

  return wl1.equal(&wl2);
}

