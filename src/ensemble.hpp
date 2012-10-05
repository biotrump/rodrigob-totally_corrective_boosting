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

#ifndef _ENSEMBLE_HPP_
#define _ENSEMBLE_HPP_

#include <vector>
#include <iostream>

#include "weak_learner.hpp"

class Ensemble;

class WeightedWeakLearner{
private:
  // Weak Learner
  const WeakLearner* wl;
  
  // Weight associated with the wl
  double wt;
  
public:
  WeightedWeakLearner(const WeakLearner* _wl, const double& _wt):wl(_wl), wt(_wt){}
  WeightedWeakLearner(const WeightedWeakLearner& wwl):wl(wwl.wl), wt(wwl.wt){}
  ~WeightedWeakLearner(){}
  
  void set_wt(double _wt){ wt = _wt; }
  double get_wt(void) const { return wt; }
  void scale_wt(double scale){ wt *= scale; }
  void add_wt(double alpha){wt += alpha;}
  std::string get_type() const {return wl->get_type();}
  double weighted_predict(const dvec& x) const { return wt*(wl->predict(x)); }
  
  double weighted_predict(const svec& x) const { return wt*(wl->predict(x)); }
  
  dvec weighted_predict(const std::vector<svec>& data) const { 
    dvec result = wl->predict(data);
    scale(result, wt); 
    return result;
  }
  
  friend 
  bool operator == (const WeightedWeakLearner& w1,
                    const WeightedWeakLearner& w2){
    return (*(w1.wl) == *(w2.wl));
  }
  

  // bool wl_equal(const CWeakLearner* _wl){ return (*wl == *_wl); };

  friend 
  std::ostream& operator << (std::ostream& os, const WeightedWeakLearner& wwl);
  
  friend
  std::istream& operator >> (std::istream& in, WeightedWeakLearner& wwl);
  
};

typedef std::vector<WeightedWeakLearner>::const_iterator wwl_citr;
typedef std::vector<WeightedWeakLearner>::iterator wwl_itr;

class Ensemble{

private:

  std::vector<WeightedWeakLearner> ensemble;
  
public:
  
  bool add(const WeightedWeakLearner& wwl);

  // predict on single examples
  double predict(const svec& x) const;
  double predict(const dvec& x) const;

  // predict on full matrix
  // Assumes matrix is read in using 
  // readlibSVM_transpose
  dvec   predict(const std::vector<svec>& data) const;

  void set_wts(const dvec& wts);

  void scale_wts(const double& scale);

  void set_wt(const double& wt, const size_t& idx);

  void add_wt(const double& wt, const size_t& idx);
  
  dvec get_wts(void) const;
  
  // bool find_wl(const CWeakLearner* wl, size_t& idx);
  
  size_t size(void) const { return ensemble.size(); } 
  
  friend
  std::ostream& operator << (std::ostream& os, const Ensemble& e);

  friend
  std::istream& operator >> (std::istream& in, Ensemble& e);
  
};

# endif
