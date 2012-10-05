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

#include <iostream>
#include <cassert>
#include "booster.hpp"

AbstractBooster::AbstractBooster(AbstractOracle* &oracle, 
                   const int& num_pt, 
                   const int& max_iter):
  oracle(oracle), num_pt(num_pt), max_iter(max_iter), disp_freq(1){

  assert(oracle);
  
  // Initialize to uniform distribution
  dist.dim = num_pt;
  dist.val = new double[num_pt];
  for(size_t i = 0; i < dist.dim; i++) dist.val[i] = (1.0/num_pt);

  return;
}

AbstractBooster::AbstractBooster(AbstractOracle* &oracle, 
                   const int& num_pt, 
                   const int& max_iter,
		   const int& disp_freq):
  oracle(oracle), num_pt(num_pt), max_iter(max_iter), disp_freq(disp_freq){

  assert(oracle);
  
  // Initialize to uniform distribution
  dist.dim = num_pt;
  dist.val = new double[num_pt];
  for(size_t i = 0; i < dist.dim; i++) dist.val[i] = (1.0/num_pt);
  return;
}



AbstractBooster::~CBooster(){ 
  return;
}

size_t AbstractBooster::boost(std::ostream& os){
  int i = 0;
  size_t num_models = 0;
  for(i = 0; i < max_iter; i++){
    timer.start();
    WeakLearner* wl = oracle->max_edge_wl(dist);
    update_stopping_criterion(*wl);
    if(stopping_criterion(os)){
      break;
    }
    update_linear_ensemble(*wl);
    update_weights(*wl);
    timer.stop();
    
    std::cout << "Iteration : " << i << std::endl;
    
    if((i+1)%disp_freq==0){
      os << "Iteration : " << i << std::endl;
      if(disp_freq==1){
        os << "Time for this iteration: " 
           << timer.last_cpu << std::endl << std::endl; 
      }
      else
        os << "Cumulative Time: " << timer.total_cpu<< std::endl;
      os << model << std::endl;
      num_models++;
      if(dist.dim < 20)
        os << dist << std::endl;
    }
    
  }
  int num_iter;
  if(i == max_iter){
    os << "Max iterations exceeded!" << std::endl;
    num_iter = max_iter;
  }else{
    os << "Finished in " << i << " iterations " << std::endl;
    num_iter = i;
  }

  if(num_iter%disp_freq!=0){
    os << model << std::endl;
    os << "Cumulative Time: " << timer.total_cpu<< std::endl;
    num_models++;
  }
  
  os << "Total CPU time expended: " 
     << timer.total_cpu << std::endl; 
  os << "Maximum iteration time: " 
     << timer.max_cpu << std::endl;
  os << "Minimum iteration time: " 
     << timer.min_cpu << std::endl;
  os << "Average time per iteration: " 
     << timer.avg_cpu() << std::endl;
  //os << model;
  return num_models;
}
