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
 * Last Updated: (28/03/2009)   
 */

#include <cassert>
#include <limits>

#include "rawdata.hpp"

CRawData::CRawData(std::vector<svec>& data, 
                   std::vector<int>& labels, 
                   const bool& transposed,
                   const bool& reflexive):
  COracle(data, labels, transposed), reflexive(reflexive){}

CRawData::~CRawData(){ }

CWeakLearner* CRawData::max_edge_wl(const dvec& dist){
  
  dvec edges;
  
  assert(labels.size() == (size_t) dist.dim);
  
  dvec dist_labels(dist.dim);
  for(size_t i = 0; i < dist.dim; i++)
    dist_labels.val[i] = dist.val[i]*labels[i];
  
  if(transposed)
    dot(data, dist_labels, edges);
  else
    transpose_dot(data, dist_labels, edges);
  
  size_t max_idx = 0;
  double max_edge = -std::numeric_limits<double>::max();
  
  size_t min_idx = 0;
  double min_edge = std::numeric_limits<double>::max(); 
  
  for(size_t i = 0; i < edges.dim; i++){
    // identify the max
    if(edges.val[i] > max_edge){
      max_edge = edges.val[i];
      max_idx = i;
    }
    // identify the min
    if(edges.val[i] < min_edge){
      min_edge = edges.val[i];
      min_idx = i;
    }
  }
  
  if(reflexive && (min_edge < 0) && (-min_edge > max_edge)){
    
    // return the negation of the min_edge feature
    svec wt(edges.dim, 1);

    wt.idx[0] = min_idx;
    wt.val[0] = -1.0;
    
    svec prediction;
    if(transposed){
      prediction = data[min_idx];
      scale(prediction, -1.0);
    }else
      dot(data, wt, prediction);
    
    
    // Multiply the predictions with the labels here
    for(size_t i = 0; i < prediction.nnz; i++)
      prediction.val[i] *= labels[prediction.idx[i]];
    
    double edge = -min_edge;
    
    // When the weak learner goes out of scope
    // wt and prediction vectors are deleted and the memory is freed
    
    //return CWeakLearner(wt, edge, prediction);
    CWeakLearner* wl = new CWeakLearner(wt, edge, prediction);
    //std::cout << "hyp: " << min_idx << std::endl;
    return wl;
  }
  
  // return the max_edge feature

  svec wt(edges.dim, 1);
  wt.idx[0] = max_idx;
  wt.val[0] = 1.0;
  
  svec prediction;
  if(transposed)
    prediction = data[max_idx];
  else
    dot(data, wt, prediction);
  
  
  // Multiply the predictions with the labels here
  for(size_t i = 0; i < prediction.nnz; i++)
    prediction.val[i] *= labels[prediction.idx[i]];
  
  double edge = max_edge;
  
  CWeakLearner* wl = new CWeakLearner(wt, edge, prediction);
  //std::cout << "hyp: " << min_idx << std::endl;
  return wl;
}
