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
 * Last Updated: (11/06/2009)   
 */

#ifndef _DVEC_HPP_
#define _DVEC_HPP_

#include <cassert>
#include <vector>
#include <iostream>

// Dense vector

class dvec{
public:
  // List of values
  double *val;
  
  // Overall dimension
  size_t dim;
  
  // default constructor 
  dvec(void){ dim = 0; val = NULL; }

  // Other constructors: Syntactic sugar 
  dvec (const size_t& dim): dim(dim) { 
    val = new double[dim]; 
    for(size_t i = 0; i < dim; i++)
      val[i] = 0.0;
  }

  dvec (const size_t& dim, const double& _val): dim(dim){
    val = new double[dim];
    for(size_t i = 0; i < dim; i++)
      val[i] = _val;
    return;
  }
    
  dvec (const size_t& dim, const double* _val): dim(dim){
    val = new double[dim];
    for(size_t i = 0; i < dim; i++) 
      val[i] = _val[i];
    return;
  }
  
  // Copy constructor 
  dvec(const dvec& d): dim(d.dim){
    val = new double[dim];
    for(size_t i = 0; i < dim; i++)
      val[i] = d.val[i];
    return;
  }
  
  ~dvec(void){
    if(val != NULL) delete[] val; val = NULL;
    dim = 0; 
    return;
  }
  
  // WARNING: Old data will be lost
  void resize(const size_t& _dim){
    if(val != NULL) delete[] val; val = NULL;
    dim = _dim;
    val = new double[dim];
    for(size_t i = 0; i < dim; i++)
      val[i] = 0.0;
    return;
  }

  dvec& operator=(const dvec &rhs){
    if(val != NULL) delete [] val;
    dim = rhs.dim;
    val = new double[dim];
    for(size_t i = 0; i < dim; i++)
      val[i] = rhs.val[i];
    return *this;
  }
  
  // clear contents of current vector 
  void clear(void){
    if(val != NULL) delete[] val; val = NULL;
    dim = 0;
    return;
  }
  friend 
  std::ostream& operator << (std::ostream& os, const dvec& d);
  
  friend 
  bool operator == (const dvec& s1, const dvec& s2);
  
};

# endif
