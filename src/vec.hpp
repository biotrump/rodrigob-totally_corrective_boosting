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

#ifndef _VEC_HPP_
#define _VEC_HPP_

#include "svec.hpp"
#include "dvec.hpp"
#include "ivec.hpp"
/** Encapsulate operations on vectors */

template <class T, class X> 
void dot(const std::vector<T>& mat, const X& vec, svec& res);

template <class T, class X> 
void dot(const std::vector<T>& mat, const X& vec, dvec& res);

template <class T> 
void transpose_dot(const std::vector<T>& mat, const dvec& vec, dvec& res);

void transpose_dot(const std::vector<svec>& mat, const svec& vec, svec& res);

void scale(svec& a, const double& s);
void scale(dvec& a, const double& s);

double dot(const svec& a, const svec& b);
double dot(const svec& a, const dvec& b);
double dot(const dvec& a, const dvec& b);
double dot(const dvec& a, const svec& b);

// double dot(const dvec& a, const std::vector<double>& b);
// double dot(const svec& a, const std::vector<double>& b);

double relent(const dvec& d);

double binary_relent(const dvec& d, const double& nu);

void normalize(svec& a);
void normalize(dvec& a);

void hadamard(const svec& a, const svec& b, svec& res);

double sum(const svec& a);
double sum(const dvec& a);

void copy(const dvec& source, dvec& target);

void axpy(const double& a, const dvec& x, const dvec& y, dvec& res);

void axpy(const double& a, const dvec& x, const svec& y, dvec& res);

void axpy(const double& a, const svec& x, const dvec& y, dvec& res);

void axpy(const double& a, const svec& x, const svec& y, svec& res);

double diffnorm(const dvec& a, const dvec& b);

double max(const dvec& a);
size_t argmax(const dvec& a);

double min(const dvec& a);
size_t argmin(const dvec& a);

double abs_max(const dvec& a);
size_t abs_argmax(const dvec& a);

# endif
