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

namespace totally_corrective_boosting
{

/** Encapsulate operations on vectors */

template <class T, class X> 
void dot(const std::vector<T>& mat, const X& vec, SparseVector& res);

template <class T, class X> 
void dot(const std::vector<T>& mat, const X& vec, DenseVector& res);

template <class T> 
void transpose_dot(const std::vector<T>& mat, const DenseVector& vec, DenseVector& res);

void transpose_dot(const std::vector<SparseVector>& mat, const SparseVector& vec, SparseVector& res);

void scale(SparseVector& a, const double& s);
void scale(DenseVector& a, const double& s);

double dot(const SparseVector& a, const SparseVector& b);
double dot(const SparseVector& a, const DenseVector& b);
double dot(const DenseVector& a, const DenseVector& b);
double dot(const DenseVector& a, const SparseVector& b);

// double dot(const dvec& a, const std::vector<double>& b);
// double dot(const svec& a, const std::vector<double>& b);

double relent(const DenseVector& d);

double binary_relent(const DenseVector& d, const double& nu);

void normalize(SparseVector& a);
void normalize(DenseVector& a);

void hadamard(const SparseVector& a, const SparseVector& b, SparseVector& res);

double sum(const SparseVector& a);
double sum(const DenseVector& a);

void copy(const DenseVector& source, DenseVector& target);

void axpy(const double& a, const DenseVector& x, const DenseVector& y, DenseVector& res);

void axpy(const double& a, const DenseVector& x, const SparseVector& y, DenseVector& res);

void axpy(const double& a, const SparseVector& x, const DenseVector& y, DenseVector& res);

void axpy(const double& a, const SparseVector& x, const SparseVector& y, SparseVector& res);

double diffnorm(const DenseVector& a, const DenseVector& b);

double max(const DenseVector& a);
size_t argmax(const DenseVector& a);

double min(const DenseVector& a);
size_t argmin(const DenseVector& a);

double abs_max(const DenseVector& a);
size_t abs_argmax(const DenseVector& a);

} // end of namespace totally_corrective_boosting


# endif
