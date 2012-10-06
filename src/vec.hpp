
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

/// Relative entropy with respect to the uniform distribution
double relative_entropy(const DenseVector& d);

/// Binary relative entropy with respect to the uniform distribution
/// Elements restricted to 1/nu
double binary_relative_entropy(const DenseVector& d, const double& nu);

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
