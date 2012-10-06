
#include "vec.hpp"

#include <iostream>
#include <cmath>
#include <limits>

namespace totally_corrective_boosting
{

// dot product of matrix with vector 
// store result in sparse vector res
// We will allocate memory for the result.
// To explicitly encourage the callers to deallocate 
// we will assert that res.idx and res.val are NULL
template <class T, class X>
void dot(const std::vector<T>& mat, const X& vec, SparseVector& res){
  
  // paranoia
  assert(res.val == NULL);
  assert(res.idx == NULL);
  
  assert(mat.size());
  // might slow things down ...
  assert(mat[0].dim == vec.dim);
  
  res.nnz = mat.size();
  res.val = new double[res.nnz];
  res.idx = new size_t[res.nnz];
  res.dim = res.nnz;
  
  size_t i =0;
  typename std::vector<T>::const_iterator it;
  for (it = mat.begin(); it!=mat.end(); it++){ 
    res.val[i] = dot(*it, vec);
    res.idx[i] = i;
    i++;
  }
 
  return;
}

template 
void dot(const std::vector<SparseVector>& mat, const SparseVector& vec, SparseVector& res);

template 
void dot(const std::vector<DenseVector>& mat, const SparseVector& vec, SparseVector& res);

// svnvish: BUGBUG
// don't need it for now 
// template 
// void dot(const std::vector<svec>& mat, const dvec& vec, svec& res);

// template 
// void dot(const std::vector<dvec>& mat, const dvec& vec, svec& res);


// dot product of matrix with vector 
// store result in dense vector res
// We will allocate memory for the result.
// To explicitly encourage the callers to deallocate 
// we will assert that res.val is NULL

template <class T, class X>
void dot(const std::vector<T>& mat, const X& vec, DenseVector& res){
  
  // paranoia
  assert(res.val == NULL);
  
  assert(mat.size());
  // might slow things down ...
  assert(mat[0].dim == vec.dim);
  
  res.val = new double[mat.size()];
  res.dim = mat.size();
  
  size_t i = 0;
  typename std::vector<T>::const_iterator it;
  for (it = mat.begin(); it!=mat.end(); it++){ 
    res.val[i] = dot(*it, vec);
    i++;
  }
  return;
}

template 
void dot(const std::vector<SparseVector>& mat, const SparseVector& vec, DenseVector& res);

template 
void dot(const std::vector<SparseVector>& mat, const DenseVector& vec, DenseVector& res);

template 
void dot(const std::vector<DenseVector>& mat, const SparseVector& vec, DenseVector& res);

template 
void dot(const std::vector<DenseVector>& mat, const DenseVector& vec, DenseVector& res);

// dot product of transpose of matrix with vector 
// store result in dense vector res
// We will allocate memory for the result.
// To explicitly encourage the callers to deallocate 
// we will assert that res.val is NULL

template <class T>
void transpose_dot(const std::vector<T>& mat, const DenseVector& vec, DenseVector& res){
  
  // paranoia
  assert(res.val == NULL);
  
  assert(mat.size());
  // might slow things down ...
  assert(mat.size() == (size_t) vec.dim);

  res.resize(mat[0].dim);

  typename std::vector<T>::const_iterator it;
  double* vec_val;
  for (it = mat.begin(), vec_val = vec.val; it!=mat.end(); vec_val++, it++)
    axpy(*vec_val, *it, res, res);
  
  return;
}

template
void transpose_dot(const std::vector<SparseVector>& mat, const DenseVector& vec, DenseVector& res);

template
void transpose_dot(const std::vector<DenseVector>& mat, const DenseVector& vec, DenseVector& res);

// dot product of transpose of matrix with vector 
// store result in dense vector res
// We will allocate memory for the result.
// To explicitly encourage the callers to deallocate 
// we will assert that res.val is NULL

void transpose_dot(const std::vector<SparseVector>& mat, const SparseVector& vec, SparseVector& res){
  
  // paranoia
  assert(res.val == NULL);
  
  assert(mat.size());
  // might slow things down ...
  assert(mat.size() == (size_t) vec.dim);
  
  // res as a svec is redundant but that is what the caller wants
  res.resize(mat[0].dim, mat[0].dim);
  
  for(size_t i = 0; i < vec.nnz; i++){
    size_t idx = vec.idx[i];
    double scale = vec.val[i];
    SparseVector row = mat[idx];
    for(size_t i = 0; i < row.nnz; i++)
      res.val[row.idx[i]] += scale*row.val[i];
  }
  
  return;
}

double dot(const SparseVector& a, const SparseVector& b){
  
  assert(a.dim == b.dim);
  
  size_t i=0, j=0;
  double val = 0;
  
  while((i<a.nnz) && (j<b.nnz)){
    if(a.idx[i] < b.idx[j]) i++;
    if(a.idx[i] > b.idx[j]) j++;
    if(a.idx[i] == b.idx[j]){
      val += a.val[i]*b.val[j];
      i++;
      j++;
    } 
  }
  return val;
}

double dot(const SparseVector& a, const DenseVector& b){
  
  assert(a.dim == b.dim);
  double val = 0;
  
  for(size_t i = 0; i < a.nnz; i++){
    val += b.val[a.idx[i]]*a.val[i];
  }
  return val;
}

double dot(const DenseVector& a, const SparseVector& b){
  return dot(b, a);
}
// double dot(const svec& a, const std::vector<double>& b){
  
//   assert(a.dim == b.size());
//   double val = 0;
  
//   for(size_t i = 0; i < a.nnz; i++){
//     val += b[a.idx[i]]*a.val[i];
//   }
//   return val;
// }

// double dot(const dvec& a, const std::vector<double>& b){
//   assert(a.dim == b.size());
//   double val = 0.0;
//   for(size_t i = 0; i < a.dim; i++)
//     val += b[i]*a.val[i];
//   return val;
// }

double dot(const DenseVector& a, const DenseVector& b){
  assert(a.dim == b.dim);
  double val = 0.0;
  for(size_t i = 0; i < a.dim; i++)
    val += b.val[i]*a.val[i];
  return val;
}

// Hadamard product of a and b
// store result in res
// We will allocate memory for the result.
// To explicitly encourage the callers to deallocate 
// we will assert that res.idx and res.val are NULL

void hadamard(const SparseVector& a, const SparseVector& b, SparseVector& res){

  // paranoia
  assert(res.val == NULL);
  assert(res.idx == NULL);
  
  assert(a.dim == b.dim);
  
  size_t i = 0, j = 0;
  std::vector<double> val;
  std::vector<size_t> idx;
  
  while((i<a.nnz) && (j<b.nnz)){
    if(a.idx[i] < b.idx[j]) i++;
    if(a.idx[i] > b.idx[j]) j++;
    if(a.idx[i] == b.idx[j]){
      val.push_back(a.val[i]*b.val[j]);
      idx.push_back(a.idx[i]);
      i++;
      j++;
    } 
  }
  
  res.nnz = val.size();
  res.dim = a.dim;
  
  if(res.nnz){
    
    res.val = new double[res.nnz];
    res.idx = new size_t[res.nnz];
    
    for(size_t i = 0; i < res.nnz; i++){
      res.val[i] = val[i];
      res.idx[i] = idx[i];
    }
  } 
  
  return;
}

// scale a by scalar s
void scale(SparseVector& a, const double& s){

  for(size_t i = 0; i < a.nnz; i++) 
    a.val[i] *= s;
  
  return;
}

// scale a by scalar s
void scale(DenseVector& a, const double& s){
  
  for(size_t i = 0; i < a.dim; i++) 
    a.val[i] *= s;
  
  return;
}

// copy values
void copy(const DenseVector& source, DenseVector& target){
  assert(source.dim == target.dim);
  for(size_t i = 0; i < target.dim; i++)
    target.val[i] = source.val[i];
  return;
}

void axpy(const double& a, const DenseVector& x, const DenseVector& y, DenseVector& res){
  assert(x.dim == y.dim);
  assert(y.dim == res.dim);
  for(size_t i = 0; i < res.dim; i++)
    res.val[i] = a*x.val[i] + y.val[i];
  return;
}

void axpy(const double& a, const DenseVector& x, const SparseVector& y, DenseVector& res){
  assert(x.dim == y.dim);
  assert(y.dim == res.dim);
  for(size_t i = 0; i < res.dim; i++)
    res.val[i] = a*x.val[i];

  // svnvish:BUGBUG
  // Not the most efficient way to do this 
  for(size_t i = 0; i < y.nnz; i++)
    res.val[y.idx[i]] += y.val[i];
  
  return;
}

void axpy(const double& a, const SparseVector& x, const DenseVector& y, DenseVector& res){
  assert(x.dim == y.dim);
  assert(y.dim == res.dim);
  for(size_t i = 0; i < res.dim; i++)
    res.val[i] = y.val[i];

  // svnvish:BUGBUG
  // Not the most efficient way to do this   
  for(size_t i = 0; i < x.nnz; i++)
    res.val[x.idx[i]] += a*x.val[i];
    
  return;
}


// sum values
double sum(const SparseVector& a){
  double sum = 0.0;
  for(size_t i = 0; i < a.nnz; i++) sum += a.val[a.idx[i]];
  return sum;
}

// sum values
double sum(const DenseVector& a){
  double sum = 0.0;
  for(size_t i = 0; i < a.dim; i++) sum += a.val[i];
  return sum;
}

// normalize so that entries sum to one 
void normalize(DenseVector& a){
  
  double sum = 0.0;
  for(size_t i = 0; i < a.dim; i++) sum += a.val[i];
  for(size_t i = 0; i < a.dim; i++) a.val[i] /= sum;
  return;
}

// normalize so that entries sum to one 
void normalize(SparseVector& a){
  
  double sum = 0.0;
  for(size_t i = 0; i < a.nnz; i++) sum += a.val[i];
  for(size_t i = 0; i < a.nnz; i++) a.val[i] /= sum;
  return;
}

double diffnorm(const DenseVector& a, const DenseVector& b){
  assert(a.dim == b.dim);
  double diff = 0.0;
  for(size_t i = 0; i < a.dim; i++)
    diff += ((a.val[i] - b.val[i])*(a.val[i] - b.val[i]));
  return diff;
}

size_t argmin(const DenseVector& a){
  double min = std::numeric_limits<double>::max();
  size_t idx = -1;
  for(size_t i = 0; i < a.dim; i++){
    if(a.val[i] < min){
      min = a.val[i];
      idx = i;
    }
  }
  return idx;
}

size_t argmax(const DenseVector& a){
  double max = -std::numeric_limits<double>::max();
  size_t idx = -1;
  for(size_t i = 0; i < a.dim; i++){
    if(a.val[i] > max){
      max = a.val[i];
      idx = i;
    }
  }
  return idx;
}

size_t abs_argmax(const DenseVector& a){
  double max = -std::numeric_limits<double>::max();
  size_t idx = -1;
  for(size_t i = 0; i < a.dim; i++){
    if(std::abs(a.val[i]) > max){
      max = std::abs(a.val[i]);
      idx = i;
    }
  }
  return idx;
}

double max(const DenseVector& a){
  double max = -std::numeric_limits<double>::max();
  for(size_t i = 0; i < a.dim; i++){
    if(a.val[i] > max) max = a.val[i];
  }
  return max;
}

double abs_max(const DenseVector& a){
  double max = 0.0; 
  for(size_t i = 0; i < a.dim; i++){
    if(std::abs(a.val[i]) > max) max = std::abs(a.val[i]);
  }
  return max;
}

double min(const DenseVector& a){
  double min = std::numeric_limits<double>::max();
  for(size_t i = 0; i < a.dim; i++){
    if(a.val[i] < min) min = a.val[i];
  }
  return min;
}

// Relative entropy with respect to the uniform distribution
double relent(const DenseVector& d){
  double ent = 0.0;
  for(size_t i = 0; i < d.dim; i ++){
    if(d.val[i] != 0.0)
      ent += (d.val[i]*log(d.val[i]*d.dim));
  }
  return ent; 
}

// Binary relative entropy with respect to the uniform distribution
// Elements restricted to 1/nu
double binary_relent(const DenseVector& d, const double& nu){
  double ent = 0.0;
  for(size_t i = 0; i < d.dim; i ++){
    if(d.val[i] != 0.0)
      ent += (d.val[i]*log(d.val[i]*d.dim));
    
    if( d.val[i] != (1/nu))
      ent += ((1/nu) - d.val[i])*log(((1/nu) - d.val[i])/((1/nu)-(1.0/d.dim)));
  }
  return ent; 
}

std::ostream& operator << (std::ostream& os, const DenseVector& d) {
  for(size_t i = 0; i < d.dim; i++){
    os << "[" << i << "] " << d.val[i];
    if( i != (d.dim -1))
      os << "  ";
  }
  os << std::endl;
  return os;
}

std::ostream& operator << (std::ostream& os, const DenseIntegerVector& d) {
  for(size_t i = 0; i < d.dim; i++){
    os << "[" << i << "] " << d.val[i];
    if( i != (d.dim -1))
      os << "  ";
  }
  os << std::endl;
  return os;
}

std::ostream& operator << (std::ostream& os, const SparseVector& s) {
  os << "(" << s.dim << ")" << " ";
  for(size_t i = 0; i < s.nnz; i++){
    os << "[" << s.idx[i] << "]  "<< s.val[i];
    if (i != (s.nnz-1))
      os << "  ";
  }
  os << std::endl;
  return os;
}

std::istream& operator >> (std::istream& in, SparseVector& s) {
  try {

    std::vector<int> idxs;
    std::vector<double> vals;
    char v;
    int idx;
    int dim;

    // get the dimension
    in >> v;
    if( v != '(')
      throw std::string("must start with (");
    in >> dim;
    in >> v;
    if( v != ')')
      throw std::string("must end with )");


    // get all of the idx,val pairs of the sparse vector
    while(in.peek()!= '\n'){
      

      in >> v;
      if( v != '[')
	throw std::string("must start with [");
      in >> idx;
      in >> v;
      if( v != ']')
	throw std::string("must end with ]");
   	  
      double wt;
      in >> wt;

      idxs.push_back(idx);
      vals.push_back(wt);

    }

    int nnz = (int)idxs.size();
    SparseVector tmp(dim,nnz);
    for(int i = 0; i < nnz; i++){
      tmp.val[i] = vals[i];
      tmp.idx[i] = idxs[i];
    }
    s = tmp; 
  }catch (std::string st) {
    std::cout << st << std::endl;
  }
  
  return in;
}


bool operator == (const SparseVector& s1, const SparseVector& s2){
  return (s1.dim == s2.dim) && 
    (s1.nnz == s2.nnz) &&
    std::equal(s1.val, s1.val+s1.nnz, s2.val) &&
    std::equal(s1.idx, s1.idx+s1.nnz, s2.idx);
}

bool operator == (const DenseVector& d1, const DenseVector& d2){
  return (d1.dim == d2.dim) && 
    std::equal(d1.val, d1.val+d1.dim, d2.val);
}

} // end of namespace totally_corrective_boosting
