
#ifndef _LIBSVMREADER_HPP_
#define _LIBSVMREADER_HPP_

#include "svec.hpp"

#include <string>
#include <vector>

namespace totally_corrective_boosting
{


class LibSVMReader{
  
private:
  bool is_blank(const std::string& line);
  
public:
  int readlibSVM(const std::string& filename, 
                 std::vector<SparseVector>& data,
                 std::vector<int>& labels);
  
  // Read data and store it as a vector
  // Each element of a the vector is a feature
  // The indices represent the data point # which contains that feature 
  // This is the transpose of the normal representation 
  // where each data point is a svec 
  int readlibSVM_transpose(const std::string& filename, 
                           std::vector<SparseVector>& data,
                           std::vector<int>& labels);

  int readlibSVM_transpose_fast(const std::string& filename, 
                           std::vector<SparseVector>& data,
                           std::vector<int>& labels);
  
};

typedef std::vector<double>::iterator dbl_itr;
typedef std::vector<size_t>::iterator uint_itr;
typedef std::vector<SparseVector>::iterator svec_itr;

} // end of namespace totally_corrective_boosting

# endif
