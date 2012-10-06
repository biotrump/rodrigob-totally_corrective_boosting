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
 * Last Updated: (23/04/2008)   
 */

#ifndef _LIBSVMREADER_HPP_
#define _LIBSVMREADER_HPP_

#include <string>
#include <vector>
#include "svec.hpp"

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
