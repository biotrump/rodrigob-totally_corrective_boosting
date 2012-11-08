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

#ifndef _SVEC_HPP_
#define _SVEC_HPP_

#include <cassert>
#include <vector>
#include <iostream>

namespace totally_corrective_boosting
{

/// Class to encapsulate a sparse vector
class SparseVector
{

public:
    // List of values
    double *val;

    // List of indices
    size_t *idx;

    // Number of non zero elements
    size_t nnz;

    // Overall dimension
    size_t dim;

    // default constructor
    SparseVector(void):val(NULL), idx(NULL), nnz(0), dim(0) { }

    // Copy constructor
    SparseVector(const SparseVector& s): nnz(s.nnz), dim(s.dim)
    {
        val = new double[nnz];
        idx = new size_t[nnz];
        for(size_t i = 0; i < nnz; i++)
        {
            val[i] = s.val[i];
            idx[i] = s.idx[i];
        }
        return;
    }

    // Construct a sparse vector with specified dimension
    // and specified number of non zero elements
    SparseVector(const size_t& dim, const size_t& nnz):nnz(nnz), dim(dim)
    {
        val = new double[nnz];
        idx = new size_t[nnz];
        for(size_t i = 0; i < nnz; i++)
        {
            val[i] = 0.0;
            idx[i] = 0;
        }
        return;
    }
    void resize(const size_t& _dim, const size_t& _nnz)
    {
        if(val != NULL) delete[] val;
        val = NULL;
        if(idx != NULL) delete[] idx;
        idx = NULL;
        dim = _dim;
        nnz = _nnz;
        val = new double[nnz];
        idx = new size_t[nnz];
        for(size_t i = 0; i < nnz; i++)
        {
            val[i] = 0.0;
            idx[i] = 0;
        }
        return;
    }

    ~SparseVector(void)
    {
        reset();
        return;
    }

    SparseVector& operator=(const SparseVector &rhs)
    {
        if(val != NULL) delete [] val;
        if(idx != NULL) delete [] idx;
        dim = rhs.dim;
        nnz = rhs.nnz;
        val = new double[nnz];
        idx = new size_t[nnz];
        for(size_t i = 0; i < nnz; i++)
        {
            idx[i] = rhs.idx[i];
            val[i] = rhs.val[i];
        }
        return *this;
    }

    void reset(void)
    {
        if(val != NULL) delete[] val;
        val = NULL;
        if(idx != NULL) delete [] idx;
        idx = NULL;
        nnz = 0;
        dim = 0;
        return;
    }

    friend
    std::ostream& operator << (std::ostream& os, const SparseVector& s);
    friend
    std::istream& operator >> (std::istream& in, SparseVector& s);
    friend
    bool operator == (const SparseVector& s1, const SparseVector& s2);

};

} // end of namespace totally_corrective_boosting

# endif
