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

#ifndef _IVEC_HPP_
#define _IVEC_HPP_

#include <cassert>
#include <vector>
#include <iostream>

namespace totally_corrective_boosting
{

/** Class to encapsulate a dense vector */

// svnvish: BUGBUG
// Maybe make it a template class?

class DenseIntegerVector
{
public:
    // List of values
    size_t *val;

    // Overall dimension
    size_t dim;

    // default constructor
    DenseIntegerVector(void)
    {
        dim = 0;
        val = NULL;
    }

    // Other constructors: Syntactic sugar
    DenseIntegerVector (const size_t& dim): dim(dim)
    {
        val = new size_t[dim];
        for(size_t i = 0; i < dim; i++)
            val[i] = 0;
    }

    DenseIntegerVector (const size_t& dim, const size_t& _val): dim(dim)
    {
        val = new size_t[dim];
        for(size_t i = 0; i < dim; i++)
            val[i] = _val;
        return;
    }

    DenseIntegerVector (const size_t& dim, const size_t* _val): dim(dim)
    {
        val = new size_t[dim];
        for(size_t i = 0; i < dim; i++)
            val[i] = _val[i];
        return;
    }

    // Copy constructor
    DenseIntegerVector(const DenseIntegerVector& d): dim(d.dim)
    {
        val = new size_t[dim];
        for(size_t i = 0; i < dim; i++)
            val[i] = d.val[i];
        return;
    }

    ~DenseIntegerVector(void)
    {
        if(val != NULL) delete[] val;
        val = NULL;
        dim = 0;
        return;
    }

    // WARNING: Old data will be lost
    void resize(const size_t& _dim)
    {
        if(val != NULL) delete[] val;
        val = NULL;
        dim = _dim;
        val = new size_t[dim];
        for(size_t i = 0; i < dim; i++)
            val[i] = 0;
        return;
    }

    DenseIntegerVector& operator=(const DenseIntegerVector &rhs)
    {
        if(val != NULL) delete [] val;
        dim = rhs.dim;
        val = new size_t[dim];
        for(size_t i = 0; i < dim; i++)
            val[i] = rhs.val[i];
        return *this;
    }

    friend
    std::ostream& operator << (std::ostream& os, const DenseIntegerVector& d);

};

} // end of namespace totally_corrective_boosting

# endif
