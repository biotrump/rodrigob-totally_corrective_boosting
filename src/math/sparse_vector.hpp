#ifndef _SVEC_HPP_
#define _SVEC_HPP_

#include <cassert>
#include <vector>
#include <iosfwd>

namespace totally_corrective_boosting
{

/// Class to encapsulate a sparse vector
class SparseVector
{

public:

    typedef double value_type;

    // List of values
    double *val;

    // List of indices
    size_t *index;

    // Number of non zero elements
    size_t nnz;

    // Overall dimension
    size_t dim;

    // default constructor
    SparseVector():val(NULL), index(NULL), nnz(0), dim(0) { }

    // Copy constructor
    SparseVector(const SparseVector& s): nnz(s.nnz), dim(s.dim)
    {
        val = new double[nnz];
        index = new size_t[nnz];
        for(size_t i = 0; i < nnz; i++)
        {
            val[i] = s.val[i];
            index[i] = s.index[i];
        }
        return;
    }

    // Construct a sparse vector with specified dimension
    // and specified number of non zero elements
    SparseVector(const size_t& dim, const size_t& nnz):nnz(nnz), dim(dim)
    {
        val = new double[nnz];
        index = new size_t[nnz];
        for(size_t i = 0; i < nnz; i++)
        {
            val[i] = 0.0;
            index[i] = 0;
        }
        return;
    }
    void resize(const size_t& _dim, const size_t& _nnz)
    {
        if(val != NULL) delete[] val;
        val = NULL;
        if(index != NULL) delete[] index;
        index = NULL;
        dim = _dim;
        nnz = _nnz;
        val = new double[nnz];
        index = new size_t[nnz];
        for(size_t i = 0; i < nnz; i++)
        {
            val[i] = 0.0;
            index[i] = 0;
        }
        return;
    }

    ~SparseVector()
    {
        reset();
        return;
    }

    SparseVector& operator=(const SparseVector &rhs)
    {
        if(val != NULL) delete [] val;
        if(index != NULL) delete [] index;
        dim = rhs.dim;
        nnz = rhs.nnz;
        val = new double[nnz];
        index = new size_t[nnz];
        for(size_t i = 0; i < nnz; i++)
        {
            index[i] = rhs.index[i];
            val[i] = rhs.val[i];
        }
        return *this;
    }

    void reset()
    {
        if(val != NULL)
        {
            delete[] val;
        }

        val = NULL;

        if(index != NULL)
        {
            delete [] index;
        }

        index = NULL;
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
