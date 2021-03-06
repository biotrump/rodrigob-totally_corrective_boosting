
#ifndef _DVEC_HPP_
#define _DVEC_HPP_

#include <cassert>
#include <vector>
#include <iosfwd>

namespace totally_corrective_boosting
{

/// Dense vector
class DenseVector
{
public:
    /// List of values
    double *val;

    /// Overall dimension
    size_t dim;

    /// default constructor
    DenseVector()
    {
        dim = 0;
        val = NULL;
        return;
    }


    /// Other constructors: Syntactic sugar
    DenseVector (const size_t& dim): dim(dim)
    {
        val = new double[dim];
        for(size_t i = 0; i < dim; i++)
        {
            val[i] = 0.0;
        }
        return;
    }


    DenseVector (const size_t& dim, const double& _val): dim(dim)
    {
        val = new double[dim];
        for(size_t i = 0; i < dim; i++)
            val[i] = _val;
        return;
    }


    DenseVector (const size_t& dim, const double* _val): dim(dim)
    {
        val = new double[dim];
        for(size_t i = 0; i < dim; i++)
            val[i] = _val[i];
        return;
    }


    /// Copy constructor
    DenseVector(const DenseVector& d): dim(d.dim)
    {
        val = new double[dim];
        for(size_t i = 0; i < dim; i++)
            val[i] = d.val[i];
        return;
    }


    ~DenseVector()
    {
        if(val != NULL)
        {
            delete[] val;
        }
        val = NULL;
        dim = 0;
        return;
    }


    /// WARNING: Old data will be lost
    void resize(const size_t& _dim)
    {
        if(val != NULL)
        {
            delete[] val;
        }

        val = NULL;
        dim = _dim;
        val = new double[dim];
        for(size_t i = 0; i < dim; i++)
        {
            val[i] = 0.0;
        }
        return;
    }


    DenseVector& operator=(const DenseVector &rhs)
    {
        if(val != NULL) delete [] val;
        dim = rhs.dim;
        val = new double[dim];
        for(size_t i = 0; i < dim; i++)
        {
            val[i] = rhs.val[i];
        }
        return *this;
    }


    /// clear contents of current vector
    void clear()
    {
        if(val != NULL)
        {
            delete[] val;
        }

        val = NULL;
        dim = 0;
        return;
    }

    friend
    std::ostream& operator << (std::ostream& os, const DenseVector& d);

    friend
    bool operator == (const DenseVector& s1, const DenseVector& s2);

};

} // end of namespace totally_corrective_boosting

# endif
