
#include "sparse_vector.hpp"

#include <iostream>


namespace totally_corrective_boosting
{


std::ostream& operator << (std::ostream& os, const SparseVector& s)
{
    os << "(" << s.dim << ")" << " ";
    for(size_t i = 0; i < s.nnz; i++)
    {
        os << "[" << s.index[i] << "]  "<< s.val[i];
        if (i != (s.nnz-1))
            os << "  ";
    }
    os << std::endl;
    return os;
}

std::istream& operator >> (std::istream& in, SparseVector& s)
{
    try
    {

        std::vector<int> indices;
        std::vector<double> vals;
        char v;
        int index;
        int dim;

        // get the dimension
        in >> v;
        if( v != '(')
            throw std::string("must start with (");
        in >> dim;
        in >> v;
        if( v != ')')
            throw std::string("must end with )");


        // get all of the index,val pairs of the sparse vector
        while(in.peek()!= '\n')
        {


            in >> v;
            if( v != '[')
                throw std::string("must start with [");
            in >> index;
            in >> v;
            if( v != ']')
                throw std::string("must end with ]");

            double wt;
            in >> wt;

            indices.push_back(index);
            vals.push_back(wt);

        }

        int nnz = (int)indices.size();
        SparseVector tmp(dim,nnz);
        for(int i = 0; i < nnz; i++)
        {
            tmp.val[i] = vals[i];
            tmp.index[i] = indices[i];
        }
        s = tmp;
    }
    catch (std::string st)
    {
        std::cout << st << std::endl;
    }

    return in;
}


bool operator == (const SparseVector& s1, const SparseVector& s2)
{
    return (s1.dim == s2.dim) and
            (s1.nnz == s2.nnz) and
            std::equal(s1.val, s1.val+s1.nnz, s2.val) and
            std::equal(s1.index, s1.index+s1.nnz, s2.index);
}


} // end of namespace totally_corrective_boosting
