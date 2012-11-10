
#include "dense_vector.hpp"

#include <iostream>

namespace totally_corrective_boosting
{

std::ostream& operator << (std::ostream& os, const DenseVector& d)
{
    for(size_t i = 0; i < d.dim; i++)
    {
        os << "[" << i << "] " << d.val[i];
        if( i != (d.dim -1))
            os << "  ";
    }
    os << std::endl;
    return os;
}


bool operator == (const DenseVector& d1, const DenseVector& d2)
{
    return (d1.dim == d2.dim) and
            std::equal(d1.val, d1.val+d1.dim, d2.val);
}

} // end of namespace totally_corrective_boosting
