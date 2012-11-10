
#include "dense_integer_vector.hpp"

#include <iostream>


namespace totally_corrective_boosting
{


std::ostream& operator << (std::ostream& os, const DenseIntegerVector& d)
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


} // end of namespace totally_corrective_boosting
