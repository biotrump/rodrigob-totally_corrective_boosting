
#include "AbstractOracle.hpp"

namespace totally_corrective_boosting
{

AbstractOracle::AbstractOracle(const std::vector<SparseVector>& data,
                               const std::vector<int>& labels,
                               const bool transposed)
    : data(data), labels(labels), transposed(transposed)
{
    // nothing to do here
    return;
}


AbstractOracle::~AbstractOracle()
{
    // nothing to do here
    return;
}


} // end of namespace totally_corrective_boosting
