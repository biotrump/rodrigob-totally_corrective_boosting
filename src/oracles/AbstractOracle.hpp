#ifndef _ORACLE_HPP_
#define _ORACLE_HPP_


#include "math/vector_operations.hpp"

#include <vector>

namespace totally_corrective_boosting
{

class AbstractWeakLearner; // forward declaration

/// Base class to encapsulate a oracle. It essentially represents a set
/// of weak learners. Given a distribution over data it picks out the
/// weak learner with the maximum edge.
/// Every new oracle has to implement the max_edge_wl function.
class AbstractOracle
{

protected:
    std::vector<SparseVector> data;
    std::vector<int> labels;

    bool transposed;

public:
    AbstractOracle(std::vector<SparseVector>& data,
                   std::vector<int>& labels,
                   const bool& transposed = false):
        data(data), labels(labels), transposed(transposed){}
    virtual ~AbstractOracle(){}

    /// given distribution return weak learner with maximum edge
    virtual AbstractWeakLearner* find_maximum_edge_weak_learner(const DenseVector& dist) = 0;
};

} // end of namespace totally_corrective_boosting


#endif
