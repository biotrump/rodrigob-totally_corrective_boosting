#ifndef _SVM_HPP_
#define _SVM_HPP_

// SVM oracle. Hypothesis are the data themselves. 
// Prediction of an oracle x on data x' is simply 
// <x, x'>

#include "AbstractOracle.hpp"

#include <vector>

namespace totally_corrective_boosting
{


class Svm: public AbstractOracle {

private:
    /// if true then training set is [data, -data]
    bool reflexive;

public:
    Svm(const std::vector<SparseVector>& data,
        const std::vector<int>& labels,
        const bool transposed,
        const bool reflexive);
    ~Svm();

    /// given distribution return weak learner with maximum edge
    AbstractWeakLearner* find_maximum_edge_weak_learner(const DenseVector& dist);

};

} // end of namespace totally_corrective_boosting

#endif
