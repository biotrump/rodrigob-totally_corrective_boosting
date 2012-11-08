
#ifndef _ENSEMBLE_HPP_
#define _ENSEMBLE_HPP_

#include "weak_learners/WeightedWeakLearner.hpp"

#include <vector>
#include <iostream>


namespace totally_corrective_boosting
{

class Ensemble
{

private:

    std::vector<WeightedWeakLearner> ensemble;

public:

    /// return false if add was successful
    /// true if weak learner already exists
    /// in that case simply add the wt to the
    /// found weak learner
    bool add(const WeightedWeakLearner& wwl);

    // predict on single examples
    double predict(const SparseVector& x) const;
    double predict(const DenseVector& x) const;

    // predict on full matrix
    // Assumes matrix is read in using
    // readlibSVM_transpose
    DenseVector   predict(const std::vector<SparseVector>& data) const;

    void set_weights(const DenseVector& wts);

    void scale_weights(const double& scale);

    void set_weight(const double& wt, const size_t& index);

    void add_weight(const double& wt, const size_t& index);

    DenseVector get_weights() const;

    // bool find_wl(const CWeakLearner* wl, size_t& index);

    size_t size() const
    {
        return ensemble.size();
    }

    friend
    std::ostream& operator << (std::ostream& os, const Ensemble& e);

    friend
    std::istream& operator >> (std::istream& in, Ensemble& e);

};

} // end of namespace totally_corrective_boosting

# endif
