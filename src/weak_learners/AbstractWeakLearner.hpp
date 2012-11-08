#ifndef TOTALLY_CORRECTIVE_BOOSTING_ABSTRACTWEAKLEARNER_HPP
#define TOTALLY_CORRECTIVE_BOOSTING_ABSTRACTWEAKLEARNER_HPP

#include "math/sparse_vector.hpp"
#include "math/dense_vector.hpp"

namespace totally_corrective_boosting {


/// Class to encapsulate a weak learner.
class AbstractWeakLearner{

protected:

    // Edge on the training dataset
    double edge;

    // Vector of predictions on the training dataset.
    SparseVector prediction;

public:
    AbstractWeakLearner();
    AbstractWeakLearner(const double& edge, const SparseVector& prediction);

    virtual ~AbstractWeakLearner();

    /// Predict on single example
    virtual double predict(const DenseVector& x) const = 0;
    virtual double predict(const SparseVector& x) const = 0;

    /// predict on a data matrix
    /// assumes it's read in using readlibSVM_transpose
    /// i.e. Data must be a vector of hypotheses
    virtual DenseVector predict(const std::vector<SparseVector>& Data) const = 0;

    /// functions to get around the fact that friends can't be virtual
    virtual void dump(std::ostream& os) const = 0;
    virtual void load(std::istream& in) = 0;
    virtual bool equal(const AbstractWeakLearner *wl) const = 0;
    virtual std::string get_type() const = 0;

    double get_edge(void) const { return edge; }
    SparseVector get_prediction(void) const { return prediction; }

    // ugly hack. Need to figure out how to avoid.
    virtual bool get_direction(void) const {return false; }
    virtual double get_threshold(void) const {return 0.0; }
    virtual size_t get_index(void) const {return 0;}

    friend
    bool operator == (const AbstractWeakLearner& wl1, const AbstractWeakLearner& wl2);

    friend
    std::ostream& operator << (std::ostream& os, const AbstractWeakLearner& wl);

    friend
    std::istream& operator >> (std::istream& in, AbstractWeakLearner& wl);

};


} // end of namespace totally_corrective_boosting

#endif // TOTALLY_CORRECTIVE_BOOSTING_ABSTRACTWEAKLEARNER_HPP
