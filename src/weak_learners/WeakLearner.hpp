#ifndef _WEAKLEARNER_HPP_
#define _WEAKLEARNER_HPP_

#include "vec.hpp"

namespace totally_corrective_boosting
{

/// Class to encapsulate a weak learner. For now our weak learner is a linear predictor.
// FIXME should separate the abstract weak learned from the linear predictor implementation
class WeakLearner{

protected:
    // svnvish: BUGBUG
    // Perhaps these should be const members?

    // The weak learner predicts with <wt, x>
    SparseVector wt;

    // Edge on the training dataset
    double edge;

    // Vector of predictions on the training dataset.
    SparseVector prediction;

public:

    WeakLearner();

    WeakLearner(const SparseVector& wt, const double& edge, const SparseVector& prediction);

    WeakLearner(const WeakLearner& wl);

    virtual ~WeakLearner(){ }

    // Predict on single example
    virtual double predict(const DenseVector& x) const;
    virtual double predict(const SparseVector& x) const;
    // predict on a data matrix
    // assumes it's read in using readlibSVM_transpose
    // i.e. Data must be a vector of hypotheses
    virtual DenseVector   predict(const std::vector<SparseVector>& Data) const;

    // functions to get around the fact that friends can't be virtual
    virtual void dump(std::ostream& os) const;
    virtual void load(std::istream& in);
    virtual bool equal(const WeakLearner *wl) const;
    virtual std::string get_type() const;

    double get_edge(void) const { return edge; }
    SparseVector get_wt(void) const { return wt;}
    SparseVector get_prediction(void) const { return prediction; }

    // ugly hack. Need to figure out how to avoid.
    virtual bool get_direction(void) const {return false; }
    virtual double get_thresh(void) const {return 0.0; }
    virtual size_t get_idx(void) const {return 0;}

    friend
    bool operator == (const WeakLearner& wl1, const WeakLearner& wl2);

    friend
    std::ostream& operator << (std::ostream& os, const WeakLearner& wl);

    friend
    std::istream& operator >> (std::istream& in, WeakLearner& wl);

};

} // end of namespace totally_corrective_boosting


#endif
