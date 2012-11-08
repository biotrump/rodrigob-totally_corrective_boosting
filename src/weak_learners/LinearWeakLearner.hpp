#ifndef TOTALLY_CORRECTIVE_BOOSTING_LinearWeakLearner_HPP
#define TOTALLY_CORRECTIVE_BOOSTING_LinearWeakLearner_HPP

#include "AbstractWeakLearner.hpp"

namespace totally_corrective_boosting {


/// Linear predictor.
class LinearWeakLearner: public AbstractWeakLearner
{

protected:
    // svnvish: BUGBUG
    // Perhaps these should be const members?

    // The weak learner predicts with <wt, x>
    SparseVector wt;

public:

    LinearWeakLearner();

    LinearWeakLearner(const SparseVector& wt, const double& edge, const SparseVector& prediction);

    LinearWeakLearner(const LinearWeakLearner& wl);

    ~LinearWeakLearner();

    // Predict on single example
    double predict(const DenseVector& x) const;
    double predict(const SparseVector& x) const;
    // predict on a data matrix
    // assumes it's read in using readlibSVM_transpose
    // i.e. Data must be a vector of hypotheses
    DenseVector predict(const std::vector<SparseVector>& Data) const;

    // functions to get around the fact that friends can't be virtual
    void dump(std::ostream& os) const;
    void load(std::istream& in);
    bool equal(const AbstractWeakLearner *other_p) const;
    std::string get_type() const;

    SparseVector get_wt() const { return wt;}

    // ugly hack. Need to figure out how to avoid.
    bool get_direction() const {return false; }
    double get_threshold() const {return 0.0; }
    size_t get_index() const {return 0;}

};


} // namespace totally_corrective_boosting

#endif // TOTALLY_CORRECTIVE_BOOSTING_LinearWeakLearner_HPP
