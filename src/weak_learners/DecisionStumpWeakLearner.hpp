#ifndef _WEAKLEARNERDSTUMP_HPP_
#define _WEAKLEARNERDSTUMP_HPP_

#include "LinearWeakLearner.hpp"

#include "math/vector_operations.hpp"

namespace totally_corrective_boosting
{

class DecisionStumpWeakLearner: public LinearWeakLearner
{

private:

    // threshold
    double threshold;

    // direction of threshold:
    // if dir == true, then x >= thresh
    // else x <= thresh
    bool direction;

    // index of hypothesis for best decision stump
    size_t index;

public:

    DecisionStumpWeakLearner();

    DecisionStumpWeakLearner(const SparseVector& wt, const double& edge, const SparseVector& prediction,
                             const double& threshold, const bool& direction, const int& index);

    /// Copy constructor
    DecisionStumpWeakLearner(const DecisionStumpWeakLearner& other);

    ~DecisionStumpWeakLearner();

    std::string get_type() const;

    // Predict on examples
    double predict(const DenseVector& x) const;
    double predict(const SparseVector& x) const;
    // predict on a data matrix
    // assumes it's read in using readlibSVM_transpose
    // i.e. Data must be a vector of hypotheses
    DenseVector predict(const std::vector<SparseVector>& Data) const;

    // methods to dump and load data
    void dump(std::ostream& os) const;
    void load(std::istream& in);
    bool equal(const AbstractWeakLearner *wl) const;

    // accessor methods
    bool get_direction() const {return direction; }
    double get_threshold() const {return threshold; }
    size_t get_index() const {return index;}

    friend
    std::ostream& operator << (std::ostream& os, const DecisionStumpWeakLearner& wl);

    friend
    std::istream& operator >> (std::istream& in, DecisionStumpWeakLearner& wl);

    friend
    bool operator == (const DecisionStumpWeakLearner& wl1, const DecisionStumpWeakLearner& wl2);

};

} // end of namespace totally_corrective_boosting

#endif
