#ifndef TOTALLY_CORRECTIVE_BOOSTING_WEIGHTEDWEAKLEARNER_HPP
#define TOTALLY_CORRECTIVE_BOOSTING_WEIGHTEDWEAKLEARNER_HPP

#include "AbstractWeakLearner.hpp"

namespace totally_corrective_boosting {


class WeightedWeakLearner
{
protected:
    /// Weak Learner
    const AbstractWeakLearner* weak_learner;

    /// Weight associated with the wl
    double weight;

public:
    WeightedWeakLearner(const AbstractWeakLearner* _wl, const double& _wt):weak_learner(_wl), weight(_wt) {}
    WeightedWeakLearner(const WeightedWeakLearner& wwl):weak_learner(wwl.weak_learner), weight(wwl.weight) {}
    ~WeightedWeakLearner() {}

    void set_weight(double _wt)
    {
        weight = _wt;
    }

    double get_weight(void) const
    {
        return weight;
    }

    void scale_weight(double scale)
    {
        weight *= scale;
    }

    void add_weight(double alpha)
    {
        weight += alpha;
    }

    std::string get_type() const
    {
        return weak_learner->get_type();
    }

    double weighted_predict(const DenseVector& x) const
    {
        return weight*(weak_learner->predict(x));
    }

    double weighted_predict(const SparseVector& x) const
    {
        return weight*(weak_learner->predict(x));
    }

    DenseVector weighted_predict(const std::vector<SparseVector>& data) const;

    friend
    bool operator == (const WeightedWeakLearner& w1,
                      const WeightedWeakLearner& w2)
    {
        return (*(w1.weak_learner) == *(w2.weak_learner));
    }


    // bool wl_equal(const AbstractWeakLearner* _wl){ return (*wl == *_wl); };

    friend
    std::ostream& operator << (std::ostream& os, const WeightedWeakLearner& wwl);

    friend
    std::istream& operator >> (std::istream& in, WeightedWeakLearner& wwl);

};


} // end of namespace totally_corrective_boosting

#endif // TOTALLY_CORRECTIVE_BOOSTING_WEIGHTEDWEAKLEARNER_HPP
