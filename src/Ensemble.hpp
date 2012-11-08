
#ifndef _ENSEMBLE_HPP_
#define _ENSEMBLE_HPP_

#include "weak_learners/AbstractWeakLearner.hpp"

#include "vector_operations.hpp"

#include <vector>
#include <iostream>


namespace totally_corrective_boosting
{

class Ensemble;

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

    DenseVector weighted_predict(const std::vector<SparseVector>& data) const
    {
        DenseVector result = weak_learner->predict(data);
        scale(result, weight);
        return result;
    }

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

typedef std::vector<WeightedWeakLearner>::const_iterator wwl_citr;
typedef std::vector<WeightedWeakLearner>::iterator wwl_itr;

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

    void set_weight(const double& wt, const size_t& idx);

    void add_weight(const double& wt, const size_t& idx);

    DenseVector get_weights(void) const;

    // bool find_wl(const CWeakLearner* wl, size_t& idx);

    size_t size(void) const
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
