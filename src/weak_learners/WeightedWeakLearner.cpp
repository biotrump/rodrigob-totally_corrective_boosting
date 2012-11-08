#include "WeightedWeakLearner.hpp"

#include "math/vector_operations.hpp"

#include "parse.hpp"

#include <cstdlib>

namespace totally_corrective_boosting {


DenseVector WeightedWeakLearner::weighted_predict(const std::vector<SparseVector>& data) const
{
    DenseVector result = weak_learner->predict(data);
    scale(result, weight);
    return result;
}


std::ostream& operator << (std::ostream& os, const WeightedWeakLearner& wwl)
{

    os << *wwl.weak_learner;
    os << "weight: " << wwl.weight << std::endl;
    return os;

}

std::istream& operator >> (std::istream& in, WeightedWeakLearner& wwl)
{

    try
    {
        in >> *(const_cast<AbstractWeakLearner *> (wwl.weak_learner));
        expect_keyword(in,"weight:");
        in >> wwl.weight;
    }
    catch (std::string s)
    {
        std::cerr << "Error when reading weighted weak learner: " << s << std::endl;
        exit(1);
    }
    return in;
}

} // end of namespace totally_corrective_boosting
