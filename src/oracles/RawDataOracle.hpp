#ifndef _RAWDATA_HPP_
#define _RAWDATA_HPP_

#include "AbstractOracle.hpp"

#include <vector>

namespace totally_corrective_boosting
{


class RawDataOracle: public AbstractOracle {
  
private:
  bool reflexive; /// if true then training set is [data, -data]
  
public:
  RawDataOracle(std::vector<SparseVector>& data,
           std::vector<int>& labels,
           const bool& transposed, 
           const bool& reflexive);
  ~RawDataOracle();
  
  /// given distribution return weak learner with maximum edge
  AbstractWeakLearner* find_maximum_edge_weak_learner(const DenseVector& dist);
  
};

} // end of namespace totally_corrective_boosting


#endif
