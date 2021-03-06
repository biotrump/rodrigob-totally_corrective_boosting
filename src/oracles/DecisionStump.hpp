
#ifndef _DECISIONSTUMP_HPP_
#define _DECISIONSTUMP_HPP_

#include "AbstractOracle.hpp"

#include "Timer.hpp"

#include "math/dense_vector.hpp"
#include "math/sparse_vector.hpp"
#include "math/dense_integer_vector.hpp"

#include <vector>
#include <iostream>

namespace totally_corrective_boosting
{


/// Decision stump oracle
class DecisionStump: public AbstractOracle
{

protected:

  // if less_than == true, the we consider x <= thresh
  // as well as x >= thresh
  // sort of correponds to reflexive
  const bool less_than;

  std::vector<DenseIntegerVector> sorted_data;
  
  // Keep track of time spent in max_edge_wl
  Timer timer;
  
public:
  DecisionStump(const std::vector<SparseVector>& data,
                 const std::vector<int>& labels,
                 const bool less_than);

  ~DecisionStump();

  /// given distribution return weak learner with maximum edge
  AbstractWeakLearner* find_maximum_edge_weak_learner(const DenseVector& dist);

  /// given a hypothesis and distribution, return the best threshold
  /// the best edge, and the direction of the best threshold
  /// if ge==true, then x >= thresh else x <= thresh
  void find_best_threshold(const size_t& index,
                const DenseVector& dist, 
                const double& init_edge,
                double& best_threshold,
                double& best_edge, 
                bool& ge) const;
  
  /// given a sorted vector of (hyp,label,dist) triplets,
  /// return the best threshold and edge for hyp <= thresh
  void find_best_threshold_less_or_equal(const size_t& index,
                   const double& dist_diff, 
                   const DenseVector& dist, 
                   const double& init_edge,
                   double& best_threshold, 
                   double& best_edge) const;

  // given a sorted vector of (hyp,label,dist) triplets,
  // return the best threshold and edge for hyp >= thresh
  void find_best_threshold_greater_or_equal(const size_t& index,
                   const double& dist_diff, 
                   const DenseVector& dist, 
                   const double& init_edge,
                   double& best_threshold, 
                   double& best_edge) const;
  
  DenseIntegerVector argsort(SparseVector unsorted);
  
};


} // end of namespace totally_corrective_boosting



#endif
