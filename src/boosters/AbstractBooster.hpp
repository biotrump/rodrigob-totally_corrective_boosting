#ifndef _BOOSTER_HPP_
#define _BOOSTER_HPP_

#include "Ensemble.hpp"
#include "oracles/AbstractOracle.hpp"
#include "Timer.hpp"

#include <vector>
#include <iostream>

namespace totally_corrective_boosting
{

/// Base Class to encapsulate a boosting algorithm. Different
/// implementations have to implement the virtual methods in this class
/// to specify a full fledged boosting algorithm.
class AbstractBooster {

protected:
    
  /// Oracle to supply hypothesis with max edge
  AbstractOracle* oracle;

  /// Number of data points
  int num_data_points;
  
  /// Maximum number of iterations
  int max_iterations;

  /// Affects how often we dump the model to file
  int display_frequency;
  
  /// The model is just an ensemble of weak hypothesis seen so far
  Ensemble model;
  
  /// Distribution on the examples
  DenseVector examples_distribution;

  /// Keep track of time per iteration
  Timer timer;

    
  /// Update the strong classifier
  /// (add the new weak learner and update the weights of the weak classifiers)
  virtual void update_linear_ensemble(const AbstractWeakLearner& wl)=0;

  /// Update the weights of the examples distribution
  virtual void update_weights(const AbstractWeakLearner& wl)=0;

  virtual void update_stopping_criterion(const AbstractWeakLearner& wl)=0;

  /// should we stop now ?
  virtual bool stopping_criterion(std::ostream& os)=0;
  
  
public:
    
  AbstractBooster(AbstractOracle* oracle_,
           const int num_data_points_,
           const int max_iterations_);


  AbstractBooster(AbstractOracle* oracle,
           const int num_data_points,
           const int max_iterations,
           const int disp_frequency_);

  virtual ~AbstractBooster();

  /// Boost and save intermediate results
  size_t boost(std::ostream& os = std::cout);
  
  Ensemble get_ensemble()
  {
      return model;
  }

};

} // end of namespace totally_corrective_boosting


#endif


