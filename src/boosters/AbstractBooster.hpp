#ifndef _BOOSTER_HPP_
#define _BOOSTER_HPP_

#include "Ensemble.hpp"
#include "oracles/AbstractOracle.hpp"
#include "Timer.hpp"

#include <boost/shared_ptr.hpp>

#include <vector>
#include <iostream>

namespace totally_corrective_boosting
{

/// Base Class to encapsulate a boosting algorithm. Different
/// implementations have to implement the virtual methods in this class
/// to specify a full fledged boosting algorithm.
class AbstractBooster
{

protected:
    
  /// Oracle to supply hypothesis with max edge
  boost::shared_ptr<AbstractOracle> oracle;

  /// Number of data points
  const int num_data_points;
  
  /// Maximum number of iterations
  const int max_iterations;

  /// Affects how often we dump the model to file
  const int display_frequency;
  
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
  virtual void update_examples_distribution(const AbstractWeakLearner& wl)=0;

  virtual void update_stopping_criterion(const AbstractWeakLearner& wl)=0;

  /// should we stop now ?
  virtual bool stopping_criterion(std::ostream& os)=0;
  
  
public:
    
  AbstractBooster(const boost::shared_ptr<AbstractOracle> &oracle_,
           const int num_data_points_,
           const int max_iterations_);


  AbstractBooster(const boost::shared_ptr<AbstractOracle> &oracle_,
           const int num_data_points,
           const int max_iterations,
           const int display_frequency_);

  virtual ~AbstractBooster();

  /// Boost and save intermediate results
  /// This is the main loop of the training,
  /// this function may take some time to finish...
  size_t boost(std::ostream& log_stream = std::cout);
  
  const Ensemble &get_ensemble() const;


  /// Helper function for debugging and external usage
  const DenseVector&  get_examples_distribution() const;

};

} // end of namespace totally_corrective_boosting


#endif


