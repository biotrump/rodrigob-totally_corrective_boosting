
#ifndef _OPTIMIZER_LBFGSB_HPP_
#define _OPTIMIZER_LBFGSB_HPP_

#include "AbstractOptimizer.hpp"
#include "lbfgsb/lbfgsb.h"

namespace totally_corrective_boosting
{


namespace LBFGSB{
  const size_t max_iter = 10000;
  const size_t lbfgsb_max_iter = 1000;
  const size_t lbfgsb_m = 5; // Past gradients stored in lbfgsb
}

/// Augmented Lagrangian solver in the w domain.
class LbfgsbOptimizer : public AbstractOptimizer {
  
private:
  // Dual variable value we are adjusting
  double lambda;    
  
  // Regularizer for the Lagrangian
  double mu;        
  
public:

  LbfgsbOptimizer(const size_t& dim,
                    const bool& transposed, 
                    const double& eta, 
                    const double& nu,
                    const double& epsilon, 
                    const bool& binary);
  ~LbfgsbOptimizer();
  
  int solve();
  
  void bounds(ap::integer_1d_array& nbd,
              ap::real_1d_array& l,
              ap::real_1d_array& u);
  
  void bounds_binary(ap::integer_1d_array& nbd,
                     ap::real_1d_array& l,
                     ap::real_1d_array& u);
  
  /// Take as input an array to return the gradient
  /// @return as output the objective value of augmented
  /// lagrangian
  double augmented_lagrangian_and_function_gradient(const ap::real_1d_array& x0,
                                                    ap::real_1d_array& g);
  
}; 

} // end of namespace totally_corrective_boosting

# endif
