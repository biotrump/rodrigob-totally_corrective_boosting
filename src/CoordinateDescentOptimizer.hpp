
#ifndef _OPTIMIZER_CD_HPP_
#define _OPTIMIZER_CD_HPP_

#include "AbstractOptimizer.hpp"

namespace totally_corrective_boosting
{


namespace CD{
  const size_t max_iter = 50000;
  const size_t ls_max_iter = 20;
}

/// Implement coordinate descent. Basically this is nothing but the
/// corrective algorithm which is run in a loop.
class CoordinateDescentOptimizer : public AbstractOptimizer {
  
private:
  double proj_simplex(DenseVector& dist, const double& exp_max);

  double line_search(DenseVector& W, 
                     const DenseVector& grad_w, 
                     const size_t& idx);
    
public:
  
  CoordinateDescentOptimizer(const size_t& dim,
                const bool& transposed, 
                const double& eta, 
                const double& nu,
                const double& epsilon,
                const bool& binary);
  ~CoordinateDescentOptimizer(void){ }
  
  int solve(void);
  
}; 

} // end of namespace totally_corrective_boosting

# endif
