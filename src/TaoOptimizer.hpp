
#ifndef _OPTIMIZER_TAO_HPP_
#define _OPTIMIZER_TAO_HPP_

#if not defined(USE_TAO)
#error This file should only be included when the TAO library is used.
#endif

#include "AbstractOptimizer.hpp"

#include <tao.h>
#include <petscvec.h>


namespace totally_corrective_boosting
{


namespace TAO{
  const size_t max_iter = 10000;
}

/// Augmented Lagrangian solver in the w domain.
class TaoOptimizer : public AbstractOptimizer {
  
private:
  // Dual variable value we are adjusting
  double lambda;    
  
  // Regularizer for the Lagrangian
  double mu;        
  
public:

  TaoOptimizer(const size_t& dim,
                 const bool& transposed, 
                 const double& eta, 
                 const double& nu,
                 const double& epsilon, 
                 const bool& binary,
                 int& argc, 
                 char** argv);
  
  ~TaoOptimizer(void);
  
  int solve(void);
  
  int bounds(Vec& xl, Vec& xu);
  
  int bounds_binary(Vec& xl, Vec& xu);
  
  // Take as input an array to return the gradient
  // Return as output the objective value of augmented
  // lagrangian 
  double aug_lag_fg(const Vec& X, Vec& G);
  
}; 

int tao_fun_grad(TAO_APPLICATION taoapp,
                 Vec X,
                 double *obj,
                 Vec G,
                 void *ctx);

int tao_bounds(TAO_APPLICATION taoapp, 
               Vec xl, 
               Vec xu, 
               void *ctx);

int tao_bounds_binary(TAO_APPLICATION taoapp, 
                      Vec xl, 
                      Vec xu, 
                      void *ctx);

} // end of namespace totally_corrective_boosting

# endif
