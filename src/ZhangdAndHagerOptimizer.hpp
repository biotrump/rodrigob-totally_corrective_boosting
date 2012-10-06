
#ifndef _OPTIMIZER_HZ_HPP_
#define _OPTIMIZER_HZ_HPP_

#include "ProjectedGradientOptimizer.hpp"


namespace totally_corrective_boosting
{


// Magic parameters of the algorithm
// Do not mess!
// svnvish: BUGBUG
// All constants are from Paul J S Silva's ipg code

namespace ProjGrad_HZ{
  const double alpha_min = 1.0e-30;
  const double alpha_max = 10; // 1.0e+30;
  const double gamma = 1e-4;
  const double sigma1 = 0.1;
  const double sigma2 = 0.9;
  
  const double etadown = 0.75;
  const double etaup = 0.999;
  // assert(0.0 < etadown && etadown <= 1.0);
  // assert(0.0 < etadown && etadown <= etaup);
  const double min_step = 1e-64;
  const double decrease = 0.5;
  
  const size_t max_iter = 100000;
}


/// Implements the  Zhang and Hager projected gradient algorithm
///
/// H. Zhang and W. W. Hager
/// A Nonmonotone line search technique and its application to unconstrained optimization
/// SIAM J. Optim., Vol. 14, No. 4, pp. 1043-1056
class ZhangdAndHagerOptimizer:public ProjectedGradientOptimizer {

public:
  
  ZhangdAndHagerOptimizer(const size_t& dim,
                const bool& transposed, 
                const double& eta, 
                const double& nu,
                const double& epsilon, 
                const bool& binary);
  ~ZhangdAndHagerOptimizer(void){ }
  
  int solve(void);
  
}; 

} // end of namespace totally_corrective_boosting

# endif
