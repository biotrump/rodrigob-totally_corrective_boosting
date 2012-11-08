
#ifndef _OPTIMIZER_PG_HPP_
#define _OPTIMIZER_PG_HPP_

#include "AbstractOptimizer.hpp"


namespace totally_corrective_boosting
{

// Implement the projected gradient algorithm in w domain

/* Credits:
 *
 * Dai-Fletcher Projected Gradient method for SVM [1].
 * Modified from the GPDT software [2] for inequality constraints.
 * Projected gradient method [3] for simplex constraints.
 *
 *
 * References:
 *
 *   [1] Y. H. Dai and R. Fletcher,
 *       New algorithms for singly linearly constrained quadratic programs
 *       subject to lower and upper bounds,
 *       Math. Program., 2006.
 *
 *   [2] L. Zanni, T. Serafini, and G. Zanghirati,
 *       Parallel Software for Training Large Scale Support Vector Machines
 *       on Multiprocessor Systems,
 *       JMLR 7, 2006.
 *
 *   [3] E. G. Birgin, J. M. Martinez, and M. Raydan
 *       Nonmonotone Spectral Projected Gradient Methods
 *       On Convex Sets,
 *       SIAM J. Optim., Vol. 10, No. 4, pp. 1196-1211
 *
 */



namespace DaiAndFletcher{
const double tol_r = 1e-16;
const double tol_lam = 1e-15;
const size_t max_iter = 10000;
}

// Magic parameters of the Projected Gradient algorithm
// Do not mess!
// svnvish: BUGBUG
// All constants are arbitrary

namespace ProjectedGradient{
const double alpha_min = 1e-30;
const double alpha_max = 10;
const double gamma = 1e-4;
const double sigma1 = 0.1;
const double sigma2 = 0.9;
const size_t M = 10;
const size_t max_iter = 10000;
}

double phi(DenseVector& x,
           const DenseVector& a,
           const double& b,
           const DenseVector& z,
           const DenseVector& l,
           const DenseVector& u,
           const double& lambda);

class ProjectedGradientOptimizer : public AbstractOptimizer {

private:

    size_t project(DenseVector& x,
                   const DenseVector& a,
                   const double& b,
                   const DenseVector& z,
                   const DenseVector& l,
                   const DenseVector& u,
                   const size_t& max_iter);

protected:
    void project_erlp(DenseVector& z);

    void project_binary(DenseVector& z);

public:

    ProjectedGradientOptimizer(const size_t& dim,
                               const bool& transposed,
                               const double& eta,
                               const double& nu,
                               const double& epsilon,
                               const bool& binary);
    ~ProjectedGradientOptimizer();

    int solve();
}; 

} // end of namespace totally_corrective_boosting

# endif
