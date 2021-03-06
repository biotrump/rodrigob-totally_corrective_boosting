
#include "ProjectedGradientOptimizer.hpp"

#include "math/vector_operations.hpp"

#include <limits>
#include <cmath>
#include <iostream>

namespace totally_corrective_boosting
{

ProjectedGradientOptimizer::ProjectedGradientOptimizer(const size_t& dim, 
                                                       const bool& transposed,
                                                       const double& eta,
                                                       const double& nu,
                                                       const double& epsilon,
                                                       const bool& binary):
    AbstractOptimizer(dim, transposed, eta, nu, epsilon, binary)
{
    // nothing to do here
    return;
}


ProjectedGradientOptimizer::~ProjectedGradientOptimizer()
{
    // nothing to do here
    return;
}

/// Compute solution to the Dai Fletcher projection problem
///
/// min_x 0.5*x'*x - x'*z - lambda*(a'*x - b)
/// s.t. l \leq x \leq u
///
/// @returns the optimal value in x
double phi(DenseVector& x,
           const DenseVector& a,
           const double& b,
           const DenseVector& z,
           const DenseVector& l,
           const DenseVector& u,
           const double& lambda){
    double r = -b;

    for (size_t i = 0; i < x.dim; i++){
        x.val[i] = z.val[i] + lambda*a.val[i];
        if (x.val[i] > u.val[i])
            x.val[i] = u.val[i];
        else if(x.val[i] < l.val[i])
            x.val[i] = l.val[i];
        r += a.val[i]*x.val[i];
    }
    return r;
}

// Dai-Fletcher Algorithm 1 (special case):
//
// Compute solution to the following QP
// 
// min_x  0.5*x'*x - x'*z
// s.t.   a'*x = b
//        l \leq x \leq u  
//

size_t ProjectedGradientOptimizer::project(DenseVector& x,
                                           const DenseVector& a,
                                           const double& b,
                                           const DenseVector& z,
                                           const DenseVector& l,
                                           const DenseVector& u,
                                           const size_t& max_iter){

    double r, r_l, r_u, s;
    double d_lambda = 0.5, lambda = 0.0;
    double lambda_l, lambda_u, lambda_new;

    size_t inner_iter = 1;

    // Bracketing
    r = phi(x, a, b, z, l, u, lambda);

    if (r < 0){
        lambda_l = lambda;
        r_l = r;
        lambda += d_lambda;
        r = phi(x, a, b, z, l, u, lambda);
        while(r < 0 and d_lambda < Optimizer::infinity){
            lambda_l = lambda;
            s = std::max((r_l/r) - 1.0, 0.1);
            d_lambda += (d_lambda/s);
            lambda += d_lambda;
            r_l = r;
            r = phi(x, a, b, z, l, u, lambda);
        }
        lambda_u = lambda;
        r_u = r;
    }else{
        lambda_u = lambda;
        r_u = r;
        lambda -= d_lambda;
        r = phi(x, a, b, z, l, u, lambda);
        while(r > 0 and d_lambda > -Optimizer::infinity){
            lambda_u = lambda;
            s = std::max((r_u/r) - 1.0, 0.1);
            d_lambda += (d_lambda/s);
            lambda -= d_lambda;
            r_u = r;
            r = phi(x, a, b, z, l, u, lambda);
        }
        lambda_l = lambda;
        r_l = r;
    }

    if(std::abs(d_lambda) > Optimizer::infinity) {
        std::cout << "ERROR: Detected Infeasible QP!" << std::endl;
        return -1;
    }

    if(r_u == 0){
        lambda = lambda_u;
        r = phi(x, a, b, z, l, u, lambda);
        return inner_iter;
    }

    // Secant phase

    s = 1.0 - (r_l/r_u);
    d_lambda = d_lambda/s;
    lambda = lambda_u - d_lambda;
    r = phi(x, a, b, z, l, u, lambda);

    while((std::abs(r) > DaiAndFletcher::tol_r) and
          (d_lambda > DaiAndFletcher::tol_lam * (1.0 + std::abs(lambda)))
          and inner_iter < max_iter ){

        inner_iter++;
        if(r > 0){
            if(s <= 2.0){
                lambda_u = lambda;
                r_u = r;
                s = 1.0 - r_l/r_u;
                d_lambda = (lambda_u - lambda_l)/s;
                lambda = lambda_u - d_lambda;
            }else{
                s = std::max(r_u/r - 1.0,0.1);
                d_lambda = (lambda_u - lambda)/s;
                lambda_new = std::max(lambda - d_lambda,
                                      0.75*lambda_l + 0.25*lambda);
                lambda_u = lambda;
                r_u = r;
                lambda= lambda_new;
                s = (lambda_u - lambda_l)/(lambda_u - lambda);
            }
        }else{
            if(s >= 2.0){
                lambda_l = lambda;
                r_l = r;
                s = 1.0 - r_l/r_u;
                d_lambda = (lambda_u - lambda_l)/s;
                lambda = lambda_u - d_lambda;
            }else{
                s = std::max(r_l/r - 1.0, 0.1);
                d_lambda = (lambda - lambda_l)/s;
                lambda_new = std::min(lambda + d_lambda,
                                      0.75*lambda_u + 0.25*lambda);
                lambda_l = lambda;
                r_l = r;
                lambda = lambda_new;
                s = (lambda_u - lambda_l) / (lambda_u-lambda);
            }
        }
        r = phi(x, a, b, z, l, u, lambda);
    }

    if(inner_iter >= max_iter)
        std::cout << "WARNING: DaiFletcher max iterations " << std::endl;

    return inner_iter;

}

void ProjectedGradientOptimizer::project_erlp(DenseVector& x){

    // Create vectors of zeros
    DenseVector a(x.dim, 0.0);
    // Fill the first num_wl values corresponding to w with 1
    // The rest which correspond to psi are simply 0
    for(size_t i = 0; i < num_weak_learners; i++)
        a.val[i] = 1.0;

    double b = 1.0;

    // z is simply a copy of x before projection
    DenseVector z(x);

    // Lower bound for all our variables is 0.0
    DenseVector l(x.dim, 0.0);

    // Create vector of ones
    // That is the upper bound on w
    DenseVector u(x.dim, 1.0);

    // psi have essentially no upper bound
    for(size_t i = num_weak_learners; i < x.dim; i++)
    {
        u.val[i] = Optimizer::infinity;
    }

    project(x, a, b, z, l, u, DaiAndFletcher::max_iter);

    return;
}

void ProjectedGradientOptimizer::project_binary(DenseVector& x){

    // Create vectors of ones
    DenseVector a(x.dim, 1.0);
    // Set coefficient for beta = 0
    a.val[num_weak_learners] = 0.0;
    double b = 1.0;

    // z is simply a copy of x before projection
    DenseVector z(x);

    // Lower bound for all our variables is 0.0
    DenseVector l(x.dim, 0.0);
    // Except for beta it does not matter
    l.val[num_weak_learners] = -Optimizer::infinity;

    // Create vector of ones
    // That is the upper bound on w
    DenseVector u(x.dim, 1.0);
    // Except for beta it does not matter
    u.val[num_weak_learners] = Optimizer::infinity;

    project(x, a, b, z, l, u, DaiAndFletcher::max_iter);

    return;
}

int ProjectedGradientOptimizer::solve(){

    // k-th iterate and gradient
    // Initialize with initial guess
    DenseVector xk(x);

    // k-th descent direction
    DenseVector dk(x.dim);

    // Intermediate iterate
    DenseVector xplus(x.dim);

    // store last M function values
    double* fk = new double[ProjectedGradient::M];
    for(size_t j = 0; j < ProjectedGradient::M; j++)
    {
        fk[j] = -std::numeric_limits<double>::max();
    }

    // Compute f and its gradient at x_{0}
    // Push into fk array
    double obj_max = function();

    DenseVector gradk = gradient();

    fk[0] = obj_max;

    // double min_primal = primal();

    double sksk = 0.0;
    double skyk = 0.0;

    double alpha = 1.0; // Stepsize
    double lambda; // proposed stepsize

    for(size_t i = 1; i <= ProjectedGradient::max_iter; i++){

        // Step 1: Detect if we have already converged
        if(duality_gap_met()){
            report_statistics();
            delete [] fk;
            return 0;
        }

        // Step 2: Backtracking

        // set lambda = 1
        lambda = 1.0;

        // Step 2.1: Compute d_{k} = P(x_{k} - \alpha_{k} g_{k})
        axpy(-alpha, gradk, xk, dk);
        // Project onto Simplex
        if(binary)
            project_binary(dk);
        else
            project_erlp(dk);
        // dk now contains projected values
        // subtract xk from it
        axpy(-1.0, xk, dk, dk);

        // Compute dtg = \gamma \lambda \inner{d_{k}}{g_{k}}
        double dtg = dot(dk, gradk);

        while(true){

            // Step 2.2: Set x_{+} = x_{k} + \lambda d_{k}
            axpy(lambda, dk, xk, xplus);
            copy(xplus, x);

            // Step 2.3
            // if f(x_{+}) \leq obj_max + \gamma * \inner{d_{k}}{g_{k}}}
            double objplus = function();

            if(objplus <=  (obj_max + ProjectedGradient::gamma*lambda*dtg) ){
                // Success in step 2.3
                // x_{k+1} = x_{+}
                // s_{k} = x_{k+1} - x_{k}
                // y_{k} = g_{k+1} - g_{k}

                DenseVector gradplus = gradient();

                sksk = diffnorm(x, xk);
                skyk = 0.0;
                for(size_t j = 0; j < x.dim; j++){
                    skyk += (x.val[j] - xk.val[j])*(gradplus.val[j] - gradk.val[j]);
                }
                // adjust iterate and gradient values for next time
                copy(xplus, xk);
                copy(gradplus, gradk);
                // Store function value
                fk[i%ProjectedGradient::M] = objplus;
                obj_max = fk[0];
                for(size_t j = 1; j < ProjectedGradient::M; j++){
                    if(fk[j] > obj_max) obj_max = fk[j];
                }
                if(skyk <= 0.0){
                    alpha = ProjectedGradient::alpha_max;
                } else{
                    alpha = std::min(ProjectedGradient::alpha_max, std::max(ProjectedGradient::alpha_min, sksk/skyk));
                }
                break;
            }else{
                // Failure in step 2.3
                // Compute a safeguarded new trial steplength
                double obj = fk[(i-1)%ProjectedGradient::M];
                double lambdanew = - lambda * lambda * dtg / (2*(objplus - obj -lambda*dtg));

                // svnvish: BUGBUG
                // Need to confirm if this is a typo in the paper or not
                // if((lambdanew >= ProjGrad::sigma1) and
                //    (lambdanew <= ProjGrad::sigma2*lambda)){
                //   lambda = lambdanew;
                // } else {
                //   lambda = lambda/2.0;
                // }
                lambda = std::max(ProjectedGradient::sigma1*lambda,
                                  std::min(ProjectedGradient::sigma2*lambda, lambdanew));
                continue;
            }
        }
    }

    delete [] fk;

    std::cout << "Failure in projected gradient optimizer " << std::endl;
    return 1;
}

} // end of namespace totally_corrective_boosting
