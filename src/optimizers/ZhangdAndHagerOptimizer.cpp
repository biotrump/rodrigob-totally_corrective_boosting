
#include "ZhangdAndHagerOptimizer.hpp"

#include "math/vector_operations.hpp"

#include <limits>
#include <cmath>
#include <iostream>


namespace totally_corrective_boosting
{


ZhangdAndHagerOptimizer::ZhangdAndHagerOptimizer(const size_t& dim, 
                                                 const bool& transposed,
                                                 const double& eta,
                                                 const double& nu,
                                                 const double& epsilon,
                                                 const bool& binary):
    ProjectedGradientOptimizer(dim, transposed, eta, nu, epsilon, binary){ }

int ZhangdAndHagerOptimizer::solve(){

    // k-th iterate and gradient
    // Initialize with initial guess
    DenseVector xk(x);

    // k-th descent direction
    DenseVector dk(x.dim);

    // Intermediate iterate
    DenseVector xplus(x.dim);

    double etak = ProjGrad_HZ::etaup;
    double qk = 0.0;
    double ck = 0.0;

    // Compute f and its gradient at x_{0}
    double obj = function();

    DenseVector gradk = gradient();

    // double min_primal = primal();

    double sksk = 0.0;
    double skyk = 0.0;

    double alpha = 1.0;
    double lambda; // proposed stepsize

    for(size_t i = 1; i <= ProjGrad_HZ::max_iter; i++)
    {

        // Step 1: Detect if we have already converged
        if(duality_gap_met())
        {
            report_statistics();
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

        // Compute dtg = \inner{d_{k}}{g_{k}}
        double dtg = dot(dk, gradk);

        const double newqk = etak*qk + 1.0;
        ck = (etak*qk*ck + obj)/newqk;
        qk = newqk;
        
        while(true)
        {

            // Step 2.2: Set x_{+} = x_{k} + \lambda d_{k}
            axpy(lambda, dk, xk, xplus);
            copy(xplus, x);

            // Step 2.3
            // if f(x_{+}) \leq obj_max + \gamma * \inner{d_{k}}{g_{k}}}
            double objplus = function();

            if(objplus <=  (ck + ProjGrad_HZ::gamma*lambda*dtg) )
            {
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
                obj = objplus;
                if(skyk <= 0.0)
                    alpha = ProjGrad_HZ::alpha_max;
                else
                    alpha = std::min(ProjGrad_HZ::alpha_max,
                                     std::max(ProjGrad_HZ::alpha_min, sksk/skyk));
                break;
            }
            else
            {
                // Failure in step 2.3
                // Compute a safeguarded new trial steplength
                double lambdanew = - lambda * lambda * dtg / (2*(objplus - obj -lambda*dtg));
                lambda = std::max(ProjGrad_HZ::sigma1*lambda,
                                  std::min(ProjGrad_HZ::sigma2*lambda, lambdanew));
                continue;
            }

        } // end of "while true"
    } // end of "for each iteration"
    std::cout << "Failure in HZ optimizer " << std::endl;
    return 1;
}

} // end of namespace totally_corrective_boosting
