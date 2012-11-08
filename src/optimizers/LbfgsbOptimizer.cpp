
#include "LbfgsbOptimizer.hpp"

#include "math/vector_operations.hpp"

#include <limits>
#include <iostream>
#include <cmath>
#include <algorithm>


namespace totally_corrective_boosting
{

LbfgsbOptimizer::LbfgsbOptimizer(const size_t dim, 
                                 const bool transposed,
                                 const double eta,
                                 const double nu,
                                 const double epsilon,
                                 const bool binary)
    : AbstractOptimizer(dim, transposed, eta, nu, epsilon, binary),
      lambda(0.0), mu(1.0)
{
    // nothing to do here
    return;
}


LbfgsbOptimizer::~LbfgsbOptimizer()
{
    // nothing to do here
    return;
}


int LbfgsbOptimizer::solve()
{

    // Solution vector
    ap::real_1d_array x0;
    x0.setbounds(1, x.dim);

    // copy current x into x0
    for(size_t i = 0; i < x.dim; i++)
        x0(i+1) = x.val[i];

    ap::integer_1d_array nbd;
    ap::real_1d_array l;
    ap::real_1d_array u;
    nbd.setbounds(1, x.dim);
    l.setbounds(1, x.dim);
    u.setbounds(1, x.dim);

    // Set bounds
    if(binary)
    {
        bounds_binary(nbd, l, u);
    }
    else
    {
        bounds(nbd, l, u);
    }

    // Lancelot: gamma_bar = 0.1
    double gamma_bar = 0.1; // < 1 :

    // Lancelot: tau = 0.1
    double tau = 0.1; // # < 1

    // Lancelot: alpha_eta = 0.1
    double alpha_eta = 0.01; // # min(1, self._alpha_omega)

    //Lancelot: beta_eta = 0.9
    double beta_eta = 0.9; // # min(1, self._beta_omega)

    //Lancelot: alpha_omega = 1.0
    double alpha_omega = 4.0;

    //Lancelot: beta_omega = 1.0
    double beta_omega = 4.0;

    // Lancelot: omega_bar = firtsg/pow(std::min(mu, gamma_bar), alpha_omega);
    double omega_bar = 0.5;

    // Lancelot: eta_bar = firtsc/pow(std::min(mu, gamma_bar), alpha_eta);
    double eta_bar = 1.0;

    double mu_bar = 0.9; // must be <= 1

    mu = mu_bar;

    // Lancelot: alpha  = min(mu, gamma_bar)
    double alpha = std::min(mu, gamma_bar);

    // Lancelot: eta = max(etamin, eta_bar*pow(alpha, alpha_eta))
    // svnvish: BUGBUG what is etamin?
    // etamin is the minimum norm of the constraint violation
    double eta = eta_bar*pow(alpha, alpha_eta);

    // Lancelot: omega = max(omemin, omega_bar*pow(alpha, alpha_omega))
    // svnvish: BUGBUG what is omemin?
    // omemin is the tolerance for kkt gap
    double omega = omega_bar*pow(alpha, alpha_omega);

    DenseVector W;
    W.val = x.val;
    W.dim = num_weak_learners;

    for(size_t iteration = 0; iteration < LBFGSB::max_iter; iteration++)
    {
        double epsg = omega;
        double epsf = 0;
        double epsx = 0;
        int info;

        lbfgsbminimize(x.dim,
                       std::min(x.dim, LBFGSB::lbfgsb_m),
                       x0,
                       epsg,
                       epsf,
                       epsx,
                       LBFGSB::lbfgsb_max_iter,
                       nbd,
                       l,
                       u,
                       (void*) this,
                       info);

        // copy current solution into x
        for(size_t i = 0; i < x.dim; i++)
            x.val[i] = x0(i+1);

        double w_gap = sum(W) - 1.0;

        // std::cout << "info: " << info << std::endl;
        // assert(info > 0);
        assert(info == 4);

        if(std::abs(w_gap) < eta)
        {

            if(std::abs(w_gap) < Optimizer::wt_sum_tol)
            {

                if(duality_gap_met())
                {
                    // // svnvish: BUGBUG
                    // // Extra gradient computation happening here
                    // // Better to use norm gaps
                    // dvec gradk = grad();

                    // if(converged(grad())){
                    report_statistics();
                    break;
                }
            }

            lambda = lambda - (w_gap/mu);
            //mu = mu;
            alpha = mu;
            eta = eta*pow(alpha, beta_eta);
            omega = omega*pow(alpha, beta_omega);

        }
        else
        {
            //lambda = lambda;
            mu *= tau;
            alpha = mu*gamma_bar;
            eta = eta_bar*pow(alpha, beta_eta);
            omega = omega_bar*pow(alpha, beta_omega);
        }

    } // end of "for each iteration"

    // This is not a memory leak!
    W.val = NULL;
    W.dim = 0;

    return 0;
} // end of LbfgsbOptimizer::solve()

double LbfgsbOptimizer::augmented_lagrangian_and_function_gradient(
        const ap::real_1d_array& x0,
        ap::real_1d_array& g)
{

    // Copy current iterate into solver solution vector
    for(size_t i = 0; i < x.dim; i++)
        x.val[i] = x0(i+1);

    // Compute objective function
    double obj = function();

    DenseVector W;
    W.val = x.val;
    W.dim = num_weak_learners;
    double w_sum = sum(W);
    W.val = NULL;
    W.dim = 0;
    obj -= lambda*(w_sum - 1.0);
    obj += (0.5*(w_sum - 1.0)*(w_sum - 1.0)/mu);

    // Compute gradient
    DenseVector _grad = gradient();

    for(size_t i = 0; i < num_weak_learners; i++)
    {
        g(i+1) = _grad.val[i] - lambda + ((w_sum - 1.0)/mu);
    }

    for(size_t i = num_weak_learners; i < _grad.dim; i++)
    {
        g(i+1) = _grad.val[i];
    }

    return obj;
}


void LbfgsbOptimizer::bounds(ap::integer_1d_array& nbd,
                             ap::real_1d_array& l,
                             ap::real_1d_array& u)
{

    // By default set lower bound to 0.0
    // By default set upper bound to 1.0
    for(size_t i = 1; i <= x.dim; i++)
    {
        l(i) = 0.0;
        u(i) = 1.0;
        nbd(i) = 2;
    }

    // psi have essentially no upper bound
    for(size_t i = num_weak_learners+1; i <= x.dim; i++)
    {
        nbd(i) = 1;
        u(i) = Optimizer::infinity;
    }

    return;
}

void LbfgsbOptimizer::bounds_binary(ap::integer_1d_array& nbd,
                                    ap::real_1d_array& l,
                                    ap::real_1d_array& u)
{

    // By default set lower bound to 0.0
    // By default set upper bound to 1.0
    for(size_t i = 1; i <= x.dim; i++)
    {
        l(i) = 0.0;
        u(i) = 1.0;
        nbd(i) = 2;
    }

    // beta has no lower bound
    nbd(num_weak_learners+1) = 0;
    l(num_weak_learners+1) = -Optimizer::infinity;
    u(num_weak_learners+1) = Optimizer::infinity;

    return;
}

} // end of namespace totally_corrective_boosting
