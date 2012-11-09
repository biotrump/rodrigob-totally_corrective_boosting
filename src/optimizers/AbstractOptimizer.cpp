
#include "AbstractOptimizer.hpp"

#include "math/vector_operations.hpp"

#include <limits>
#include <cmath>
#include <algorithm>

#include <stdexcept>
#include <sstream>

namespace totally_corrective_boosting
{

double log_one_plus_x(const double& x)
{
    if (x <= -1.0){
        std::stringstream os;
        os << "Invalid input argument (" << x << "); must be greater than -1.0";
        throw std::invalid_argument(os.str());
    }

    if (fabs(x) > 0.375)
    {
        // x is sufficiently large that the obvious evaluation is OK
        return log(1.0 + x);
    }

    // For smaller arguments we use a rational approximation
    // to the function log(1+x) to avoid the loss of precision
    // that would occur if we simply added 1 to x then took the log.

    const double p1 =  -0.129418923021993e+01;
    const double p2 =   0.405303492862024e+00;
    const double p3 =  -0.178874546012214e-01;
    const double q1 =  -0.162752256355323e+01;
    const double q2 =   0.747811014037616e+00;
    const double q3 =  -0.845104217945565e-01;
    double t, t2, w;

    t = x/(x + 2.0);
    t2 = t*t;
    w = (((p3*t2 + p2)*t2 + p1)*t2 + 1.0)/(((q3*t2 + q2)*t2 + q1)*t2 + 1.0);
    return 2.0*t*w;
}

AbstractOptimizer::AbstractOptimizer(const size_t& dim, 
                                     const bool& transposed,
                                     const double& eta,
                                     const double& nu,
                                     const double& epsilon,
                                     const bool& binary):
    gap(std::numeric_limits<double>::max()),
    num_weak_learners(0), dim(dim), transposed(transposed),
    eta(eta), nu(nu), epsilon(epsilon), binary(binary), edge(0.0), x(0),
    min_primal(std::numeric_limits<double>::max()),
    dual_obj(std::numeric_limits<double>::max()){

    if(binary)
    {
        x.resize(num_weak_learners + 1);
    } else
    {
        x.resize(num_weak_learners + dim);
    }
    // Note: dist is still un-initialized at this point
    return;
}

AbstractOptimizer::~AbstractOptimizer()
{
    // Give up reference to dist
    distribution.val = NULL;
    distribution.dim = 0;
    return;
}


void AbstractOptimizer::set_distribution(const DenseVector& _distribution)
{
    // dist is just a reference to the true array
    distribution.val = _distribution.val;
    distribution.dim = _distribution.dim;
    return;
}


void AbstractOptimizer::push_back(const SparseVector& u)
{

    double alpha = 0.0;

    if(num_weak_learners != 0)
    {
        // lets do one corrective step
        DenseVector UW;
        DenseVector W;
        W.val = x.val;
        W.dim = num_weak_learners;

        if(transposed)
        {
            transpose_dot(U, W, UW);
        }
        else
        {
            dot(U, W, UW);
        }

        W.val = NULL;
        W.dim = 0;

        axpy(-1.0, UW, u, UW);

        double dx = dot(distribution, UW);
        double maxx = max(UW);
        alpha = std::max(0.0, std::min(1.0, dx/(eta*maxx*maxx)));
    }

    DenseVector u_dense(u.dim);
    for(size_t i = 0; i < u.nnz; i++)
    {
        u_dense.val[u.index[i]] = u.val[i];
    }

    U.push_back(u_dense);

    // U.push_back(u);

    DenseVector x_tmp(x);

    // Increase size of x by one
    x.resize(x.dim + 1);

    // // svnvish: BUGBUG
    // // start off with uniform distribution
    // for(size_t i = 0; i <= num_wl; i++)
    //   x.val[i] = 1.0/(num_wl+1);

    // Copy previous value into x
    // set 0 wt for new weak learner
    for(size_t i = 0; i < num_weak_learners; i++)
    {
        x.val[i] = (1.0 - alpha)*x_tmp.val[i];
    }

    if(num_weak_learners == 0)
    {
        x.val[num_weak_learners] = 1.0;
    }
    else
    {
        x.val[num_weak_learners] = alpha;
    }

    // Now copy the rest of x back
    for(size_t i = num_weak_learners+1; i < x.dim; i++)
    {
        x.val[i] = x_tmp.val[i-1];
    }

    num_weak_learners++;

    // reset the min_primal value seen so far
    min_primal = std::numeric_limits<double>::max();

    // reset the dual objective value
    dual_obj = std::numeric_limits<double>::max();

    // reset the duality gap
    gap = std::numeric_limits<double>::max();

    return;
}


double AbstractOptimizer::erlp_function()
{
    function_timer.start();

    DenseVector W;
    W.val = x.val;
    W.dim = num_weak_learners;

    // psi is obtained by simply offsetting num_wl elements of x
    // Whatever is left is psi
    double* psi = x.val + num_weak_learners;
    double psi_sum = 0.0;
    for(size_t i = 0; i < dim; i++)
    {
        psi_sum += psi[i];
    }

    DenseVector tmp_distribution;
    // since the booster stores U transpose do transpose dot
    if(transposed)
    {
        transpose_dot(U, W, tmp_distribution);
    }
    else
    {
        dot(U, W, tmp_distribution);
    }

    // Find max element
    double exp_max = -std::numeric_limits<double>::max();
    for(size_t i = 0; i < tmp_distribution.dim; i++)
    {
        tmp_distribution.val[i] = - eta*(tmp_distribution.val[i] + psi[i]);
        if(tmp_distribution.val[i] > exp_max)
        {
            exp_max = tmp_distribution.val[i];
        }
    }

    // Safe exponentiation
    dual_obj = 0.0;
    for(size_t i = 0; i < tmp_distribution.dim; i++)
    {
        distribution.val[i] = exp(tmp_distribution.val[i] - exp_max)/dim;
        dual_obj += distribution.val[i];
    }

    scale(distribution, 1.0/dual_obj);

    dual_obj += 1e-10;
    dual_obj = (log(dual_obj)+ exp_max)/eta;
    dual_obj += (psi_sum/nu);

    // This is not a memory leak! (because W.val is only pointing to x.val)
    W.val = NULL;
    W.dim = 0;

    function_timer.stop();
    return dual_obj;
}


// Compute objective function value 
// Just a dummy forwarding function 
double AbstractOptimizer::function()
{
    if(binary)
    {
        return binary_function();
    }
    else
    {
        return erlp_function();
    }
}


DenseVector AbstractOptimizer::erlp_gradient()
{
    gradient_timer.start();
    DenseVector grad_w;

    // since the booster stores U transpose do normal dot and not
    // transpose dot
    if(transposed)
    {
        dot(U, distribution, grad_w);
    }
    else
    {
        transpose_dot(U, distribution, grad_w);
    }

    edge = max(grad_w);
    // Adjust the gradient
    scale(grad_w, -1.0);

    DenseVector grad(num_weak_learners + dim);

    // copy grad_w
    for(size_t i = 0; i < num_weak_learners; i++)
    {
        grad.val[i] = grad_w.val[i];
    }


    // set grad_psi = -dist + 1.0/nu
    for(size_t i = 0; i < dim; i++)
    {
        grad.val[i+num_weak_learners] = (1.0/nu) - distribution.val[i];
    }

    // The lowest primal objective we have seen so far
    min_primal = std::min(min_primal, primal());

    // duality gap w.r.t last known function value
    gap = min_primal + dual_obj;

    gradient_timer.stop();
    return grad;
}

// Return gradient of objective function 
// Assume that dist has been set by previous call to fun
// compute duality gap as a side effect 
// Just a dummy forwarding function 
DenseVector AbstractOptimizer::gradient()
{
    if(binary)
    {
        return binary_gradient();
    }
    else
    {
        return erlp_gradient();
    }
}

// Return primal objective function 
// Assume that x, dist, and edge have already been set by previous calls
// to grad  
double AbstractOptimizer::primal()
{
    if(binary)
    {
        return edge + (binary_relative_entropy(distribution, nu)/eta);
    }

    return edge + (relative_entropy(distribution)/eta);

}

bool AbstractOptimizer::duality_gap_met()
{
    // return gap < 0.05*epsilon;
    return gap < 0.5*epsilon;
}


// Compute objective function value for binary boost  
double AbstractOptimizer::binary_function()
{

    function_timer.start();
    DenseVector W;
    W.val = x.val;
    W.dim = num_weak_learners;

    // beta is last element of x
    double beta = x.val[num_weak_learners];

    DenseVector tmp_dist;

    // since the booster stores U transpose do transpose dot
    if(transposed)
        transpose_dot(U, W, tmp_dist);
    else
        dot(U, W, tmp_dist);


    // This is not a memory leak!
    W.val = NULL;
    W.dim = 0;

    // We want to compute:
    // f(x) = log(1 - nu.d + nu.d.exp(-x))
    // g(x) = d.exp(-x)/(1 - nu.d + nu.d.exp(-x))
    // (in our code x = eta*(tmp_dist.val[i] + beta))

    // Lets first tackle f(x)

    // The safe way to compute it is to multiply and divide by
    // nu.d.exp(x) to write
    // f(x) = log(1 + ((1 -nu.d)/nu.d)exp(x)) + log(nu.d) - x

    // To avoid underflow error we solve
    // ((1 -nu.d)/nu.d)exp(x) = epsilon
    // where epsilon is the machine precision to obtain
    // x_min = log(nu.d.epsilon/(1- nu.d))
    // if x < x_min then 1 + ((1 -nu.d)/nu.d)exp(x) will underflow
    // Therefore we simply use log(1 + y) \approx y to set
    // f(x) = ((1-nu.d)/nu.d)exp(x) + log(nu.d) - x

    // To avoid overflow error we solve
    // ((1 -nu.d)/nu.d)exp(x) = (1/epsilon)
    // where epsilon is the machine precision to obtain
    // x_max = log(nu.d/(epsilon.(1- nu.d)))
    // if x > x_max then 1 + ((1 -nu.d)/nu.d)exp(x) equals
    // ((1 -nu.d)/nu.d)exp(x) to within machine precision
    // Therefore we simply set
    // f(x) = log(((1 -nu.d)/nu.d)exp(x)) + log(nu.d) - x
    // which simplifies to
    // f(x) = log(1-nu.d)

    // Finally we call the log_one_plus_x routine with
    // ((1 -nu.d)/nu.d)exp(x) for the other cases

    // Now turning to g(x) we rewrite it as
    // g(x) = (1/nu) (1/(1 + ((1-nu.d)/nu.d)exp(x)))
    // It is easy to figure out that
    // if x < x_min then g(x) = 1/nu
    // if x > x_min then g(x) = 0
    // For the rest we compute things explicitly

    dual_obj = 0.0;
    double nu_d = nu/dim;
    double x_min = log(nu_d*std::numeric_limits<double>::epsilon()/(1- nu_d));
    double x_max = log(nu_d/(std::numeric_limits<double>::epsilon()*(1- nu_d)));

    for(size_t i = 0; i < tmp_dist.dim; i++)
    {
        tmp_dist.val[i] = eta*(tmp_dist.val[i] + beta);
        if(tmp_dist.val[i] < x_min)
        {
            dual_obj += ((1-nu_d)/nu_d)*exp(tmp_dist.val[i]) + log(nu_d) - tmp_dist.val[i];
            distribution.val[i] = 1.0/nu;
            continue;
        }
        if(tmp_dist.val[i] > x_max)
        {
            dual_obj += log(1-nu_d);
            distribution.val[i] = 0.0;
            continue;
        }
        double tmp = (1.0 - nu_d)*exp(tmp_dist.val[i])/nu_d;
        dual_obj += log_one_plus_x(tmp) + log(nu_d) - tmp_dist.val[i];
        distribution.val[i] = 1.0/(nu*(1.0 + tmp));
    }

    dual_obj /= (nu*eta);
    dual_obj += beta;
    
    function_timer.stop();
    return dual_obj;
}


// Assume that dist has been set by previous call to fun
DenseVector AbstractOptimizer::binary_gradient()
{

    gradient_timer.start();
    DenseVector grad_w;

    // since the booster stores U transpose do normal dot and not
    // transpose dot
    if(transposed)
        dot(U, distribution, grad_w);
    else
        transpose_dot(U, distribution, grad_w);

    edge = max(grad_w);
    // Adjust the gradient
    scale(grad_w, -1.0);

    DenseVector grad(num_weak_learners + 1);

    // copy grad_w
    for(size_t i = 0; i < num_weak_learners; i++)
        grad.val[i] = grad_w.val[i];

    // grad w.r.t beta
    grad.val[num_weak_learners] = 1.0 - sum(distribution);

    // The lowest primal objective we have seen so far
    min_primal = std::min(min_primal, primal());

    // duality gap w.r.t last known function value
    gap = min_primal + dual_obj;

    gradient_timer.stop();
    return grad;
}


void AbstractOptimizer::report_statistics()
{
    // dvec W;
    // W.val = x.val;
    // W.dim = num_wl;
    // std::cout << "W: " << W << std::endl;
    // W.val = NULL;
    // W.dim = 0;
    // std::cout << "X: " << x << std::endl;
    // std::cout << "dist: " << dist << std::endl;
    std::cout << "Time spent in " <<  function_timer.num_calls
              << " function evaluations: "
              << function_timer.total_cpu << " seconds" << std::endl;
    std::cout << "Time spent in " << gradient_timer.num_calls
              << " gradient evaluations: "
              << gradient_timer.total_cpu << " seconds" << std::endl;
    function_timer.reset();
    gradient_timer.reset();
    return;
}


// bool COptimizer::kkt_gap_met(const dvec& gradk){

//   double malpha = -std::numeric_limits<double>::max();
//   double Malpha = std::numeric_limits<double>::max();

//   for(size_t j = 0; j < num_wl; j++){
//     if((x.val[j] < 1.0) and (malpha < -gradk.val[j]))
//       malpha = -gradk.val[j];

//     if((x.val[j] > 0.0) and (Malpha > -gradk.val[j]))
//       Malpha = -gradk.val[j];
//   }
//   // std::cout << "KKT Gap : " << (malpha - Malpha) << std::endl; 
//   return (malpha - Malpha) < Optimizer::kkt_gap_tol;
// }

// bool COptimizer::pgnorm_met(const dvec& gradk){
//   // For the psi just compute the norm of the projected gradient  
//   double pgnorm = 0.0;
//   for(size_t j = num_wl; j < gradk.dim; j++){
//     double tmp = gradk.val[j];
//     if(x.val[j] == 0)
//       tmp = std::min(gradk.val[j], 0.0);
//     pgnorm += tmp*tmp;
//   }
//   // std::cout << "sqrt of pgnorm: " << sqrt(pgnorm) << std::endl;
//   return sqrt(pgnorm) < Optimizer::pgnorm_tol;
// }

// bool COptimizer::converged(const dvec& gradk){
//   if(binary){
//     // for binary erlpboost
//     // converged = true if kkt_gap < kkt_gap_tol and 
//     // norm of gradient w.r.t beta < pgnorm_tol
//     double gradbetak = gradk.val[num_wl];
//     return kkt_gap_met(gradk) and (std::abs(gradbetak) < Optimizer::pgnorm_tol);
//   } else {
//     // for erlpboost
//     // converged = true if kkt_gap < kkt_gap_tol and 
//     // norm of projected gradient of psi < pgnorm_tol
//     return kkt_gap_met(gradk) and pgnorm_met(gradk); 
//   }
//   return false;
// }

} // end of namespace totally_corrective_boosting
