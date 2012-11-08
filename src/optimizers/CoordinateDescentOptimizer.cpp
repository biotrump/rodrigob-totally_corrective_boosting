
#include "CoordinateDescentOptimizer.hpp"

#include "math/vector_operations.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace totally_corrective_boosting
{


CoordinateDescentOptimizer::CoordinateDescentOptimizer(const size_t& dim,
                                                       const bool& transposed,
                                                       const double& eta,
                                                       const double& nu,
                                                       const double& epsilon,
                                                       const bool& binary):
    AbstractOptimizer(dim, transposed, eta, nu, epsilon, binary){ }

int CoordinateDescentOptimizer::solve(){

    // svnvish: BUGBUG
    // only hard margin case
    if(nu != 1)
    {
        throw std::runtime_error("CoordinateDescentOptimizer is only valid for nu == 1");
    }

    // svnvish: BUGBUG
    // no binary ERLPboost yet
    if(binary == true)
    {
        throw std::runtime_error("CoordinateDescentOptimizer is only valid for binary == false");
    }

    DenseVector W;
    W.val = x.val;
    W.dim = num_weak_learners;

    DenseVector UW;

    // since the booster stores U transpose do transpose dot
    if(transposed)
        transpose_dot(U, W, UW);
    else
        dot(U, W, UW);

    double min_primal = std::numeric_limits<double>::max();

    // Loop through many times
    for(size_t j = 0; j < CD::max_iter; j++){

        // Find max element for safe exponentiation
        double exp_max = -std::numeric_limits<double>::max();

        for(size_t i = 0; i < distribution.dim; i++){
            distribution.val[i] = - eta*(UW.val[i]);
            if(distribution.val[i] > exp_max) exp_max = distribution.val[i];
        }

        // Safe exponentiation
        double obj = 0.0;
        for(size_t i = 0; i < distribution.dim; i++) {
            distribution.val[i] = exp(distribution.val[i] - exp_max)/dim;
            obj += distribution.val[i];
        }

        proj_simplex(distribution, exp_max);

        obj += 1e-10;
        obj = (log(obj)+ exp_max)/eta;

        DenseVector grad_w;

        // since the booster stores U transpose do normal dot and not
        // transpose dot
        if(transposed)
            dot(U, distribution, grad_w);
        else
            transpose_dot(U, distribution, grad_w);

        edge = max(grad_w);

        // This is the lowest primal objective we have seen so far
        min_primal = std::min(min_primal, primal());
        double gap = min_primal + obj;

        if(gap < 0.05*epsilon){
            std::cout << "Converged in " << j << " iterations" << std::endl;
            report_statistics();
            // This is not a memory leak!
            W.val = NULL;
            W.dim = 0;
            return 0;
        }

        size_t index = argmax(grad_w);
        assert(grad_w.val[index] > 0);

        DenseVector store(UW);

        if(transposed){
            DenseVector tmp(num_weak_learners);
            tmp.val[index] = 1.0;
            DenseVector U_index;
            transpose_dot(U, tmp, U_index);
            axpy(-1.0, UW, U_index, UW);
        }else
            axpy(-1.0, UW, U[index], UW);

        double eta_t = line_search(W, grad_w, index);
        // double denom = abs_max(UW);
        // denom *= denom;
        // double eta_t = std::max(0.0,
        //                         std::min(1.0, dot(dist, UW)/eta/denom));

        scale(W, (1.0-eta_t));
        W.val[index] += eta_t;
        axpy(eta_t, UW, store, UW);
    }

    // This is not a memory leak!
    W.val = NULL;
    W.dim = 0;

    return 1;
}

double CoordinateDescentOptimizer::line_search(DenseVector& W,
                                               const DenseVector& grad_w,
                                               const size_t& index){

    double lower = 0.0;
    double upper = 1.0;

    DenseVector orig_W(W);

    for(size_t i = 0; i < CD::ls_max_iter; i++){

        double mid = lower + ((upper - lower)/2.0);

        if (std::abs(mid-lower) < 1e-8)
            break;

        axpy(mid, grad_w, orig_W, W);
        function();
        DenseVector G = gradient();

        if(G.val[index] > 0)
            lower = mid;
        else
            upper = mid;

    }

    for(size_t i = 0; i < num_weak_learners; i++)
        W.val[i] = orig_W.val[i];

    return lower + ((upper - lower)/2.0);
}

// Return the dual objective function after projection
double CoordinateDescentOptimizer::proj_simplex(DenseVector& dist, const double& exp_max){

    double cap = 1.0/nu;
    double theta = 0.0;
    size_t dim = dist.dim;
    int N = dist.dim - 1;
    DenseVector tmpdist(dist);
    // sort dist from smallest to largest
    std::sort(tmpdist.val, tmpdist.val+tmpdist.dim);
    // store the sum of the dist in Z
    double Z = sum(tmpdist);
    // find theta
    for(size_t i = 0; i < tmpdist.dim; i++){
        theta = (1.0 - cap * i)/Z;
        if(theta * tmpdist.val[N-i] <= cap){break;}
        else{Z -= tmpdist.val[N-i];}
    }

    double ubndsum = 0.0;
    double psisum = 0.0;
    size_t bnd = 0;

    for(size_t i = 0; i < dist.dim; i++){
        if(theta*dist.val[i] > cap){
            bnd++;
            psisum -= log(dist.val[i]*dim)/eta;
            dist.val[i] = cap;
        }else{
            ubndsum += dist.val[i];
            dist.val[i] = theta*dist.val[i];
        }
    }

    double obj = exp_max + log(ubndsum/(1.0 - cap*bnd));
    obj *= ((bnd*cap - 1.0)/eta);
    obj -= ((bnd*cap/eta)*log(nu/dim));
    obj += psisum;

    return obj;
}

} // end of namespace totally_corrective_boosting
