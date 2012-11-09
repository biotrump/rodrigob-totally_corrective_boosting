
#include "CorrectiveBoost.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>

namespace totally_corrective_boosting
{


CorrectiveBoost::CorrectiveBoost(AbstractOracle* oracle,
                                 const int num_pt,
                                 const int max_iter,
                                 const double eps,
                                 const double nu,
                                 const bool linesearch,
                                 const int disp_freq)
    : AbstractBooster(oracle, num_pt, max_iter,disp_freq),
      linesearch(linesearch),
      minPt1dt1(-1.0),
      minPqdq1(1.0),
      eps(eps),
      nu(nu)
{
    eta = 2.0*log(num_pt/nu)/eps;
    return;
}


CorrectiveBoost::CorrectiveBoost(AbstractOracle* oracle,
                                 const int num_pt,
                                 const int max_iter,
                                 const double eps,
                                 const double eta,
                                 const double nu,
                                 const bool linesearch,
                                 const int disp_freq)
    : AbstractBooster(oracle, num_pt, max_iter,disp_freq),
      linesearch(linesearch),
      minPt1dt1(-1.0),
      minPqdq1(1.0),
      eps(eps),
      nu(nu),
      eta(eta)
{
    return;
}

CorrectiveBoost::~CorrectiveBoost()
{
    // nothing to do here
    return;
}

void CorrectiveBoost::update_weights(const AbstractWeakLearner &wl){

    double exp_max = 0.0;

    for(size_t i = 0; i < examples_distribution.dim; i++){
        examples_distribution.val[i] = -eta * UW.val[i];
        exp_max = std::max(exp_max,examples_distribution.val[i]);
    }

    double dual_obj = 0.0;
    for(size_t i = 0; i < examples_distribution.dim; i++){
        examples_distribution.val[i] = exp(examples_distribution.val[i] - exp_max)/examples_distribution.dim;
        dual_obj += examples_distribution.val[i];
    }

    minPt1dt1 = proj_simplex(examples_distribution, exp_max);

    return;
}


void CorrectiveBoost::update_linear_ensemble(const AbstractWeakLearner &weak_learner)
{
    std::cout << "Number of weak learners: " << model.size() << std::endl;

    SparseVector ut = weak_learner.get_prediction();

    DenseVector x(ut.dim);
    for(size_t i = 0; i < ut.nnz; i++)
    {
        x.val[ut.index[i]] = ut.val[i];
    }

    const DenseVector ut_dense(x);

    if(UW.dim>0)
    {
        DenseVector w = model.get_weights();
        // subtract Uw from x and store in x
        for(size_t i = 0; i < x.dim; i++)
        {
            x.val[i] -=  UW.val[i];
        }
    }

    // set alpha
    const double dx = dot(examples_distribution, x);

    // find the maximum value of x
    const double max_x = max(x);
    const double denom = pow(max_x,2);
    double alpha;

    // svnvish: BUGBUG
    // silently ignore linesearch request
    alpha = std::max(0.0, std::min(1.0, dx/eta/denom));

    // if(linesearch){
    //   alpha = line_search(ut_dense);
    // }else{
    //   alpha = std::max(0.0, std::min(1.0, dx/eta/denom));
    // }

    // update the other weights
    model.scale_weights(1.0-alpha);

    WeightedWeakLearner wwl(&weak_learner, alpha);
    model.add(wwl);

    if(UW.dim ==0)
    {
        UW.dim = x.dim;
        UW.val = new double[x.dim];
        for(size_t i = 0; i < UW.dim; i++)
        {
            UW.val[i] = alpha *ut_dense.val[i];
        }
    }
    else
    {
        for(size_t i = 0; i < UW.dim; i++)
        {
            UW.val[i] = (1.0 - alpha)*UW.val[i] + alpha * ut_dense.val[i];
        }
    }

    return;
}


bool CorrectiveBoost::stopping_criterion(std::ostream& log_stream)
{
    log_stream << "min of Obj Values : " << minPqdq1 << std::endl;
    log_stream << "min Lower Bound : " << minPt1dt1 << std::endl;
    log_stream << "epsilon gap: " <<  minPqdq1 - minPt1dt1<< std::endl;
    return(minPqdq1 <=  minPt1dt1 + eps/2.0);
}


void CorrectiveBoost::update_stopping_criterion(const AbstractWeakLearner &wl)
{
    minPqdq1 = std::min(wl.get_edge()+(relative_entropy(examples_distribution)/eta), minPqdq1);
    return;
}


/// @returns the dual objective function after projection
double CorrectiveBoost::proj_simplex(DenseVector& dist, const double& exp_max)
{

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
    for(size_t i = 0; i < tmpdist.dim; i++)
    {
        theta = (1.0 - cap * i)/Z;
        if(theta * tmpdist.val[N-i] <= cap)
        {
            break;
        }
        else
        {
            Z -= tmpdist.val[N-i];
        }
    }

    double ubndsum = 0.0;
    double psisum = 0.0;
    size_t bnd = 0;

    for(size_t i = 0; i < dist.dim; i++)
    {
        if(theta*dist.val[i] > cap)
        {
            bnd++;
            psisum -= log(dist.val[i]*dim)/eta;
            dist.val[i] = cap;
        }
        else
        {
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
