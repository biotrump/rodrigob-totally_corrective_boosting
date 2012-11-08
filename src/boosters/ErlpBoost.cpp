
#include "ErlpBoost.hpp"
#include "vector_operations.hpp"

#include <iostream>
#include <cmath>


namespace totally_corrective_boosting
{

ErlpBoost::ErlpBoost(AbstractOracle* &oracle,
                     const int& num_pt,
                     const int& max_iter,
                     const double& eps,
                     const double& nu,
                     const bool& binary,
                     AbstractOptimizer* &solver):
    AbstractBooster(oracle, num_pt, max_iter), found(false),
    binary(binary), minPt1dt1(-1.0), minPqdq1(1.0),
    eps(eps), nu(nu), solver(solver){

    if(binary){
        eta = 2.0*(1.0+log(num_pt/nu))/eps;
    }else{
        eta = 2.0*log(num_pt/nu)/eps;
    }
    assert(solver);
    // Set reference to the dist array in the solver
    solver->set_distribution(examples_distribution);
    return;
}

ErlpBoost::ErlpBoost(AbstractOracle* &oracle,
                     const int& num_pt,
                     const int& max_iter,
                     const double& eps,
                     const double& eta,
                     const double& nu,
                     const bool& binary,
                     AbstractOptimizer* &solver):
    AbstractBooster(oracle, num_pt, max_iter), found(false),
    binary(binary), minPt1dt1(-1.0), minPqdq1(1.0), eps(eps),
    nu(nu), eta(eta), solver(solver){

    assert(solver);
    // Set reference to the dist array in the solver
    solver->set_distribution(examples_distribution);
    return;
}


ErlpBoost::~ErlpBoost()
{
    // nothing to do here
    return;
}

void ErlpBoost::update_linear_ensemble(const AbstractWeakLearner& wl){

    WeightedWeakLearner wwl(&wl, 0.0);
    found = model.add(wwl);

    return;
}

bool ErlpBoost::stopping_criterion(std::ostream& os){
    std::cout << "min of Obj Values : " << minPqdq1 << std::endl;
    std::cout << "min Lower Bound : " << minPt1dt1 << std::endl;
    std::cout << "epsilon gap: " <<  minPqdq1 - minPt1dt1<< std::endl;
    os << "epsilon gap: " <<  minPqdq1 - minPt1dt1<< std::endl;
    return(minPqdq1 <=  minPt1dt1 + eps/2.0);
}


void ErlpBoost::update_stopping_criterion(const AbstractWeakLearner& wl){

    double gamma = wl.get_edge();
    if(binary){
        gamma += (binary_relative_entropy(examples_distribution, nu)/eta);
    }else{
        gamma += (relative_entropy(examples_distribution)/eta);
    }

    if(gamma < minPqdq1) minPqdq1 = gamma;
    return;
}

void ErlpBoost::update_weights(const AbstractWeakLearner& wl){

    // The predictions are already pre-multiplied with the labels already
    // in the weak learner

    if(!found){
        // need to push into the solver
        SparseVector prediction = wl.get_prediction();
        solver->push_back(prediction);
    }

    // Call the solver
    const int info = solver->solve();
    assert(!info);

    // We get back the distribution and max edge for free.
    // Only need to read wts back
    // BEWARE:
    // Only the relevant entries of solver->x are copied over.

    model.set_weights(solver->x);

    minPt1dt1 = -solver->dual_obj;

    return;
}

} // end of namespace totally_corrective_boosting
