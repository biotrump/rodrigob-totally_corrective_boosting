
#include "LpBoost.hpp"

#include "math/vector_operations.hpp"

#include <iostream>
#include <cmath>

namespace totally_corrective_boosting
{

LpBoost::LpBoost(AbstractOracle *oracle,
                 const int num_data_points,
                 const int max_iterations,
                 const double epsilon_,
                 const double nu_):
    AbstractBooster(oracle, num_data_points, max_iterations), minPt1dt1(-1.0), minPqdq1(1.0), epsilon(epsilon_), nu(nu_)
{

    solver.setLogLevel(0);
    solver.resize(0, 1+num_data_points);

    solver.setObjCoeff(0, 1.0);
    for(int i=0; i<num_data_points; i++)
        solver.setObjCoeff(i+1, 0.0);

    // set bounds for \xi
    solver.setColumnBounds(0, -COIN_DBL_MAX, COIN_DBL_MAX);

    // set bounds for d_i
    for (int i=0; i<num_data_points; i++)
    {
        solver.setColumnBounds(i+1, 0.0, 1.0/nu);
    }

    // add summation to one constraint
    std::vector<double> new_row(1+num_data_points);
    std::vector<int> new_row_index(1+num_data_points);

    new_row[0] = 0;
    new_row_index[0] = 0;
    for(int i = 0; i < num_data_points; i++)
    {
        new_row[i+1] = 1.0;
        new_row_index[i+1] = i+1;
    }
    solver.addRow(1+num_data_points, new_row_index.data(), new_row.data(), 1.0, 1.0);

    return;
}

LpBoost::~LpBoost()
{
    // nothing to do here
    return;
}

void LpBoost::update_linear_ensemble(const AbstractWeakLearner &wl){
    WeightedWeakLearner wwl(&wl, 1.0);
    model.add(wwl);
    return;
}

bool LpBoost::stopping_criterion(std::ostream& os){
    os << "epsilon gap: " <<  minPqdq1 - minPt1dt1<< std::endl;
    return(minPqdq1 <=  minPt1dt1 + epsilon/2.0);
}


void LpBoost::update_stopping_criterion(const AbstractWeakLearner &wl){

    double gamma = wl.get_edge();
    if(gamma < minPqdq1) minPqdq1 = gamma;
    return;
}

void LpBoost::update_weights(const AbstractWeakLearner &wl){

    // The predictions are already pre-multiplied with the labels already
    // in the weak learner

    SparseVector pred = wl.get_prediction();

    // std::cout << pred
    //           << "Number of rows: " << solver.getNumRows() << std::endl
    //           << "Number of cols: " << solver.getNumCols() << std::endl;

    double* new_row = new double[1+pred.nnz];
    int* new_row_index = new int[1+pred.nnz];

    new_row[0] = -1;
    new_row_index[0] = 0;
    for(size_t i = 0; i < pred.nnz; i++){
        new_row[i+1] = pred.val[i];
        new_row_index[i+1] = pred.index[i]+1;
        // std::cout << "new_row[" << i+1 << "]: " << new_row[i+1] << "  " << new_row_index[i+1] << std::endl;
    }
    solver.addRow(1+pred.nnz, new_row_index, new_row, -COIN_DBL_MAX, 0);

    delete new_row;
    delete new_row_index;

    // solve in the primal
    solver.primal();
    // solver.dual();

    // read back the distribution
    double *sol = solver.primalColumnSolution();
    assert(sol);
    // std::cout << sol[0] << std::endl << std::endl;
    for(int i = 0; i < num_data_points; i++){
        examples_distribution.val[i] = sol[i+1];
        // std::cout << dist.val[i] << std::endl;
    }

    // Now read the weights
    double *wt = solver.dualRowSolution();
    assert(wt);
    // std::cout << std::endl << wt[0] << std::endl << std::endl;
    // wt[0] is the summation to one constraint

    DenseVector wtvec(model.size());
    for(size_t i = 0; i < wtvec.dim; i++)
        wtvec.val[i] = -wt[i+1];

    model.set_weights(wtvec);

    // for(size_t i = 0; i < model.ensemble.size(); i++){
    //   model.ensemble[i].set_wt(-wt[i+1]);
    //   // std::cout << wt[i+1] << std::endl;
    // }

    // update lower bound on \xi for next iteration
    solver.setColumnBounds(0, sol[0], COIN_DBL_MAX);

    // minimum edge from the solver
    minPt1dt1 = sol[0];

    // std::cout << "Objective value: " << solver.objectiveValue()
    //           << "  xi value: " << sol[0] << std::endl;
    return;
}

} // end of namespace totally_corrective_boosting
