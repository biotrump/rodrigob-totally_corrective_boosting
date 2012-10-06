/* Copyright (c) 2009, S V N Vishwanathan
 * All rights reserved. 
 * 
 * The contents of this file are subject to the Mozilla Public License 
 * Version 1.1 (the "License"); you may not use this file except in 
 * compliance with the License. You may obtain a copy of the License at 
 * http://www.mozilla.org/MPL/ 
 * 
 * Software distributed under the License is distributed on an "AS IS" 
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the 
 * License for the specific language governing rights and limitations 
 * under the License. 
 * 
 * Authors: S V N Vishwanathan 
 *
 * Created: (04/05/2009) 
 *
 * Last Updated: (04/05/2009)   
 */

#include "LpBoost.hpp"

#include "vec.hpp"

#include <iostream>
#include <cmath>

namespace totally_corrective_boosting
{

LpBoost::LpBoost(AbstractOracle* &oracle,
                   const int& num_pt, 
                   const int& max_iter,
                   const double& eps,
                   const double& nu):
  AbstractBooster(oracle, num_pt, max_iter), minPt1dt1(-1.0), minPqdq1(1.0), eps(eps), nu(nu){
  
  solver.setLogLevel(0);
  solver.resize(0, 1+num_pt);
  
  solver.setObjCoeff(0, 1.0);
  for(int i=0; i<num_pt; i++) 
    solver.setObjCoeff(i+1, 0.0);
  
  // set bounds for \xi
  solver.setColumnBounds(0, -COIN_DBL_MAX, COIN_DBL_MAX);
  
  // set bounds for d_i
  for (int i=0; i<num_pt; i++)
    solver.setColumnBounds(i+1, 0.0, 1.0/nu);
  
  // add summation to one constraint
  double* newrow = new double[1+num_pt];
  int* newrowidx = new int[1+num_pt];
  
  newrow[0] = 0;
  newrowidx[0] = 0;
  for(int i = 0; i < num_pt; i++){
    newrow[i+1] = 1.0;
    newrowidx[i+1] = i+1;
  }
  solver.addRow(1+num_pt, newrowidx, newrow, 1.0, 1.0);
  
  delete newrow;
  delete newrowidx;
  
  return;
}

LpBoost::~LPBoost(){
  
}

void LpBoost::update_linear_ensemble(const WeakLearner& wl){
  WeightedWeakLearner wwl(&wl, 1.0);
  model.add(wwl);
  return;
}

bool LpBoost::stopping_criterion(std::ostream& os){
  os << "epsilon gap: " <<  minPqdq1 - minPt1dt1<< std::endl;
  return(minPqdq1 <=  minPt1dt1 + eps/2.0);
}


void LpBoost::update_stopping_criterion(const WeakLearner& wl){

  double gamma = wl.get_edge(); 
  if(gamma < minPqdq1) minPqdq1 = gamma;
  return;
}

void LpBoost::update_weights(const WeakLearner& wl){
  
  // The predictions are already pre-multiplied with the labels already
  // in the weak learner
  
  SparseVector pred = wl.get_prediction();

  // std::cout << pred
  //           << "Number of rows: " << solver.getNumRows() << std::endl
  //           << "Number of cols: " << solver.getNumCols() << std::endl;

  double* newrow = new double[1+pred.nnz];
  int* newrowidx = new int[1+pred.nnz];
  
  newrow[0] = -1;
  newrowidx[0] = 0;
  for(size_t i = 0; i < pred.nnz; i++){
    newrow[i+1] = pred.val[i];
    newrowidx[i+1] = pred.idx[i]+1;
    // std::cout << "newrow[" << i+1 << "]: " << newrow[i+1] << "  " << newrowidx[i+1] << std::endl;
  }
  solver.addRow(1+pred.nnz, newrowidx, newrow, -COIN_DBL_MAX, 0);

  delete newrow;
  delete newrowidx;
  
  solver.primal();
  // solver.dual();
  
  // read back the distribution
  double *sol = solver.primalColumnSolution(); 
  assert(sol);
  // std::cout << sol[0] << std::endl << std::endl;
  for(int i = 0; i < num_pt; i++){
    dist.val[i] = sol[i+1];
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
  
  model.set_wts(wtvec);
  
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
