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
 * Authors: S V N Vishwanathan (vishy@stat.purdue.edu)
 * 
 *
 * Created: (28/03/2009) 
 *
 * Last Updated: (15/04/2008)   
 */

#include "TaoOptimizer.hpp"

#include <limits>
#include <iostream>
#include <cmath>
#include <algorithm>


namespace totally_corrective_boosting
{

TaoOptimizer::TaoOptimizer(const size_t& dim, 
                               const bool& transposed, 
                               const double& eta, 
                               const double& nu,
                               const double& epsilon,
                               const bool& binary, 
                               int& argc, 
                               char** argv):
  AbstractOptimizer(dim, transposed, eta, nu, epsilon, binary), 
  lambda(0.0), mu(1.0) { 
  
  static  char help[] = "Augmented Lagrangian solver using TAO";

  PetscInitialize(&argc, &argv,(char *)0,help);
  TaoInitialize(&argc, &argv,(char *)0,help);
  return;
}

TaoOptimizer::~Optimizer_TAO(){

  PetscFinalize();
  TaoFinalize();
  return;
}

int TaoOptimizer::solve(){
  int        info;
  
  TAO_SOLVER tao;
  TAO_APPLICATION taoapp;
  TaoMethod  method = "tao_blmvm"; 
  
  // TAO solver
  info = TaoCreate(PETSC_COMM_SELF,method,&tao); CHKERRQ(info);
  info = TaoApplicationCreate(PETSC_COMM_SELF,&taoapp); CHKERRQ(info);
  // Solution vector
  Vec X;
  VecCreateSeqWithArray(PETSC_COMM_SELF, x.dim, x.val, &X); CHKERRQ(info);
  
  info = TaoAppSetInitialSolutionVec(taoapp, X); CHKERRQ(info);
  // Set routines for function and gradient evaluation
  info = TaoAppSetObjectiveAndGradientRoutine(taoapp,
                                              tao_fun_grad,
                                              (void *)this); CHKERRQ(info);
  // Set bounds
  if(binary){
    info = TaoAppSetVariableBoundsRoutine(taoapp,
                                          tao_bounds_binary,
                                          (void*)this); CHKERRQ(info);
  }else{
    info = TaoAppSetVariableBoundsRoutine(taoapp,
                                          tao_bounds,
                                          (void*)this); CHKERRQ(info);
  }
  
  // Lancelot: eta_0 = firtsc/pow(std::min(mu, gamma_bar), alpha_eta); 
  PetscReal eta_0 = 1.0; 

  PetscReal mu_0 = 0.1; // must be <= 1

  // Lancelot: omega_0 = firtsg/pow(std::min(mu, gamma_bar), alpha_omega); 
  PetscReal omega_0 = 1.0; // 0.5; 

  // Lancelot: tau = 0.1  
  PetscReal tau = 0.01; // # < 1
  
  // Lancelot: gamma_1 = 0.1
  PetscReal gamma_1 = 0.1; // < 1 : 

    //Lancelot: alpha_omega = 1.0
  PetscReal alpha_omega = 1.0;

  //Lancelot: beta_omega = 1.0
  PetscReal beta_omega = 1.0;
  
  // Lancelot: alpha_eta = 0.1
  PetscReal alpha_eta = 0.1; // # min(1, self._alpha_omega)
  
  //Lancelot: beta_eta = 0.9
  PetscReal beta_eta = 0.9; // # min(1, self._beta_omega)
    
  mu = mu_0;
  
  // Lancelot: alpha  = min(mu, gamma_1)
  PetscReal alpha = std::min(mu, gamma_1);

  // Lancelot: omega = max(omemin, omega_0*pow(alpha, alpha_omega))
  // svnvish: BUGBUG what is omemin?
  // omemin is the tolerance for kkt gap
  PetscReal omega = omega_0*pow(alpha, alpha_omega);

  // Lancelot: eta = max(etamin, eta_0*pow(alpha, alpha_eta))
  // svnvish: BUGBUG what is etamin?
  // etamin is the minimum norm of the constraint violation 
  PetscReal eta = eta_0*pow(alpha, alpha_eta);
  
  DenseVector W;
  W.val = x.val;
  W.dim = num_wl;
  info = TaoSetTolerances(tao, 0, 0, 0, 0); CHKERRQ(info);
  
  for(size_t iter = 0; iter < TAO::max_iter; iter++){
    
    info = TaoSetGradientTolerances(tao, omega, 0, 0); CHKERRQ(info);
    // seed with old solution
    info = TaoAppSetInitialSolutionVec(taoapp, X); CHKERRQ(info);
    // solve the problem
    info = TaoSolveApplication(taoapp, tao); CHKERRQ(info);
    
    double w_gap = sum(W) - 1.0;
    
    int it;
    double ff, gnorm, cnorm;
    TaoTerminateReason reason;
    
    // Get termination information
    info = TaoGetSolutionStatus(tao, &it, &ff, &gnorm, &cnorm, 0, &reason); CHKERRQ(info);
    
    if(std::abs(w_gap) < eta){
      
      if(std::abs(w_gap) < Optimizer::wt_sum_tol){
        if(duality_gap_met()){
          report_stats();
          break;
        } 
      } 
      lambda = lambda + (w_gap/mu);
      //mu = mu;
      alpha = std::min(mu, gamma_1);
      omega = omega*pow(alpha, beta_omega);
      eta = eta*pow(alpha, beta_eta);
      
    }else{
      //lambda = lambda;
      mu *= tau;
      alpha = std::min(mu, gamma_1);
      omega = omega_0*pow(alpha, alpha_omega);
      eta = eta_0*pow(alpha, alpha_eta);
    }
  }
  
  // This is not a memory leak!
  W.val = NULL;
  W.dim = 0;
 
  // Free TAO data structures
  info = TaoDestroy(tao); CHKERRQ(info);
  info = TaoAppDestroy(taoapp); CHKERRQ(info);
  info = VecDestroy(X); CHKERRQ(info);
  
  return 0;
}

double TaoOptimizer::aug_lag_fg(const Vec& X, Vec& G){

  // Copy TAO iterate into solver solution vector 
  PetscScalar  *x_array, *g_array;
  int info = VecGetArray(X, &x_array); CHKERRQ(info);
  for(size_t i = 0; i < x.dim; i++)
    x.val[i] = x_array[i];
  info = VecRestoreArray(X, &x_array); CHKERRQ(info);

  // Compute objective function 
  double obj = fun();
  
  DenseVector W;
  W.val = x.val;
  W.dim = num_wl;
  double w_gap = sum(W) - 1.0;
  W.val = NULL;
  W.dim = NULL;
  obj += lambda*w_gap;
  obj += ((0.5*w_gap*w_gap)/mu);
  
  // Compute gradient 
  DenseVector _grad = grad();
  
  info = VecGetArray(G, &g_array); CHKERRQ(info);
  for(size_t i = 0; i < num_wl; i++)
    g_array[i] = _grad.val[i] + lambda + (w_gap/mu);
  for(size_t i = num_wl; i < _grad.dim; i++)
    g_array[i] = _grad.val[i];
  
  info = VecRestoreArray(G, &g_array); CHKERRQ(info);
  
  return obj;
}

int TaoOptimizer::bounds(Vec& xl, Vec& xu){
  
  PetscScalar  *xu_array;
  
  // Lower bound for all our variables is 0.0
  int info = VecSet(xl, 0.0); CHKERRQ(info);
  
  // Create vector of ones
  // That is the upper bound on w  
  info = VecSet(xu, 1.0); CHKERRQ(info);

  info = VecGetArray(xu, &xu_array); CHKERRQ(info);
  
  // psi have essentially no upper bound
  for(size_t i = num_wl; i < num_wl+dim; i++)
    xu_array[i] = Optimizer::INFTY;
  
  info = VecRestoreArray(xu, &xu_array); CHKERRQ(info);
  
  return 0;
}

int TaoOptimizer::bounds_binary(Vec& xl, Vec& xu){
  
  PetscScalar  *xl_array, *xu_array;
  
  // Lower bound for all our variables is 0.0
  int info = VecSet(xl, 0.0); CHKERRQ(info);

  info = VecGetArray(xl, &xl_array); CHKERRQ(info);

  // beta has no lower bound
  xl_array[num_wl] = -Optimizer::INFTY;

  info = VecRestoreArray(xl, &xl_array); CHKERRQ(info);

  // Create vector of ones
  // That is the upper bound on w  
  info = VecSet(xu, 1.0); CHKERRQ(info);

  info = VecGetArray(xu, &xu_array); CHKERRQ(info);
  
  // beta has no upper bound
  xu_array[num_wl] = Optimizer::INFTY;
  
  info = VecRestoreArray(xu, &xu_array); CHKERRQ(info);
  
  return 0;
}


// Dummy forwarding function 
int tao_fun_grad(TAO_APPLICATION taoapp,
                 Vec X,
                 double *obj,
                 Vec G,
                 void *ctx){
  TaoOptimizer * solver = (TaoOptimizer *)ctx;
  *obj = solver->aug_lag_fg(X, G);  
  return 0;
}

// dummy forwarding function
int tao_bounds(TAO_APPLICATION taoapp, 
               Vec xl, 
               Vec xu, 
               void *ctx){
  TaoOptimizer * solver = (TaoOptimizer *)ctx;
  return solver->bounds(xl, xu);
}

// dummy forwarding function
int tao_bounds_binary(TAO_APPLICATION taoapp, 
                      Vec xl, 
                      Vec xu, 
                      void *ctx){
  TaoOptimizer * solver = (TaoOptimizer *)ctx;
  return solver->bounds_binary(xl, xu);
}

} // end of namespace totally_corrective_boosting
