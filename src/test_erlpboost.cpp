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
 * Created: (28/03/2009) 
 *
 * Last Updated: (28/03/2008)   
 */

#include <iostream>
#include <fstream>

#include <sstream>
#include <stdexcept>

#include <erlpboost-config.h>
#include "reader.hpp"
#include "rawdata.hpp"
#include "decisionstump.hpp"
#include "svm.hpp"
#include "erlpboost.hpp"
#include "adaboost.hpp"
#include "corrective.hpp"


#include "optimizer_pg.hpp"
#include "optimizer_hz.hpp"
#include "optimizer_lbfgsb.hpp"
#include "optimizer_cd.hpp"

#include "evaluate.hpp"
#include "parse.hpp"
#include "ConfigFile.hpp"



#ifdef ERLPBOOST_HAVE_TAO
#include "optimizer_tao.hpp"
#endif

#ifdef ERLPBOOST_HAVE_CLP
#include "lpboost.hpp"
#endif

int main(int argc, char **argv){
    
  if(argc != 2){
    std::stringstream os;
    os <<"You need to run this program as: erlpboost name_of_config_file" 
       << std::endl
       << "See the config directory for examples of config files" 
       << std::endl;
		throw std::invalid_argument(os.str());
  }
  
  char* filename = argv[1];
  std::string config_file = filename;

  ConfigFile config(config_file);

  std::string train_file;
  config.readInto(train_file, "train_file");

  std::string test_file;
  config.readInto(test_file, "test_file");

  std::string valid_file;
  config.readInto(valid_file, "valid_file", std::string("no_valid"));

  std::string output_file;
  config.readInto(output_file, "output_file");
  
  std::string oracle_type;
  config.readInto(oracle_type, "oracle_type");
  
  bool reflexive;
  config.readInto(reflexive, "reflexive");

  int max_iter = 0;
  config.readInto(max_iter, "max_iter");
  
  std::ofstream of;
  of.open(output_file.c_str());
  if(!of.good()) {
    std::stringstream os;
    os <<"Cannot open data file : " << output_file << std::endl;
		throw std::invalid_argument(os.str());
  }

  CLibSVMReader c;
  std::vector<svec> data;
  std::vector<int> labels;
  bool xposed = true;
  
  if(xposed)
    c.readlibSVM_transpose(train_file, data, labels);
  else
    c.readlibSVM(train_file, data, labels);

  COracle* oracle = NULL;
  
  if(oracle_type == "rawdata")
    oracle = new CRawData(data, labels, xposed, reflexive);
  else if(oracle_type == "svm")
    oracle = new CSVM(data, labels, xposed, reflexive);
  else if(oracle_type == "decisionstump")
    oracle = new CDecisionStump(data, labels, reflexive);
  
  std::string booster_type;
  config.readInto(booster_type, "booster_type", std::string("ERLPBoost"));
  
  CBooster* eb = NULL;
  COptimizer* solver = NULL;
  
  if(booster_type == "ERLPBoost"){
    
    double eps = 0;
    config.readInto(eps, "eps", 0.001);
     
    double nu = 0;
    config.readInto(nu, "nu", 1.0);
    
    bool binary = false;
    config.readInto(binary, "binary", false);
    
    std::string optimizer_type;
    config.readInto(optimizer_type, "optimizer_type", std::string("lbfgsb"));
    
    bool transposed = true;
    double eta = 0.0;
    
    if(binary)
      config.readInto(eta, "eta", 2.0*(1.0+log(labels.size()/nu))/eps);
    else
      config.readInto(eta, "eta", 2.0*log(labels.size()/nu)/eps);
    
    of << "Maximum Iterations: " << max_iter << std::endl;
    of << "Epsilon (Tolerance): " << eps << std::endl;
    of << "Nu (softening): " << 1.0/nu << std::endl << std::endl;  
    of << "eta: " << eta << std::endl << std::endl;
    
    if(optimizer_type == "tao"){
#ifdef ERLPBOOST_HAVE_TAO
      solver = new COptimizer_TAO(labels.size(), transposed, eta, nu, eps, binary, argc, argv);
#else                                           
      std::stringstream os;
      os << "You must compile with TAO support enabled to use TAO" 
         << std::endl << "See README for details" << std::endl; 
      throw std::invalid_argument(os.str());
#endif
    } else if(optimizer_type == "pg")
      solver = new COptimizer_PG(labels.size(), transposed, eta, nu, eps, binary);
    else if(optimizer_type == "hz")
      solver = new COptimizer_HZ(labels.size(), transposed, eta, nu, eps, binary);
    else if(optimizer_type == "lbfgsb")
      solver = new COptimizer_LBFGSB(labels.size(), transposed, eta, nu, eps, binary);
    else if(optimizer_type == "cd")
      solver = new COptimizer_CD(labels.size(), transposed, eta, nu, eps, binary);
    
    eb = new CERLPBoost(oracle, labels.size(), max_iter, eps, eta, nu, binary, solver);
  } 
  else if(booster_type == "LPBoost"){
    std::cout << "running lpboost" << std::endl;
    double eps = 0;
    config.readInto(eps, "eps", 0.001);
    
    double nu = 0;
    config.readInto(nu, "nu", 1.0);
#ifdef ERLPBOOST_HAVE_CLP
    eb = new CLPBoost(oracle, labels.size(), max_iter, eps, nu);
#else
    std::stringstream os;
    os << "You must compile with COIN-OR LP solver support enabled to use LPBoost" 
       << std::endl << "See README for details" << std::endl; 
    throw std::invalid_argument(os.str());
#endif
  }
  else if(booster_type == "AdaBoost"){
    std::cout << "running adaboost" << std::endl;
    eb = new CAdaBoost(oracle, labels.size(), max_iter,250);
  }
  else if(booster_type == "Corrective"){

    double eps = 0;
    config.readInto(eps, "eps", 0.001);
     
    double nu = 0;
    config.readInto(nu, "nu", 1.0);

    bool linesearch;
    config.readInto(linesearch, "linesearch", false);
    
    double eta = 0.0;
    config.readInto(eta, "eta", 2.0*log(labels.size()/nu)/eps);

    eb = new CCorrective(oracle, labels.size(), max_iter, eps, eta, nu, linesearch,10);
  }
  
  assert(eb);
  
  size_t num_models = eb->boost(of);
  
  CEnsemble model = eb->get_ensemble();
  // of << "model" << std::endl << model;

  CEvaluate score;

  // get training error
  dvec trainpred = model.predict(data);
  int train_loss;
  double train_err;
  score.binary_loss(trainpred, labels, train_loss, train_err);
  std::cout << "training error: " << train_err*100 << "%" << std::endl;
  of << "training error: " << train_err*100 << "%" << std::endl;

  // get test error
  std::vector<svec> test_data;
  std::vector<int> test_labels;
  c.readlibSVM_transpose(test_file, test_data, test_labels);
  // backfill
  while(test_data.size() < data.size()){
    svec empty(data[0].dim,1);
    test_data.push_back(empty);
  }

  dvec testpred = model.predict(test_data);
  int test_loss;
  double test_err;
  score.binary_loss(testpred, test_labels, test_loss, test_err);
  std::cout  << "test error: " << test_err*100 << "%" << std::endl;
  of << "test error: " << test_err*100 << "%" << std::endl;

  // get validation error
  std::vector<svec> valid_data;
  std::vector<int> valid_labels;
  if(valid_file != "no_valid"){
    c.readlibSVM_transpose(valid_file, valid_data, valid_labels);

    // backfill
    while(valid_data.size() < data.size()){
      svec empty(data[0].dim,1);
      valid_data.push_back(empty);
    }
    int valid_loss;
    double valid_err;
    dvec validpred = model.predict(valid_data);
    score.binary_loss(validpred,valid_labels, valid_loss, valid_err);
    std::cout << "validation error: " << valid_err*100 << "%" << std::endl;
    of << "validation error: " << valid_err*100 << "%" << std::endl;
  }
  of.close();

  // get generalization error per iteration
  std::ifstream in;
  in.open(output_file.c_str());
  if(!in.good()) {
    std::stringstream os;
    os <<"Cannot open data file : " << output_file << std::endl;
		throw std::invalid_argument(os.str());
  }



  std::string s;
  int gen_loss;
  double gen_err;
  
  int val_loss;
  double val_err;

  double *gen_error = new double[num_models];
  double *val_error = new double[num_models];

  
  for(size_t i = 0; i <num_models; i++){
    CEnsemble gen_model;
    in >> gen_model;
    // get gen error per iteration
    dvec genpred = gen_model.predict(test_data);
    score.binary_loss(genpred, test_labels, gen_loss, gen_err);
    gen_error[i] = gen_err;
    // get valid error per iteration
    dvec valpred = gen_model.predict(valid_data);
    score.binary_loss(valpred, valid_labels, val_loss, val_err);
    val_error[i] = val_err;
  }
  in.close();

  // apepend validation and generalization error per iter to output file
  of.open(output_file.c_str(),std::ofstream::app);
  if(!of.good()) {
    std::stringstream os;
    os <<"Cannot open data file : " << output_file << std::endl;
		throw std::invalid_argument(os.str());
  }

  of << "val iter ";
  for(size_t i = 0; i < num_models; i++){
    of << val_error[i] << " ";
  }
  of << std::endl;

  of << "generalization iter ";
  for(size_t i = 0; i < num_models; i++){
    of << gen_error[i] << " ";
  }
  of << std::endl;
  of.close();
  
  delete [] gen_error;
  delete [] val_error;

  if(solver) delete solver;  
  if(oracle) delete oracle;
  
  return 0;
}
