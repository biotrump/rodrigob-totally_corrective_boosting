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


#include "LibSvmReader.hpp"
#include "oracles/RawDataOracle.hpp"
#include "oracles/DecisionStump.hpp"
#include "oracles/Svm.hpp"
#include "boosters/ErlpBoost.hpp"
#include "boosters/tKlBoost.hpp"
#include "boosters/AdaBoost.hpp"
#include "boosters/CorrectiveBoost.hpp"

#include "optimizers/ProjectedGradientOptimizer.hpp"
#include "optimizers/ZhangdAndHagerOptimizer.hpp"
#include "optimizers/LbfgsbOptimizer.hpp"
#include "optimizers/CoordinateDescentOptimizer.hpp"

#include "EvaluateLoss.hpp"
#include "parse.hpp"
#include "ConfigFile.hpp"

#ifdef USE_TAO
#include "optimizers/TaoOptimizer.hpp"
#endif

#ifdef USE_CLP
#include "boosters/LpBoost.hpp"
#endif

#include <iostream>
#include <fstream>

#include <sstream>
#include <stdexcept>


using namespace totally_corrective_boosting;

/// Oracles factory
AbstractOracle *new_oracle_instance(const ConfigFile &config,
                                    const std::vector<SparseVector> &data,
                                    const std::vector<int> &labels,
                                    const bool transposed)
{
    bool reflexive;
    config.readInto(reflexive, "reflexive");

    std::string oracle_type;
    config.readInto(oracle_type, "oracle_type");

    AbstractOracle* oracle = NULL;

    if(oracle_type == "rawdata")
    {
        oracle = new RawDataOracle(data, labels, transposed, reflexive);
    }
    else if(oracle_type == "svm")
    {
        oracle = new Svm(data, labels, transposed, reflexive);
    }
    else if(oracle_type == "decisionstump")
    {
        oracle = new DecisionStump(data, labels, reflexive);
    }
    else
    {
        printf("oracle_type == %s\n", oracle_type.c_str());
        throw std::runtime_error("Received an unknown value for oracle_type");
    }


    return oracle;
}


/// Optimizers factory
AbstractOptimizer* new_optimizer_instance(
        const ConfigFile &config,
        const size_t labels_size,
        const bool transposed,
        const double eta, const double nu, const double epsilon,
        const bool binary)
{


    std::string optimizer_type;
    config.readInto(optimizer_type, "optimizer_type", std::string("lbfgsb"));

    printf("Using optimizer_type == %s\n", optimizer_type.c_str());

    AbstractOptimizer* optimizer = NULL;

    if(optimizer_type == "tao")
    {
#ifdef USE_TAO
        solver = new TaoOptimizer(labels_size, transposed, eta, nu, eps, binary, argc, argv);
#else
        std::stringstream os;
        os << "You must compile with TAO support enabled to use TAO"
           << std::endl << "See readme.text for details" << std::endl;
        throw std::invalid_argument(os.str());
#endif
    } else if(optimizer_type == "pg")
    {
        optimizer = new ProjectedGradientOptimizer(labels_size, transposed, eta, nu, epsilon, binary);
    }
    else if(optimizer_type == "hz")
    {
        optimizer = new ZhangdAndHagerOptimizer(labels_size, transposed, eta, nu, epsilon, binary);
    }
    else if(optimizer_type == "lbfgsb")
    {
        optimizer = new LbfgsbOptimizer(labels_size, transposed, eta, nu, epsilon, binary);
    }
    else if(optimizer_type == "cd")
    {
        optimizer = new CoordinateDescentOptimizer(labels_size, transposed, eta, nu, epsilon, binary);
    }
    else
    {
        throw std::runtime_error("Received an unknown value for optimizer_type");
    }

    return optimizer;
}


int main(int argc, char **argv)
{
    
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



    int max_iter = 0;
    config.readInto(max_iter, "max_iter");

    std::ofstream output_stream;
    output_stream.open(output_file.c_str());
    if(!output_stream.good())
    {
        std::stringstream os;
        os <<"Cannot open data file : " << output_file << std::endl;
        throw std::invalid_argument(os.str());
    }

    // read input data --
    LibSVMReader svm_reader;
    std::vector<SparseVector> data;
    std::vector<int> labels;
    const bool transposed = true;

    if(transposed)
    {
        svm_reader.readlibSVM_transpose(train_file, data, labels);
    }
    else
    {
        svm_reader.readlibSVM(train_file, data, labels);
    }

    // create oracle --
    AbstractOracle * const oracle = new_oracle_instance(config, data, labels, transposed);


    // create booster --
    std::string booster_type;
    config.readInto(booster_type, "booster_type", std::string("ERLPBoost"));

    AbstractOptimizer* solver = NULL;
    AbstractBooster* ensemble_booster = NULL;


    const bool
            is_kl_boost = (booster_type == "ERLPBoost"
                           or booster_type == "ErlpBoost"
                           or booster_type == "KlBoost"),
            is_tKl_boost = (booster_type == "tKlBoost");

    if(is_kl_boost or is_tKl_boost)
    {
        double epsilon = 0;
        config.readInto(epsilon, "eps", 0.001);

        double nu = 0;
        config.readInto(nu, "nu", 1.0);

        bool binary = false;
        config.readInto(binary, "binary", false);


        bool transposed = true;
        double eta = 0.0;

        if(binary)
        {
            config.readInto(eta, "eta", 2.0*(1.0+log(labels.size()/nu))/epsilon);
        }
        else
        {
            config.readInto(eta, "eta", 2.0*log(labels.size()/nu)/epsilon);
        }


        double capital_dee = 1/7.0;
        config.readInto(capital_dee, "D", 1/7.0);

        if(is_tKl_boost)
        {
            nu = tKlBoost::compute_nu(capital_dee);
            eta = tKlBoost::compute_eta(labels.size(), epsilon, capital_dee);
        }

        output_stream << "Maximum Iterations: " << max_iter << std::endl;
        output_stream << "Epsilon (Tolerance): " << epsilon << std::endl;
        output_stream << "1/Nu (softening): " << 1.0/nu << std::endl << std::endl;
        output_stream << "eta: " << eta << std::endl << std::endl;

        solver = new_optimizer_instance(config,
                                                                 labels.size(),
                                                                 transposed,
                                                                 eta, nu, epsilon,
                                                                 binary);

        ensemble_booster = new ErlpBoost(oracle, labels.size(), max_iter, epsilon, eta, nu, binary, solver);
    }
    else if(booster_type == "LPBoost")
    {
        std::cout << "running lpboost" << std::endl;
        double eps = 0;
        config.readInto(eps, "eps", 0.001);

        double nu = 0;
        config.readInto(nu, "nu", 1.0);
#ifdef USE_CLP
        eb = new LpBoost(oracle, labels.size(), max_iter, eps, nu);
#else
        std::stringstream os;
        os << "You must compile with COIN-OR LP solver support enabled to use LPBoost"
           << std::endl << "See readme.text for details" << std::endl;
        throw std::invalid_argument(os.str());
#endif
    }
    else if(booster_type == "AdaBoost")
    {
        std::cout << "running adaboost" << std::endl;
        ensemble_booster = new AdaBoost(oracle, labels.size(), max_iter,250);
    }
    else if(booster_type == "Corrective")
    {

        double eps = 0;
        config.readInto(eps, "eps", 0.001);

        double nu = 0;
        config.readInto(nu, "nu", 1.0);

        bool linesearch;
        config.readInto(linesearch, "linesearch", false);

        double eta = 0.0;
        config.readInto(eta, "eta", 2.0*log(labels.size()/nu)/eps);

        ensemble_booster = new CorrectiveBoost(oracle, labels.size(), max_iter, eps, eta, nu, linesearch,10);
    }
    else
    {
        printf("booster_type == %s\n", booster_type.c_str());
        throw std::runtime_error("Received an unknown value for booster_type");
    }

    assert(ensemble_booster);

    size_t num_models = ensemble_booster->boost(output_stream);

    Ensemble model = ensemble_booster->get_ensemble();
    // output_stream << "model" << std::endl << model;

    std::cout << std::endl << "-----------------------" << std::endl;
    output_stream << std::endl << "-----------------------" << std::endl;

    EvaluateLoss score;

    // get training error
    DenseVector trainpred = model.predict(data);
    int train_loss;
    double train_err;
    score.binary_loss(trainpred, labels, train_loss, train_err);

    std::cout << "training error: " << train_err*100 << "%" << std::endl;
    std::cout << std::endl << "-----------------------" << std::endl;

    output_stream << "training error: " << train_err*100 << "%" << std::endl;
    output_stream << std::endl << "-----------------------" << std::endl;

    // get test error
    std::vector<SparseVector> test_data;
    std::vector<int> test_labels;
    svm_reader.readlibSVM_transpose(test_file, test_data, test_labels);
    // backfill
    while(test_data.size() < data.size()){
        SparseVector empty(data[0].dim,1);
        test_data.push_back(empty);
    }

    DenseVector testpred = model.predict(test_data);
    int test_loss;
    double test_err;
    score.binary_loss(testpred, test_labels, test_loss, test_err);

    std::cout << "test error: " << test_err*100 << "%" << std::endl;
    std::cout << std::endl << "-----------------------" << std::endl;

    output_stream << "test error: " << test_err*100 << "%" << std::endl;
    output_stream << std::endl << "-----------------------" << std::endl;

    // get validation error
    std::vector<SparseVector> valid_data;
    std::vector<int> valid_labels;
    if(valid_file != "no_valid")
    {
        svm_reader.readlibSVM_transpose(valid_file, valid_data, valid_labels);

        // backfill
        while(valid_data.size() < data.size())
        {
            SparseVector empty(data[0].dim,1);
            valid_data.push_back(empty);
        }
        int valid_loss;
        double valid_err;
        DenseVector validpred = model.predict(valid_data);
        score.binary_loss(validpred,valid_labels, valid_loss, valid_err);

        std::cout << "validation error: " << valid_err*100 << "%" << std::endl;
        std::cout << std::endl << "-----------------------" << std::endl;

        output_stream << "validation error: " << valid_err*100 << "%" << std::endl;
        output_stream << std::endl << "-----------------------" << std::endl;
    }
    output_stream.close();

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


    for(size_t i = 0; i <num_models; i++)
    {
        Ensemble gen_model;
        in >> gen_model;
        // get gen error per iteration
        DenseVector genpred = gen_model.predict(test_data);
        score.binary_loss(genpred, test_labels, gen_loss, gen_err);
        gen_error[i] = gen_err;
        // get valid error per iteration
        DenseVector valpred = gen_model.predict(valid_data);
        score.binary_loss(valpred, valid_labels, val_loss, val_err);
        val_error[i] = val_err;
    }
    in.close();

    // apepend validation and generalization error per iter to output file
    output_stream.open(output_file.c_str(),std::ofstream::app);
    if(!output_stream.good()) {
        std::stringstream os;
        os <<"Cannot open data file : " << output_file << std::endl;
        throw std::invalid_argument(os.str());
    }

    output_stream << "val iter ";
    for(size_t i = 0; i < num_models; i++){
        output_stream << val_error[i] << " ";
    }
    output_stream << std::endl;

    output_stream << "generalization iter ";
    for(size_t i = 0; i < num_models; i++){
        output_stream << gen_error[i] << " ";
    }
    output_stream << std::endl;
    output_stream.close();

    delete [] gen_error;
    delete [] val_error;

    // FIXME should use smart pointers
    if(solver)
    {
        delete solver;
    }

    if(oracle)
    {
        delete oracle;
    }

    return 0;
}
