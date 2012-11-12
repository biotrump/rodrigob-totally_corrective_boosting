
#include "LibSvmReader.hpp"

#include "oracles/oracles_factory.hpp"

#include "boosters/AbstractBooster.hpp"
#include "boosters/boosters_factory.hpp"

#include "EvaluateLoss.hpp"
#include "parse.hpp"
#include "ConfigFile.hpp"

#include <boost/shared_ptr.hpp>

#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>


#include <iostream>
#include <fstream>

#include <sstream>
#include <stdexcept>


using namespace totally_corrective_boosting;



int main(int argc, char **argv)
{
    
    if(argc != 2)
    {
        std::stringstream os;
        os <<"You need to run this program as: erlpboost name_of_config_file"
          << std::endl
          << "See the config directory for examples of config files"
          << std::endl;
        throw std::invalid_argument(os.str());
    }

    const char * const filename = argv[1];
    std::string config_file = filename;

    ConfigFile config(config_file);

    std::string train_filepath;
    config.readInto(train_filepath, "train_file");

    std::string test_filepath;
    config.readInto(test_filepath, "test_file");

    std::string valid_filepath;
    config.readInto(valid_filepath, "valid_file", std::string("no_valid"));

    std::string log_filepath;
    config.readInto(log_filepath, "output_file");


    std::ofstream log_file_stream;
    log_file_stream.open(log_filepath.c_str());
    if(not log_file_stream.good())
    {
        std::stringstream os;
        os <<"Cannot open log file : " << log_filepath << std::endl;
        throw std::invalid_argument(os.str());
    }

    // Tee code based on http://stackoverflow.com/questions/999120
    typedef boost::iostreams::tee_device<std::ostream, std::ofstream> tee_device_t;
    typedef boost::iostreams::stream<tee_device_t> tee_stream_t;

    tee_device_t log_tee_device(std::cout, log_file_stream);
    tee_stream_t log_stream(log_tee_device);

    // read input data --
    LibSVMReader svm_reader;
    std::vector<SparseVector> data;
    std::vector<int> labels;
    const bool transposed = true;

    if(transposed)
    {
        svm_reader.readlibSVM_transpose(train_filepath, data, labels);
    }
    else
    {
        svm_reader.readlibSVM(train_filepath, data, labels);
    }

    // create oracle and booster
    boost::shared_ptr<AbstractOracle> oracle( new_oracle_instance(config, data, labels, transposed, log_stream) );
    boost::shared_ptr<AbstractBooster> ensemble_booster( new_booster_instance(config, labels, oracle, log_stream) );

    if(not ensemble_booster)
    {
        throw std::runtime_error("Failed to create an ensemble booster. Check your configuration file.");
    }

    // Key call, this is where all the action is happening
    const size_t num_models = ensemble_booster->boost(log_stream);

    Ensemble model = ensemble_booster->get_ensemble();
    // output_stream << "model" << std::endl << model;

    log_stream << std::endl << "-----------------------" << std::endl;

    EvaluateLoss score;

    // get training error --
    {
        DenseVector train_predictions = model.predict(data);
        int train_loss;
        double train_err;
        score.binary_loss(train_predictions, labels, train_loss, train_err);

        log_stream << "Evaluated " << train_predictions.dim << " predictions" << std::endl;
        log_stream << "training error: " << train_err*100 << "% (accuracy " <<  100 - train_err*100 << " %)" << std::endl;
        log_stream << std::endl << "-----------------------" << std::endl;
    }

    // get test error --
    std::vector<SparseVector> test_data;
    std::vector<int> test_labels;
    {
        svm_reader.readlibSVM_transpose(test_filepath, test_data, test_labels);
        // backfill
        while(test_data.size() < data.size())
        {
            SparseVector empty(data[0].dim,1);
            test_data.push_back(empty);
        }

        DenseVector test_predictions = model.predict(test_data);
        int test_loss;
        double test_err;
        score.binary_loss(test_predictions, test_labels, test_loss, test_err);

        log_stream << "test error: " << test_err*100 << "% (accuracy " <<  100 - test_err*100 << " %)" << std::endl;
        log_stream << std::endl << "-----------------------" << std::endl;
    }

    // get validation error --
    std::vector<SparseVector> validation_data;
    std::vector<int> valid_labels;
    {
        if(valid_filepath != "no_valid")
        {
            svm_reader.readlibSVM_transpose(valid_filepath, validation_data, valid_labels);

            // backfill
            while(validation_data.size() < data.size())
            {
                SparseVector empty(data[0].dim,1);
                validation_data.push_back(empty);
            }
            int valid_loss;
            double valid_err;
            const DenseVector validation_predictions = model.predict(validation_data);
            score.binary_loss(validation_predictions,valid_labels, valid_loss, valid_err);

            log_stream << "validation error: " << valid_err*100 << "% (accuracy " <<  100 - valid_err*100 << " %)" << std::endl;
            log_stream << std::endl << "-----------------------" << std::endl;
        }

    }

    // get generalization error per iteration --
    {
        std::ifstream input_stream;
        input_stream.open(log_filepath.c_str());
        if(not input_stream.good())
        {
            std::stringstream os;
            os <<"Cannot open data file : " << log_filepath << std::endl;
            throw std::invalid_argument(os.str());
        }


        std::string s;
        int gen_loss;
        double gen_err;

        int val_loss;
        double val_err;

        std::vector<double> test_data_error(num_models);
        std::vector<double> validation_data_error(num_models);


        for(size_t i = 0; i < num_models; i++)
        {
            Ensemble learned_model;
            input_stream >> learned_model;
            // get gen error per iteration
            DenseVector test_data_prediction = learned_model.predict(test_data);
            score.binary_loss(test_data_prediction, test_labels, gen_loss, gen_err);
            test_data_error[i] = gen_err;

            // get valid error per iteration
            DenseVector validation_data_prediction = learned_model.predict(validation_data);
            score.binary_loss(validation_data_prediction, valid_labels, val_loss, val_err);
            validation_data_error[i] = val_err;
        }
        input_stream.close();

        // append validation and generalization error per iter to output file
        /*log_stream.open(log_filepath.c_str(), std::ofstream::app);
        if(not log_stream.good()) {
            std::stringstream os;
            os <<"Cannot open data file : " << log_filepath << std::endl;
            throw std::invalid_argument(os.str());
        }*/

        log_stream << "validation data error for each N? iterations: ";
        for(size_t i = 0; i < num_models; i++)
        {
            log_stream << validation_data_error[i] << " ";
        }
        log_stream << std::endl;

        log_stream << "test data error for each N? iterations: ";
        for(size_t i = 0; i < num_models; i++)
        {
            log_stream << test_data_error[i] << " ";
        }
        log_stream << std::endl;

    }

    log_stream.close();

    // Reseting the smart pointer will push for memory de-allocation
    oracle.reset();
    ensemble_booster.reset();

    log_stream << "End of game. Have a nice day!" << std::endl;

    return EXIT_SUCCESS;
}
