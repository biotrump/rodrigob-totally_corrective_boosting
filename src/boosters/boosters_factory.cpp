#include "boosters_factory.hpp"

#include "boosters/ErlpBoost.hpp"
#include "boosters/tKlBoost.hpp"
#include "boosters/AdaBoost.hpp"
#include "boosters/CorrectiveBoost.hpp"

#ifdef USE_CLP
#include "boosters/LpBoost.hpp"
#endif

#include "optimizers/optimizers_factory.hpp"

#include <cmath>
#include <stdexcept>

namespace totally_corrective_boosting
{

AbstractBooster *new_booster_instance(const ConfigFile &config,
                                      const std::vector<int> &labels,
                                      const boost::shared_ptr<AbstractOracle> &oracle,
                                      std::ostream &log_stream)
{

    if(not oracle)
    {
        throw std::runtime_error("new_booster_instance requires an initialized oracle");
    }

    std::string booster_type;
    config.readInto(booster_type, "booster_type", std::string("ERLPBoost"));

    log_stream << "Using booster_type == " << booster_type << std::endl;


    int max_iterations = 0;
    config.readInto(max_iterations, "max_iter");

    AbstractBooster *ensemble_booster = NULL;



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
            config.readInto(eta, "eta", 2.0*(1.0+std::log(labels.size()/nu))/epsilon);
        }
        else
        {
            config.readInto(eta, "eta", 2.0*std::log(labels.size()/nu)/epsilon);
        }


        double capital_dee = 1/7.0;
        config.readInto(capital_dee, "D", 1/7.0);

        if(is_tKl_boost)
        {
            nu = tKlBoost::compute_nu(capital_dee);
            eta = tKlBoost::compute_eta(labels.size(), epsilon, capital_dee);
        }

        log_stream << "Maximum Iterations: " << max_iterations << std::endl;
        log_stream << "Epsilon (Tolerance): " << epsilon << std::endl;
        log_stream << "1/Nu (softening): " << 1.0/nu << std::endl << std::endl;
        log_stream << "eta: " << eta << std::endl << std::endl;

        boost::shared_ptr<AbstractOptimizer>
                solver( new_optimizer_instance(config,
                                               labels.size(), transposed,
                                               eta, nu, epsilon,
                                               binary,
                                               log_stream) );

        if(is_kl_boost)
        {
            std::cout << "Running Erlpboost (also known as KlBoost)" << std::endl;
            ensemble_booster = new ErlpBoost(oracle, labels.size(), max_iterations, epsilon, eta, nu, binary, solver);
        }
        else if(is_tKl_boost)
        {
            ensemble_booster = new tKlBoost(oracle, labels.size(), max_iterations, epsilon, capital_dee, binary, solver);
        }
        else
        {
            throw std::runtime_error("This should never happen.");
        }

    }
    else if(booster_type == "LPBoost")
    {
        std::cout << "Running lpboost" << std::endl;
        double epsilon = 0;
        config.readInto(epsilon, "eps", 0.001);

        double nu = 0;
        config.readInto(nu, "nu", 1.0);
#ifdef USE_CLP
        ensemble_booster = new LpBoost(oracle, labels.size(), max_iterations, epsilon, nu);
#else
        std::stringstream os;
        os << "You must compile with COIN-OR LP solver support enabled to use LPBoost"
           << std::endl << "See readme.text for details" << std::endl;
        throw std::invalid_argument(os.str());
#endif
    }
    else if(booster_type == "AdaBoost")
    {
        std::cout << "Running adaboost" << std::endl;
        ensemble_booster = new AdaBoost(oracle, labels.size(), max_iterations, 250);
    }
    else if(booster_type == "Corrective")
    {

        double epsilon = 0;
        config.readInto(epsilon, "eps", 0.001);

        double nu = 0;
        config.readInto(nu, "nu", 1.0);

        bool linesearch;
        config.readInto(linesearch, "linesearch", false);

        double eta = 0.0;
        config.readInto(eta, "eta", 2.0*log(labels.size()/nu)/epsilon);

        ensemble_booster = new CorrectiveBoost(oracle, labels.size(), max_iterations, epsilon, eta, nu, linesearch,10);
    }
    else
    {
        printf("booster_type == %s\n", booster_type.c_str());
        throw std::runtime_error("Received an unknown value for booster_type");
    }

    return ensemble_booster;
}

} // end of namespace totally_corrective_boosting
