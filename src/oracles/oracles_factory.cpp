#include "oracles_factory.hpp"

#include "RawDataOracle.hpp"
#include "Svm.hpp"
#include "DecisionStump.hpp"

#include <string>
#include <stdexcept>

namespace totally_corrective_boosting
{

/// Oracles factory
AbstractOracle *new_oracle_instance(const ConfigFile &config,
                                    const std::vector<SparseVector> &data,
                                    const std::vector<int> &labels,
                                    const bool transposed,
                                    std::ostream &log_stream)
{
    bool reflexive = false;
    config.readInto(reflexive, "reflexive");

    std::string oracle_type;
    config.readInto(oracle_type, "oracle_type");

    log_stream << "Using oracle_type == " << oracle_type << std::endl;


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


} // end of namespace totally_corrective_boosting
