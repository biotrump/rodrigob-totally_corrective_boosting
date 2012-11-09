
#include "AbstractBooster.hpp"

#include <iostream>
#include <cassert>

namespace totally_corrective_boosting
{


AbstractBooster::AbstractBooster(AbstractOracle* oracle_,
                                 const int num_data_points_,
                                 const int max_iterations_)
    : oracle(oracle_), num_data_points(num_data_points_),
      max_iterations(max_iterations_), display_frequency(10)
{

    assert(oracle);

    // Initialize to uniform distribution
    examples_distribution.dim = num_data_points_;
    examples_distribution.val = new double[num_data_points_];
    for(size_t i = 0; i < examples_distribution.dim; i++)
    {
        examples_distribution.val[i] = (1.0/num_data_points_);
    }

    return;
}


AbstractBooster::AbstractBooster(AbstractOracle* oracle_,
                                 const int num_data_points_,
                                 const int max_iterations_,
                                 const int display_frequency_)
    : oracle(oracle_), num_data_points(num_data_points_),
      max_iterations(max_iterations_), display_frequency(display_frequency_)
{

    assert(oracle);

    // Initialize to uniform distribution
    examples_distribution.dim = num_data_points_;
    examples_distribution.val = new double[num_data_points_];
    for(size_t i = 0; i < examples_distribution.dim; i++)
    {
        examples_distribution.val[i] = (1.0/num_data_points_);
    }

    return;
}



AbstractBooster::~AbstractBooster()
{
    // nothing to do here
    return;
}


size_t AbstractBooster::boost(std::ostream& log_stream)
{
    int i = 0;
    size_t num_models = 0;
    for(i = 0; i < max_iterations; i++)
    {
        timer.start();
        AbstractWeakLearner* new_weak_learner = oracle->find_maximum_edge_weak_learner(examples_distribution);
        update_stopping_criterion(*new_weak_learner);
        if(stopping_criterion(log_stream))
        {
            break;
        }
        update_linear_ensemble(*new_weak_learner);
        update_weights(*new_weak_learner);
        timer.stop();

        if(((i+1)%display_frequency)==0)
        {
            log_stream << "Iteration : " << i << std::endl;
            if(display_frequency==1)
            {
                log_stream << "Time for this iteration: "
                           << timer.last_cpu << " seconds" << std::endl << std::endl;
            }
            else
            {
                log_stream << "Cumulative time: " << timer.total_cpu << " seconds" << std::endl;
            }
            log_stream << model << std::endl;

            num_models+=1;
            if(examples_distribution.dim < 20)
            {
                log_stream << "Examples distribution: " << examples_distribution << std::endl;
            }
        }

    } // end of "for each boosting iteration"


    int num_iter;
    if(i == max_iterations)
    {
        log_stream << "Max iterations exceeded!" << std::endl;
        num_iter = max_iterations;
    }
    else
    {
        log_stream << "Finished in " << i << " iterations " << std::endl;
        num_iter = i;
    }

    if( (num_iter%display_frequency) !=0)
    {
        log_stream << model << std::endl;
        log_stream << "Cumulative Time: " << timer.total_cpu<< std::endl;
        num_models++;
    }


    log_stream << "Total CPU time expended: "
                  << timer.total_cpu << " seconds" << std::endl;
    log_stream << "Maximum iteration time: "
                  << timer.max_cpu << " seconds" << std::endl;
    log_stream << "Minimum iteration time: "
                  << timer.min_cpu << " seconds" << std::endl;
    log_stream << "Average time per iteration: "
                  << timer.average_cpu() << " seconds" << std::endl;
    //os << model;
    return num_models;
}


const Ensemble &AbstractBooster::get_ensemble() const
{
    return model;
}

} // end of namespace totally_corrective_boosting
