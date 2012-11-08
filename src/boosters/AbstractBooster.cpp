
#include "AbstractBooster.hpp"

#include <iostream>
#include <cassert>

namespace totally_corrective_boosting
{

AbstractBooster::AbstractBooster(AbstractOracle* &oracle, 
                                 const int& num_pt,
                                 const int& max_iter):
    oracle(oracle), num_data_points(num_pt), max_iterations(max_iter), display_frequency(1){

    assert(oracle);

    // Initialize to uniform distribution
    examples_distribution.dim = num_pt;
    examples_distribution.val = new double[num_pt];
    for(size_t i = 0; i < examples_distribution.dim; i++) examples_distribution.val[i] = (1.0/num_pt);

    return;
}

AbstractBooster::AbstractBooster(AbstractOracle* &oracle, 
                                 const int& num_pt,
                                 const int& max_iter,
                                 const int& disp_freq):
    oracle(oracle), num_data_points(num_pt), max_iterations(max_iter), display_frequency(disp_freq){

    assert(oracle);

    // Initialize to uniform distribution
    examples_distribution.dim = num_pt;
    examples_distribution.val = new double[num_pt];
    for(size_t i = 0; i < examples_distribution.dim; i++) examples_distribution.val[i] = (1.0/num_pt);
    return;
}



AbstractBooster::~AbstractBooster()
{
    // nothing to do here
    return;
}

size_t AbstractBooster::boost(std::ostream& output_stream){
    int i = 0;
    size_t num_models = 0;
    for(i = 0; i < max_iterations; i++)
    {
        timer.start();
        AbstractWeakLearner* new_weak_learner = oracle->find_maximum_edge_weak_learner(examples_distribution);
        update_stopping_criterion(*new_weak_learner);
        if(stopping_criterion(output_stream))
        {
            break;
        }
        update_linear_ensemble(*new_weak_learner);
        update_weights(*new_weak_learner);
        timer.stop();

        std::cout << "Iteration : " << i << std::endl;

        if((i+1)%display_frequency==0){
            output_stream << "Iteration : " << i << std::endl;
            if(display_frequency==1){
                output_stream << "Time for this iteration: "
                              << timer.last_cpu << std::endl << std::endl;
            }
            else
            {
                output_stream << "Cumulative Time: " << timer.total_cpu<< std::endl;
            }
            output_stream << model << std::endl;

            num_models+=1;
            if(examples_distribution.dim < 20)
            {
                output_stream << examples_distribution << std::endl;
            }
        }

    } // end of "for each boosting iteration"


    int num_iter;
    if(i == max_iterations){
        output_stream << "Max iterations exceeded!" << std::endl;
        num_iter = max_iterations;
    }else{
        output_stream << "Finished in " << i << " iterations " << std::endl;
        num_iter = i;
    }

    if(num_iter%display_frequency!=0){
        output_stream << model << std::endl;
        output_stream << "Cumulative Time: " << timer.total_cpu<< std::endl;
        num_models++;
    }

    output_stream << "Total CPU time expended: "
                  << timer.total_cpu << std::endl;
    output_stream << "Maximum iteration time: "
                  << timer.max_cpu << std::endl;
    output_stream << "Minimum iteration time: "
                  << timer.min_cpu << std::endl;
    output_stream << "Average time per iteration: "
                  << timer.avg_cpu() << std::endl;
    //os << model;
    return num_models;
}

} // end of namespace totally_corrective_boosting
