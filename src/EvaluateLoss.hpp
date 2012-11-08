
#ifndef _EVALUATE_HPP_
#define _EVALUATE_HPP_

#include <vector>
#include <iostream>
#include "vector_operations.hpp"

namespace totally_corrective_boosting
{


// evaluates binary loss
class EvaluateLoss
{

    // private:
    // if tie_breaking == true, break ties in favor of +1 class
    // else break ties in favor of -1 class
    bool tie_breaking;

public:

    EvaluateLoss();

    EvaluateLoss(bool tie_breaking);

    ~EvaluateLoss();

    // evaluate loss on single example
    // returns:
    //     0 if prediction has same sign as label
    //     1 otherwise
    int binary_loss(const double pred, const int label) const;

    // evaluate binary loss on all examples
    // side effect:
    // total_loss is the total binary loss:
    //     0 if prediction has same sign as label
    //     1 otherwise
    // percent_err is the error rate
    void binary_loss(const DenseVector pred, const std::vector<int>& labels,
                     int& total_loss, double& percent_err) const;

};

} // end of namespace totally_corrective_boosting

# endif
