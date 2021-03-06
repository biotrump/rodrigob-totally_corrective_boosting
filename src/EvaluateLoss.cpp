
#include "EvaluateLoss.hpp"

namespace totally_corrective_boosting
{


EvaluateLoss::EvaluateLoss():
    tie_breaking(true)
{
    // nothing to do here
    return;
}

EvaluateLoss::EvaluateLoss(bool tie_breaking):
    tie_breaking(tie_breaking)
{
    // nothing to do here
    return;
}


EvaluateLoss::~EvaluateLoss()
{
    // nothing to do here
    return;
}


int EvaluateLoss::binary_loss(const double prediction, const int label) const
{

    int result = 0;
    double tmp_pred = prediction;

    // tie breaking is needed
    if(tmp_pred == 0.0)
    {
        // if tie_breaking==true, tie broken in favor of 1
        if(tie_breaking)
        {
            tmp_pred = 1.0;
        }
        else
        {
            tmp_pred = -1.0;
        }
    }

    if((double)label * tmp_pred < 0.0)
    {
        result = 1;
    }

    return result;

}


void EvaluateLoss::binary_loss(const DenseVector predictions,
                               const std::vector<int>& labels,
                               int& total_loss,
                               double& percent_err) const
{

    int tmp_loss = 0;

    for(size_t i = 0; i < predictions.dim; i++)
    {

        double tmp_pred = predictions.val[i];

        if(tmp_pred == 0.0)
        {
            if(tie_breaking)
            {
                tmp_pred = 1.0;
            }
            else
            {
                tmp_pred = -1.0;
            }
        }

        if((double)labels[i] * tmp_pred < 0.0)
        {
            tmp_loss += 1;
        }

    }

    total_loss = tmp_loss;
    percent_err = ((double)tmp_loss) / ((double)predictions.dim);

    return;
}

} // end of namespace totally_corrective_boosting
