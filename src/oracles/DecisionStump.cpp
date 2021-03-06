
#include "DecisionStump.hpp"

#include "weak_learners/DecisionStumpWeakLearner.hpp"

#include <algorithm>

namespace totally_corrective_boosting
{

DecisionStump::DecisionStump(const std::vector<SparseVector>& data,
                             const std::vector<int>& labels,
                             const bool less_than):
    AbstractOracle(data, labels), less_than(less_than)
{

    // sorted_data is a matrix where each column is the
    // argsort value of the corresponding hypothesis
    // we sort the data upon initialization so we only have to
    // do it once.

    Timer sort_timer;
    std::vector<SparseVector>::const_iterator it;
    sort_timer.start();
    for(it = data.begin(); it != data.end(); ++it)
    {
        DenseIntegerVector tmp = argsort(*it);
        sorted_data.push_back(tmp);
    }
    sort_timer.stop();
    std::cout << "Sorting time: " << sort_timer.last_cpu << std::endl;
    return;
}


DecisionStump::~DecisionStump()
{
    std::cout << "Total time spent in the DecisionStump weak learner (aka the oracle): "
              << timer.total_cpu << " seconds" << std::endl;
    return;
}


AbstractWeakLearner* DecisionStump::find_maximum_edge_weak_learner(const DenseVector& dist)
{

    double best_threshold = 1.0;
    double best_edge = -1.0;
    bool ge;
    size_t max_index = 0;
    size_t size = data.size();
    double init_edge = 0.0;

    timer.start();

    // compute the initial edge for a hypothesis
    // that always predicts 1
    for(size_t i = 0; i < labels.size(); i++)
    {
        init_edge += dist.val[i]*labels[i];
    }

    for(size_t i = 0; i < size; i++){
        double tmp_threshold;
        double tmp_edge;
        bool tmp_ge;

        find_best_threshold(i, dist, init_edge, tmp_threshold, tmp_edge, tmp_ge);

        if(tmp_edge > best_edge){
            best_edge = tmp_edge;
            best_threshold = tmp_threshold;
            ge = tmp_ge;
            max_index = i;
        }
    }

    SparseVector prediction(data[max_index].dim, data[max_index].dim);

    // initially set result to zero
    for(size_t i = 0; i < prediction.dim; i++)
    {
        prediction.val[i] = 0.0;
        prediction.index[i] = i;
    }

    // copy nonzero elements of best hypothesis into result
    for(size_t i = 0; i < data[max_index].nnz; i++)
    {
        size_t index = data[max_index].index[i];
        prediction.val[index] = data[max_index].val[i];
    }

    double edge = 0.0; // just for checking that we're thresholding well
    for(size_t i = 0; i < prediction.dim; i++)
    {

        // threshold prediction
        if((ge and (prediction.val[i] >= best_threshold)) or
                ((not ge) and (prediction.val[i] <= best_threshold)))
        {
            prediction.val[i] = 1.0;
        }
        else
        {
            prediction.val[i] = -1.0;
        }

        // multiply by label
        prediction.val[i] *= labels[i];
        edge += prediction.val[i]*dist.val[i];
    }
    SparseVector wt(size, 1);
    wt.index[0] = max_index;
    wt.val[0] = 1.0;

    //std::cout << "thresh: " << best_threshold << " dir: " << ge;
    //std::cout << " index: " << max_index << " edge: " << edge << std::endl;

    AbstractWeakLearner* wl = new DecisionStumpWeakLearner(wt,
                                                           edge,
                                                           prediction,
                                                           best_threshold,
                                                           ge,
                                                           max_index);

    timer.stop();
    std::cout << "Weak learner time: " << timer.last_cpu << " seconds" << std::endl;
    return wl;
}


void DecisionStump::find_best_threshold_greater_or_equal(const size_t& index,
                                                         const double& dist_diff,
                                                         const DenseVector& dist,
                                                         const double& init_edge,
                                                         double& best_threshold,
                                                         double& best_edge) const
{

    DenseIntegerVector indices = sorted_data[index]; // sorted indices for hyp index
    double max_so_far = init_edge;
    double edgeChunk = 0.0;
    double edge;
    double tmp_threshold;
    double prev;

    if((int)indices.val[0] >= 0)
    {
        tmp_threshold = data[index].val[indices.val[0]];
    }
    else
    {
        tmp_threshold = 0.0;
    }


    edge = max_so_far;
    prev = tmp_threshold;

    for(size_t i = 0; i < indices.dim; i++)
    {
        size_t sparseindex = indices.val[i];
        size_t denseindex = data[index].index[indices.val[i]];
        double tmpdata;

        if((int)sparseindex >=0)
        {
            tmpdata = data[index].val[sparseindex];
        }
        else
        {
            tmpdata = 0.0;
        }

        if(tmpdata != prev)
        {
            edge -= 2*edgeChunk;
            edgeChunk = 0.0;
        }

        if (edge > max_so_far)
        {
            tmp_threshold = tmpdata;
            max_so_far = edge;
        }

        if((int)sparseindex >=0)
        {
            edgeChunk += dist.val[denseindex] * labels[denseindex];
        }
        else
        {
            edgeChunk += dist_diff;
        }

        prev = tmpdata;
    }

    best_threshold = tmp_threshold;
    best_edge = max_so_far;

    return;
}


void DecisionStump::find_best_threshold_less_or_equal(const size_t& index,
                                                      const double& dist_diff,
                                                      const DenseVector& dist,
                                                      const double& init_edge,
                                                      double& best_threshold,
                                                      double& best_edge) const
{

    DenseIntegerVector indices = sorted_data[index]; // sorted indices for hyp index
    double max_so_far = init_edge;
    double edgeChunk = 0.0;
    double edge;
    double tmp_threshold;
    double prev;
    int N = indices.dim-1;

    if((int)indices.val[N] >= 0)
        tmp_threshold = data[index].val[indices.val[N]];
    else
        tmp_threshold = 0.0;

    edge = max_so_far;
    prev = tmp_threshold;

    for(int i = N; i >= 0; i--){
        size_t sparseindex = indices.val[i];
        size_t denseindex = data[index].index[indices.val[i]];
        double tmpdata;

        if((int)sparseindex >=0)
            tmpdata = data[index].val[sparseindex];
        else
            tmpdata = 0.0;

        if(tmpdata != prev){
            edge -= 2*edgeChunk;
            edgeChunk = 0.0;
        }

        if (edge > max_so_far){
            tmp_threshold = tmpdata ;
            max_so_far = edge;
        }

        if((int)sparseindex >=0)
            edgeChunk += dist.val[denseindex] * labels[denseindex];
        else
            edgeChunk += dist_diff;

        prev = tmpdata;
    }

    best_threshold = tmp_threshold;
    best_edge = max_so_far;

    return;
}


void DecisionStump::find_best_threshold(const size_t& index,
                                        const DenseVector& dist,
                                        const double& init_edge,
                                        double& best_threshold,
                                        double& best_edge,
                                        bool& ge) const
{

    double ge_edge;
    double ge_threshold;
    double le_edge;
    double le_threshold;
    double dist_diff;


    dist_diff = init_edge;
    for(size_t i = 0; i < data[index].nnz; i++){
        size_t tmpindex = data[index].index[i];
        //tmplabels[tmpindex] = 0;
        dist_diff -= dist.val[tmpindex] * labels[tmpindex];
    }


    // get the best ge threshold
    find_best_threshold_greater_or_equal(index,dist_diff,dist,init_edge, ge_threshold, ge_edge);

    best_edge = ge_edge;
    best_threshold = ge_threshold;
    ge = true;

    // potentially get the best le threshold
    if(less_than==true){
        find_best_threshold_less_or_equal(index,dist_diff,dist,init_edge, le_threshold, le_edge);
        if( le_edge > ge_edge){
            best_edge = le_edge;
            best_threshold = le_threshold;
            ge = false;
        }
    }

    return;
}


DenseIntegerVector DecisionStump::argsort(SparseVector unsorted)
{

    size_t dim = unsorted.nnz;

    std::vector<std::pair<double,size_t> > pair_vec;

    for(size_t i = 0; i < unsorted.nnz; i++)
    {

        std::pair<double,size_t> tmp(unsorted.val[i],i);
        pair_vec.push_back(tmp);
    }

    if(unsorted.nnz != unsorted.dim)
    {
        std::pair<double,size_t>tmp(0.0,-1);
        pair_vec.push_back(tmp);
        dim++;
    }

    sort(pair_vec.begin(), pair_vec.end());

    // extract the indices into a dvec
    std::vector<std::pair<double,size_t> >::iterator it;

    size_t* resval = new size_t[dim];

    size_t j = 0;
    for(it=pair_vec.begin(); it!=pair_vec.end(); ++it,j++)
    {
        resval[j] = it->second;
    }
    DenseIntegerVector result;
    result.dim = dim;
    result.val = resval;

    return result;
}

} // end of namespace totally_corrective_boosting
