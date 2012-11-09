
#include "RawDataOracle.hpp"

#include "weak_learners/LinearWeakLearner.hpp"

#include "math/vector_operations.hpp"

#include <cassert>
#include <limits>

namespace totally_corrective_boosting
{

RawDataOracle::RawDataOracle(
        const std::vector<SparseVector>& data,
        const std::vector<int>& labels,
        const bool transposed,
        const bool reflexive):
    AbstractOracle(data, labels, transposed), reflexive(reflexive)
{
    // nothing to do here
    return;
}

RawDataOracle::~RawDataOracle()
{
    // nothing to do here
    return;
}

AbstractWeakLearner* RawDataOracle::find_maximum_edge_weak_learner(const DenseVector& dist){

    DenseVector edges;

    assert(labels.size() == (size_t) dist.dim);

    DenseVector dist_labels(dist.dim);
    for(size_t i = 0; i < dist.dim; i++)
    {
        dist_labels.val[i] = dist.val[i]*labels[i];
    }

    if(transposed)
        dot(data, dist_labels, edges);
    else
        transpose_dot(data, dist_labels, edges);

    size_t max_index = 0;
    double max_edge = -std::numeric_limits<double>::max();

    size_t min_index = 0;
    double min_edge = std::numeric_limits<double>::max();

    for(size_t i = 0; i < edges.dim; i++){
        // identify the max
        if(edges.val[i] > max_edge){
            max_edge = edges.val[i];
            max_index = i;
        }
        // identify the min
        if(edges.val[i] < min_edge){
            min_edge = edges.val[i];
            min_index = i;
        }
    }

    if(reflexive and (min_edge < 0) and (-min_edge > max_edge))
    {

        // return the negation of the min_edge feature
        SparseVector wt(edges.dim, 1);

        wt.index[0] = min_index;
        wt.val[0] = -1.0;

        SparseVector prediction;
        if(transposed){
            prediction = data[min_index];
            scale(prediction, -1.0);
        }else
            dot(data, wt, prediction);


        // Multiply the predictions with the labels here
        for(size_t i = 0; i < prediction.nnz; i++)
            prediction.val[i] *= labels[prediction.index[i]];

        double edge = -min_edge;

        // When the weak learner goes out of scope
        // wt and prediction vectors are deleted and the memory is freed

        //return CWeakLearner(wt, edge, prediction);
        AbstractWeakLearner* wl = new LinearWeakLearner(wt, edge, prediction);
        //std::cout << "hyp: " << min_index << std::endl;
        return wl;
    }

    // return the max_edge feature

    SparseVector wt(edges.dim, 1);
    wt.index[0] = max_index;
    wt.val[0] = 1.0;

    SparseVector prediction;
    if(transposed)
        prediction = data[max_index];
    else
        dot(data, wt, prediction);


    // Multiply the predictions with the labels here
    for(size_t i = 0; i < prediction.nnz; i++)
        prediction.val[i] *= labels[prediction.index[i]];

    double edge = max_edge;

    AbstractWeakLearner* wl = new LinearWeakLearner(wt, edge, prediction);
    //std::cout << "hyp: " << min_index << std::endl;
    return wl;
}

} // end of namespace totally_corrective_boosting
