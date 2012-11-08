#include "Svm.hpp"

#include "math/vector_operations.hpp"

#include "weak_learners/LinearWeakLearner.hpp"

#include <cassert>
#include <limits>

namespace totally_corrective_boosting
{


Svm::Svm(std::vector<SparseVector>& data, 
         std::vector<int>& labels,
         const bool& transposed,
         const bool& reflexive):
    AbstractOracle(data, labels, transposed), reflexive(reflexive){}

Svm::~Svm()
{
    // nothing to do here
    return;
}


AbstractWeakLearner* Svm::find_maximum_edge_weak_learner(const DenseVector& dist){

    DenseVector edges;

    assert(labels.size() == (size_t) dist.dim);

    DenseVector dist_labels(dist.dim);
    for(size_t i = 0; i < dist.dim; i++)
        dist_labels.val[i] = dist.val[i]*labels[i];

    // Compute X^{\top} X dist_labels
    if(transposed){
        DenseVector tmp_edges;
        dot(data, dist_labels, tmp_edges);
        transpose_dot(data, tmp_edges, edges);
    }else{
        DenseVector tmp_edges;
        transpose_dot(data, dist_labels, tmp_edges);
        dot(data, tmp_edges, edges);
    }

    // std::cout << "edges: " << edges << std::endl;
    size_t max_idx = 0;
    double max_edge = -std::numeric_limits<double>::max();

    size_t min_idx = 0;
    double min_edge = std::numeric_limits<double>::max();

    for(size_t i = 0; i < edges.dim; i++){
        // identify the max
        if(edges.val[i] > max_edge){
            max_edge = edges.val[i];
            max_idx = i;
        }
        // identify the min
        if(edges.val[i] < min_edge){
            min_edge = edges.val[i];
            min_idx = i;
        }
    }

    // std::cout << min_edge << "  " << min_idx << "  " << max_edge << " " << max_idx << std::endl;
    if(reflexive and (min_edge < 0) and (-min_edge > max_edge)){
        // wt = -x of the point with the max edge
        SparseVector wt;
        SparseVector prediction;
        if(transposed){
            SparseVector tmp(edges.dim, 1);
            tmp.idx[0] = min_idx;
            tmp.val[0] = -1.0;
            dot(data, tmp, wt);
            transpose_dot(data, wt, prediction);
        }else{
            wt = data[min_idx];
            scale(wt, -1.0);
            dot(data, wt, prediction);
        }
        
        // Multiply the predictions with the labels here
        for(size_t i = 0; i < prediction.nnz; i++)
            prediction.val[i] *= labels[prediction.idx[i]];

        double edge = -min_edge;
        // std::cout << "reflexive : " << min_idx << "  " << edge << std::endl;

        AbstractWeakLearner* wl = new LinearWeakLearner(wt, edge, prediction);
        return wl;
    }

    // return the max_edge feature

    // wt = x of the point with the max edge
    SparseVector wt;
    SparseVector prediction;
    if(transposed){
        SparseVector tmp(edges.dim, 1);
        tmp.idx[0] = max_idx;
        tmp.val[0] = 1.0;
        dot(data, tmp, wt);
        transpose_dot(data, wt, prediction);
    }else{
        wt = data[max_idx];
        dot(data, wt, prediction);
    }

    // Multiply the predictions with the labels here
    for(size_t i = 0; i < prediction.nnz; i++)
        prediction.val[i] *= labels[prediction.idx[i]];

    double edge = max_edge;
    // std::cout << "non reflexive : " << max_idx << "  " << edge << std::endl;

    AbstractWeakLearner* wl = new LinearWeakLearner(wt, edge, prediction);
    return wl;
}

} // end of namespace totally_corrective_boosting
