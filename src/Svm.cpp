/* Copyright (c) 2009, S V N Vishwanathan
 * All rights reserved.
 *
 * The contents of this file are subject to the Mozilla Public License
 * Version 1.1 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS"
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
 * License for the specific language governing rights and limitations
 * under the License.
 *
 * Authors: S V N Vishwanathan
 *
 * Created: (28/03/2009)
 *
 * Last Updated: (09/06/2009)
 */

#include <cassert>
#include <limits>

#include "Svm.hpp"

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

WeakLearner* Svm::max_edge_wl(const DenseVector& dist){

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
    if(reflexive && (min_edge < 0) && (-min_edge > max_edge)){
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

        WeakLearner* wl = new WeakLearner(wt, edge, prediction);
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

    WeakLearner* wl = new WeakLearner(wt, edge, prediction);
    return wl;
}

} // end of namespace totally_corrective_boosting
