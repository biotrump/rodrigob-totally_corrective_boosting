
#include "DecisionStumpWeakLearner.hpp"

#include "parse.hpp"
#include "math/vector_operations.hpp"

#include <cassert>
#include <iostream>

namespace totally_corrective_boosting
{

DecisionStumpWeakLearner::DecisionStumpWeakLearner():
    LinearWeakLearner(),
    threshold(0), direction(true), index(0)
{
    // nothing to do here
    return;
}


DecisionStumpWeakLearner::DecisionStumpWeakLearner(
        const SparseVector& wt,
        const double& edge,
        const SparseVector& prediction,
        const double& threshold_,
        const bool& direction_,
        const int& index_)
    : LinearWeakLearner(wt, edge, prediction),
      threshold(threshold_), direction(direction_), index(index_)
{
    // nothing to do here
    return;
}



DecisionStumpWeakLearner::DecisionStumpWeakLearner(const DecisionStumpWeakLearner& other)
    : LinearWeakLearner(other.wt, other.edge, other.prediction),
      threshold(other.threshold),
      direction(other.direction)
{
    // nothing to do here
    return;
}


DecisionStumpWeakLearner::~DecisionStumpWeakLearner()
{
    // nothing to do here
    return;
}


std::string DecisionStumpWeakLearner::get_type() const
{
    return "DSTUMP";
}


// these are the wrong thing to do
double DecisionStumpWeakLearner::predict(const DenseVector& x) const
{
    double tmp = dot(wt,x);
    double result;

    if(direction){
        if( tmp >= threshold){result = 1.0;}
        else {result = -1.0;}
    }
    else{
        if( tmp <= threshold){result = 1.0;}
        else {result = -1.0;}
    }

    return result;
}


double DecisionStumpWeakLearner::predict(const SparseVector& x) const
{

    double tmp = dot(wt,x);
    double result;

    if(direction){
        if( tmp >= threshold){result = 1.0;}
        else {result = -1.0;}
    }
    else{
        if( tmp <= threshold){result = 1.0;}
        else {result = -1.0;}
    }

    return result;
}


DenseVector DecisionStumpWeakLearner::predict(const std::vector<SparseVector>& data) const
{

    DenseVector result(data[index].dim);

    for(size_t i = 0; i < data[index].nnz; i++){
        int tmpindex = data[index].index[i];
        result.val[tmpindex] = data[index].val[i];
    }

    for(size_t i = 0; i < data[index].dim; i++){

        if(direction){
            if( result.val[i] >= threshold){result.val[i] = 1.0;}
            else {result.val[i] = -1.0;}
        }
        else{
            if( result.val[i] <= threshold){result.val[i] = 1.0;}
            else {result.val[i] = -1.0;}
        }

    }

    return result;
}


void DecisionStumpWeakLearner::dump(std::ostream& os) const
{
    os  << wt;
    os << "edge: " << edge << std::endl;
    os << "thresh: " << threshold << std::endl;
    os << "dir: " << direction << std::endl;
    os << "index: " << index << std::endl;
    return;
}


void DecisionStumpWeakLearner::load(std::istream& in)
{
    try {
        in >> wt;
        expect_keyword(in,"edge:");
        in >> edge;
        expect_keyword(in,"thresh:");
        in >> threshold;
        expect_keyword(in,"dir:");
        in >> direction;
        expect_keyword(in,"index:");
        in >> index;
    }
    catch (std::string s) {
        std::cerr << "Error when reading ensemble: " << s << std::endl;
        exit(1);
    }
    return;
}


bool DecisionStumpWeakLearner::equal(const AbstractWeakLearner *other_p) const
{
    const DecisionStumpWeakLearner *wl_p = dynamic_cast<const DecisionStumpWeakLearner *>(other_p);
    return (wl_p != NULL)
            and ( this->threshold == wl_p->get_threshold()
                  and this->direction == wl_p->get_direction()
                  and this->index == wl_p->get_index()
                  and this->wt == wl_p->get_wt());
}


std::ostream& operator << (std::ostream& os, const DecisionStumpWeakLearner& wl){
    wl.dump(os);
    return os;

}


std::istream& operator >> (std::istream& in, DecisionStumpWeakLearner& wl){
    wl.load(in);
    return in;
}


bool operator == (const DecisionStumpWeakLearner& wl1, const DecisionStumpWeakLearner& wl2){

    return wl1.equal(&wl2);
}

} // end of namespace totally_corrective_boosting
