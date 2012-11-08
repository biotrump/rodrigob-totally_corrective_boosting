#include "LinearWeakLearner.hpp"

#include "parse.hpp"

#include "vector_operations.hpp"

namespace totally_corrective_boosting {


LinearWeakLearner::LinearWeakLearner()
    : AbstractWeakLearner(), wt()
{
    // nothing to do here
    return;
}


LinearWeakLearner::LinearWeakLearner(const SparseVector& wt,
                                     const double& edge,
                                     const SparseVector& prediction)
    : AbstractWeakLearner(edge, prediction),
      wt(wt)
{
    // nothing to do here
    return;
}


LinearWeakLearner::LinearWeakLearner(const LinearWeakLearner& wl)
    : AbstractWeakLearner(wl.edge, wl.prediction),
      wt(wl.wt)
{
    // nothing to do here
    return;
}

LinearWeakLearner::~LinearWeakLearner()
{
    // nothing to do here
    return;
}

double LinearWeakLearner::predict(const DenseVector& x) const{
    return dot(wt, x);
}

double LinearWeakLearner::predict(const SparseVector& x) const{
    return dot(wt, x);
}

DenseVector LinearWeakLearner::predict(const std::vector<SparseVector>& Data) const{


    DenseVector result(Data[0].dim);

    if(wt.nnz == 1){
        int idx = wt.idx[0];
        for(size_t i = 0; i < Data[idx].nnz; i++){
            int tmpidx = Data[idx].idx[i];
            result.val[tmpidx] = wt.val[0]*Data[idx].val[i];
        }
    }
    else{
        SparseVector tmp;

        transpose_dot(Data,wt,tmp);
        for(size_t i = 0; i < tmp.nnz; i++){
            result.val[i] = tmp.val[i];
        }
    }
    return result;
}

std::string LinearWeakLearner::get_type() const {
    return "RAWDATA";
}

void LinearWeakLearner::dump(std::ostream& os) const{

    os << wt;
    os << "Edge: " << edge << std::endl;
    return;
}

void LinearWeakLearner::load(std::istream& in){
    try {
        in >> wt;
        expect_keyword(in, "Edge:");
        in >> edge;
    }
    catch (std::string s) {
        std::cerr << "Error when reading raw data weak learner: " << s << std::endl;
        exit(1);
    }
    return;
}


bool LinearWeakLearner::equal(const AbstractWeakLearner *other_p) const
{
    const LinearWeakLearner *wl_p = dynamic_cast<const LinearWeakLearner *>(other_p);
    return (wl_p != NULL) and (this->wt == wl_p->get_wt());
}


} // end of namespace totally_corrective_boosting
