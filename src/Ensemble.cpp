
#include "oracles/RawDataOracle.hpp"
#include "oracles/Svm.hpp"
#include "oracles/DecisionStump.hpp"
#include "weak_learners/AbstractWeakLearner.hpp"
#include "weak_learners/DecisionStumpWeakLearner.hpp"

#include "LibSvmReader.hpp"
#include "Ensemble.hpp"

#include "vector_operations.hpp"
#include "parse.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>

namespace totally_corrective_boosting
{

// predict on a single example
double Ensemble::predict(const DenseVector& x) const
{

    double result = 0.0;
    for(wwl_citr it = ensemble.begin(); it != ensemble.end(); it++)
        result += (*it).weighted_predict(x);

    // for(size_t i = 0; i < ensemble.size(); i++){
    //   weighted_wl wwl = ensemble[i];
    //   double pred = wwl.wl->predict(x);
    //   double wt = wwl.get_wt();
    //   result += pred*wt;
    // }
    return result;
}

// predict on a single example
double Ensemble::predict(const SparseVector& x) const
{


    double result = 0.0;
    for(wwl_citr it = ensemble.begin(); it != ensemble.end(); it++)
        result += (*it).weighted_predict(x);

    // for(size_t i = 0; i < ensemble.size(); i++){
    //   weighted_wl wwl = ensemble[i];
    //   double pred = wwl.wl->predict(x);
    //   double wt = wwl.get_wt();
    //   result += pred*wt;
    // }

    return result;
}


DenseVector Ensemble::predict(const std::vector<SparseVector>& data) const
{

    DenseVector result(data[0].dim);

    for(wwl_citr it = ensemble.begin(); it != ensemble.end(); it++)
    {
        DenseVector pred = (*it).weighted_predict(data);
        axpy(1.0, result, pred, result);
    }
    // for(size_t i = 0; i < ensemble.size(); i++){
    //   weighted_wl wwl = ensemble[i];
    //   dvec pred = wwl.wl->predict(Data);
    //   double wt = wwl.get_wt();

    //   for(size_t j = 0; j < Data[0].dim; j++){
    //     result.val[j] += pred.val[j]*wt;
    //   }
    // }

    return result;
}

// // input: wl pointer
// // returns: true if wl is already in the ensemble, else false
// // side effect: if true, then index contains the index of the wl
// bool CEnsemble::find_wl(const CWeakLearner* wl, size_t& idx){
//   size_t tmpidx = 0;
//   for(wwl_itr it = ensemble.begin(); it != ensemble.end(); it++,tmpidx++){
//     bool found = it->wl_equal(wl);
//     if( found ){
//       idx = tmpidx;
//       return true;
//     }
//   }
//   idx = 0;
//   return false;
// }

DenseVector Ensemble::get_weights() const
{
    int size = (int)ensemble.size();
    DenseVector result(size);

    for(int i = 0; i < size; i++)
    {
        result.val[i] = ensemble[i].get_weight();
    }
    return result;
}

void Ensemble::set_weights(const DenseVector& wts)
{
    assert(wts.dim >= ensemble.size());
    wwl_itr it = ensemble.begin();
    double* val = wts.val;
    for(; it != ensemble.end(); it++, val++)
    {
        it->set_weight(*val);
    }
    return;
}

void Ensemble::scale_weights(const double& scale)
{
    for(wwl_itr it = ensemble.begin(); it != ensemble.end(); it++)
    {
        it->scale_weight(scale);
    }
    return;

}

void Ensemble::set_weight(const double& wt, const size_t& idx)
{
    assert(idx < ensemble.size());
    ensemble[idx].set_weight(wt);
    return;
}

void Ensemble::add_weight(const double& wt, const size_t& idx)
{
    assert(idx < ensemble.size());
    ensemble[idx].add_weight(wt);
    return;
}

/// return false if add was successful
/// true if weak learner already exists
/// in that case simply add the wt to the
/// found weak learner
bool Ensemble::add(const WeightedWeakLearner& wwl)
{

    std::vector<WeightedWeakLearner>::iterator it =
        std::find(ensemble.begin(), ensemble.end(), wwl);

    if(it == ensemble.end())
    {
        // did not find it in the ensemble
        ensemble.push_back(wwl);
        return false;
    }

    (*it).add_weight(wwl.get_weight());
    return true;
}

std::ostream& operator << (std::ostream& os, const WeightedWeakLearner& wwl)
{

    os << *wwl.weak_learner;
    os << "weight: " << wwl.weight << std::endl;
    return os;

}

std::istream& operator >> (std::istream& in, WeightedWeakLearner& wwl)
{

    try
    {
        in >> *(const_cast<AbstractWeakLearner *> (wwl.weak_learner));
        expect_keyword(in,"weight:");
        in >> wwl.weight;
    }
    catch (std::string s)
    {
        std::cerr << "Error when reading weighted weak learner: " << s << std::endl;
        exit(1);
    }
    return in;
}

std::ostream& operator << (std::ostream& os, const Ensemble& e)
{
    std::string typ;

    os << "MODEL BEGIN" << std::endl;
    if (e.ensemble.size() > 0)
    {
        typ = e.ensemble[0].get_type();
    }
    else
    {
        typ = "NONE";
    }
    os << "TYPEWL " << typ << std::endl;
    os << "NUMWL " << e.ensemble.size() << std::endl;
    for (wwl_citr it = e.ensemble.begin(); it != e.ensemble.end(); it++)
        os << *it;
    os << "MODEL END" << std::endl;
    return os;

}


std::istream& operator >> (std::istream& in, Ensemble& e)
{
    int num_wls;
    std::string wl_type;
    try
    {
        chomp_input_until(in, "MODEL");
        expect_keyword(in, "BEGIN");
        expect_keyword(in, "TYPEWL");
        wl_type = expect_word(in);
        expect_keyword(in, "NUMWL");
        num_wls = expect_int(in);
        e.ensemble.clear();
        for (int i = 0; i < num_wls; i++)
        {
            if (wl_type == "DSTUMP")
            {
                DecisionStumpWeakLearner *wl = new DecisionStumpWeakLearner;
                WeightedWeakLearner wwl(wl,0.0);
                in >> wwl;
                e.ensemble.push_back(wwl);
            }
            else if (wl_type == "RAWDATA")
            {
                LinearWeakLearner *wl = new LinearWeakLearner;
                WeightedWeakLearner wwl(wl,0.0);
                in >> wwl;
                e.ensemble.push_back(wwl);
            }
            //     else if (wl_type == "SVM") {
            //       CWeakLearnerSVM *wl = new CWeakLearnerSVM;
            // weighted_wl wwl(wl,0.0);
            // in >> wwl;
            // e.ensemble.push_back(wwl);
            //     }
            else
            {
                throw std::string("Unknown weak learner type.");
            }
        }
        expect_keyword(in,"MODEL");
        expect_keyword(in,"END");
    }
    catch (std::string s)
    {
        std::cerr << "Error when reading ensemble: " << s << std::endl;
        exit(1);
    }
    return in;
}

} // end of namespace totally_corrective_boosting
