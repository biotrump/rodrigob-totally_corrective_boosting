#include "AbstractWeakLearner.hpp"

namespace totally_corrective_boosting {

AbstractWeakLearner::AbstractWeakLearner()
    : edge(0), prediction()
{
    // nothing to do here
    return;
}

AbstractWeakLearner::AbstractWeakLearner(const double &edge_, const SparseVector &prediction_)
    : edge(edge_), prediction(prediction_)
{
    // nothing to do here
    return;
}


AbstractWeakLearner::~AbstractWeakLearner()
{
    // nothing to do here
    return;
}


std::ostream& operator << (std::ostream& os, const AbstractWeakLearner& wl){
  wl.dump(os);
  return os;
}


std::istream& operator >> (std::istream& in, AbstractWeakLearner& wl){
  wl.load(in);
  return in;
}


bool operator == (const AbstractWeakLearner& wl1, const AbstractWeakLearner& wl2){

  return wl1.equal(&wl2);
}


} // end of namespace totally_corrective_boosting
