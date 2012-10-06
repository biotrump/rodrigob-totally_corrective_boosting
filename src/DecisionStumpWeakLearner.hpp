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
 * Authors: Karen Glocer, S V N Vishwanathan
 *
 * Created: (20/04/2009)
 *
 * Last Updated: (23/04/2009)
 */
#ifndef _WEAKLEARNERDSTUMP_HPP_
#define _WEAKLEARNERDSTUMP_HPP_

#include "WeakLearner.hpp"

#include "vec.hpp"

namespace totally_corrective_boosting
{

class DecisionStumpWeakLearner: public WeakLearner{

private:


    // threshold
    double thresh;

    // direction of threshold:
    // if dir == true, then x >= thresh
    // else x <= thresh
    bool direction;

    // index of hypothesis for best decision stump
    size_t idx;

public:

    DecisionStumpWeakLearner();

    DecisionStumpWeakLearner(const SparseVector& wt, const double& edge, const SparseVector& prediction,
                             const double& thresh, const bool& direction, const int& idx);

    DecisionStumpWeakLearner(const DecisionStumpWeakLearner& wl);

    ~DecisionStumpWeakLearner(){ }

    std::string get_type() const;

    // Predict on examples
    double predict(const DenseVector& x) const;
    double predict(const SparseVector& x) const;
    // predict on a data matrix
    // assumes it's read in using readlibSVM_transpose
    // i.e. Data must be a vector of hypotheses
    DenseVector   predict(const std::vector<SparseVector>& Data) const;

    // methods to dump and load data
    void dump(std::ostream& os) const;
    void load(std::istream& in);
    bool equal(const WeakLearner *wl) const;

    // accessor methods
    bool get_direction(void) const {return direction; }
    double get_thresh(void) const {return thresh; }
    size_t get_idx(void) const {return idx;}

    friend
    std::ostream& operator << (std::ostream& os, const DecisionStumpWeakLearner& wl);

    friend
    std::istream& operator >> (std::istream& in, DecisionStumpWeakLearner& wl);

    friend
    bool operator == (const DecisionStumpWeakLearner& wl1, const DecisionStumpWeakLearner& wl2);

};

} // end of namespace totally_corrective_boosting

#endif
