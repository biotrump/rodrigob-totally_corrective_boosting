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
 * Authors: Karen Glocer 
 *
 * Created: (28/04/2009) 
 *
 * Last Updated: (28/04/2008)   
 */

#include "evaluate.hpp"

namespace totally_corrective_boosting
{


EvaluateLoss::EvaluateLoss():
  tie_breaking(true){}

EvaluateLoss::EvaluateLoss(bool tie_breaking):
  tie_breaking(tie_breaking){}



int EvaluateLoss::binary_loss(const double pred, const int label) const{

  int result = 0;
  double tmp_pred = pred;

  // tie breaking is needed
  if(tmp_pred == 0.0){
    // if tie_breaking==true, tie broken in favor of 1
    if(tie_breaking){tmp_pred = 1.0;}
    else{tmp_pred = -1.0;}
  }

  if((double)label * tmp_pred < 0.0){result = 1;}

  return result;

}


void EvaluateLoss::binary_loss(const DenseVector pred, 
			    const std::vector<int>& labels,
			    int& total_loss,
			    double& percent_err) const{

  int tmp_loss = 0;

  for(size_t i = 0; i < pred.dim; i++){

    double tmp_pred = pred.val[i];

    if(tmp_pred == 0.0){
      if(tie_breaking){tmp_pred = 1.0;}
      else{tmp_pred = -1.0;}
    }

    if((double)labels[i] * tmp_pred < 0.0){tmp_loss += 1;}
    
  }

  total_loss = tmp_loss;
  percent_err = ((double)tmp_loss) / ((double)pred.dim);

  return;
}

} // end of namespace totally_corrective_boosting
