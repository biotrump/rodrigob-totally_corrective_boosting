/* Copyright (c) 2009, Choon Hui Teo and S V N Vishwanathan
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
 * Authors      : Choon Hui Teo (ChoonHui.Teo@anu.edu.au)
 *                S V N Vishwanathan (vishy@stat.purdue.edu)
 * Created      : 08/01/2007
 * Last Updated : 28/04/2009
 */


#ifndef _TIMER_CPP_
#define _TIMER_CPP_

#include <limits>
#include <stdexcept>
#include <sstream>

#include "timer.hpp"

// #include <cstdio>


CTimer::CTimer()
  :_start_cpu(-1),
   _start_wc(-1),
   num_calls(0),
   total_cpu(0),
   max_cpu(-std::numeric_limits<double>::max()),
   min_cpu(std::numeric_limits<double>::max()),
   total_wc(0),
   max_wc(-std::numeric_limits<double>::max()),
   min_wc(std::numeric_limits<double>::max()){}

void CTimer::start(){
  struct tms start;
  times(&start); 
  _start_cpu = (double(start.tms_utime) + double(start.tms_stime))/TIMES_TICKS_PER_SEC;
  
  timeval wc;
  gettimeofday(&wc, NULL);
  _start_wc = double(wc.tv_sec) + double(wc.tv_usec)/1e6;
  return;
}


void CTimer::stop(){
  
  if (_start_cpu < 0){
    std::stringstream os;
    os << "Need to call start() before calling stop()" << std::endl;
		throw std::runtime_error(os.str());
  }
  
  struct tms end;
  times(&end);
  last_cpu = (double(end.tms_utime) + double(end.tms_stime))/TIMES_TICKS_PER_SEC - _start_cpu;
  
  _start_cpu = -1;
  total_cpu += last_cpu; 
  max_cpu = std::max(max_cpu, last_cpu);
  min_cpu = std::min(min_cpu, last_cpu);
  
  timeval wc;
  gettimeofday(&wc, NULL);
  last_wc = (double(wc.tv_sec) + double(wc.tv_usec)/1e6) - _start_wc;
  
  _start_wc = -1;
  total_wc += last_wc;
  max_wc = std::max(max_wc, last_wc);
  min_wc = std::min(min_wc, last_wc);
  
  num_calls++;
  return;
}

void CTimer::reset(){
  num_calls = 0;
  _start_cpu = -1;
  _start_wc = -1;
  total_cpu = 0;
  max_cpu = -std::numeric_limits<double>::max();
  min_cpu = std::numeric_limits<double>::max();
  total_wc = 0;
  max_wc = -std::numeric_limits<double>::max();
  min_wc = std::numeric_limits<double>::max();
  return; 
}

#endif
