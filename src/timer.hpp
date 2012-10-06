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


#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#include <time.h>
#include <sys/time.h>
#include <sys/param.h>
#include <sys/times.h>
#include <sys/types.h>


#if defined(CLK_TCK)
#define TIMES_TICKS_PER_SEC double(CLK_TCK)
#elif defined(_SC_CLK_TCK)
#define TIMES_TICKS_PER_SEC double(sysconf(_SC_CLK_TCK))
#elif defined(HZ)
#define TIMES_TICKS_PER_SEC double(HZ)
#endif

// Keep track of CPU and wall-clock time (in seconds) of program segments

class Timer {

private:
  
  
  double _start_cpu;    // CPU time at start of stopwatch 
  
  double _start_wc;     // wall clock time at start of stopwatch  
  
public:

  int  num_calls;        // number of intervals

  // CPU time 
  double total_cpu;     // total 
  double max_cpu;       // longest recorded interval
  double min_cpu;       // shortest recorded interval
  double last_cpu;      // last recorded interval 

  // Wall clock time 
  double total_wc;     // total 
  double max_wc;       // longest recorded interval
  double min_wc;       // shortest recorded interval
  double last_wc;      // last recorded interval 
  
  Timer();
  virtual ~Timer(){}
  
  void   start();          // start stopwatch
  void   stop();           // stop stopwatch
  void   reset();          // reset 
  
  double avg_cpu() const { return total_cpu/num_calls; }           
  // double CurrentCPUTotal(); // return current cpu_total_time
  
  double avg_wc() const { return total_wc/num_calls; }         
  // double CurrentWallclockTotal();// return current wallclock_total_time
};

#endif
