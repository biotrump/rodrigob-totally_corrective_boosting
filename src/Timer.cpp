

#ifndef _TIMER_CPP_
#define _TIMER_CPP_


#include "Timer.hpp"

#include <limits>
#include <stdexcept>
#include <sstream>

// #include <cstdio>

namespace totally_corrective_boosting
{



Timer::Timer()
    :_start_cpu(-1),
     _start_wc(-1),
     num_calls(0),
     total_cpu(0),
     max_cpu(-std::numeric_limits<double>::max()),
     min_cpu(std::numeric_limits<double>::max()),
     total_wc(0),
     max_wc(-std::numeric_limits<double>::max()),
     min_wc(std::numeric_limits<double>::max()) {}

void Timer::start()
{
    struct tms start;
    times(&start);
    _start_cpu = (double(start.tms_utime) + double(start.tms_stime))/TIMES_TICKS_PER_SEC;

    timeval wc;
    gettimeofday(&wc, NULL);
    _start_wc = double(wc.tv_sec) + double(wc.tv_usec)/1e6;
    return;
}


void Timer::stop()
{

    if (_start_cpu < 0)
    {
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

void Timer::reset()
{
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

} // end of namespace totally_corrective_boosting

#endif
