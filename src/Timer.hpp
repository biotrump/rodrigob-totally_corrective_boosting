

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

namespace totally_corrective_boosting
{

/// Keep track of CPU and wall-clock time (in seconds) of program segments
class Timer
{

protected:

    /// CPU time at start of stopwatch
    double _start_cpu;

    /// wall clock time at start of stopwatch
    double _start_wall_clock;

public:

    int  num_calls;        // number of intervals

    // CPU time
    double total_cpu;     // total
    double max_cpu;       // longest recorded interval
    double min_cpu;       // shortest recorded interval
    double last_cpu;      // last recorded interval

    // Wall clock time
    double total_wall_clock;     // total
    double max_wall_clock;       // longest recorded interval
    double min_wall_clock;       // shortest recorded interval
    double last_wall_clock;      // last recorded interval

    Timer();
    virtual ~Timer();

    void   start();          // start stopwatch
    void   stop();           // stop stopwatch
    void   reset();          // reset

    double average_cpu() const
    {
        return total_cpu/num_calls;
    }
    // double CurrentCPUTotal(); // return current cpu_total_time

    double average_wall_clock() const
    {
        return total_wall_clock/num_calls;
    }
    // double CurrentWallclockTotal();// return current wallclock_total_time
};

} // end of namespace totally_corrective_boosting

#endif
