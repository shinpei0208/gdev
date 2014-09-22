/*!	\file LowLevelTimer.cpp
*
*	\brief Source file for the LowLevelTimer class
*
*	Author: Gregory Diamos
*
*
*/

#ifndef LOW_LEVEL_TIMER_CPP_INCLUDED
#define LOW_LEVEL_TIMER_CPP_INCLUDED

#ifdef _WIN32
	#include <windows.h>
#elif __APPLE__
	#include <mach/mach_time.h>
#endif

#include <hydrazine/interface/LowLevelTimer.h>

namespace hydrazine
{

	LowLevelTimer::LowLevelTimer()
	{
		start();
	}

	void LowLevelTimer::start()
	{
		beginning = rdtsc();
	    beginningS = ( beginning + 0.0 ) * 1.0e-9;        
		running = true;
	}

	void LowLevelTimer::stop()
	{
		ending = rdtsc();

		endingS = ( ending + 0.0 ) * 1.0e-9;

		running = false;
	}

	LowLevelTimer::Cycle LowLevelTimer::cycles() const
	{
		if( running )
		{
			return ( rdtsc() - beginning );
		}
		else
		{
			return ( ending - beginning );
		}
	}

	LowLevelTimer::Second LowLevelTimer::seconds() const
	{
        if( running )
        {
            return ( ( ( rdtsc() + 0.0 ) * 1.0e-9 ) - beginningS );
        }
        else
        {
            return endingS - beginningS;
        }
	}

	LowLevelTimer::Second LowLevelTimer::absolute() const
	{
        if( running )
        {
            return ( ( ( rdtsc() + 0.0 ) * 1.0e-9 ) );
        }
        else
        {
            return endingS;
        }
	}
	
	extern "C" 
	{
		LowLevelTimer::Cycle LowLevelTimer::rdtsc()
		{
#ifdef _WIN32
			Cycle cycles = 0;
			Cycle frequency = 0;
			QueryPerformanceFrequency((LARGE_INTEGER*) &frequency);
			QueryPerformanceCounter((LARGE_INTEGER*) &cycles);
			return cycles / frequency;

#elif __APPLE__ 
            uint64_t absolute_time = mach_absolute_time();
            mach_timebase_info_data_t info = {0,0};
            
            if (info.denom == 0) mach_timebase_info(&info);
            uint64_t elapsednano = absolute_time * (info.numer / info.denom);
            
            timespec spec;
            spec.tv_sec  = elapsednano * 1e-9;
            spec.tv_nsec = elapsednano - (spec.tv_sec * 1e9);
			return spec.tv_nsec + (Cycle)spec.tv_sec * 1e9;
#else
			timespec spec;
		
			clock_gettime( CLOCK_REALTIME, &spec );

			return spec.tv_nsec + (Cycle)spec.tv_sec * 1e9;
#endif
		
		}
	}
}

#endif

