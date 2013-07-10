/*!	\file LowLevelTimer.h
	\brief Header file for the LowLevelTimer set of functions
	\author Gregory Diamos
	\date : Monday May 4, 2009
*/

#ifndef LOW_LEVEL_TIMER_H_INCLUDED
#define LOW_LEVEL_TIMER_H_INCLUDED

#ifdef HAVE_CONFIG_H
	#include <configure.h>
	
	#ifdef HAVE_FLOAT_H
		#include <float.h>
		#ifndef TIMER_SMALL_SECOND
			#define TIMER_SMALL_SECOND DBL_MIN
		#endif
		#ifndef TIMER_LARGE_SECOND
			#define TIMER_LARGE_SECOND DBL_MAX
		#endif
	#endif	

#endif

#ifndef TIMER_SMALL_SECOND
	#define TIMER_SMALL_SECOND 5e-324
#endif

#ifndef TIMER_LARGE_SECOND
	#define TIMER_LARGE_SECOND 5e324
#endif

#include <ctime>

namespace hydrazine
{

	class LowLevelTimer
	{
		public:
			/*! \brief A type for seconds */
			typedef double Second;
		
			/*! \brief A type for representing clock ticks */			
			typedef long long unsigned Cycle;

		private:
			/*! An integer representing the value of the cycle counter when the 
				last start() function was called
			*/
			Cycle beginning;
			
			/*! An integer representing the value of the cycle counter when the 
				last stop() function was called
			*/
			Cycle ending;
			
			/*! A floating point number representing the value of the system 
				clock when the last start() function was called
			*/
			Second beginningS;

			/*! A floating point number representing the value of the system 
				clock when the last stop() function was called
			*/
			Second endingS;
			
			/*! Read a cycle counter using either assembly or an OS interface.
				\return a 64 bit value representing the current number of 
				clock cycles since the last reset
			*/			
			static Cycle rdtsc();
			
			/*! \brief Is the Timer running? */
			bool running;
			
		public:
		
			/*! \brief The constructor initializes the private variables
				and makes sure that the Timer is not running.
			*/
			LowLevelTimer();
		
			/*! A function that is used to set beginning to the value of 
				the hardware Timer
			*/
			void start();

			/*! A function that is used to set ending to the value of 
				the hardware Timer
			*/			
			void stop();
			
			/*! A function that is used to determine the number of clock cycles 
				between the last time start was called and the last time that 
				end was called.
				\return the difference between ending and beginning
			*/				
			Cycle cycles() const;
			
			/*! A function that is used to determine the number of seconds 
				between the last time start was called and the last time that 
				end was called.
				\return the difference between ending and beginning
			*/				
			Second seconds() const;
			
			/*! \brief Get the absolute number of seconds elapsed since system
					start
				\return That time
			*/
			Second absolute() const;
		
	};

}

#endif

