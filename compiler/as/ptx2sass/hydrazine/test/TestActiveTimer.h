/*!

	\file TestActiveTimer.h
	
	\date September 25, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	
	\brief The header file for the TestActiveTimer class.

*/

#ifndef TEST_ACTIVE_TIMER_H_INCLUDED
#define TEST_ACTIVE_TIMER_H_INCLUDED

#include <hydrazine/interface/Test.h>
#include <hydrazine/interface/ActiveTimer.h>
#include <boost/random/mersenne_twister.hpp>

namespace test
{
	/*	\brief A unit test for asynchronous timers 
		
		Test Point 1: Launch several timers, each set to fire at a specific 
			time.  Make sure that they fire in order of increasing time.
	*/
	class TestActiveTimer : public Test
	{
		private:
			class Timer : public hydrazine::ActiveTimer
			{
				public:
					hydrazine::Timer timer;
					void fired();
			
			};

		private:
			bool doTest( );
		
		public:
			TestActiveTimer();
			
			unsigned int objects;
			hydrazine::Timer::Second delayStep;
			hydrazine::Timer::Second minDelay;
	
	};

}

int main( int argc, char** argv );

#endif

