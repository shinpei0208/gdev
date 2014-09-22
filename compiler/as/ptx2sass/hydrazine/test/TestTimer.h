/*!
	\file TestTimer.h
	\author Greory Diamos
	\date June 7, 2008
	\brief The header file for the TestTimer function.
*/

#ifndef TEST_TIMER_H_INCLUDED
#define TEST_TIMER_H_INCLUDED

#ifndef DEFAULT_ITERATIONS
#define DEFAULT_ITERATIONS 100000
#endif

#include <hydrazine/implementation/Timer.h>
#include <hydrazine/implementation/ArgumentParser.h>
#include <hydrazine/interface/Test.h>

namespace test
{

	/*!
		\brief A simple function to test the functionality of the Timer
		class. 
		
		Print out the current time.  Start and stop the Timer, print out
		the elapsed time in seconds and cycles.
	*/
	class TestTimer : public Test
	{
		private:
			bool doTest();
	
		public:
			TestTimer();
			unsigned int iterations;
	
	};

}

int main( int argc, char** argv );

#endif

