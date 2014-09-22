/*!
	\file TestTimer.cpp
	\author Greory Diamos
	\date June 7, 2008
	\brief The source file for the TestTimer function.
*/


#ifndef TEST_TIMER_CPP_INCLUDED
#define TEST_TIMER_CPP_INCLUDED

#include "TestTimer.h"

namespace test
{

	bool TestTimer::doTest()
	{
		hydrazine::Timer Timer;
		status << "Timer starting " << Timer.toString() << ".\n";
		Timer.start();
		
		bool test = false;
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			test = !test;
		}
		
		Timer.stop();

		hydrazine::Timer::Second seconds = Timer.seconds();
		hydrazine::Timer::Cycle cycles = Timer.cycles();
		
		status << "Timer ending " << Timer.toString() << ".\n";

		for( unsigned int i = 0; i < iterations; i++ )
		{
			test = !test;
		}

		bool result;

		if( cycles == Timer.cycles() && seconds == Timer.seconds() ) 
		{
			result = true;
		}
		else
		{
			result = false;
		}
	
		if( test )
		{
			Timer.stop();
		}
		
		return result;	
	}

	TestTimer::TestTimer()
	{
		name = "TestTimer";
		
		description = "A simple test to spin for a number of iterations and ";
		description += "print out the time recorded by a Timer in the spin ";
		description += "loop.";
	}

}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	
	test::TestTimer test;
	parser.description( test.testDescription() );
	
	parser.parse( "-i", test.iterations, DEFAULT_ITERATIONS, 
		"How many iterations to spin around in." );
	parser.parse( "-v", test.verbose, false, 
		"Print out status message when the test is over." );
	parser.parse();

	test.test();

	return test.passed();
}

#endif
