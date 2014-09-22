/*!

	\file TestActiveTimer.cpp
	
	\date September 25, 2008
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	
	\brief The source file for the TestActiveTimer class.

*/

#ifndef TEST_ACTIVE_TIMER_CPP_INCLUDED
#define TEST_ACTIVE_TIMER_CPP_INCLUDED

#include "TestActiveTimer.h"
#include <hydrazine/implementation/ArgumentParser.h>
#include <ctime>
#include <vector>

namespace test
{
	
	////////////////////////////////////////////////////////////////////////////
	// TestActiveTimer::Timer
	void TestActiveTimer::Timer::fired()
	{
		timer.stop();
	}
	////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////
	// TestActiveTimer
	bool TestActiveTimer::doTest( )
	{
		typedef std::vector< Timer > Vector;
		Vector timers;
		
		timers.resize( objects );

		unsigned int count = objects - 1;
		
		hydrazine::Timer::Second time = ( count * delayStep/1000.0 )  + 
			( minDelay / 1000.0 );
		
		for( Vector::reverse_iterator fi = timers.rbegin(); 
			fi != timers.rend(); ++fi )
		{
			fi->timer.start();
			fi->start( time );
			status << "Timer " << count-- << " will fire after " << time 
				<< " seconds.\n";
			time -= delayStep/1000.0;
		}
		
		for( Vector::iterator fi = timers.begin(); fi != timers.end(); ++fi )
		{
			fi->wait();
		}
		
		count = 0;
		
		for( Vector::iterator fi = timers.begin(); fi != timers.end(); ++fi )
		{
			if( fi != timers.begin() )
			{
				if( fi->timer.seconds() < time )
				{
					status << "Timer fired after " << fi->timer.seconds() 
						<< " before previous timer which fired at " 
						<< time << "\n";
					return false;
				}
				else
				{
					status << "Timer " << count++ << " fired after " 
					<< fi->timer.seconds() 
						<< " seconds.\n";					
				}
			}
			else
			{
				status << "Timer " << count++ << " fired after " 
					<< fi->timer.seconds() 
					<< " seconds.\n";	
			}
			
			time = fi->timer.seconds();
		}
		
		status << "Test passed.\n";
		return true;
	}
		
	TestActiveTimer::TestActiveTimer()
	{
		name = "TestActiveTimer";
		
		description = "Create a bunch of timers, record when they start, ";
		description += "tell them to fire in order.  Make sure that they ";
		description += "actually do.";
	}
	////////////////////////////////////////////////////////////////////////////
			
}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestActiveTimer test;
	
	parser.description( test.testDescription() );

	parser.parse( "-v", test.verbose, false, 
		"Print info when the test is over." );	
	parser.parse( "-s", test.seed, 0, "Random seed for repeatability." );
	parser.parse( "-t", test.objects, 10, "Number of timers to fire." );
	parser.parse( "-m", test.minDelay, 50, 
		"Min amount of microseconds to wait." );
	parser.parse( "-d", test.delayStep, 50, 
		"Number of microseconds to wait between objects." );
	parser.parse();
	
	test.test();
	return test.passed();
}

#endif

