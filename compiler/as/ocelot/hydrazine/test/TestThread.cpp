#ifndef TEST_THREAD_CPP_INCLUDED
#define TEST_THREAD_CPP_INCLUDED

#include "TestThread.h"
#include <hydrazine/implementation/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace test
{
	void RingThread::execute()
	{
		Map* touches;
		
		report( "Starting ring thread " << id() << "." );
		
		if( id() == source )
		{
			report( "Thread " << id() << " is the source, waiting for " 
				<< "touch map from controller." );
	
			threadReceive( touches, THREAD_CONTROLLER_ID );

			report( "Thread " << id() << " got the touch map." );
			report( "Thread " << id() << " zeroing out the touch map." );
			
			for( Map::iterator fi = touches->begin(); 
				fi != touches->end(); fi++ )
			{
				fi->second = 0;
			}
		}
		else
		{
			report( "Thread " << id() << " is the not the source, waiting " 
				<< "for the touch map." );
			threadReceive( touches );
			report( "Thread " << id() << " got the touch map." );
		}
		
		while( true )
		{
			if( (*touches)[id()] < loops )
			{
				(*touches)[id()]++;

				report( "Thread " << id() 
					<< " incrementing the touch counter, it is now " 
					<< (*touches)[id()] << "." );	

				if( (*touches)[destination] < loops )
				{
					report( "Thread " << id() << " sending map to " 
						<< destination << "." );	
					threadSend( touches, destination );
				}
			}
			
			if( (*touches)[id()] < loops )
			{
				report( "Thread " << id() << " waiting for next map." );	
				threadReceive( touches );
			}
			else
			{
				break;
			}
		}
		
		if( destination == source )
		{
			report( "Thread " << id() 
				<< " done looping, sending map back to host." );	
			threadSend( touches );
		}
		
		report( "Thread " << id() << " is done, returning." );	
	}

	RingThread* startRing( unsigned int threads, unsigned int loops )
	{
	
		Map* touches = 0;
		
		RingThread* ring = 0;
		
		report( "Starting ring with " << threads << " threads." );
		
		if( threads > 0 )
		{
		
			touches = new Map;
			ring = new RingThread[ threads ];
	
		}
		
		report( "Initializing threads." );
		
		for( unsigned int i = 0; i < threads; i++ )
		{
		
			if( i > 0 )
			{
			
				ring[i].associate( &ring[0] );
			
			}
			
			ring[i].source = ring[0].id();
			ring[i].destination = ring[ ( i + 1 ) % threads ].id();
			
			ring[i].loops = loops;
			ring[i].start();
			
			(*touches)[ ring[i].id() ] = 0;
		
		}
		
		if( threads > 0 )
		{
		
			report( "Sending touch map to the source thread." );
		
			ring[0].send( touches );
	
		}
			
		return ring;
		
	}
		
	bool TestThread::testMessage( )
	{
		RingThread* ring = startRing( threads, loops );
		Map* touches;
		
		bool pass = true;
		
		if( threads > 0 )
		{
			hydrazine::Thread::Id finalThread = ring[0].testGroup( true ).first;
			hydrazine::Thread* final = ring[0].find( finalThread );
		
			final->receive( touches );
				
			for( Map::iterator fi = touches->begin(); 
				fi != touches->end(); fi++ )
			{
				report( "Thread " << fi->first << " touched the shared object " 
					<< fi->second << " times." );
			
				if( fi->second != loops )
				{
					status << "The shared object was touched " 
						<< fi->second << "/" << loops << " by thread " 
						<< fi->first << ".\n";  
					pass = false;
				}
			}
			
			delete touches;
			delete[] ring;
		}
		
		return pass;
	}
	
	bool TestThread::doTest()
	{
		bool pass = true;

		if( testMessage() )
		{
			status << "Test message passed.\n";
		}
		else
		{
			status << "Test message failed.\n";
			pass = false;
		}
		
		return pass;
	}
	
	TestThread::TestThread()
	{
		name = "TestThread";
		
		description = "A test program to test the basic communication ";
		description += "functions in the thread wrapper class.";
	}
	
}

int main( int argc, char* argv[] )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestThread test;

	parser.description( test.testDescription() );

	parser.parse( "-t", test.threads, 20, "The number of threads to use." );
	parser.parse( "-l", test.loops, 200, "Number of times to loop." );
	
	parser.parse( "-v", test.verbose, false, "Print out status information." );
	parser.parse( "-s", test.seed, 0, "Random seed." );
	parser.parse();

	test.test();
	return test.passed();
}

#endif

