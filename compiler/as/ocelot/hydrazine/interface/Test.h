/*!
	\file Test.h
	\author Gregory Diamos
	\date July 4 2008
	\brief Contains the class prototype for the Test infertace
*/

#ifndef TEST_H_INCLUDED
#define TEST_H_INCLUDED

#include <hydrazine/interface/Timer.h>
#include <cassert>
#include <sstream>
#include <boost/random/mersenne_twister.hpp>

/*! \brief A namespace for creating test programs for individual classes. */
namespace test
{

	/*!
		\brief An interface for generating a test for a specific class.  
		
		The idea here it to make tests of individual classes all have a single
		interface for running the test and gathering functional and performance
		information about whether or not the test passed and if it failed,
		why it failed.
	*/
	class Test
	{
		protected:
			typedef boost::random::mersenne_twister
			<
				unsigned int,
				32,
				351,
				175,
				19,
				0xccab8ee7,
				11,
				7,
				0x31b6ab00,
				15,
				0xffe50000,
				17,
				0xa37d3c92
			> 
			MersenneTwister;

		private:
		
			hydrazine::Timer::Second _time;
			bool _testRun;
			bool _passed;
		
		protected:
			std::string name;
			std::string description;
			std::stringstream status;
			MersenneTwister random;
		
		protected:
			virtual bool doTest( ) = 0;
			void _seed();
	
		public:		
			unsigned int seed;
			bool verbose;
	
		public:
			Test();
			virtual ~Test();
			
			void test();
			std::string toString() const;
			std::string testStatus() const;
			const std::string& testName() const;
			std::string testDescription() const;
			bool passed() const;
			bool run() const;			
			hydrazine::Timer::Second time() const;
	
	};

}

#endif

