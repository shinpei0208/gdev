/*!
	\file TestMath.cpp
	\author Gregory Diamos
	\date July 8, 2008
	\brief THe source for a test program for the functions in 
			math.h
*/

#ifndef TEST_MATH_CPP_INCLUDED
#define TEST_MATH_CPP_INCLUDED

#include "TestMath.h"

namespace test
{
	
	bool TestMath::testIsPowerOfTwo()
	{
		bool pass = true;
		
		// test for correctness
		unsigned int powerOfTwoValue = 1;
		unsigned int doubles = sizeof( unsigned int ) * 8 - 1;
		
		for( unsigned int i = 0; i < doubles; i++ )
		{
			if( hydrazine::isPowerOfTwo( powerOfTwoValue ) )
			{
				status << "Correctly detected " << powerOfTwoValue 
					<< " as a power of two.\n";
			}
			else
			{
				status << "Incorrectly did not detect " << powerOfTwoValue 
					<< " as a power of two.\n";
				pass = false;
			}
		
			powerOfTwoValue *= 2;
		}
		
		for( unsigned int i = 0; i < correctnessIterations; i++ )
		{
			unsigned int value = random();
		
			if( hydrazine::isPowerOfTwo( value ) )
			{
				if( naiveIsPowerOfTwo( value ) )
				{
					status << "Both implementations agreed that " << value 
						<< " was a power of two.\n";
				}
				else
				{
					status << "Experimental thought that " << value 
						<< " was a power of two, but reference did not.\n";
					pass = false;
				}
			}
			else
			{
				if( naiveIsPowerOfTwo( value ) )
				{
					status << "Experimental thought that " << value 
						<< " was not a power of two, but reference did.\n";
					pass = false;
				}
				else
				{
					status << "Both implementations agreed that " << value 
						<< " was not a power of two.\n";
				}
			}
		}
		
		hydrazine::Timer Timer;
		
		bool swap = 0;
		
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			swap ^= hydrazine::isPowerOfTwo( i );
		}
		
		Timer.stop();
		
		status << "Experimental isPowerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " << swap 
			<< " )\n";
		
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			swap ^= naiveIsPowerOfTwo( i );
		}
		
		Timer.stop();
		
		status << "Reference isPowerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " << swap 
			<< " )\n";
		
		// same thing for int

		int powerOfTwoIntValue = 1;
		
		for( unsigned int i = 0; i < doubles; i++ )
		{
			if( hydrazine::isPowerOfTwo( powerOfTwoIntValue ) )
			{
				status << "Correctly detected " << powerOfTwoIntValue 
					<< " as a power of two.\n";
			}
			else
			{
				status << "Incorrectly did not detect " << powerOfTwoIntValue 
					<< " as a power of two.\n";
				pass = false;	
			}
			powerOfTwoIntValue *= 2;
		}
		
		for( unsigned int i = 0; i < correctnessIterations; i++ )
		{
			int value = random();
		
			if( hydrazine::isPowerOfTwo( value ) )
			{
				if( naiveIsPowerOfTwo( value ) )
				{
					status << "Both implementations agreed that " << value 
						<< " was a power of two.\n";
				}
				else
				{
					status << "Experimental thought that " << value 
						<< " was a power of two, but reference did not.\n";
					pass = false;
				}
			}
			else
			{
				if( naiveIsPowerOfTwo( value ) )
				{
					status << "Experimental thought that " << value 
						<< " was not a power of two, but reference did.\n";
					pass = false;
				}
				else
				{
					status << "Both implementations agreed that " << value 
						<< " was not a power of two.\n";
				}
			}
		}
		
		int swapInt = 0;
		
		Timer.start();
		
		for( int i = 0; i < (int) iterations; i++ )
		{
			swapInt ^= hydrazine::isPowerOfTwo( i );
		}
		
		Timer.stop();
		
		status << "Experimental isPowerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " 
			<< swapInt << " )\n";
		
		Timer.start();
		
		for( int i = 0; i < (int) iterations; i++ )
		{
			swapInt ^= naiveIsPowerOfTwo( i );
		}
		
		Timer.stop();
		
		status << "Reference isPowerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " 
			<< swapInt << " )\n";
		
		return pass;
	}

	bool TestMath::testModPowerOfTwo()
	{
		bool pass = true;
		
		// test for correctness
		unsigned int powerOfTwoValue;
		
		unsigned int doubles = sizeof( unsigned int ) * 8 - 1;
		
		for( unsigned int i = 0; i < correctnessIterations; i++ )
		{
		
			unsigned int value = abs( random() );
		
			powerOfTwoValue = 1;
		
			for( unsigned int i = 0; i < doubles; i++ )
			{
				unsigned int experimental = hydrazine::modPowerOfTwo( 
					value, powerOfTwoValue );
				unsigned int reference = naiveModPowerOfTwo( value, 
					powerOfTwoValue );
			
				if( experimental == reference )
				{
					status << "Both implementations agree that " << value 
						<< " % " << powerOfTwoValue << " = " << experimental 
						<< ".\n";
				}
				else
				{
					status << "Experimental thinks that " << value << " % " 
						<< powerOfTwoValue << " = " << experimental 
						<< ", but reference = " << reference << ".\n";
					pass = false;
				}
			
				powerOfTwoValue *= 2;
			}
		}
		
		hydrazine::Timer Timer;
		
		unsigned int swap = 0;
		
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			swap ^= hydrazine::modPowerOfTwo( i, (unsigned int)2 );
		}
		
		Timer.stop();
		
		status << "Experimental modPowerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " << swap 
			<< " )\n";
		
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			swap ^= naiveModPowerOfTwo<unsigned int>( i, 2 );
		}
		
		Timer.stop();
		
		status << "Reference modPowerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " << swap 
			<< " )\n";
		
		// same thing for int

		int powerOfTwoIntValue = 1;
		
		doubles = sizeof( unsigned int ) * 8 - 1;
		
		for( unsigned int i = 0; i < correctnessIterations; i++ )
		{
			int value = abs( random() );
		
			powerOfTwoIntValue = 1;
		
			for( unsigned int i = 0; i < doubles; i++ )
			{
				int experimental = hydrazine::modPowerOfTwo( value, 
					powerOfTwoIntValue );
				int reference = naiveModPowerOfTwo( value, powerOfTwoIntValue );
				
				if( experimental == reference )
				{
				
					status << "Both implementations agree that "<< value 
						<< " % " << powerOfTwoIntValue << " = " << experimental
						<< ".\n";
				
				}
				else
				{
					status << "Experimental thinks that " << value << " % " 
						<< powerOfTwoIntValue << " = " << experimental 
						<< ", but reference = " << reference << ".\n";
					pass = false;
				}
				
				powerOfTwoIntValue *= 2;
			}
		
		}
				
		int swapInt = 0;
		
		Timer.start();
		
		for( int i = 0; i < (int) iterations; i++ )
		{
		
			swapInt ^= hydrazine::modPowerOfTwo( i, (int)2 );
		
		}
		
		Timer.stop();
		
		status << "Experimental modPowerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " << swapInt 
			<< " )\n";
		
		Timer.start();
		
		for( int i = 0; i < (int)iterations; i++ )
		{
		
			swapInt ^= naiveModPowerOfTwo( i, 2 );
		
		}
		
		Timer.stop();
		
		status << "Reference modPowerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " << swapInt 
			<< " )\n";
				
		return pass;
	}

	bool TestMath::testPowerOfTwo()
	{
		bool pass = true;
		
		for( unsigned int i = 0; i < correctnessIterations; i++ )
		{
		
			unsigned int value = random();
			unsigned int experimental = hydrazine::powerOfTwo( value );
			unsigned int reference = naivePowerOfTwo( value );
			
			if( experimental == reference )
			{
			
				status << "Both implementations agree that the next " 
					<< "power of two of "<< value << " = " << experimental 
					<< ".\n";
			
			}
			else
			{
			
				status << "Experimental thinks that the next power of two of "
					<< value << " = " << experimental << ", but reference = " 
					<< reference << ".\n";
			
				pass = false;
			
			}
			
		}
		
		hydrazine::Timer Timer;
		
		unsigned int swap = 0;
		
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
		
			swap ^= hydrazine::powerOfTwo( i );
		
		}
		
		Timer.stop();
	
		status << "Experimental powerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " 
			<< swap << " )\n";
		
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
		
			swap ^= naivePowerOfTwo( i );
		
		}
		
		Timer.stop();
	
		status << "Reference powerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " << swap 
			<< " )\n";
		
		// do for int
		for( unsigned int i = 0; i < correctnessIterations; i++ )
		{
			int value = abs( random() );
			int experimental = hydrazine::powerOfTwo( value );
			int reference = naivePowerOfTwo( value );
			
			if( experimental == reference )
			{
				status << "Both implementations agree that the next " 
					<< "power of two of "<< value << " = " << experimental 
					<< ".\n";
			}
			else
			{
				status << "Experimental thinks that the next power of two of "
					<< value << " = " << experimental << ", but reference = " 
					<< reference << ".\n";
				pass = false;
			}
		}
		
		int swapInt = 0;
		
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			swapInt ^= hydrazine::powerOfTwo( i );
		}
		
		Timer.stop();
	
		status << "Experimental powerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " 
			<< swapInt << " )\n";
		
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
		
			swapInt ^= naivePowerOfTwo( i );
		
		}
		
		Timer.stop();
	
		status << "Reference powerOfTwo time was " << Timer.seconds() 
			<< " seconds, ( " << Timer.cycles() << " ) cycles. ( " 
			<< swapInt << " )\n";
		
		return pass;	
	}

	bool TestMath::doTest()
	{
		bool pass;
		
		pass = testIsPowerOfTwo();
		pass &= testModPowerOfTwo();
		pass &= testPowerOfTwo();
		
		return pass;
	}

	TestMath::TestMath()
	{
		name = "TestMath";
		
		description = "\nA simple test for the functions in math.cpp.  These use\n";
		description += "bitwise operations to do fast power of 2 operations, this\n";
		description += "test benchmarks the operations and tests them against\n";
		description += "equivalent slower implemenations.\n";
	}

}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestMath test;
	
	parser.description( test.testDescription() );
	
	parser.parse( "-i", test.iterations, DEFAULT_ITERATIONS, 
		"How many iterations of each instruction to do for timing." );
	parser.parse( "-c", test.correctnessIterations, CORRECTNESS_ITERATIONS, 
		"How many random values to test for correctness." );
	parser.parse( "-s", test.seed, DEFAULT_SEED, 
		"Random seed to use for repeatability." );
	parser.parse( "-v", test.verbose, false,
		"Print out status message when the test is over." );
	parser.parse();

	test.test();
	return test.passed();
	
}

#endif


