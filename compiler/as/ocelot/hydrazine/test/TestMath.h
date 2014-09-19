/*!
	\file TestMath.h
	\author Gregory Diamos
	\date July 8, 2008
	\brief THe interface for a test program for the functions in 
			math.h
*/

#ifndef TEST_MATH_H_INCLUDED
#define TEST_MATH_H_INCLUDED

#include <hydrazine/implementation/math.h>
#include <hydrazine/implementation/ArgumentParser.h>
#include <hydrazine/interface/Test.h>

#include <boost/random/mersenne_twister.hpp>

#ifndef DEFAULT_ITERATIONS 
#define DEFAULT_ITERATIONS 1000000
#endif

#ifndef CORRECTNESS_ITERATIONS
#define CORRECTNESS_ITERATIONS 10
#endif

#ifndef DEFAULT_SEED
#define DEFAULT_SEED 0
#endif

namespace test
{

	/*!
	
		\brief A class to test the functions in math.cpp
	
	*/
	class TestMath : public Test
	{
	
		private:
			template< class T >
			inline T naivePowerOfTwo( T value ) const;
		
			template< class T >
			inline bool naiveIsPowerOfTwo( T value ) const;
		
			template< class T >
			inline T naiveModPowerOfTwo( T value, T value1 ) const;
		
			bool testIsPowerOfTwo();
			bool testModPowerOfTwo();
			bool testPowerOfTwo();
			bool doTest();
			
		public:
			TestMath();
			
			unsigned int iterations;
			unsigned int correctnessIterations;
			unsigned int seed;
	
	};
	
	template< class T >
	inline T TestMath::naivePowerOfTwo( T value ) const
	{
		T result = 1;		
		unsigned int count = sizeof(T) * 8;
		
		for( unsigned int i = 0; i < count; i++ )
		{
			if( (unsigned int)result >= (unsigned int)value )
			{	
				break;
			}
			result *= 2;
		}
		
		return result;
	}

	template< class T >
	inline bool TestMath::naiveIsPowerOfTwo( T value ) const
	{
		T result = 1;
		unsigned int count = sizeof(T) * 8 - 1;
		
		for( unsigned int i = 0; i < count; i++ )
		{
			if( result == value )
			{
				return true;
			}
		
			result *= 2;
		}
		
		return false;
	}

	template< class T >
	inline T TestMath::naiveModPowerOfTwo( T value, T value1 ) const
	{
		return value % value1;
	}

}

int main( int argc, char** argv );

#endif

