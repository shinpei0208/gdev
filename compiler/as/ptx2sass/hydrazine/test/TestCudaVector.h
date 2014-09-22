/*!
	\file TestCudaVector.h
	\date Tuesday May 12, 2009
	\author Gregory Diamos <gregory.diamos>
	\brief The header file for the TestCudaVector class
*/

#ifndef TEST_CUDA_VECTOR_H_INCLUDED
#define TEST_CUDA_VECTOR_H_INCLUDED

#include <hydrazine/interface/Test.h>

#ifndef ITERATIONS 
#define ITERATIONS 10
#endif

namespace test
{
	
	/*!
		\brief Test for cuda vector interface.
		
		Test Points: 
			1) Try reserving, resizing, and clearing the vector.
			2) Copy data from host vector to gpu vector to another gpu vector 
				back to host vector
			3) Iterate over the array incrementing values with array index 
				and iterator
	*/
	class TestCudaVector : public Test
	{
		private:
			bool testAllocation( );
			bool testCopy( );
			bool testIteration( );
			void benchmark();
			
			bool doTest( );
			
		public:
			//! Max memory size to use
			unsigned int max;

			//! Run the bandwidth test
			bool performanceTest;
			
			//! Set the description
			TestCudaVector();
	};

}

int main( int argc, char** argv ); 

#endif

