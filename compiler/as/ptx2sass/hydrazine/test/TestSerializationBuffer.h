/*!
	\file TestSerializationBuffer.h
	\author Greory Diamos
	\date July 8, 2008
	\brief The header file for the TestSerializationBuffer test suite.
*/


#ifndef TEST_SERIALIZATION_BUFFER_H_INCLUDED
#define TEST_SERIALIZATION_BUFFER_H_INCLUDED

#ifndef ITERATIONS
#define ITERATIONS 100000
#endif

#ifndef MAX_ELEMENT_BYTES
#define MAX_ELEMENT_BYTES 100
#endif

#ifndef DEFAULT_SEED
#define DEFAULT_SEED 0
#endif

#include <hydrazine/implementation/ArgumentParser.h>
#include <hydrazine/interface/Test.h>
#include <hydrazine/implementation/SerializationBuffer.h>
#include <boost/random/mersenne_twister.hpp>

namespace test
{

	/*!
		\brief A test for the serialization buffer.  
		
		Write in some values
		then read them back and make sure they match.
		
		Take in an external buffer and read some values out.
		
		Write in some strings then read them out.
	*/	
	class TestSerializationBuffer : public Test
	{
		private:
			bool testReadWrite();
			bool testExternalBuffer();
			bool testStrings();
		
			void performanceTest();
			bool doTest();
	
		public:
			TestSerializationBuffer();
			
			unsigned int iterations;
			unsigned int maxSize;
	
	};

}

int main( int argc, char** argv );

#endif

