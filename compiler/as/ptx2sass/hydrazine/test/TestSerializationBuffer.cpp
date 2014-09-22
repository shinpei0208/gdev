/*!
	\file TestSerializationBuffer.h
	\author Greory Diamos
	\date July 8, 2008
	\brief The source file for the TestSerializationBuffer test suite.
*/

#ifndef TEST_SERIALIZATION_BUFFER_CPP_INCLUDED
#define TEST_SERIALIZATION_BUFFER_CPP_INCLUDED

#include "TestSerializationBuffer.h"

namespace test
{
	
	bool TestSerializationBuffer::testReadWrite()
	{
		bool pass = true;

		std::queue< std::vector< char > > values;
		hydrazine::SerializationBuffer buffer;
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			unsigned int size = random() % maxSize;
			
			std::vector< char > element( size );
			
			for( std::vector< char >::iterator fi = element.begin(); 
				fi != element.end(); fi++ )
			{
				*fi = random();
			}
			
			buffer.write( &element[0], size );
			values.push( element );
			
			while( ( ( random() & 0x1 ) > 0 ) && !values.empty() )
			{
				std::vector< char > readElement( values.front().size() );
				
				buffer.read( &readElement[0], readElement.size() );
			
				std::vector< char >::iterator vi = values.front().begin();
			
				for( std::vector< char >::iterator fi = readElement.begin(); 
					fi != readElement.end(); fi++ )
				{
					assert( vi != values.front().end() );
				
					unsigned int val = *fi;
					unsigned int ref = *vi;
				
					if( val != ref )
					{
						status << "Read write test failed at iteration " << i 
							<< ", read value " << val 
							<< " did not match reference " << ref << ".\n";
						pass = false;
					}
				
					++vi;
				}
				
				if( !pass )
				{
					break;
				}
				
				values.pop();
			}
			
			if( !pass )
			{
				break;
			}
		}
		
		while( !values.empty() )
		{		
			std::vector< char > readElement( values.front().size() );

			buffer.read( &readElement[0], readElement.size() );
		
			std::vector< char >::iterator vi = values.front().begin();
		
			for( std::vector< char >::iterator fi = readElement.begin(); 
				fi != readElement.end(); fi++ )
			{
				assert( vi != values.front().end() );
			
				unsigned int val = *fi;
				unsigned int ref = *vi;
			
				if( val != ref )
				{
					status << "Read write test failed at end, read value " 
						<< val << " did not match reference " << ref << ".\n";
					pass = false;
				}
			
				++vi;
			}
			
			if( !pass )
			{
				break;
			}
			
			values.pop();
		}

		return pass;
	}
	
	bool TestSerializationBuffer::testExternalBuffer()
	{
		hydrazine::SerializationBuffer buffer;
		std::queue< char > reference;
	
		bool pass = true;
		
		unsigned int size = random() % MAX_ELEMENT_BYTES;
		char* myBuffer = new char[ size ];
		
		for( unsigned int i = 0; i < size; i++ )
		{
			char value = random();
			myBuffer[i] = value;
			reference.push( value );
		}
		
		buffer.ptr( myBuffer, size );
		
		for( unsigned int i = 0; i < size; i++ )
		{
			char value;
			buffer.read( &value, 1 );
			
			unsigned int val = value;
			unsigned int ref = reference.front();
			
			if( value != reference.front() )
			{
				pass = false;
				status << "Read value " << val << " does not match reference " 
					<< ref << " at index " << i << ".\n";
			}
			
			reference.pop();
		}
		
		if( !buffer.empty() )
		{
			pass = false;
			status << "Buffer should have been empty after reading out " 
				<< "individual elements, it had " << buffer.size() 
				<< " elements.\n";
		}
		
		return pass;
	}
	
	bool TestSerializationBuffer::testStrings()
	{
		hydrazine::SerializationBuffer buffer;
		std::stringstream stream;
	
		bool pass = true;
		unsigned int size = random() % MAX_ELEMENT_BYTES;
		
		std::string input;
		std::string output;
		
		for( unsigned int i = 0; i < size; i++ )
		{
			char value = random();
			stream << (unsigned int)value;
		}
		
		input = stream.str();
		
		buffer.write( input );
		buffer.read( output );

		if( input != output )
		{
			status << "Input string " << input << " != output string " 
				<< output << ".\n";
			pass = false;
		}
		else
		{
			status << "Input string " << input << " == output string " 
				<< output << ".\n";
		}

		return pass;	
	}
	
	void TestSerializationBuffer::performanceTest()
	{
		hydrazine::SerializationBuffer buffer;
		hydrazine::Timer Timer;

		char* staticBuffer = new char[ maxSize ];		
		
		// test writes
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			buffer.write( staticBuffer, maxSize );
		}
		
		Timer.stop();
		
		status << "Write of " << iterations << " buffers of " << maxSize 
			<< " took " << Timer.seconds() << " seconds ( " << Timer.cycles() 
			<< " ).\n";
		
		// test reads
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			buffer.read( staticBuffer, maxSize );
		}
		
		Timer.stop();
		
		status << "Read of " << iterations << " buffers of " << maxSize 
			<< " took " << Timer.seconds() << " seconds ( " << Timer.cycles() 
			<< " ).\n";
		
		Timer.start();
		
		unsigned int reads = 0;
		unsigned int writes = 0;
		
		// test rw
		for( unsigned int i = 0; i < iterations; i++ )
		{
			unsigned int randomValue = random() % maxSize;
			bool read = randomValue & 0x1;
			
			if( read && ( buffer.size() > randomValue ) )
			{
				buffer.read( staticBuffer, randomValue );
				++reads;
			}
			else
			{
				buffer.write( staticBuffer, randomValue );
				++writes;
			}
		}
		
		Timer.stop();
		
		status << "Read of " << reads << " buffers and writes of " << writes 
			<< " buffers took " << Timer.seconds() << " seconds ( " 
			<< Timer.cycles() << " ).\n";
		
		delete[] staticBuffer;
	}

	bool TestSerializationBuffer::doTest()
	{
		bool pass = true;
		
		pass &= testReadWrite( );
		pass &= testExternalBuffer( );
		pass &= testStrings( );
		performanceTest( );
		
		return pass;
	}

	TestSerializationBuffer::TestSerializationBuffer()
	{
		name = "TestSerializationBuffer";
		
		description = "A test for the serialization buffer.\n\n";
		description += "Write in some values\n";
		description += "then read them back and make sure they match.\n";
		description += "Take in an external buffer and read some values out.\n";
		description += "Write in some strings then read them out.";
	}
			
}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestSerializationBuffer test;
	
	parser.description( test.testDescription() );

	parser.parse( "-i", test.iterations, ITERATIONS, 
		"How many iterations of random numbers to try serializing." );
	parser.parse( "-m", test.maxSize, MAX_ELEMENT_BYTES, 
		"Max number of bytes to serialize for each iteration." );
	parser.parse( "-s", test.seed, DEFAULT_SEED, 
		"Random seed for repeatability." );
	parser.parse( "-v", test.verbose, false, 
		"Print out status message when the test is over." );

	test.test();

	return test.passed();
}

#endif

