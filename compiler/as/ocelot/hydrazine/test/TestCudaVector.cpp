/*!
	\file TestCudaVector.cpp
	\date Tuesday May 12, 2009
	\author Gregory Diamos <gregory.diamos>
	\brief The source file for the TestCudaVector class
*/

#ifndef TEST_CUDA_VECTOR_CPP_INCLUDED
#define TEST_CUDA_VECTOR_CPP_INCLUDED

#include <hydrazine/test/TestCudaVector.h>
#include <hydrazine/cuda/Vector.h>
#include <vector>
#include <hydrazine/implementation/ArgumentParser.h>

namespace test
{
	
	bool TestCudaVector::testAllocation( )
	{
		typedef hydrazine::cuda::Vector< char > DeviceVector;

		size_t size = ( random() % max ) + 1;
		
		status << "1) Allocation\n";
		status << " Size is " << size << "\n";
		
		DeviceVector vector( size );
		
		if( size > 0 && vector.empty() )
		{
			status << " Vector reported empty after being initialized.\n";
			return false;
		}
		
		if( vector.size() != size )
		{
			status << " Initialization failed, vector reported size " 
				<< vector.size() << " which should be " << size << "\n";
			return false;
		}
		
		vector.resize( size / 2 );
		
		if( vector.size() != size / 2 )
		{
			status << " Resize failed, vector reported size " 
				<< vector.size() << " which should be " << size / 2 << "\n";
			return false;
		}

		vector.reserve( size * 2 );
		
		if( vector.capacity() < size * 2 )
		{
			status << " Reserve failed, vector reported capacity " 
				<< vector.capacity() << " which was less than " 
				<< size * 2 << "\n";
			return false;
		}
		
		vector.clear();
		
		if( !vector.empty() )
		{
			status << " Clear failed, vector reported not empty\n";
			return false;
		}
		
		if( vector.size() != 0 )
		{
			status << " Clear failed, vector reported size " 
				<< vector.size() << "\n";
			return false;			
		}
	
		status << " Passed.\n";
		return true;
		
	}

	bool TestCudaVector::testCopy( )
	{
		typedef hydrazine::cuda::Vector< char > DeviceVector;
		typedef std::vector< char > HostVector;
		
		size_t size = ( random() % max ) + 1;

		status << "2) Copy\n";
		status << " Size is " << size << "\n";
		
		HostVector initial( size );
		
		for( HostVector::iterator it = initial.begin(); 
			it != initial.end(); ++it )
		{
			*it = random();
		}
		
		DeviceVector vector1( initial.begin(), initial.end() );
		DeviceVector vector2( vector1 );

		HostVector result( vector2.size() );
		vector2.read( &result[0], vector2.size() );
		
		if( result.size() != initial.size() )
		{
			status << " Copy failed, initial size " << initial.size() 
				<< " did not match result size " << result.size() << ".\n";
			return false;
		}
		
		for( unsigned int i = 0; i < size; ++i )
		{
			if( result[ i ] != initial[ i ] )
			{
				status << " Copy failed, at index " << i << " initial " 
					<< (int) initial[ i ] << " does not match result " 
					<< (int) result[ i ] << "\n";
				return false;
			}
		}

		vector2.clear();		
		vector2 = vector1;

		vector2.read( &result[0], vector2.size() );
		
		if( result.size() != initial.size() )
		{
			status << " Copy failed, initial size " << initial.size() 
				<< " did not match result size " << result.size() << ".\n";
			return false;
		}
		
		for( unsigned int i = 0; i < size; ++i )
		{
			if( result[ i ] != initial[ i ] )
			{
				status << " Copy failed, at index " << i << " initial " 
					<< (int) initial[ i ] << " does not match result " 
					<< (int) result[ i ] << "\n";
				return false;
			}
		}

		status << " Passed.\n";
		return true;
	}
	bool TestCudaVector::testIteration( )
	{
		typedef hydrazine::cuda::Vector< char > DeviceVector;
		typedef std::vector< char > HostVector;
		
		size_t size = ( random() % max ) + 1;

		status << "3) Iteration\n";
		status << " Size is " << size << "\n";
		
		HostVector initial( size );
		
		for( HostVector::iterator it = initial.begin(); 
			it != initial.end(); ++it )
		{
			*it = random();
		}
		
		DeviceVector vector;
		vector.write( &initial[0], initial.size() );
		
		for( HostVector::iterator hi = initial.begin(); 
			hi != initial.end(); ++hi )
		{
			*hi += 1;
		}
		
		for( DeviceVector::iterator di = vector.begin(); 
			di != vector.end(); ++di )
		{
			*di = *di + 1;
		}
		
		HostVector result( vector.size() );
		vector.read( &result[0], vector.size() );
		
		if( result.size() != initial.size() )
		{
			status << " Iteration failed, initial size " << initial.size() 
				<< " did not match result size " << result.size() << ".\n";
			return false;
		}
		
		for( unsigned int i = 0; i < size; ++i )
		{
			if( result[ i ] != initial[ i ] )
			{
				status << " Copy failed, at index " << i << " initial " 
					<< (int) initial[ i ] << " does not match result " 
					<< (int) result[ i ] << "\n";
				return false;
			}
		}
		
		status << " Passed.\n";
		return true;
	}
	
	void TestCudaVector::benchmark()
	{
		typedef hydrazine::cuda::Vector< char > DeviceVector;
		typedef std::vector< char > HostVector;

		hydrazine::Timer timer;

		status << "Determining optimal data transfer size.\n";

		DeviceVector device;
		unsigned int maxSize = device.max_size();
		
		device.resize( maxSize );
		HostVector host( maxSize );
				
		timer.start();
		
		for( unsigned int i = 0; i < ITERATIONS; ++i )
		{
			device.write( &host[0], host.size() );		
		}
		
		timer.stop();
		
		double maxBandwidth = maxSize * ITERATIONS 
			/ ( 1048576 * timer.seconds() );
		
		status << " Max bandwidth: " << maxBandwidth << " MB/s\n";
		
		for( unsigned int i = 1024; i < maxSize; i *= 2 )
		{
			host.resize( i );
			timer.start();
			for( unsigned int it = 0; it < ITERATIONS; ++it )
			{
				device.write( &host[0], host.size() );
			}
			timer.stop();
			
			double bandwidth = i * ITERATIONS / ( 1048576 * timer.seconds() );
			
			if( bandwidth > .9 * maxBandwidth )
			{
				status << " Optimal size is " << i << " at " 
					<< bandwidth << " MB/s\n";
				return;
			}
			else
			{
				status << "  Intermediate size is " << i << " at " 
					<< bandwidth << " MB/s\n";
			}
		}
		
		status << " Optimal size is max size " << maxSize << "\n";
		
	}
	
	bool TestCudaVector::doTest( )
	{
		if( performanceTest )
		{
			benchmark();
			return true;
		}
		else
		{
			return testAllocation() && testCopy() && testIteration();	
		}
	}
	
	TestCudaVector::TestCudaVector()
	{
		name = "TestCudaVector";
		description = "Test for cuda vector interface. Test Points: 1) Try";
		description += " reserving, resizing, and clearing the vector. 2) ";
		description += "Copy data from host vector to gpu vector to another";
		description += " gpu vector back to host vector 3) Iterate over ";
		description += "the array incrementing values with array index ";
		description += "and iterator";
	}

}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestCudaVector test;
	parser.description( test.testDescription() );
	
	parser.parse( "-m", "--memory", test.max, 5000,
		"The maxmimum amount of elements to allocate." );
	parser.parse( "-p", "--perf-test", test.performanceTest, false,
		"Run a performance test to determine the optimal block size." );
	parser.parse( "-s", "--seed", test.seed, 0,
		"Random seed for repeatibility, 0 implies seed with time." );
	parser.parse( "-v", "--verbose", test.verbose, false,
		"Verbosely print out information after the test is over." );
	parser.parse();

	test.test();
}

#endif

