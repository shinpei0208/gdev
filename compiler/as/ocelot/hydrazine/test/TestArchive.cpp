/*!
	\file TestArchive.cpp
	\author Gregory Diamos
	\date Sunday July 20, 2008
	\brief Source file for the TestArchive class. 
*/

#ifndef TEST_ARCHIVE_CPP_INCLUDED
#define TEST_ARCHIVE_CPP_INCLUDED

#include "TestArchive.h"
#include <cstring>

namespace test
{

	////////////////////////////////////////////////////////////////////////////
	// SimpleSerializableAllocator
	hydrazine::Serializable* 
		TestArchive::SimpleSerializable::\
		SimpleSerializableAllocator::allocate() const
	{
		return new TestArchive::SimpleSerializable;
	}
	////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////
	// SimpleSerializable
	TestArchive::SimpleSerializable::SimpleSerializable()
	{
		_size = 0;
	}
					
	TestArchive::SimpleSerializable::~SimpleSerializable()
	{
		if( _size > 0 )
		{
			delete[] _data;
		}
	}
	
	unsigned int TestArchive::SimpleSerializable::size() const
	{
		return _size;
	}
	
	void TestArchive::SimpleSerializable::resize( 
		unsigned int size )
	{
		if( size != _size )
		{
			if( _size > 0 )
			{
				delete[] _data;
			}

			_size = size;

			if( _size > 0 )
			{
				_data = new char[ _size ];
			}
		}
	}
	
	void* TestArchive::SimpleSerializable::data()
	{
		return _data;
	}
	
	hydrazine::Serializable::Id 
		TestArchive::SimpleSerializable::id() const
	{
		return 0;
	}
	
	void TestArchive::SimpleSerializable::serialize( 
		hydrazine::SerializationBuffer& b ) const
	{
		b.write( &_size, sizeof( unsigned int ) );
		b.write( _data, _size );
	}
	
	void TestArchive::SimpleSerializable::deserialize( 
		hydrazine::SerializationBuffer& b )
	{
		unsigned int size;
		b.read( &size, sizeof( unsigned int ) );
		resize( size );
		b.read( _data, _size );
	}
	
	hydrazine::Serializable::Allocator* 
		TestArchive::SimpleSerializable::allocator() const
	{
		return new TestArchive::SimpleSerializable::\
			SimpleSerializableAllocator;
	}
	
	bool TestArchive::SimpleSerializable::operator!=( 
		const TestArchive::SimpleSerializable& object ) const
	{
		if( _size != object._size )
		{
			return true;
		}
		else
		{
			return memcmp( object._data, _data, _size ) != 0;
		}
	}
	////////////////////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////////////////////
	// TestArchive
	bool TestArchive::testSaveLoad()
	{
	
		bool pass = true;
	
		std::stringstream stream;
		
		hydrazine::SerializableArchive archive;
		
		TestArchive::SimpleSerializable objectIn;
		TestArchive::SimpleSerializable objectOut;
		
		archive.registerClass( &objectIn );
		objectIn.resize( size );
		
		for( unsigned int i = 0; i < objectIn.size(); i++ )
		{
			static_cast<char*>( objectIn.data() )[i] = i;
		}
		
		// test correctness
		archive.save( &objectIn );
		archive.load( &objectOut );
		
		if( objectIn != objectOut )
		{
			status << "Correctness test failed for SerializableArchive " 
				<< "load to existing object.\n";
			pass = false;
		}
		
		TestArchive::SimpleSerializable* objectClone;
		
		archive.save( &objectIn );
		objectClone = static_cast< 
			TestArchive::SimpleSerializable* >( archive.load() );
		
		if( objectIn != *objectClone )
		{
			status << "Correctness test failed for SerializableArchive " 
				<< "load to new object.\n";
			pass = false;
		}
		
		delete objectClone;
		
		hydrazine::Timer Timer;
		
		// benchmark reference
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			archive.save( &objectIn );
			archive.load( &objectOut );
		}
		
		Timer.stop();
		
		status << "SerializableArchive average save/load time for " 
			<< iterations << " iterations: " << (Timer.seconds() / iterations) 
			<< " seconds.\n";

		#ifdef HAVE_BOOST_SERIALIZATION
	
		// boost correctness
		objectOut.resize( 0 );
		
		{
			std::ofstream boostArchiveFile( "_temp.archive" );
			boost::archive::text_oarchive boostArchive( boostArchiveFile );
			boostArchive << const_cast< const 
				TestArchive::SimpleSerializable& >( objectIn );
		}
		
		{
			std::ifstream boostArchiveFile( "_temp.archive", std::ios::binary );
			boost::archive::text_iarchive boostArchive( boostArchiveFile );
			boostArchive >> objectOut;
		}
		
		if( objectIn != objectOut )
		{
			status << "Correctness test failed for BoostArchive load to " 
				<< "existing object.\n";
			pass = false;
		}
		
		// benchmark boost
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			{
				std::ofstream boostArchiveFile( "_temp.archive" );
				boost::archive::text_oarchive boostArchive( boostArchiveFile );
				boostArchive << static_cast< const 
					TestArchive::SimpleSerializable& >( objectIn );
			}
		
			{
				std::ifstream boostArchiveFile( "_temp.archive", 
					std::ios::binary );
				boost::archive::text_iarchive boostArchive( boostArchiveFile );
				boostArchive >> objectOut;
			}		
		}
		
		Timer.stop();
		
		status << "BoostArchive average save/load time for " << iterations 
			<< " iterations: " << (Timer.seconds() / iterations) 
			<< " seconds.\n";

		#endif
		
		return pass;
	}
	
	bool TestArchive::testDisk()
	{

		bool pass = true;
	
		std::stringstream stream;
		
		hydrazine::SerializableArchive archive;
		
		TestArchive::SimpleSerializable objectIn;
		TestArchive::SimpleSerializable objectOut;
		
		archive.registerClass( &objectIn );
		objectIn.resize( size );
		
		for( unsigned int i = 0; i < objectIn.size(); i++ )
		{
			static_cast<char*>( objectIn.data() )[i] = i;
		}
		
		// test correctness
		archive.save( &objectIn );
		archive.saveToFile( "_temp.archive" );
		archive.loadFromFile( "_temp.archive" );
		archive.load( &objectOut );
		
		if( objectIn != objectOut )
		{
			status << "Correctness test failed for SerializableArchive " 
				<< "load to existing object.\n";
			pass = false;
		}
		
		TestArchive::SimpleSerializable* objectClone;
		
		archive.save( &objectIn );
		objectClone = static_cast< 
			TestArchive::SimpleSerializable* >( archive.load() );
		
		if( objectIn != *objectClone )
		{
			status << "Correctness test failed for SerializableArchive " 
				<< "load to new object.\n";
			pass = false;
		}
		
		delete objectClone;
		
		hydrazine::Timer Timer;
		
		// benchmark reference
		Timer.start();
		
		for( unsigned int i = 0; i < iterations; i++ )
		{
			archive.save( &objectIn );
			archive.saveToFile( "_temp.archive" );
			archive.loadFromFile( "_temp.archive" );
			archive.load( &objectOut );
		}
		
		Timer.stop();
		
		status << "SerializableArchive average save/load time for " 
			<< iterations << " iterations: " << (Timer.seconds() / iterations)
			<< " seconds.\n";
	
		return pass;
	}
	
	#ifdef HAVE_MPI	
	
	bool TestArchive::testMpi()
	{
		bool pass = true;
	
		hydrazine::SerializableArchive archive;
		TestArchive::SimpleSerializable object;
		
		archive.registerClass( &object );
		object.resize( size );
		
		for( unsigned int i = 0; i < object.size(); i++ )
		{
			static_cast<char*>( object.data() )[i] = i;
		}
		
		switch( rank )
		{
			case 0:
			{
				hydrazine::Timer Timer;
		
				// benchmark reference
				Timer.start();
		
				for( unsigned int i = 0; i < iterations; i++ )
				{
					archive.save( &objectIn );					
					MPI_Ssend( archive.buffer().ptr(), archive.buffer().size(),
						MPI_CHAR, 1, 0, MPI_COMM_WORLD );
					archive.clear();
				}
		
				Timer.stop();
		
				status << "SerializableArchive average save/mpiSsend time for "
					<< iterations << " iterations: " 
					<< (Timer.seconds() / iterations) << " seconds.\n";
				break;
			}
			
			case 1:
			{
				for( unsigned int i = 0; i < iteratios; i++ )
				{
				
					MPI_Status status;
					TestArchive::SimpleSerializable objectOut;
					unsigned int size;

					MPI_Probe( 0, 0, MPI_COMM_WORLD, &status );
					MPI_Get_count( &status, MPI_CHAR, &size );
				
					archive.buffer().clear();
					archive.buffer().resize( size );
					archive.buffer().initializeWritePointer( size );
					
					MPI_Recv( archive.buffer().ptr(), size, MPI_CHAR, 0, 0, 
						MPI_COMM_WORLD, &status ); 
					
					archive.load( &objectOut );
					
					assert( ! ( objectOut != object ) );
				}
			}
		}

		#ifdef HAVE_BOOST_SERIALIZATION
	
		switch( rank )
		{
			case 0:
			{
				hydrazine::Timer Timer;
		
				// benchmark boost
				Timer.start();
		
				for( unsigned int i = 0; i < iterations; i++ )
				{
					{
						std::ofstream boostArchiveFile( "_temp.archive" );
		
						boost::archive::text_oarchive 
							boostArchive( boostArchiveFile );
						boostArchive << static_cast< const 
							TestArchive::SimpleSerializable& >( 
							objectIn );
					}
		
					{
						std::ifstream boostArchiveFile( "_temp.archive", 
							std::ios::binary );

						// get size of file
						file.seekg( 0, std::ifstream::end );		
						unsigned int size = file.tellg();
						file.seekg( 0 );
						
						char* buffer = new char[ size ];
						boostArchiveFile.read( buffer, size );
						
						MPI_Ssend( buffer, size, MPI_CHAR, 1, 0, 
							MPI_COMM_WORLD );
						
						delete[] buffer;
					}		
				}
		
				Timer.stop();
		
				status << "Boost Serialization average save/mpiSsend time for "
					<< iterations << " iterations: " 
					<< (Timer.seconds() / iterations) << " seconds.\n";
				break;
			}
			
			case 1:
			{
				for( unsigned int i = 0; i < iteratios; i++ )
				{
					MPI_Status status;
					TestArchive::SimpleSerializable objectOut;
					unsigned int size;

					MPI_Probe( 0, 0, MPI_COMM_WORLD, &status );					
					MPI_Get_count( &status, MPI_CHAR, &size );
				
					char* buffer = new char[ size ];
					
					MPI_Recv( buffer, size, MPI_CHAR, 0, 0, 
						MPI_COMM_WORLD, &status ); 
					
					{
						std::ofstream boostArchiveFile( "_temp_recv.archive", 
							std::ofstream::binary );
						boostArchiveFile.write( buffer, size );
					}
					
					{
						std::ifstream boostArchiveFile( "_temp_recv.archive", 
							std::ios::binary );
						boost::archive::text_iarchive 
							boostArchive( boostArchiveFile );
						boostArchive >> objectOut;
					}		
										
					assert( ! ( objectOut != object ) );
				}
			}
		}

		#endif

		return pass;	
	}
	
	#endif	
	
	bool TestArchive::doTest()
	{
		bool pass = true;
		
		if( rank == 0 )
		{
			pass &= testSaveLoad();
			pass &= testDisk();
		}

		#ifdef HAVE_MPI		
		
		if( ranks >= 2 )
		{
			pass &= testMpi();
		}
			
		#endif

		return pass;
	}
	
	TestArchive::TestArchive()
	{
		name = "TestArchive";
		
		description += "Make sure that the implementation of a Serializable\n";
		description += "archive works as intended.\n";  
	
		description += "Make sure that the performance is also reasonable.  Compare\n";
		description += "to the boost serialization library.  This should test and benchmark saving to\n";
		description += "a buffer then immediately restoring, saving to and restoring from disk,\n";
		description += "and saving to an archive, sending the archive over mpi, and restoring at the\n";
		description += "other end.";
	}
	////////////////////////////////////////////////////////////////////////////
	
}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );
	test::TestArchive test;

	#ifdef HAVE_MPI
		MPI_Init( &argc, &argv );
		MPI_Comm_size( MPI_COMM_WORLD, &test.ranks );
		MPI_Comm_rank( MPI_COMM_WORLD, &test.rank );
	#else	
		test.ranks = 0;
		test.rank = 0;	
	#endif
	
	parser.description( test.testDescription() );
	
	parser.parse( "-v", test.verbose, false, 
		"Print out status message when the test is over." );
	parser.parse( "-i", test.iterations, 1000, 
		"How many times to perform the test before computing the avg time?" );
	parser.parse( "-s", test.size, 100, 
		"How many bytes does the object being serialized contain?" );
	parser.parse();

	test.test();
	
	#ifdef HAVE_MPI
	MPI_Finalize();
	#endif

	return test.passed();

}

#endif

