#ifndef TEST_ARGUMENT_PARSER_CPP_INCLUDED
#define TEST_ARGUMENT_PARSER_CPP_INCLUDED

#include "TestArgumentParser.h"
#include <hydrazine/implementation/ArgumentParser.h>
#include <hydrazine/implementation/macros.h>
#include <map>
#include <vector>

#ifdef REPORT_BASE
#undef REPORT_BASE 
#endif

#define REPORT_BASE 0

namespace test
{

	bool TestArgumentParser::doTest( )
	{
	
		typedef std::map< std::string, std::pair< int, std::string > > intMap;
		typedef std::map< std::string, std::pair< double, 
			std::string > > doubleMap;
		typedef std::map< std::string, std::pair< std::string, 
			std::string > > stringMap;
		typedef std::map< std::string, std::pair< bool, std::string > > boolMap;

		stringMap mistakes;
		stringMap passed;

		unsigned int starting;
		
		srand( std::time(0) );
		starting = random();

		intMap ints;
		doubleMap doubles;
		stringMap strings;
		boolMap bools;
		
		unsigned int key;
		
		std::string reference("");
		
		key = 0;

		int argc = 2 * ( intCount + doubleCount + stringCount ) + boolCount;
		std::vector< char* > argv(argc);
		
		for( unsigned int i = 0; i < intCount; i++ )
		{
		
			intMap::iterator fi;
		
			int value;
			
			std::stringstream keyStream;
			std::stringstream valueStream;
			
			value = random();
			
			keyStream << "--key_" << key;
			valueStream << value;
			
			fi = ints.insert( std::make_pair( keyStream.str(), 
				std::make_pair( value, valueStream.str() ) ) ).first;

			argv[ key++ ] = const_cast< char* >( fi->first.c_str() );
			argv[ key++ ] = const_cast< char* >( fi->second.second.c_str() );
		
		}
		
		for( unsigned int i = 0; i < doubleCount; i++ )
		{
		
			double value;
			
			doubleMap::iterator fi;
			
			std::stringstream keyStream;
			std::stringstream valueStream;
			
			value = random();
			
			keyStream << "--key_" << key;
			valueStream << value;
			
			fi = doubles.insert( std::make_pair( keyStream.str(), 
				std::make_pair( value, valueStream.str() ) ) ).first;

			argv[ key++ ] = const_cast< char* >( fi->first.c_str() );
			argv[ key++ ] = const_cast< char* >( fi->second.second.c_str() );
		
		}
		
		for( unsigned int i = 0; i < stringCount; i++ )
		{
		
			stringMap::iterator fi;
		
			std::stringstream keyStream;
			std::stringstream valueStream;
			
			keyStream << "--key_" << key;
			valueStream << "string_" << random();
			
			fi = strings.insert( std::make_pair( keyStream.str(), 
				std::make_pair( valueStream.str(), 
				valueStream.str() ) ) ).first;

			argv[ key++ ] = const_cast< char* >( fi->first.c_str() );
			argv[ key++ ] = const_cast< char* >( fi->second.second.c_str() );
		
		}
		
		for( unsigned int i = 0; i < boolCount; i++ )
		{
		
			boolMap::iterator fi;
		
			bool value;
			
			std::stringstream keyStream;
			std::stringstream valueStream;
			
			value = random() & 0x1;
			
			keyStream << "--key_" << key;
			valueStream << value;
			
			fi = bools.insert( std::make_pair( keyStream.str(), 
				std::make_pair( value, valueStream.str() ) ) ).first;

			if( value )
			{
			
				argv[ key++ ] = const_cast< char* >( fi->first.c_str() );
			
			}
			else
			{
			
				argv[ key++ ] = const_cast< char* >( reference.c_str() );
			
			}
			
		}
		
		assert( key == (unsigned int) argc );
		
		hydrazine::ArgumentParser parser( argc, &argv[0] );
		
		for( intMap::iterator fi = ints.begin(); fi != ints.end(); fi++ )
		{
		
			int value;
	
			parser.parse( "-z", fi->first.c_str() , value, starting, "" );
					
			if( value != fi->second.first )
			{
			
				std::stringstream parsedValue;
	
				parsedValue << value;
				
				mistakes.insert( std::make_pair( fi->first, 
					std::make_pair( fi->second.second, parsedValue.str() ) ) );
			
			}
			else
			{
			
				std::stringstream parsedValue;
	
				parsedValue << value;
				
				passed.insert( std::make_pair( fi->first, 
					std::make_pair( fi->second.second, parsedValue.str() ) ) );
			
			}
		
		}
		
		for( doubleMap::iterator fi = doubles.begin(); 
			fi != doubles.end(); fi++ )
		{
		
			double value;
	
			parser.parse( "-z", fi->first.c_str() , value, starting, "" );
			
			std::stringstream stream;
			
			stream << value;
					
			if( stream.str() != fi->second.second )
			{
			
				std::stringstream parsedValue;
	
				parsedValue << value;
				
				mistakes.insert( std::make_pair( fi->first, 
					std::make_pair( fi->second.second, parsedValue.str() ) ) );
			
			}
			else
			{
			
				std::stringstream parsedValue;
	
				parsedValue << value;
				
				passed.insert( std::make_pair( fi->first, 
					std::make_pair( fi->second.second, parsedValue.str() ) ) );
			
			}
		
		}
		
		for( stringMap::iterator fi = strings.begin(); 
			fi != strings.end(); fi++ )
		{
			
			std::string value;
	
			parser.parse( "-z", fi->first.c_str() , value, starting, "" );
					
			if( value != fi->second.first )
			{
			
				std::stringstream parsedValue;
	
				parsedValue << value;
				
				mistakes.insert( std::make_pair( fi->first, 
					std::make_pair( fi->second.second, parsedValue.str() ) ) );
			
			}
			else
			{
			
				std::stringstream parsedValue;
	
				parsedValue << value;
				
				passed.insert( std::make_pair( fi->first, 
					std::make_pair( fi->second.second, parsedValue.str() ) ) );
			
			}
		
		}
		
		for( boolMap::iterator fi = bools.begin(); fi != bools.end(); fi++ )
		{
		
			bool value;
	
			parser.parse( "-z", fi->first.c_str() , value, false, "" );
					
			if( value != fi->second.first )
			{
			
				std::stringstream parsedValue;
	
				parsedValue << value;
				
				mistakes.insert( std::make_pair( fi->first, 
					std::make_pair( fi->second.second, parsedValue.str() ) ) );
			
			}
			else
			{
			
				std::stringstream parsedValue;
	
				parsedValue << value;
				
				passed.insert( std::make_pair( fi->first, 
					std::make_pair( fi->second.second, parsedValue.str() ) ) );
			
			}
				
		}
		
		for( stringMap::iterator fi = mistakes.begin(); 
			fi != mistakes.end(); fi++ )
		{
	
			status << "For key " << fi->first << ", reference " 
				<< fi->second.first << ", did not match parsed " 
				<< fi->second.second << "\n";
	
		}
		
		for( stringMap::iterator fi = passed.begin(); 
			fi != passed.end(); fi++ )
		{
	
			status << "For key " << fi->first << ", reference " 
				<< fi->second.first << ", matched parsed " 
				<< fi->second.second << "\n";
	
		}
	
		return mistakes.empty();
		
	}

	TestArgumentParser::TestArgumentParser()
	{
	
		name = "TestArgumentParser";
	
		description = "This test will create a fake argc and argv and ";
		description += "populate them with test values.  It will then use the ";
		description += "argument parser to initialize some variables and make ";
		description += "sure that they are set to the correct values.";
	
	}

}

int main( int argc, char** argv )
{
	hydrazine::ArgumentParser parser( argc, argv );

	test::TestArgumentParser test;
	parser.description( test.testDescription() );
	
	parser.parse( "-i", "--int_count", test.intCount, 10,
		"Number of ints to search for." );
	parser.parse( "-d", "--double_count", test.doubleCount, 10, 
		"Number of doubles to search for." );
	parser.parse( "-b", "--bool_count", test.boolCount, 10,
		"Number of bools to search for." );
	parser.parse( "-s", "--string_count", test.stringCount, 10,
		"Number of strings to search for." );
	parser.parse( "-v", "--verbose", test.verbose, false,
		"Show status info about the test after it finishes." );
	parser.parse();

	test.test();

	return test.passed();
}

#endif

