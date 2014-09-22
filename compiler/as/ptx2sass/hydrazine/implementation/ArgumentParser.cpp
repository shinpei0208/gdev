/*!	\file ArgumentParser.cpp
	\brief Source file for the ArgumentParser class.
*/

#ifndef ARGUMENT_PARSER_CPP_INCLUDED
#define ARGUMENT_PARSER_CPP_INCLUDED

#include <hydrazine/interface/ArgumentParser.h>

// Standard Library Includes
#include <cstdlib>

namespace hydrazine
{

	////////////////////////////////////////////////////////////////////////////
	// Argument parser	
	ArgumentParser::ArgumentParser(int _argc, char* _argv[])
	{
		argc = _argc;
		argv = _argv;
		description( "No description provided." );
	}
	
	void ArgumentParser::description( const std::string& d )
	{
		std::string desc = " Description: ";
		std::stringstream stream( d );	
		int repetition = MESSAGE_OFFSET - ( int )desc.size();
		std::string prefix( MAX( repetition, 0 ), ' ' );
		std::string regularPrefix( MESSAGE_OFFSET, ' ' );
		
		_description = format( stream.str(), desc + prefix, 
			regularPrefix, SCREEN_WIDTH ) + "\n";
	}


	bool ArgumentParser::isPresent(const std::string& identifier)
	{
		bool found = false;
	
		report( "Searching for " << identifier );
	
		for(int i = 0; i < argc; i++)
		{
			std::string str = argv[i];
			size_t pos = str.find(identifier);
			if( pos == 0 && str.size() == identifier.size() )
			{
				report( "Found in " << str );
				found = true;
				break;
			}
		}
	
		return found;
	}
	
	void ArgumentParser::setValue(std::string& value, const std::string& s)
	{
		value = s;
	}
	
	std::string ArgumentParser::help() const
	{
		std::stringstream stream;
		stream << "\nProgram : " << argv[0] << "\n\n";
		stream << _description;
		stream << "Arguments : \n\n";		
		stream << arguments.str();
		stream << "\n";
		return stream.str();
	}
			
	void ArgumentParser::parse(const std::string& _identifier, bool& b, 
		bool starting, const std::string& string)
	{
		assert( _identifier.size() == 2 );
		assert( _identifier[0] == '-' );

		if( isPresent( _identifier ) )
		{
			report( " is present" );
			b = !starting;
		}
		else
		{
			b = starting;
		}
		
		std::string identifier( ' ' + _identifier );

		int prefixSpacing = MESSAGE_OFFSET - ( int )identifier.size();
		
		std::string prefix( MAX( prefixSpacing, 0 ), ' ' );
		std::string regularPrefix( MESSAGE_OFFSET, ' ' );

		std::stringstream secondStream( string + '\n' );
		
		std::string result = format( secondStream.str(), prefix, 
			regularPrefix, SCREEN_WIDTH );
		
		std::stringstream thirdStream;
		thirdStream << result << regularPrefix << "value = " << std::boolalpha 
			<< b << "\n";
			
		arguments << identifier << thirdStream.str() << "\n";
	}

	void ArgumentParser::parse(const std::string& _identifier, 
		const std::string& _longIdentifier, bool& b, bool starting,
		const std::string& string)
	{
		bool inFirst = false;

		if( !_identifier.empty() )
		{
			assert( _identifier.size() == 2 );
			assert( _identifier[0] == '-' );
			inFirst = isPresent( _identifier );
		}
		
		if( inFirst || isPresent( _longIdentifier ) )
		{
			report( " is present" );
			b = !starting;
		}
		else
		{
			b = starting;
		}
		
		std::string identifier( ' ' + _identifier + '(' 
			+ _longIdentifier + ')' );

		int prefixSpacing = MESSAGE_OFFSET - ( int )identifier.size();
		
		std::string prefix( MAX( prefixSpacing, 0 ), ' ' );
		std::string regularPrefix( MESSAGE_OFFSET, ' ' );

		std::stringstream secondStream( string + '\n' );
		
		std::string result = format( secondStream.str(), prefix, 
			regularPrefix, SCREEN_WIDTH );
		
		std::stringstream thirdStream;
		thirdStream << result << regularPrefix << "value = " << std::boolalpha 
			<< b << "\n";
		
		arguments << identifier << thirdStream.str() << "\n";
	}
	
	void ArgumentParser::parse()
	{
		bool printHelp;
		parse( "-h", "--help", printHelp, false, "Print this help message." );
		if( printHelp )
		{
			std::cout << help();
			std::exit(0);
		}
	}
	////////////////////////////////////////////////////////////////////////////////

}

#endif

