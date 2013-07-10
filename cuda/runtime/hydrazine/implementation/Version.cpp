/*! \file Version.cpp
	\date Saturday January 17, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the Version class
*/

#ifndef VERSION_CPP_INCLUDED
#define VERSION_CPP_INCLUDED

#include <hydrazine/interface/Version.h>
#include <cassert>
#include <sstream>

//#include <configure.h>
#define PACKAGE_VERSION "2.1.unknown"

namespace hydrazine
{

	Version::Version()
	{
		parse( PACKAGE_VERSION );
	}

	void Version::parse( const std::string& version )
	{
	
		std::stringstream stream;
		
		std::string::const_iterator character = version.begin();
		
		for( ; character != version.end(); ++character )
		{
		
			if( *character == '.' )
			{
			
				++character;
				break;
			
			}
		
			stream << *character;
		
		}
		
		stream >> major;

		std::stringstream stream2;
		
		assert( character != version.end() );

		for( ; character != version.end(); ++character )
		{
		
			if( *character == '.' )
			{
			
				++character;
				break;
			
			}
		
			stream2 << *character;
		
		}
		
		stream2 >> minor;
		
		assert( character != version.end() );

		std::stringstream stream3;

		for( ; character != version.end(); ++character )
		{
		
			stream3 << *character;
		
		}
		
		stream3 >> changeList;
		
	}
	
	std::string Version::toString() const
	{
	
		std::stringstream stream;
		
		stream << major << ".";
		stream << minor << ".";
		stream << changeList;
		
		return stream.str();
	
	}
	
	bool Version::operator>( const Version& version )
	{
	
		if( major > version.major )
		{
		
			return true;
		
		}
		else if( major == version.major )
		{
		
			if( minor > version.minor )
			{
			
				return true;
			
			}
			
			return changeList > version.changeList;
		
		}
		
		return false;
	
	}

}

#endif
