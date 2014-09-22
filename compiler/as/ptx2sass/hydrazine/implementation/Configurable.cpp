/*!

	\file Configurable.cpp
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	
	\date Friday October 31, 2008
	
	\brief The source file for the Configurable class.

*/

#ifndef CONFIGURABLE_CPP_INCLUDED
#define CONFIGURABLE_CPP_INCLUDED

#include <hydrazine/interface/Configurable.h>

namespace hydrazine
{

	std::string Configurable::toString( const Configuration& configuration )
	{
			
		std::stringstream stream;
	
		for( Configuration::const_iterator fi = configuration.begin(); 
			fi != configuration.end(); ++fi )
		{
		
			stream << " " << fi->first <<  " = " << fi->second << "\n";
		
		}
		
		return stream.str();
	
	}

	void Configurable::parseString( 
		const std::string& identifier, std::string& value, 
		const std::string& defaultValue, 
		const Configurable::Configuration& configuration )
	{
	
		Configuration::const_iterator fi = configuration.find( identifier );
	
		if( fi != configuration.end() )
		{
		
			report( "Found parameter " << fi->first << " set to "
			 	<< fi->second );
			value = fi->second;
		
		}
		else
		{

			report( "Did not find parameter " << identifier << " set to "
			 	<< defaultValue );
			value = defaultValue;
		
		}
	
	}

}

#endif





