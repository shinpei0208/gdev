/*!

	\file Configurable.h
	
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	
	\date Wednesday October 22, 2008
	
	\brief The header file for the Configurable class.

*/

#ifndef CONFIGURABLE_H_INCLUDED
#define CONFIGURABLE_H_INCLUDED

#include <string>
#include <map>
#include <sstream>

#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE 
#endif

#define REPORT_BASE 0

namespace hydrazine
{

	/*!
	
		\brief A class that can be configured via a map from parameter names
			to values.
	
	*/
	class Configurable
	{
		
		public:
		
			/*!
			
				\brief A configuration is a map from identifiers to values.
			
			*/
			typedef std::map< std::string, std::string > Configuration;
		
		protected:
		
			/*!
			
				\brief Parse a value from a configuration.
				
				\param identifier The identifier to search for in the 
					Configuration
			
				\param defaultValue The value to set if the identifier is not
					found in the map.
					
				\param value The value to set.
				
				\param configuration The Configuration to look for the 
					identifier in.
			
			*/
			template< class T, class V >
			void parse( const std::string& identifier, T& value, 
				const V& defaultValue, const Configuration& configuration );

			void parseString( const std::string& identifier, std::string& value, 
				const std::string& defaultValue, 
				const Configuration& configuration );


		public:
		
			/*!
			
				\brief Convert a configuration to a string.
			
			*/
			static std::string toString( const Configuration& configuration );
			
		public:
		
			/*!
			
				\brief Virtual destructor
			
			*/
			virtual ~Configurable() {}
			
			/*!
			
				\brief Configure the class from a specified Configuration.
				
				\param configuration The specified Configuration.
			
			*/
			virtual void configure( const Configuration& configuration ) = 0;
	
	};
		
	template< class T, class V >
	void Configurable::parse( const std::string& identifier, T& value, 
		const V& defaultValue, 
		const Configurable::Configuration& configuration )
	{
	
		Configuration::const_iterator fi = configuration.find( identifier );
	
		if( fi != configuration.end() )
		{
		
			report( "Found parameter " << fi->first << " set to "
			 	<< fi->second );
			std::stringstream stream( fi->second );
			stream >> value;
		
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

