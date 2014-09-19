/*!	\file ArgumentParser.h
	\brief Header file for The ArgumentParser class.
	\author Gregory Diamos
	\date : 9/27/2007
*/

#ifndef ARGUMENT_PARSER_H_INCLUDED
#define ARGUMENT_PARSER_H_INCLUDED

#define MESSAGE_OFFSET 22
#define SCREEN_WIDTH 80

#include <sstream>
#include <hydrazine/interface/macros.h>
#include <cassert>
#include "debug.h"
#include <hydrazine/interface/string.h>

#ifdef REPORT_BASE
#undef REPORT_BASE 
#endif

#define REPORT_BASE 0

/*!
	\brief a namespace for common classes and functions
*/
namespace hydrazine
{
	/*! 
		\class ArgumentParser
		\brief A class that can be used to parse arguments from argv and argc
	*/
	class ArgumentParser
	{
		private:
			/*! number of strings contained in argv
			*/
			int argc;
			/*! pointer to character array of arguments	
			*/
			char** argv;
			
			/*! A string to hold the descriptions of arguments
			*/
			std::stringstream arguments;
			/*!	\brief Format a string to fit a specific length. */			

			/*!
				\brief A string to hold the description of the program.
			*/
			std::string _description;
		
		private:
		
			/*! A function for parsing an argument
				\param identifier a string used to 
					match strings in argv.
				\param value The value to set if the identifier is found
						
			*/
			template<class T>
			void find( const std::string& identifier, T& value );

			/*! A function determining if an argument is present
				\param identifier a string used to 
					match strings in argv.
				\return true if the identifier was found in argv false otherwise
						
			*/			
			bool isPresent(const std::string& identifier);

			/*! \brief Set the value of an arbitrary type given a string */
			template< typename T >
			void setValue(T& value, const std::string& s);
	
			/*! \brief override the default for strings */
			void setValue(std::string& value, const std::string& s);
			
		public:
			
			/*! The constructor used to initialize an argument parser
			*/
			ArgumentParser(int argc, char** argv);
	
			/*!
				
				\brief Set the description for the program.
			
			*/
			void description( const std::string& d );
	
			/*!
			
				\brief Parse a bool from the command line
				
				\param identifier a string used to match 
					strings in argv.
				\param b A bool to set to true if the argument is found, 
					false otherwise
				\param string The help message to print out when the 
					help function is called
			
			*/
			void parse(const std::string& identifier, bool& b, bool starting,
				const std::string& string);

			/*!
			
				\brief Parse a bool from the command line
				
				\param identifier a string used to match 
					strings in argv.
				\param longIdentifier A long string used to match strings in 
					argv.  Must begin with "--".
				\param b A bool to set to true if the argument is found, 
					false otherwise
				\param string The help message to print out when the 
					help function is called
			
			*/
			void parse(const std::string& identifier, 
				const std::string& longIdentifier, bool& b, bool starting,
				const std::string& string);
			
			/*!
			
				\brief Parse an argument from the command line
				
				\param identifier A string used to match strings in argv. Must
					be a '-' followed by a single character or blank
				\param i A refernce to an argument to set to the parsed value
				\param starting The value to assign to i if the identifier is 
					not found in argv
				\param string The help message to print out when the help 
					function is called
			
			*/
			template< class T, class V >
			void parse(const std::string& identifier, T& i, const V& starting, 
				const std::string& string);

			/*!
			
				\brief Parse a long argument from the command line
				
				\param identifier A string used to match strings in argv. Must
					be a '-' followed by a single character or blank
				\param longIdentifier A long string used to match strings in 
					argv.  Must begin with "--".
				\param i A refernce to an argument to set to the parsed value
				\param starting The value to assign to i if the identifier is 
					not found in argv
				\param string The help message to print out when the help 
					function is called
			
			*/
			template< class T, class V >
			void parse(const std::string& identifier, 
				const std::string& longIdentifier, T& i, const V& starting, 
				const std::string& string);
			
			/*!

				\brief Create a help message describing the program.
				\return A help message stored in a string.
			*/
			std::string help() const;
			
			/*!
				\brief Signal that there will be no more rules added.  It
					is now safe to search for help messages.
			*/
			void parse();
			
	};
	
	template< typename T >
	void ArgumentParser::setValue(T& value, const std::string& s)
	{
		std::stringstream stream(s);
		stream >> value;
	}
	
	void setValue(std::string& value, const std::string& s);

	template< typename T >
	void ArgumentParser::find(const std::string& identifier, T& value)
	{
		std::string str;
	
		bool found = false;
	
		for(int i = 0; i < argc; i++)
		{
			str = argv[i];
			if( str.size() > 0 )
			{
			    if( str[0] == '-' )
			    {
			        str = str.substr( 1 );
           			if( str.size() > 0 )
			        {
			            if( str[0] == '-' )
			            {
			                str = str.substr( 1 );
			            }
			        }
			    }
			    else
			    {
			    	continue;
			    }
			}
			
			report( "Searching for " << identifier << " in " << str );
			
			size_t pos = str.find(identifier);
			if( pos == 0 )
			{
			    report( " Found " << identifier );
				if( str.size() == identifier.size() )
				{
					if( i < argc - 1 )
					{
        			    found = true;
						str = argv[i+1];
						report( "  Setting to next value " << str );
						break;
					}
				}
				else
				{
					pos = identifier.size();
					if( pos < str.size() )
					{
					    if( str[pos] == '=' )
				        {
					        ++pos;
				        }
					}
					if( identifier == str )
					{
						found = true;
						str = str.substr( pos );
						report( "  Setting to substring " << str );
						break;					
					}
				}
			}
		}
	
		if( found )
		{
			setValue(value, str);
		}
		
	}

	template< class T, class V >
	void ArgumentParser::parse(const std::string& _identifier, T& i, 
		const V& starting, const std::string& _string)
	{
		assert( _identifier.size() == 2 );
		assert( _identifier[0] == '-' );

		i = starting;
		find( _identifier.substr(1), i );
		
		std::string identifier( ' ' + _identifier );

		int prefixSpacing = MESSAGE_OFFSET - ( int )identifier.size();
		
		std::string prefix( MAX( prefixSpacing, 0 ), ' ' );
		std::string regularPrefix( MESSAGE_OFFSET, ' ' );

		std::stringstream secondStream( _string + '\n' );

		std::string result = format( secondStream.str(), prefix, 
			regularPrefix, SCREEN_WIDTH );
		
		std::stringstream thirdStream;
		thirdStream << result << regularPrefix << "value = " << i << "\n";
			
		arguments << identifier << thirdStream.str() << "\n";
	}

	template< class T, class V >
	void ArgumentParser::parse(const std::string& _identifier, 
		const std::string& _longIdentifier, T& i, 
		const V& starting, const std::string& _string)
	{
		i = starting;
		
		if( !_identifier.empty() )
		{
			assert( _identifier.size() == 2 );
			assert( _identifier[0] == '-' );
			find( _identifier.substr(1), i );
		}
		
		assert( _longIdentifier.size() > 2 );
		assert( 0 == _longIdentifier.find( "--" ) );

		find( _longIdentifier.substr(2), i );
		
		std::string identifier( ' ' + _identifier 
			+ '(' + _longIdentifier + ')' );

		int prefixSpacing = MESSAGE_OFFSET - ( int )identifier.size();
		
		std::string prefix( MAX( prefixSpacing, 0 ), ' ' );
		std::string regularPrefix( MESSAGE_OFFSET, ' ' );

		std::stringstream secondStream( _string + '\n' );

		std::string result = format( secondStream.str(), prefix, 
			regularPrefix, SCREEN_WIDTH );
		
		std::stringstream thirdStream;
		thirdStream << result << regularPrefix << "value = " << i << "\n";
			
		arguments << identifier << thirdStream.str() << "\n";
	}

}

#endif

