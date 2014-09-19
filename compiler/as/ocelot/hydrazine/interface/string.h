/*! \file string.h
	\date Friday February 13, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief Function headers for common C string manipulations
*/

#ifndef STRING_H_INCLUDED
#define STRING_H_INCLUDED

#include <string>
#include <vector>

namespace hydrazine
{
	/*! \brief A vector of strings */
	typedef std::vector< std::string > StringVector;

	/*! \brief Safe string copy
		
		\param destination The target string
		\param source The source string
		\param max The max number of characters to copy
	*/
	void strlcpy( char* destination, const char* source, unsigned int max );

	/*! \brief Split a string into substrings divided on a delimiter */
	StringVector split( const std::string& string, 
		const std::string& delimiter = " " );
	
	/*! \brief Strip out substrings in a string */
	std::string strip( const std::string& string, 
		const std::string& delimiter = " ");
	
	/*! \brief Format a string to fit a specific character width */
	std::string format( const std::string& input, 
		const std::string& firstPrefix = "", const std::string& prefix = "", 
		unsigned int width = 80 );

	/*! \brief Parse a string specifying a binary number, return the number */
	long long unsigned int binaryToUint( const std::string& );

	/*! \brief Convert a string to a label that can be parsed by graphviz */
	std::string toGraphVizParsableLabel( const std::string& );

	/*! \brief Add line numbers to a very large string */
	std::string addLineNumbers( const std::string&, unsigned int begin = 1 );

	/*! \brief Convert a raw data stream into a hex representation */
	std::string dataToString(const void* data, unsigned int bytes);
}

#endif

