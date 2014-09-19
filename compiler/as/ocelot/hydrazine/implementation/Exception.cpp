/*!	\file Exception.cpp
*
*	\brief Source file for the Exception class.
*
*	\author Gregory Diamos
*
*
*/

#ifndef EXCEPTION_CPP_INCLUDED
#define EXCEPTION_CPP_INCLUDED

#include <hydrazine/interface/Exception.h>


namespace hydrazine
{
	const char* Exception::what() const throw()
	{
		return _message.c_str();
	
	}

	Exception::~Exception() throw()
	{
	
	}

	Exception::Exception( const std::string& m ) :
		_message( m )
	{
	
		
	
	}
}

#endif

