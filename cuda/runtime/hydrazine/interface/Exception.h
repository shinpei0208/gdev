/*!	\file Exception.h
	\brief Header file for the Exception class.
	\author Gregory Diamos
*/

#ifndef EXCEPTION_H_INCLUDED
#define EXCEPTION_H_INCLUDED

//	Standard Library Includes 
#include <exception>
#include <string>


/*! \brief a namespace for common classes and functions */
namespace hydrazine
{
	/*! \brief An Exception with a variable message */
	class Exception : public std::exception
	{
		public:
			Exception( const std::string& message );
			virtual ~Exception() throw();
			virtual const char* what() const throw();

		private:
			std::string _message;		
	};
}

#endif
