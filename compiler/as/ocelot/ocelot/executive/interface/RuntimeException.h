/*!
	\file RuntimeException.h

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief defines an exception to be thrown at runtime during the emulation or execution of a
		kernel

	\date 29 Jan 2009
*/

#ifndef EXECUTIVE_RUNTIMEEXCEPTION_H_INCLUDED
#define EXECUTIVE_RUNTIMEEXCEPTION_H_INCLUDED

#include <string>
#include <exception>
#include <ocelot/ir/interface/PTXInstruction.h>

namespace executive {

	/*!

	*/
	class RuntimeException : public std::exception {
	public:
		RuntimeException( );
		RuntimeException(std::string);
		RuntimeException(std::string, ir::PTXInstruction);
		RuntimeException(std::string, int, ir::PTXInstruction);
		RuntimeException(std::string, int, int, int, ir::PTXInstruction);
		~RuntimeException() throw();
		
		std::string toString() const;
		const char* what() const throw();

		/*!
			Human-readable message
		*/
		std::string message;

		/*!
			PC of offending instruction
		*/
		int PC;

		/*! \brief offending thread */
		int thread;
		
		/*! \brief offending cta */
		int cta;

		/*!
			offending PTX instruction
		*/
		ir::PTXInstruction instruction;
	};

}

#endif

