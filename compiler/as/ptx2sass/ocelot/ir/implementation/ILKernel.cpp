/*! \file ILKernel.cpp
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 27, 2010
 *  \brief The implementation file for the ILKernel class.
 */

// Ocelot includes
#include <ocelot/ir/interface/ILKernel.h>

// Hydrazine includes
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

// C++ includes
#include <iostream>

namespace ir
{
	ILKernel::ILKernel()
	{
		ISA = Instruction::CAL;
	}

	ILKernel::ILKernel(const IRKernel &k) : IRKernel(k)
	{
		ISA = Instruction::CAL;
	}

	void ILKernel::assemble()
	{
		_code.clear();

		ILStatementVector::const_iterator statement;
		for (statement = _statements.begin() ; 
				statement != _statements.end() ; statement++)
		{
			report("Assembling " + statement->toString());
			_code += statement->toString() + "\n";
		}
	}

	const std::string& ILKernel::code() const
	{
		return _code;
	}
	
}


