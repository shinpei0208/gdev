/*! \file ILKernel.h
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 27, 2010
 *  \brief The header file for the ILKernel class.
 */

#ifndef IL_KERNEL_H_INCLUDED
#define IL_KERNEL_H_INCLUDED

// Ocelot includes
#include <ocelot/ir/interface/IRKernel.h>
#include <ocelot/ir/interface/ILStatement.h>

namespace translator
{
	class PTXToILTranslator;
}

namespace ir
{
	/*! \brief A class containing a complete representation of an IL kernel */
	class ILKernel : public IRKernel
	{
		friend class translator::PTXToILTranslator;
		public:
			/*! \brief A vector of IL Statements */
			typedef std::deque< ILStatement > ILStatementVector;

			/*! \brief Assemble the IL kernel from the set of statements */
			void assemble();
			/*! \brief Get the assembly code */
			const std::string& code() const;

			/*! \brief Default constructor */
			ILKernel();
			/*! \brief Constructor from a base class */
			ILKernel(const IRKernel &k);

		private:
			/*! \brief The assembled IL kernel */
			std::string _code;
			/*! \brief The set of statements representing the kernel */
			ILStatementVector _statements;
	};
}

#endif

