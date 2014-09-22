/*! \file LLVMKernel.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Saturday August 1, 2009
	\brief The header file for the LLVMKernel class.
*/

#ifndef LLVM_KERNEL_H_INCLUDED
#define LLVM_KERNEL_H_INCLUDED

#include <ocelot/ir/interface/Kernel.h>
#include <ocelot/ir/interface/LLVMInstruction.h>
#include <ocelot/ir/interface/LLVMStatement.h>

namespace ir
{

	/*! \brief A class containing a complete representation of an LLVM kernel */
	class LLVMKernel : public Kernel
	{
		public:
			/*! \brief A vector of LLVM instructions */
			typedef std::vector< LLVMInstruction* > LLVMInstructionVector;
			/*! \brief A vector of LLVM Statements */
			typedef std::deque< LLVMStatement > LLVMStatementVector;
		
		private:
			/*! \brief The assembled LLVM kernel */
			std::string _code;
			/*! \brief The set of statements representing the kernel */
			LLVMStatementVector _statements;
			
		public:
			/*! \brief Sets the ISA */
			LLVMKernel();
			/*! \brief Initialized the base class from a kernel and executive */
			LLVMKernel( const Kernel& k );
		
		public:
			/*! \brief Add a statement to the end */
			void push_back(const LLVMStatement& statement);
			/*! \brief Add a statement to the beginning */
			void push_front(const LLVMStatement& statement);
		
		public:
			/*! \brief Assemble the LLVM kernel from the set of statements */
			void assemble();
			/*! \brief Is the kernel assembled? */
			bool assembled() const;
			/*! \brief Get the assembly code */
			const std::string& code() const;
			/*! \brief Get the assembly code with line numbers */
			std::string numberedCode() const;
			/*! \brief Get the set of statements */
			const LLVMStatementVector& llvmStatements() const;
	};

}

#endif

