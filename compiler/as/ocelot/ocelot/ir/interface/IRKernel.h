/*! \file IRKernel.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 15, 2009
	\brief The header file for the IRKernel class.
*/

#ifndef IR_KERNEL_H_INCLUDED
#define IR_KERNEL_H_INCLUDED

// Ocelot Includes
#include <ocelot/ir/interface/Kernel.h>

namespace ir {
	class ControlFlowGraph;
}

namespace ir {
	/*! A wrapper class containging common analysis for kernels */
	class IRKernel : public Kernel {

	public:
		/*!	Constructs an empty kernel */
		IRKernel( Instruction::Architecture isa = Instruction::Unknown,
			const std::string& name = "", bool isFunction = false,
			const ir::Module* module = 0, Id id = 0 );
		/*!	Destructs kernel */
		virtual ~IRKernel();
		/*! \brief Copy constructor (deep) */
		IRKernel( const IRKernel& k );
		/*! \brief Assignment operator (deep) */
		const IRKernel& operator=( const IRKernel& k );

	public:
		/*! \brief Gets the cfg */
		ControlFlowGraph* cfg();
		/*! \brief Gets the const cfg */
		const ControlFlowGraph* cfg() const;
	
	public:
		/*! \brief Get a string representation of the nearest line to
			an instruction */
		std::string getLocationString(const Instruction&) const;
	
	public:	
		/*!	Returns true if the kernel instance is derived from 
			ExecutableKernel */
		virtual bool executable() const;
		
	protected:
		/*!	Control flow graph of kernel - this is the primary store of 
				instructions belonging to the kernel */
		ControlFlowGraph* _cfg;
	};

}

#endif

