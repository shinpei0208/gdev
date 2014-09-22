/*! \file LLVMState.h
	\date Friday September 24, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMState class.
*/

#ifndef LLVM_STATE_H_INCLUDED
#define LLVM_STATE_H_INCLUDED

namespace llvm
{
	class ExecutionEngine;
	class Module;
}

namespace executive
{

/*! \brief A class for managing global llvm state */
class LLVMState
{
public:
	/*! \brief Get a reference to the jit */
	static llvm::ExecutionEngine* jit();

public:
	/*! \brief A global singleton for the LLVM JIT */ 
	static LLVMState llvmState;

private:
	class StateWrapper
	{
	public:
		/*! \brief Build the jit */
		StateWrapper();
		/*! \brief Destroy the jit */
		~StateWrapper();

	public:
		/*! \brief Get a reference to the jit */
		llvm::ExecutionEngine* jit();

	private:
		/*! \brief LLVM JIT Engine */
		llvm::ExecutionEngine* _jit;
		/*! \brief LLVM fake mofule */
		llvm::Module* _module;
	};
	
private:
	static StateWrapper _wrapper;
};

}

#endif

