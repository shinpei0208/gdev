/*! \file LLVMState.cpp
	\date Friday September 24, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMState class.
*/

#ifndef LLVM_STATE_CPP_INCLUDED
#define LLVM_STATE_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/executive/interface/LLVMState.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// LLVM Includes
#include <configure.h>

#if HAVE_LLVM
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JIT.h> 
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#endif

// Preprocessor Defines
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace executive
{

llvm::ExecutionEngine* LLVMState::jit()
{
	return _wrapper.jit();
}

LLVMState::StateWrapper::StateWrapper() : _jit(0), _module(0)
{

}

llvm::ExecutionEngine* LLVMState::StateWrapper::jit()
{
	#if HAVE_LLVM
	if(_jit == 0)
	{
		report("Bringing the LLVM JIT-Compiler online.");

		_module = new llvm::Module("Ocelot-LLVM-JIT-Blank Module", 
			llvm::getGlobalContext());
		assertM(_module != 0, "Creating global module failed.");
	
		llvm::InitializeNativeTarget();
		
		_jit = llvm::EngineBuilder(_module).create();
		_jit->DisableLazyCompilation(true);
	
		assertM(_jit != 0, "Creating the JIT failed.");
		report(" The JIT is alive.");
	}
	#endif
	return _jit;
}

LLVMState::StateWrapper::~StateWrapper()
{
	// no need to delete anything, llvm will handle it for us
}

LLVMState::StateWrapper LLVMState::_wrapper;

}

#endif

