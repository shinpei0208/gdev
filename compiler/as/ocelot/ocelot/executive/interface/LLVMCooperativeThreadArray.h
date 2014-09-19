/*! \file   LLVMCooperativeThreadArray.h
	\date   Monday September 27, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMCooperativeThreadArray class.
*/

#ifndef LLVM_COOPERATIVE_THREAD_ARRAY_H_INCLUDED
#define LLVM_COOPERATIVE_THREAD_ARRAY_H_INCLUDED

// Ocelot Includes
#include <ocelot/executive/interface/LLVMWorkerThread.h>
#include <ocelot/executive/interface/LLVMFunctionCallStack.h>
#include <ocelot/executive/interface/LLVMModuleManager.h>
#include <ocelot/executive/interface/LLVMContext.h>

// Forward Declarations
namespace executive
{
	class LLVMExecutableKernel;
}

namespace executive
{

/*! \brief A class for managing a CTA executed via LLVM translation */
class LLVMCooperativeThreadArray
{
public:
	LLVMCooperativeThreadArray(LLVMWorkerThread* worker);

public:
	/*! \brief Prepares the CTA to execute the specified kernel */
	void setup(const LLVMExecutableKernel& kernel);
	/*! \brief Execute the specified CTA from the currently selected kernel */
	void executeCta(unsigned int id);
	/*! \brief Flush all references to translated kernels */
	void flushTranslatedKernels();

private:
	typedef std::vector<LLVMModuleManager::MetaData*> FunctionTable;
	typedef std::vector<LLVMFunctionCallStack> StackVector;
	typedef std::vector<LLVMContext> LLVMContextVector;
	typedef std::vector<unsigned int> ThreadList;
	typedef std::vector<char> DataVector;
	typedef std::vector<ThreadList> ThreadListVector;

private:
	void _executeSimpleCta(unsigned int id);
	void _executeComplexCta(unsigned int id);

	void _executeThread(unsigned int id);
	void _executeWarp(ThreadList::const_iterator begin,
		ThreadList::const_iterator end);
	unsigned int _initializeNewContext(unsigned int tid, unsigned int ctaId);
	void _computeNextFunction();
	void _reclaimContext(unsigned int context);
	void _destroyContext(unsigned int context);
	bool _finishContext(unsigned int context);
	void _destroyContexts();
	unsigned int _threadId(const LLVMContext& context);

private:
	LLVMModuleManager::FunctionId _entryPoint;
	LLVMModuleManager::FunctionId _guessFunction;
	LLVMModuleManager::FunctionId _nextFunction;
	FunctionTable                 _functions;
	const LLVMExecutableKernel*   _kernel;
	DataVector                    _sharedMemory;
	DataVector                    _globallyScopedLocalMemory;
	LLVMContextVector             _contexts;
	StackVector                   _stacks;
	ThreadListVector              _queuedThreads;
	ThreadList                    _freeContexts;
	ThreadList                    _reclaimedContexts;
	unsigned int                  _warpSize;
	LLVMWorkerThread*             _worker;
};

}

#endif

