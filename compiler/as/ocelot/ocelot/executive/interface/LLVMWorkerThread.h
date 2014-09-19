/*! \file LLVMWorkerThread.h
	\date Friday September 24, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMWorkerThread class.
*/

#ifndef LLVM_WORKER_THREAD_H_INCLUDED
#define LLVM_WORKER_THREAD_H_INCLUDED

// Ocelot Includes
#include <ocelot/executive/interface/LLVMModuleManager.h>

// Hydrazine Includes
#include <hydrazine/interface/Thread.h>

// Forward Declarations
namespace executive
{
	class LLVMExecutableKernel;
}

namespace executive
{

/*! \brief A thread responsible for executing CTAs */
class LLVMWorkerThread : public hydrazine::Thread
{
public:
	LLVMWorkerThread();
	~LLVMWorkerThread();

public:
	/*! \brief Setup the state of a cta using an LLVM kernel */
	void setupCta(const LLVMExecutableKernel& kernel);

	/*! \brief Launch the specified CTA from the current kernel */
	void launchCta(unsigned int ctaId);

	/*! \brief Block until the currently executing CTA is finished */
	void finishCta();
	
	/*! \brief Flush refernces to all translated kernels */
	void flushTranslatedKernels();

public:
	/*! \brief Get the id of a translated function from the database */
	LLVMModuleManager::FunctionId getFunctionId(const std::string& moduleName,
		const std::string& functionName);

	/*! \brief Get the translated function from the database */
	LLVMModuleManager::KernelAndTranslation::MetaData* getFunctionMetaData(
		const LLVMModuleManager::FunctionId& id);

private:
	/*! \brief The 'main' entry point for the thread. */
	void execute();
};

}

#endif

