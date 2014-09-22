/*! \file LLVMExecutionManager.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Friday September 24, 2010
	\brief The header file for the LLVMExecutionManager
*/

#ifndef LLVM_EXECUTION_MANAGER_H_INCLUDED
#define LLVM_EXECUTION_MANAGER_H_INCLUDED

// Standard Library Includes
#include <string>
#include <vector>

namespace executive
{

class LLVMWorkerThread;
class LLVMExecutableKernel;

/*! \brief Controls the execution of worker threads */
class LLVMExecutionManager
{
public:
	/*! \brief Launches a kernel on a grid using a context */
	static void launch(const LLVMExecutableKernel& kernel);
	
	/*! \brief Changes the number of worker threads */
	static void setWorkerThreadCount(unsigned int threads);

	/*! \brief Gets the current number of threads */
	static unsigned int threads();
		
	/*! \brief Flush references to translated kernels */
	static void flushTranslatedKernels();

private:
	/*! \brief A vector of created threads */
	typedef std::vector<LLVMWorkerThread*> WorkerVector;

private:
	class Manager
	{
	public:
		/*! \brief Destroy all active threads */
		~Manager();
	
	public:
		/*! \brief Launches a kernel on a grid using a context */
		void launch(const LLVMExecutableKernel& kernel);
	
		/*! \brief Changes the number of worker threads */
		void setWorkerThreadCount(unsigned int threads);

		/*! \brief Gets the current number of threads */
		unsigned int threads() const;
		
		/*! \brief Flush references to translated kernels */
		void flushTranslatedKernels();
	
	private:
		/*! \brier The currently active worker threads */
		WorkerVector _workers;
	};
	
private:
	/*! \brief The global singleton execution manager */
	static Manager _manager;
};

}

#endif

