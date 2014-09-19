/*! \file   CudaWorkerThread.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday October 23, 2011
	\brief  The header file for the CudaWorkerThread class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/ir/interface/Dim3.h>

#include <ocelot/trace/interface/TraceGenerator.h>

// Hydrazine Includes
#include <hydrazine/interface/Thread.h>

// Standard Library Includes
#include <string>
#include <vector>
#include <queue>

// Forward Declarations
namespace executive { class Device;              }
namespace ir        { class ExternalFunctionSet; }

namespace cuda
{

/*! \brief A worker thread that allows asynchronous execution of
	kernels on an Ocelot device. */
class CudaWorkerThread : public hydrazine::Thread
{
public:
	/*! \brief Put the thread in a consistent state */
	CudaWorkerThread();

	/*! \brief Signal for the thread to die, wait until it joins */
	~CudaWorkerThread();

public:
	/*! \brief Set the device associated with this worker 
	
		Note: The worker must *not* be running.	
	*/
	void setDevice(executive::Device* d);

public:
	/*! \brief Launch the specified kernel on this thread */
	void launch(const std::string& module, 
		const std::string& kernel, const ir::Dim3& grid, 
		const ir::Dim3& block, size_t sharedMemory, 
		const void* argumentBlock, size_t argumentBlockSize, 
		const trace::TraceGeneratorVector& traceGenerators,
		const ir::ExternalFunctionSet* externals);

	/*! \brief Wait for all kernels that have been launched on this
		device to complete */
	void wait();
	
	/*! \brief Check if there are any running kernels */
	bool areAnyKernelsRunning();

public:
	/*! \brief The entry point to the threaded function */
	void execute();

public:
	/*! \brief A raw data array storing kernel parameters */
	typedef std::vector<char> DataVector;

	/*! \brief A description of a kernel launch */
	class Launch
	{
	public:
		std::string                    module;
		std::string                    kernel;
		ir::Dim3                       gridDim;
		ir::Dim3                       blockDim;
		size_t                         sharedMemory;
		DataVector                     parameters;
		trace::TraceGeneratorVector    generators;
		const ir::ExternalFunctionSet* externals;
	};
	
	/*! \brief A queue of kernel launches */
	typedef std::queue<Launch> LaunchQueue;

private:
	/*! \brief Launch the next kernel */
	void _launchNext();
	
private:
	LaunchQueue        _launches;
	unsigned int       _launched;
	unsigned int       _finished;
	executive::Device* _device;
};

}

