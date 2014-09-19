/*!
	\file CudaRuntimeContext.h
	\author Andrew Kerr <arkerr@gatech.edu>

	\brief defines objects used by CUDA front ends to maintain context information - note: is NOT
		specific to Ocelot's CUDA Runtime API implementation
	
	\date Sept 16 2010
	\location somewhere over Western Europe
*/

#ifndef OCELOT_CUDARUNTIMECONTEXT_H_INCLUDED
#define OCELOT_CUDARUNTIMECONTEXT_H_INCLUDED

// C++ libs
#include <string>
#include <list>
#include <vector>
#include <map>
#include <set>

// Boost libs
#include <boost/thread/thread.hpp>

// Ocelot libs
#include <ocelot/executive/interface/Device.h>
#include <ocelot/cuda/interface/cuda_runtime.h>

// Hydrazine includes
#include <hydrazine/interface/Timer.h>

namespace cuda {

	/***************************************************************/
	/*!	configuration of kernel launch */
	class KernelLaunchConfiguration {
	public:
		KernelLaunchConfiguration(dim3 grid, dim3 block, size_t shared, 
			cudaStream_t s): gridDim(grid), blockDim(block), 
			sharedMemory(shared), stream(s) { }
		KernelLaunchConfiguration(): gridDim(0,0,0), blockDim(0,0,0), sharedMemory(0) { }
		
	public:
		//! dimensions of grid
		dim3 gridDim;
		
		//! dimensions of each block
		dim3 blockDim;
		
		//! number of bytes of dynamically allocated shared memory
		size_t sharedMemory;
		
		//! stream to which kernel launch is to be recorded
		cudaStream_t stream;
	};

	typedef std::list< KernelLaunchConfiguration > KernelLaunchStack;
	
	/*!	\brief Set of thread ids */
	typedef std::set< boost::thread::id > ThreadSet;	
	
	typedef std::vector< unsigned int > IndexVector;
	typedef std::vector< unsigned int > SizeVector;
	
	/*! Host thread CUDA context consists of these */
	class HostThreadContext {	
	public:
		//! index of selected device
		int selectedDevice;
		
		//! array of valid device indices
		std::vector< int > validDevices;
	
		//! stack of launch configurations
		KernelLaunchStack launchConfigurations;
	
		//! last result returned by a CUDA call
		cudaError_t lastError;
		
		//! parameter memory
		unsigned char *parameterBlock;
		
		//! size of parameter memory
		size_t parameterBlockSize;

		//! Offsets for individual parameters
		IndexVector parameterIndices;
		
		//! Sizes for individual parameters
		SizeVector parameterSizes;
		
		//! set of trace generators to be inserted into emulated kernels
		trace::TraceGeneratorVector persistentTraceGenerators;

		//! set of trace generators to be inserted into emulated kernels
		trace::TraceGeneratorVector nextTraceGenerators;
	
	public:
		HostThreadContext();
		~HostThreadContext();

		HostThreadContext(const HostThreadContext& c);	
		HostThreadContext& operator=(const HostThreadContext& c);

		HostThreadContext(HostThreadContext&& c);	
		HostThreadContext& operator=(HostThreadContext&& c);

		void clearParameters();
		void clear();
		unsigned int mapParameters(const ir::Kernel* kernel);
	};
	
	typedef std::map< boost::thread::id, HostThreadContext > HostThreadContextMap;
	
	//! references a kernel registered to CUDA runtime
	class RegisteredKernel {
	public:
		RegisteredKernel(size_t handle = 0, const std::string& module = "", 
			const std::string& kernel = "");

	public:
		//! cubin handle
		size_t handle;
		
		//! name of module
		std::string module;
		
		//! name of kernel
		std::string kernel;
	};
	
	typedef std::map< void*, RegisteredKernel > RegisteredKernelMap;

	/*!	\brief Class allowing sharing of a fat binary among threads	*/
	class FatBinaryContext {
	public:
		FatBinaryContext(const void *_ptr): cubin_ptr(_ptr) { }
		//! pointer to CUBIN structure
		const void *cubin_ptr;
		
	public:
		const char *name() const;
		const char *ptx() const;
	};

	class RegisteredTexture
	{
		public:
			RegisteredTexture(const std::string& module = "", 
				const std::string& texture = "", bool norm = false);
	
		public:
			/*! \brief The module that the texture is declared in */
			std::string module;
			/*! \brief The name of the texture */
			std::string texture;
			// Should the texture be normalized?
			bool norm;
	};
	
	class RegisteredGlobal
	{
		public:
			RegisteredGlobal(const std::string& module = "", 
				const std::string& global = "");
	
		public:
			/*! \brief The module that the global is declared in */
			std::string module;
			/*! \brief The name of the global */
			std::string global;
	};
	
	class Dimension
	{
		public:
			/*! \brief Initializing constructor */
			Dimension(int x = 0, int y = 0, int z = 0, 
				const cudaChannelFormatDesc& f = 
				cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindNone));
	
			/*! \brief Get the pitch of the array */
			size_t pitch() const;
	
		public:
			/*! \brief X dimension */
			int x;
			/*! \brief Y dimension */
			int y;
			/*! \brief Z dimension */
			int z;
			/*! \brief Format */
			cudaChannelFormatDesc format;
	};
	
	typedef std::vector< FatBinaryContext > FatBinaryVector;
	typedef std::map< void*, RegisteredGlobal > RegisteredGlobalMap;
	typedef std::map< void*, RegisteredTexture > RegisteredTextureMap;
	typedef std::map< void*, Dimension > DimensionMap;
	typedef std::map< std::string, ir::Module > ModuleMap;
	typedef std::unordered_map<unsigned int, void*> GLBufferMap;
	typedef executive::DeviceVector DeviceVector;

}

#endif

