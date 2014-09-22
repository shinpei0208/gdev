/*! \file CudaRuntime.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief implements the CUDA Runtime API for Ocelot
*/

#ifndef OCELOT_CUDA_RUNTIME_H_INCLUDED
#define OCELOT_CUDA_RUNTIME_H_INCLUDED

// C++ libs
#include <string>
#include <list>
#include <vector>
#include <map>
#include <set>

// Boost libs
#include <boost/thread/thread.hpp>

// Ocelot libs
#include <ocelot/cuda/interface/CudaRuntimeInterface.h>
#include <ocelot/cuda/interface/FatBinaryContext.h>
#include <ocelot/cuda/interface/CudaWorkerThread.h>
#include <ocelot/executive/interface/Device.h>
#include <ocelot/ir/interface/ExternalFunctionSet.h>

// Hydrazine includes
#include <hydrazine/interface/Timer.h>

// Forward Declarations

namespace transforms { class Pass; }

namespace cuda {

	/***************************************************************/
	/*!	configuration of kernel launch */
	class KernelLaunchConfiguration {
	public:
		KernelLaunchConfiguration(dim3 grid, dim3 block, size_t shared, 
			cudaStream_t s): gridDim(grid), blockDim(block), 
			sharedMemory(shared), stream(s) { }
			
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
	
	typedef std::map<boost::thread::id, HostThreadContext> HostThreadContextMap;
	
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
				cudaChannelFormatDesc({8,0,0,0,cudaChannelFormatKindNone}));
	
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
	
	/*! \brief Set of PTX passes */
	typedef std::set< transforms::Pass* > PassSet;

	typedef std::map< unsigned int, FatBinaryContext > FatBinaryMap;
	typedef std::map< void*, RegisteredGlobal > RegisteredGlobalMap;
	typedef std::map< void*, RegisteredTexture > RegisteredTextureMap;
	typedef std::map< void*, Dimension > DimensionMap;
	typedef std::map< std::string, ir::Module > ModuleMap;
	typedef std::unordered_map<unsigned int, void*> GLBufferMap;
	typedef executive::DeviceVector DeviceVector;

	/*! \brief List of worker threads */
	typedef std::vector<CudaWorkerThread> ThreadVector;

	////////////////////////////////////////////////////////////////////////////
	/*! Cuda runtime context */
	class CudaRuntime: public CudaRuntimeInterface {
	private:
		/*! \brief Memory copy */
		void _memcpy(void* dst, const void* src, size_t count, 
			enum cudaMemcpyKind kind);
		/*! \brief Report a memory error and throw an exception */
		void _memoryError(const void* address, size_t count, 
			const std::string& function = "");		
		/*! \brief Create devices if they do not already exist */
		void _enumerateDevices();
		//! \brief acquires mutex and locks the runtime
		void _lock();
		//! \brief releases mutex
		void _unlock();
		//! \brief sets the last error state for the CudaRuntime object
		cudaError_t _setLastError(cudaError_t result);
		//! \brief Bind the current thread to a device context
		HostThreadContext& _bind();
		//! \brief Unbind the current thread
		void _unbind();
		//! \brief Lock the mutex and bind the the thread
		void _acquire();
		/*! \brief Unbind the thread and unlock the mutex */
		void _release();
		/*! \brief Wait for all running kernels to finish */
		void _wait();
		//! \brief gets the current device for the current thread
		executive::Device& _getDevice();
		//! \brief gets the current worker thread for the current thread
		CudaWorkerThread& _getWorkerThread();
		//! \brief returns an Ocelot-formatted error message
		std::string _formatError(const std::string & message);
		// Get the current thread, create it if it doesn't exist
		HostThreadContext& _getCurrentThread();
		// Load module and register it with all devices
		void _registerModule(ModuleMap::iterator module);
		// Load module and register it with all devices
		void _registerModule(const std::string& name);
		// Load all modules and register them with all devices
		void _registerAllModules();

	private:
		//! locking object for cuda runtime
		boost::mutex _mutex;
		
		//! worker threads for each device
		ThreadVector _workers;
		
		//! Registered modules
		ModuleMap _modules;
		
		//! map of pthreads to thread contexts
		HostThreadContextMap _threads;
		
		//! maps kernel symbols to module-kernels
		RegisteredKernelMap _kernels;
		
		//! maps texture symbols to module-textures
		RegisteredTextureMap _textures;

		//! maps symbol pointers onto their device names
		RegisteredGlobalMap _globals;
		
		//! The dimensions for multi-dimensional allocations
		DimensionMap _dimensions;
		
		//! Registered opengl buffers and mapping to graphics resources
		GLBufferMap _buffers;
		
		//! The total number of enabled devices in the system
		unsigned int _deviceCount;
		
		//! Device vector
		DeviceVector _devices;
		
		//! Have the devices been loaded?
		bool _devicesLoaded;
		
		//! Currently selected device
		int _selectedDevice;
		
		//! the next symbol for dynamically registered kernels
		int _nextSymbol;
		
		//! The minimum supoported compute capability
		int _computeCapability;
		
		//! The device flags
		unsigned int _flags;
		
		//! fatbinaries
		FatBinaryMap _fatBinaries;
		
		//! optimization level
		translator::Translator::OptimizationLevel _optimization;
	
		//! external functions
		ir::ExternalFunctionSet _externals;
		
		//! PTX passes
		PassSet _passes;

		//! set of trace generators to be inserted into emulated kernels
		trace::TraceGeneratorVector _persistentTraceGenerators;

		//! set of trace generators to be inserted into emulated kernels
		trace::TraceGeneratorVector _nextTraceGenerators;
	
	private:
		cudaError_t _launchKernel(const std::string& module, 
			const std::string& kernel);
		
	public:
		CudaRuntime();
		~CudaRuntime();

	public:
		//
		// FatBinary, function, variable, and texture register functions
		//
		virtual void** cudaRegisterFatBinary(void *fatCubin);

		virtual void cudaUnregisterFatBinary(void **fatCubinHandle);

		virtual void cudaRegisterVar(void **fatCubinHandle, char *hostVar, 
			char *deviceAddress, const char *deviceName, int ext, int size, 
			int constant, int global);

		virtual void cudaRegisterTexture(
			void **fatCubinHandle,
			const struct textureReference *hostVar,
			const void **deviceAddress,
			const char *deviceName,
			int dim,
			int norm,
			int ext
		);

		virtual void cudaRegisterShared(
			void **fatCubinHandle,
			void **devicePtr
		);

		virtual void cudaRegisterSharedVar(
			void **fatCubinHandle,
			void **devicePtr,
			size_t size,
			size_t alignment,
			int storage
		);

		virtual void cudaRegisterFunction(
			void **fatCubinHandle,
			const char *hostFun,
			char *deviceFun,
			const char *deviceName,
			int thread_limit,
			uint3 *tid,
			uint3 *bid,
			dim3 *bDim,
			dim3 *gDim,
			int *wSize
		);
		
		virtual cudaError_t cudaGetExportTable(const void **ppExportTable,
			const cudaUUID_t *pExportTableId);

	public:
		//
		// Memory - malloc and free
		//
		virtual cudaError_t  cudaMalloc(void **devPtr, size_t size);
		virtual cudaError_t  cudaMallocHost(void **ptr, size_t size);
		virtual cudaError_t  cudaMallocPitch(void **devPtr, size_t *pitch, 
			size_t width, size_t height);
		virtual cudaError_t  cudaMallocArray(struct cudaArray **array, 
			const struct cudaChannelFormatDesc *desc, size_t width, 
				size_t height = 1);
		virtual cudaError_t  cudaFree(void *devPtr);
		virtual cudaError_t  cudaFreeHost(void *ptr);
		virtual cudaError_t  cudaFreeArray(struct cudaArray *array);
	
		virtual cudaError_t  cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, 
			struct cudaExtent extent);
		virtual cudaError_t  cudaMalloc3DArray(struct cudaArray** arrayPtr, 
			const struct cudaChannelFormatDesc* desc, struct cudaExtent extent);

		// Host Interface

		virtual cudaError_t  cudaHostAlloc(void **pHost, size_t bytes, 
			unsigned int flags);
		virtual cudaError_t  cudaHostGetDevicePointer(void **pDevice, 
			void *pHost, unsigned int flags);
		virtual cudaError_t  cudaHostGetFlags(unsigned int *pFlags, 
			void *pHost);
		virtual cudaError_t cudaHostRegister(void *pHost, size_t bytes, 
			unsigned int flags);
		virtual cudaError_t cudaHostUnregister(void *pHost);


	public:
		//
		// Memcpy
		//
		virtual cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, 
			enum cudaMemcpyKind kind);
		virtual cudaError_t cudaMemcpyToSymbol(const char *symbol, 
			const void *src, size_t count, 
			size_t offset, enum cudaMemcpyKind kind = cudaMemcpyHostToDevice);
		virtual cudaError_t  cudaMemcpyFromSymbol(void *dst, 
			const char *symbol, size_t count, size_t offset = 0, 
			enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
		virtual cudaError_t  cudaMemcpyAsync(void *dst, const void *src, 
			size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

		virtual cudaError_t  cudaMemcpyToArray(struct cudaArray *dst, 
			size_t wOffset, size_t hOffset, 
			const void *src, size_t count, enum cudaMemcpyKind kind);
		virtual cudaError_t  cudaMemcpyFromArray(void *dst, 
			const struct cudaArray *src, size_t wOffset, size_t hOffset, 
			size_t count, enum cudaMemcpyKind kind);
		virtual cudaError_t  cudaMemcpyArrayToArray(struct cudaArray *dst, 
			size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, 
			size_t wOffsetSrc, size_t hOffsetSrc, size_t count, 
			enum cudaMemcpyKind kind);

		virtual cudaError_t  cudaMemcpy2D(void *dst, size_t dpitch, 
			const void *src, size_t spitch, 
			size_t width, size_t height, enum cudaMemcpyKind kind);
		virtual cudaError_t  cudaMemcpy2DToArray(struct cudaArray *dst, 
			size_t wOffset, size_t hOffset, const void *src, size_t spitch, 
			size_t width, size_t height, enum cudaMemcpyKind kind);
		virtual cudaError_t  cudaMemcpy2DFromArray(void *dst, size_t dpitch, 
			const struct cudaArray *src, size_t wOffset, size_t hOffset, 
			size_t width, size_t height, enum cudaMemcpyKind kind);
		
		virtual cudaError_t  cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
		virtual cudaError_t  cudaMemcpy3DAsync(
			const struct cudaMemcpy3DParms *p, cudaStream_t stream);

	public:
		//
		// Memset
		//
		virtual cudaError_t  cudaMemset(void *devPtr, int value, size_t count);
		virtual cudaError_t  cudaMemset2D(void *devPtr, size_t pitch, 
			int value, size_t width, size_t height);
		virtual cudaError_t  cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, 
			int value, struct cudaExtent extent);
		
	public:
		//
		// global variable accessors
		//
		virtual cudaError_t cudaGetSymbolAddress(void **devPtr, 
			const char *symbol);
		virtual cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol);
	
	public:
		//
		// CUDA device management
		//
		virtual cudaError_t cudaGetDeviceCount(int *count);
		virtual cudaError_t cudaGetDeviceProperties(
			struct cudaDeviceProp *prop, int device);
		virtual cudaError_t cudaChooseDevice(int *device, 
			const struct cudaDeviceProp *prop);
		virtual cudaError_t cudaSetDevice(int device);
		virtual cudaError_t cudaGetDevice(int *device);
		virtual cudaError_t cudaSetValidDevices(int *device_arr, int len);
		virtual cudaError_t cudaSetDeviceFlags( int flags );
		
	public:
		//
		// texture binding
		//
		virtual cudaError_t cudaBindTexture(size_t *offset, 
			const struct textureReference *texref, 
			const void *devPtr, const struct cudaChannelFormatDesc *desc, 
			size_t size = UINT_MAX);
		virtual cudaError_t cudaBindTexture2D(size_t *offset,
			const struct textureReference *texref,
			const void *devPtr, const struct cudaChannelFormatDesc *desc,
			size_t width, size_t height, size_t pitch);
		virtual cudaError_t cudaBindTextureToArray(
			const struct textureReference *texref, 
			const struct cudaArray *array, 
			const struct cudaChannelFormatDesc *desc);
		virtual cudaError_t cudaUnbindTexture(
			const struct textureReference *texref);
		virtual cudaError_t cudaGetTextureAlignmentOffset(size_t *offset, 
			const struct textureReference *texref);
		virtual cudaError_t cudaGetTextureReference(
			const struct textureReference **texref, const char *symbol);
	
	public:
		//
		// channel creation
		//
		virtual cudaError_t  cudaGetChannelDesc(
			struct cudaChannelFormatDesc *desc, const struct cudaArray *array);
		virtual struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, 
			int z, int w, enum cudaChannelFormatKind f);

	public:
		virtual cudaError_t cudaGetLastError(void);
		virtual cudaError_t cudaPeekAtLastError(void);

	public:
		//
		// kernel configuration and launch procedures
		//
		virtual cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, 
			size_t sharedMem = 0, cudaStream_t stream = 0);
		virtual cudaError_t cudaSetupArgument(const void *arg, size_t size, 
			size_t offset);
		virtual cudaError_t cudaLaunch(const char *entry);
		virtual cudaError_t cudaFuncGetAttributes(
			struct cudaFuncAttributes *attr, const char *func);	
		virtual cudaError_t cudaFuncSetCacheConfig(const char *func, 
			enum cudaFuncCache cacheConfig);
	
	public:
		//
		// event creation
		//
		virtual cudaError_t  cudaEventCreate(cudaEvent_t *event);
		virtual cudaError_t  cudaEventCreateWithFlags(cudaEvent_t *event, 
			int flags);
		virtual cudaError_t  cudaEventRecord(cudaEvent_t event, 
			cudaStream_t stream);
		virtual cudaError_t  cudaEventQuery(cudaEvent_t event);
		virtual cudaError_t  cudaEventSynchronize(cudaEvent_t event);
		virtual cudaError_t  cudaEventDestroy(cudaEvent_t event);
		virtual cudaError_t  cudaEventElapsedTime(float *ms, cudaEvent_t start, 
			cudaEvent_t end);
	
	public:
		//
		// stream creation
		//
		virtual cudaError_t  cudaStreamCreate(cudaStream_t *pStream);
		virtual cudaError_t  cudaStreamDestroy(cudaStream_t stream);
		virtual cudaError_t  cudaStreamSynchronize(cudaStream_t stream);
		virtual cudaError_t  cudaStreamQuery(cudaStream_t stream);
	
	public:
		/*
			Version accessors
		*/
		virtual cudaError_t cudaDriverGetVersion(int *driverVersion);
		virtual cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
	
	public:
		/*
			Device synchronization
		*/
		virtual cudaError_t cudaDeviceReset(void);
		virtual cudaError_t cudaDeviceSynchronize(void);
		virtual cudaError_t cudaDeviceSetLimit(enum cudaLimit limit,
			size_t value);
		virtual cudaError_t cudaDeviceGetLimit(size_t *pValue,
			enum cudaLimit limit);
		virtual cudaError_t cudaDeviceGetCacheConfig(
			enum cudaFuncCache *pCacheConfig);
		virtual cudaError_t cudaDeviceSetCacheConfig(
			enum cudaFuncCache cacheConfig);
	
	public:
		//
		// kernel thread synchronization
		//
		virtual cudaError_t cudaThreadExit(void);
		virtual cudaError_t cudaThreadSynchronize(void);
		
	public:
		//
		// OpenGL interoperability - deprecated
		//
		virtual cudaError_t cudaGLMapBufferObject(void **devPtr, GLuint bufObj);
		virtual cudaError_t cudaGLMapBufferObjectAsync(void **devPtr, 
			GLuint bufObj, cudaStream_t stream);
		virtual cudaError_t cudaGLRegisterBufferObject(GLuint bufObj);
		virtual cudaError_t cudaGLSetBufferObjectMapFlags(GLuint bufObj, 
			unsigned int flags);
		virtual cudaError_t cudaGLSetGLDevice(int device);
		virtual cudaError_t cudaGLUnmapBufferObject(GLuint bufObj);
		virtual cudaError_t cudaGLUnmapBufferObjectAsync(GLuint bufObj, 
			cudaStream_t stream);
		virtual cudaError_t cudaGLUnregisterBufferObject(GLuint bufObj);

	public:
		//
		// Graphics interoperability
		//
		virtual cudaError_t cudaGraphicsGLRegisterBuffer(
			struct cudaGraphicsResource **resource, GLuint buffer, 
			unsigned int flags);
		virtual cudaError_t cudaGraphicsGLRegisterImage(
			struct cudaGraphicsResource **resource, GLuint image, int target, 
			unsigned int flags);
		
		virtual cudaError_t cudaGraphicsUnregisterResource(
			struct cudaGraphicsResource *resource);
		virtual cudaError_t cudaGraphicsResourceSetMapFlags(
			struct cudaGraphicsResource *resource, unsigned int flags); 
		virtual cudaError_t cudaGraphicsMapResources(int count, 
			struct cudaGraphicsResource **resources, cudaStream_t stream = 0);
		virtual cudaError_t cudaGraphicsUnmapResources(int count, 
			struct cudaGraphicsResource **resources, cudaStream_t stream = 0);
		virtual cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, 
			size_t *size, struct cudaGraphicsResource *resource);
		virtual cudaError_t cudaGraphicsSubResourceGetMappedArray(
			struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, 
			unsigned int arrayIndex, unsigned int mipLevel);
		
	public:

		virtual void addTraceGenerator( trace::TraceGenerator& gen, 
			bool persistent = false );
		virtual void clearTraceGenerators();

		virtual void addPTXPass(transforms::Pass &pass);
		virtual void removePTXPass(transforms::Pass &pass);
		virtual void clearPTXPasses();
		virtual void limitWorkerThreads( unsigned int limit = 1024 );
		virtual void registerPTXModule(std::istream& stream, 
			const std::string& name);
		virtual void registerTexture(const void* texref,
			const std::string& moduleName,
			const std::string& textureName, bool normalize);
		virtual void clearErrors();
		virtual void reset();
		virtual ocelot::PointerMap contextSwitch( 
			unsigned int destinationDevice, unsigned int sourceDevice);
		virtual void unregisterModule(const std::string& name);
		virtual void launch(const std::string& moduleName, 
			const std::string& kernelName);
		virtual void setOptimizationLevel(
			translator::Translator::OptimizationLevel l);
		virtual void registerExternalFunction(const std::string& name,
			void* function);
		virtual void removeExternalFunction(const std::string& name);
		virtual bool isExternalFunction(const std::string& name);
		virtual void getDeviceProperties(executive::DeviceProperties &, int deviceIndex = -1);
	};

}

#endif

