/*!
	\file CudaDriverFrontend.h

	\author Andrew Kerr <arkerr@gatech.edu>
	\brief implements a CUDA Driver API front-end interface for GPU Ocelot
	\date Sept 16 2010
	\location somewhere over western Europe
*/

#ifndef OCELOT_CUDADRIVERFRONTEND_H_INCLUDED
#define OCELOT_CUDADRIVERFRONTEND_H_INCLUDED

// C++ libs
#include <deque>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <set>

// Boost libs
#include <boost/thread/thread.hpp>

// Ocelot libs
#include <ocelot/cuda/interface/CudaDriverInterface.h>
#include <ocelot/executive/interface/Device.h>
#include <ocelot/cuda/interface/CudaRuntimeContext.h>

// Hydrazine includes
#include <hydrazine/interface/Timer.h>

namespace cuda {

	/*!
		\brief implements the CUDA Driver API front-end to GPU Ocelot
	*/
	class CudaDriverFrontend : public CudaDriverInterface {
	public:
	
		typedef std::set< CUevent > EventSet;
		typedef std::set< CUstream > StreamSet;

		//! \brief CUDA Driver API context
		class Context {
		public:

			enum MemcpyKind {
				HostToHost = 0,
				HostToDevice = 1,
				DeviceToHost = 2,
				DeviceToDevice = 3,
				MemcpyKind_invalid
			};

		public:

			Context();
			~Context();

		public:
			//! \brief performs a memcpy on selected device
			void _memcpy(void *dst, const void *src, size_t count, MemcpyKind kind);

			//! \brief report a memory error and throw an exception
			void _memoryError(const void *address, size_t count, const std::string &func = "");

			//! \brief create devices if they do not exist
			void _enumerateDevices();

			//! \brief gets the current device for the current thread
			executive::Device& _getDevice();
			//! \brief returns an Ocelot-formatted error message
			std::string _formatError(const std::string & message);
			// Load module and register it with all devices
			void _registerModule(ModuleMap::iterator module);
			// Load module and register it with all devices
			void _registerModule(const std::string& name);
			// Load all modules and register them with all devices
			void _registerAllModules();

		public:
		
			//! Registered modules
			ModuleMap _modules;
		
			//! \brief object storing context information needed while configuring calls
			HostThreadContext _hostThreadContext;
			
			//! \brieflaunch configuration of next kernel
			KernelLaunchConfiguration _launchConfiguration;
		
			//! maps kernel symbols to module-kernels
			RegisteredKernelMap _kernels;
		
			//! maps texture symbols to module-textures
			RegisteredTextureMap _textures;

			//! maps symbol pointers onto their device names
			RegisteredGlobalMap _globals;
			
			//! set of live events
			EventSet _events;
			
			//! set of live streams
			StreamSet _streams;
		
			//! The dimensions for multi-dimensional allocations
			DimensionMap _dimensions;
		
			//! Registered opengl buffers and mapping to graphics resources
			GLBufferMap _buffers;
		
			//! device
			executive::Device *_device;
			
			//! index of device bound to context
			int _selectedDevice;
		
			//! the next symbol for dynamically registered kernels
			int _nextSymbol;
		
			//! The device flags
			unsigned int _flags;
		
			//! fatbinaries
			FatBinaryVector _fatBinaries;

			//! optimization level
			translator::Translator::OptimizationLevel _optimization;

			//! \brief number of references to this context
			int _referenceCount;
		};

		typedef std::deque<Context*> ContextQueue;
		typedef std::map< boost::thread::id , ContextQueue > ContextQueueThreadMap;

	public:

		CudaDriverFrontend();
		virtual ~CudaDriverFrontend();

		static CudaDriverFrontend *get();

	public:
		/*********************************
		** Initialization
		*********************************/
		CUresult cuInit(unsigned int Flags);

		/*********************************
		** Driver Version Query
		*********************************/
		CUresult cuDriverGetVersion(int *driverVersion);

		/************************************
		**
		**    Device management
		**
		***********************************/

		CUresult cuDeviceGet(CUdevice *device, int ordinal);
		CUresult cuDeviceGetCount(int *count);
		CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
		CUresult cuDeviceComputeCapability(int *major, int *minor, 
			CUdevice dev);
		CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);
		CUresult cuDeviceGetProperties(CUdevprop *prop, 
			CUdevice dev);
		CUresult cuDeviceGetAttribute(int *pi, 
			CUdevice_attribute attrib, CUdevice dev);

		/************************************
		**
		**    Context management
		**
		***********************************/

		CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, 
			CUdevice dev );
		CUresult cuCtxDestroy( CUcontext ctx );
		CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags);
		CUresult cuCtxDetach(CUcontext ctx);
		CUresult cuCtxPushCurrent( CUcontext ctx );
		CUresult cuCtxPopCurrent( CUcontext *pctx );
		CUresult cuCtxGetDevice(CUdevice *device);
		CUresult cuCtxSynchronize(void);

		/************************************
		**
		**    Module management
		**
		***********************************/

		CUresult cuModuleLoad(CUmodule *module, const char *fname);
		CUresult cuModuleLoadData(CUmodule *module, 
			const void *image);
		CUresult cuModuleLoadDataEx(CUmodule *module, 
			const void *image, unsigned int numOptions, 
			CUjit_option *options, void **optionValues);
		CUresult cuModuleLoadFatBinary(CUmodule *module, 
			const void *fatCubin);
		CUresult cuModuleUnload(CUmodule hmod);
		CUresult cuModuleGetFunction(CUfunction *hfunc, 
			CUmodule hmod, const char *name);
		CUresult cuModuleGetGlobal(CUdeviceptr *dptr, 
			size_t *bytes, CUmodule hmod, const char *name);
		CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, 
			const char *name);

		/************************************
		**
		**    Memory management
		**
		***********************************/

		CUresult cuMemGetInfo(size_t *free, 
			size_t *total);

		CUresult cuMemAlloc( CUdeviceptr *dptr, 
			unsigned int bytesize);
		CUresult cuMemAllocPitch( CUdeviceptr *dptr, 
					          size_t *pPitch,
					          unsigned int WidthInBytes, 
					          unsigned int Height, 
					          unsigned int ElementSizeBytes
					         );
		CUresult cuMemFree(CUdeviceptr dptr);
		CUresult cuMemGetAddressRange( CUdeviceptr *pbase, 
			size_t *psize, CUdeviceptr dptr );

		CUresult cuMemAllocHost(void **pp, unsigned int bytesize);
		CUresult cuMemFreeHost(void *p);

		CUresult cuMemHostAlloc(void **pp, 
			unsigned long long bytesize, unsigned int Flags );

		CUresult cuMemHostGetDevicePointer( CUdeviceptr *pdptr, 
			void *p, unsigned int Flags );
		CUresult cuMemHostGetFlags( unsigned int *pFlags, void *p );

		/************************************
		**
		**    Synchronous Memcpy
		**
		** Intra-device memcpy's done with these functions may execute 
		**	in parallel with the CPU,
		** but if host memory is involved, they wait until the copy is 
		**	done before returning.
		**
		***********************************/

		// 1D functions
		// system <-> device memory
		CUresult cuMemcpyHtoD (CUdeviceptr dstDevice, 
			const void *srcHost, unsigned int ByteCount );
		CUresult cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, 
			unsigned int ByteCount );
		CUresult cuMemcpyHtoH (void *dstHost, const void *srcHost, 
			unsigned int ByteCount );

		// device <-> device memory
		CUresult cuMemcpyDtoD (CUdeviceptr dstDevice, 
			CUdeviceptr srcDevice, unsigned int ByteCount );

		// device <-> array memory
		CUresult cuMemcpyDtoA ( CUarray dstArray, 
			unsigned int dstIndex, CUdeviceptr srcDevice, 
			unsigned int ByteCount );
		CUresult cuMemcpyAtoD ( CUdeviceptr dstDevice, 
			CUarray hSrc, unsigned int SrcIndex, unsigned int ByteCount );

		// system <-> array memory
		CUresult cuMemcpyHtoA( CUarray dstArray, 
			unsigned int dstIndex, const void *pSrc, 
			unsigned int ByteCount );
		CUresult cuMemcpyAtoH( void *dstHost, CUarray srcArray, 
			unsigned int srcIndex, unsigned int ByteCount );

		// array <-> array memory
		CUresult cuMemcpyAtoA( CUarray dstArray, 
			unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, 
			unsigned int ByteCount );

		// 2D memcpy

		CUresult cuMemcpy2D( const CUDA_MEMCPY2D *pCopy );
		CUresult cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy );

		// 3D memcpy

		CUresult cuMemcpy3D( const CUDA_MEMCPY3D *pCopy );

		/************************************
		**
		**    Asynchronous Memcpy
		**
		** Any host memory involved must be DMA'able (e.g., 
		** allocated with cuMemAllocHost).
		** memcpy's done with these functions execute in parallel with 
		** the CPU and, if
		** the hardware is available, may execute in parallel with the GPU.
		** Asynchronous memcpy must be accompanied by appropriate stream 
		** synchronization.
		**
		***********************************/

		// 1D functions
		// system <-> device memory
		CUresult cuMemcpyHtoDAsync (CUdeviceptr dstDevice, 
		const void *srcHost, unsigned int ByteCount, CUstream hStream );
		CUresult cuMemcpyDtoHAsync (void *dstHost, 
		CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );

		// system <-> array memory
		CUresult cuMemcpyHtoAAsync( CUarray dstArray, 
			unsigned int dstIndex, const void *pSrc, 
			unsigned int ByteCount, CUstream hStream );
		CUresult cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, 
			unsigned int srcIndex, unsigned int ByteCount, 
			CUstream hStream );

		// 2D memcpy
		CUresult cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, 
			CUstream hStream );

		// 3D memcpy
		CUresult cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, 
			CUstream hStream );

		/************************************
		**
		**    Memset
		**
		***********************************/
		CUresult cuMemsetD8( CUdeviceptr dstDevice, 
			unsigned char uc, unsigned int N );
		CUresult cuMemsetD16( CUdeviceptr dstDevice, 
			unsigned short us, unsigned int N );
		CUresult cuMemsetD32( CUdeviceptr dstDevice, 
			unsigned int ui, unsigned int N );

		CUresult cuMemsetD2D8( CUdeviceptr dstDevice,
			unsigned int dstPitch, unsigned char uc, unsigned int Width, 
			unsigned int Height );
		CUresult cuMemsetD2D16( CUdeviceptr dstDevice, 
			unsigned int dstPitch, unsigned short us, unsigned int Width, 
			unsigned int Height );
		CUresult cuMemsetD2D32( CUdeviceptr dstDevice, 
			unsigned int dstPitch, unsigned int ui, unsigned int Width, 
			unsigned int Height );

		/************************************
		**
		**    Function management
		**
		***********************************/

		CUresult cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z);
		CUresult cuFuncSetSharedSize (CUfunction hfunc, unsigned int bytes);
		CUresult cuFuncGetAttribute (int *pi, CUfunction_attribute attrib, 
			CUfunction hfunc);
		CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache pconfig);

		/************************************
		**
		**    Array management 
		**
		***********************************/

		CUresult cuArrayCreate( CUarray *pHandle, 
			const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
		CUresult cuArrayGetDescriptor( 
			CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
		CUresult cuArrayDestroy( CUarray hArray );

		CUresult cuArray3DCreate( CUarray *pHandle, 
			const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray );
		CUresult cuArray3DGetDescriptor( 
			CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );


		/************************************
		**
		**    Texture reference management
		**
		***********************************/
		CUresult cuTexRefCreate( CUtexref *pTexRef );
		CUresult cuTexRefDestroy( CUtexref hTexRef );

		CUresult cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, 
			unsigned int Flags );
		CUresult cuTexRefSetAddress( size_t *ByteOffset, 
			CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes );
		CUresult cuTexRefSetAddress2D( CUtexref hTexRef, 
			const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, 
			unsigned int Pitch);
		CUresult cuTexRefSetFormat( CUtexref hTexRef, 
			CUarray_format fmt, int NumPackedComponents );
		CUresult cuTexRefSetAddressMode( CUtexref hTexRef, int dim, 
			CUaddress_mode am );
		CUresult cuTexRefSetFilterMode( CUtexref hTexRef, 
			CUfilter_mode fm );
		CUresult cuTexRefSetFlags( CUtexref hTexRef, 
			unsigned int Flags );

		CUresult cuTexRefGetAddress( CUdeviceptr *pdptr, 
			CUtexref hTexRef );
		CUresult cuTexRefGetArray( CUarray *phArray, 
			CUtexref hTexRef );
		CUresult cuTexRefGetAddressMode( CUaddress_mode *pam, 
			CUtexref hTexRef, int dim );
		CUresult cuTexRefGetFilterMode( CUfilter_mode *pfm, 
			CUtexref hTexRef );
		CUresult cuTexRefGetFormat( CUarray_format *pFormat, 
			int *pNumChannels, CUtexref hTexRef );
		CUresult cuTexRefGetFlags( unsigned int *pFlags, 
			CUtexref hTexRef );

		/************************************
		**
		**    Parameter management
		**
		***********************************/

		CUresult cuParamSetSize (CUfunction hfunc, 
			unsigned int numbytes);
		CUresult cuParamSeti    (CUfunction hfunc, int offset, 
			unsigned int value);
		CUresult cuParamSetf    (CUfunction hfunc, int offset, 
			float value);
		CUresult cuParamSetv    (CUfunction hfunc, int offset, 
			void * ptr, unsigned int numbytes);
		CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, 
			CUtexref hTexRef);

		/************************************
		**
		**    Launch functions
		**
		***********************************/

		CUresult cuLaunch ( CUfunction f );
		CUresult cuLaunchGrid (CUfunction f, int grid_width, 
			int grid_height);
		CUresult cuLaunchGridAsync( CUfunction f, int grid_width, 
			int grid_height, CUstream hStream );

		/************************************
		**
		**    Events
		**
		***********************************/
		CUresult cuEventCreate( CUevent *phEvent, 
			unsigned int Flags );
		CUresult cuEventRecord( CUevent hEvent, CUstream hStream );
		CUresult cuEventQuery( CUevent hEvent );
		CUresult cuEventSynchronize( CUevent hEvent );
		CUresult cuEventDestroy( CUevent hEvent );
		CUresult cuEventElapsedTime( float *pMilliseconds, 
			CUevent hStart, CUevent hEnd );

		/************************************
		**
		**    Streams
		**
		***********************************/
		CUresult cuStreamCreate( CUstream *phStream, 
			unsigned int Flags );
		CUresult cuStreamQuery( CUstream hStream );
		CUresult cuStreamSynchronize( CUstream hStream );
		CUresult cuStreamDestroy( CUstream hStream );

		/************************************
		**
		**    Graphics
		**
		***********************************/
		CUresult cuGraphicsUnregisterResource(
			CUgraphicsResource resource);
		CUresult cuGraphicsSubResourceGetMappedArray(
			CUarray *pArray, CUgraphicsResource resource, 
			unsigned int arrayIndex, unsigned int mipLevel );
		CUresult cuGraphicsResourceGetMappedPointer(
			CUdeviceptr *pDevPtr, size_t *pSize, 
			CUgraphicsResource resource );
		CUresult cuGraphicsResourceSetMapFlags(
			CUgraphicsResource resource, unsigned int flags ); 
		CUresult cuGraphicsMapResources(unsigned int count, 
			CUgraphicsResource *resources, CUstream hStream );
		CUresult cuGraphicsUnmapResources(unsigned int count, 
			CUgraphicsResource *resources, CUstream hStream );

		/************************************
		**
		**    OpenGL
		**
		***********************************/
		CUresult cuGLInit();
		CUresult cuGLCtxCreate(CUcontext *pCtx, 
			unsigned int Flags, CUdevice device);
		CUresult cuGraphicsGLRegisterBuffer( 
			CUgraphicsResource *pCudaResource, unsigned int buffer, 
			unsigned int Flags );
		CUresult cuGraphicsGLRegisterImage( 
			CUgraphicsResource *pCudaResource, unsigned int image, 
			int target, unsigned int Flags);
			
		/*
			CUDA Driver API Support Functions
		*/
		CUresult cuGetExportTable(const void **ppExportTable,
			const CUuuid *pExportTableId);

		std::string toString(CUresult result);

	private:
		//! \brief gets active context
		Context * _getContext();

		//! \brief gets the current thread's context queue
		ContextQueue & _getThreadContextQueue();

		//! \brief gets the calling thread's ID
		boost::thread::id _getThreadId();

		//! \brief locks context thread map
		void _lock();

		//! \brief unlocks context thread map
		void _unlock();

		//! \brief locks and gets the thread's active context
		Context *_bind();

		//! \brief unlocks thread's active context
		void _unbind();
		
		//! \brief lists devices present
		void _enumerateDevices();

	public:
		
		//
		unsigned int _flags;

		//! locking object for _contexts queue [each contex has its own mutex]
		boost::mutex _mutex;

		//! \brief contexts
		ContextQueueThreadMap _contexts;

		//! \brief singleton instance of front end
		static CudaDriverFrontend *_instance;
		
		//! \brief true if devices are loaded
		bool _devicesLoaded;

		//! The minimum supoported compute capability
		int _computeCapability;
		
		//! set of available devices
		executive::DeviceVector _devices;
	};

}

#endif

