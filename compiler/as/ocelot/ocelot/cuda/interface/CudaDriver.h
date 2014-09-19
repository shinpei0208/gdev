/*! \file CudaDriver.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Friday March 26, 2010
	\brief The header file for the CudaDriver class.
*/
#ifndef CUDA_DRIVER_H_INCLUDED
#define CUDA_DRIVER_H_INCLUDED

// Ocelot Includes
#include <ocelot/cuda/interface/cuda_internal.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// OpenGL Includes
#include <GL/gl.h>

// Standard Library Includes
#include <string>

namespace cuda
{

/*! \brief Dynamic interface to the cuda driver */
class CudaDriver
{
	public:
		/*! \brief Container for pointers to the actual functions */
		class Interface
		{
			private:
				/*! \brief The handle to the driver dll */
				void* _driver;
				/*! \brief The driver version */
				int _version;
				
			public:
				//! name of library to load
				std::string _libname;
				
			public:
				CUresult (*cuInit)(unsigned int Flags);
				CUresult (*cuDriverGetVersion)(int *driverVersion);
				CUresult (*cuDeviceGet)(CUdevice *device, int ordinal);
				CUresult (*cuDeviceGetCount)(int *count);
				CUresult (*cuDeviceGetName)(char *name, int len, 
					CUdevice dev);
				CUresult (*cuDeviceComputeCapability)(int *major, 
					int *minor, CUdevice dev);
				CUresult (*cuDeviceTotalMem)(size_t *bytes, 
					CUdevice dev);
				CUresult (*cuDeviceGetProperties)(CUdevprop *prop, 
					CUdevice dev);
				CUresult (*cuDeviceGetAttribute)(int *pi, 
					CUdevice_attribute attrib, CUdevice dev);	
				CUresult (*cuCtxGetApiVersion)(CUcontext ctx, unsigned int *version);
				CUresult (*cuCtxCreate)(CUcontext *pctx, 
					unsigned int flags, CUdevice dev );
				CUresult (*cuCtxGetLimit)(size_t *, CUlimit);
				CUresult (*cuCtxDestroy)( CUcontext ctx );
				CUresult (*cuCtxAttach)(CUcontext *pctx, 
					unsigned int flags);
				CUresult (*cuCtxDetach)(CUcontext ctx);
				CUresult (*cuCtxPushCurrent)( CUcontext ctx );
				CUresult (*cuCtxPopCurrent)( CUcontext *pctx );
				CUresult (*cuCtxGetDevice)(CUdevice *device);
				CUresult (*cuCtxSynchronize)(void);
				CUresult (*cuModuleLoad)(CUmodule *module, 
					const char *fname);
				CUresult (*cuModuleLoadData)(CUmodule *module, 
					const void *image);
				CUresult (*cuModuleLoadDataEx)(CUmodule *module, 
					const void *image, unsigned int numOptions, 
					CUjit_option *options, void **optionValues);
				CUresult (*cuModuleLoadFatBinary)(CUmodule *module, 
					const void *fatCubin);
				CUresult (*cuModuleUnload)(CUmodule hmod);
				CUresult (*cuModuleGetFunction)(CUfunction *hfunc, 
					CUmodule hmod, const char *name);
				CUresult (*cuModuleGetGlobal)(CUdeviceptr *dptr, 
					size_t *bytes, CUmodule hmod, const char *name);
				CUresult (*cuModuleGetTexRef)(CUtexref *pTexRef, 
					CUmodule hmod, const char *name);
				CUresult (*cuMemGetInfo)(size_t *free, 
					size_t *total);
				CUresult (*cuMemAlloc)( CUdeviceptr *dptr, 
					unsigned int bytesize);
				CUresult (*cuMemAllocPitch)( CUdeviceptr *dptr, 
					  size_t *pPitch,
					  unsigned int WidthInBytes, 
					  unsigned int Height, 
					  unsigned int ElementSizeBytes
					 );
				CUresult (*cuMemFree)(CUdeviceptr dptr);
				CUresult (*cuMemGetAddressRange)( CUdeviceptr *pbase, 
					size_t *psize, CUdeviceptr dptr );
				CUresult (*cuMemAllocHost)(void **pp, 
					unsigned int bytesize);
				CUresult (*cuMemFreeHost)(void *p);
				CUresult (*cuMemHostAlloc)(void **pp, 
					unsigned long long bytesize, unsigned int Flags );
				CUresult (*cuMemHostRegister)(void *pp, 
					unsigned long long bytesize, unsigned int Flags );
				CUresult (*cuMemHostUnregister)(void *pp);


				CUresult (*cuMemHostGetDevicePointer)( CUdeviceptr *pdptr, 
					void *p, unsigned int Flags );
				CUresult (*cuMemHostGetFlags)( unsigned int *pFlags, 
					void *p );
				CUresult (*cuMemcpyHtoD)(CUdeviceptr dstDevice, 
					const void *srcHost, unsigned int ByteCount );
				CUresult (*cuMemcpyDtoH)(void *dstHost, 
					CUdeviceptr srcDevice, unsigned int ByteCount );
				CUresult (*cuMemcpyDtoD)(CUdeviceptr dstDevice, 
					CUdeviceptr srcDevice, unsigned int ByteCount );
				CUresult (*cuMemcpyDtoA)( CUarray dstArray, 
					unsigned int dstIndex, CUdeviceptr srcDevice, 
					unsigned int ByteCount );
				CUresult (*cuMemcpyAtoD)( CUdeviceptr dstDevice, 
					CUarray hSrc, unsigned int SrcIndex, 
					unsigned int ByteCount );
				CUresult (*cuMemcpyHtoA)( CUarray dstArray, 
					unsigned int dstIndex, const void *pSrc, 
					unsigned int ByteCount );
				CUresult (*cuMemcpyAtoH)( void *dstHost, CUarray srcArray, 
					unsigned int srcIndex, unsigned int ByteCount );
				CUresult (*cuMemcpyAtoA)( CUarray dstArray, 
					unsigned int dstIndex, CUarray srcArray, 
					unsigned int srcIndex, unsigned int ByteCount );
				CUresult (*cuMemcpy2D)( const CUDA_MEMCPY2D *pCopy );
				CUresult (*cuMemcpy2DUnaligned)( 
					const CUDA_MEMCPY2D *pCopy );
				CUresult (*cuMemcpy3D)( const CUDA_MEMCPY3D *pCopy );
				CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr dstDevice, 
				const void *srcHost, unsigned int ByteCount, 
					CUstream hStream );
				CUresult (*cuMemcpyDtoHAsync)(void *dstHost, 
				CUdeviceptr srcDevice, unsigned int ByteCount, 
					CUstream hStream );
				CUresult (*cuMemcpyHtoAAsync)( CUarray dstArray, 
					unsigned int dstIndex, const void *pSrc, 
					unsigned int ByteCount, CUstream hStream );
				CUresult (*cuMemcpyAtoHAsync)( void *dstHost, 
					CUarray srcArray, unsigned int srcIndex, 
					unsigned int ByteCount, CUstream hStream );
				CUresult (*cuMemcpy2DAsync)( const CUDA_MEMCPY2D *pCopy, 
					CUstream hStream );
				CUresult (*cuMemcpy3DAsync)( const CUDA_MEMCPY3D *pCopy, 
					CUstream hStream );
				CUresult (*cuMemsetD8)( CUdeviceptr dstDevice, 
					unsigned char uc, unsigned int N );
				CUresult (*cuMemsetD16)( CUdeviceptr dstDevice, 
					unsigned short us, unsigned int N );
				CUresult (*cuMemsetD32)( CUdeviceptr dstDevice, 
					unsigned int ui, unsigned int N );
				CUresult (*cuMemsetD2D8)( CUdeviceptr dstDevice, 
					unsigned int dstPitch, unsigned char uc, 
					unsigned int Width, unsigned int Height );
				CUresult (*cuMemsetD2D16)( CUdeviceptr dstDevice, 
					unsigned int dstPitch, unsigned short us, 
					unsigned int Width, unsigned int Height );
				CUresult (*cuMemsetD2D32)( CUdeviceptr dstDevice, 
					unsigned int dstPitch, unsigned int ui, 
					unsigned int Width, unsigned int Height );
				CUresult (*cuFuncSetBlockShape)(CUfunction hfunc, int x, 
					int y, int z);
				CUresult (*cuFuncSetSharedSize)(CUfunction hfunc, 
					unsigned int bytes);
				CUresult (*cuFuncGetAttribute)(int *pi, 
					CUfunction_attribute attrib, CUfunction hfunc);
				CUresult (*cuFuncSetCacheConfig)(CUfunction hFunc, CUfunc_cache config);
				
				CUresult (*cuArrayCreate)( CUarray *pHandle, 
					const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
				CUresult (*cuArrayGetDescriptor)( 
					CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, 
					CUarray hArray );
				CUresult (*cuArrayDestroy)( CUarray hArray );
				CUresult (*cuArray3DCreate)( CUarray *pHandle, 
					const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray );
				CUresult (*cuArray3DGetDescriptor)( 
					CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, 
					CUarray hArray );
				CUresult (*cuTexRefCreate)( CUtexref *pTexRef );
				CUresult (*cuTexRefDestroy)( CUtexref hTexRef );

				CUresult (*cuTexRefSetArray)( CUtexref hTexRef, 
					CUarray hArray, unsigned int Flags );
				CUresult (*cuTexRefSetAddress)( size_t *ByteOffset, 
					CUtexref hTexRef, CUdeviceptr dptr, 
					size_t bytes );
				CUresult (*cuTexRefSetAddress2D)( CUtexref hTexRef, 
					const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, 
					unsigned int Pitch);
				CUresult (*cuTexRefSetFormat)( CUtexref hTexRef, 
					CUarray_format fmt, int NumPackedComponents );
				CUresult (*cuTexRefSetAddressMode)( CUtexref hTexRef, 
					int dim, CUaddress_mode am );
				CUresult (*cuTexRefSetFilterMode)( CUtexref hTexRef, 
					CUfilter_mode fm );
				CUresult (*cuTexRefSetFlags)( CUtexref hTexRef, 
					unsigned int Flags );

				CUresult (*cuTexRefGetAddress)( CUdeviceptr *pdptr, 
					CUtexref hTexRef );
				CUresult (*cuTexRefGetArray)( CUarray *phArray, 
					CUtexref hTexRef );
				CUresult (*cuTexRefGetAddressMode)( CUaddress_mode *pam, 
					CUtexref hTexRef, int dim );
				CUresult (*cuTexRefGetFilterMode)( CUfilter_mode *pfm, 
					CUtexref hTexRef );
				CUresult (*cuTexRefGetFormat)( CUarray_format *pFormat, 
					int *pNumChannels, CUtexref hTexRef );
				CUresult (*cuTexRefGetFlags)( unsigned int *pFlags, 
					CUtexref hTexRef );
				CUresult (*cuParamSetSize)(CUfunction hfunc, 
					unsigned int numbytes);
				CUresult (*cuParamSeti)(CUfunction hfunc, int offset, 
					unsigned int value);
				CUresult (*cuParamSetf)(CUfunction hfunc, int offset, 
					float value);
				CUresult (*cuParamSetv)(CUfunction hfunc, int offset, 
					void * ptr, unsigned int numbytes);
				CUresult (*cuParamSetTexRef)(CUfunction hfunc, int texunit, 
					CUtexref hTexRef);
				CUresult (*cuLaunch)( CUfunction f );
				CUresult (*cuLaunchGrid)(CUfunction f, int grid_width, 
					int grid_height);
				CUresult (*cuLaunchGridAsync)( CUfunction f, 
					int grid_width, int grid_height, CUstream hStream );
				CUresult (*cuEventCreate)( CUevent *phEvent, 
					unsigned int Flags );
				CUresult (*cuEventRecord)( CUevent hEvent, 
					CUstream hStream );
				CUresult (*cuEventQuery)( CUevent hEvent );
				CUresult (*cuEventSynchronize)( CUevent hEvent );
				CUresult (*cuEventDestroy)( CUevent hEvent );
				CUresult (*cuEventElapsedTime)( float *pMilliseconds, 
					CUevent hStart, CUevent hEnd );
				CUresult (*cuStreamCreate)( CUstream *phStream, 
					unsigned int Flags );
				CUresult (*cuStreamQuery)( CUstream hStream );
				CUresult (*cuStreamSynchronize)( CUstream hStream );
				CUresult (*cuStreamDestroy)( CUstream hStream );

				CUresult (*cuGraphicsUnregisterResource)(
					CUgraphicsResource resource);
				CUresult (*cuGraphicsSubResourceGetMappedArray)(
					CUarray *pArray, CUgraphicsResource resource, 
					unsigned int arrayIndex, unsigned int mipLevel );
				CUresult (*cuGraphicsResourceGetMappedPointer)(
					CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource );
				CUresult (*cuGraphicsResourceSetMapFlags)(
					CUgraphicsResource resource, unsigned int flags ); 
				CUresult (*cuGraphicsMapResources)(unsigned int count, 
					CUgraphicsResource *resources, CUstream hStream );
				CUresult (*cuGraphicsUnmapResources)(unsigned int count, 
					CUgraphicsResource *resources, CUstream hStream );
				CUresult (*cuGetExportTable)(const void **ppExportTable,
					const CUuuid *pExportTableId);

				CUresult (*cuGLInit)();
				CUresult (*cuGLCtxCreate)(CUcontext *pCtx, 
					unsigned int Flags, CUdevice device);
				CUresult (*cuGLRegisterBufferObject)(GLuint bufferobj);
				CUresult (*cuGraphicsGLRegisterBuffer)( 
					CUgraphicsResource *pCudaResource, unsigned int buffer, 
					unsigned int Flags );
				CUresult (*cuGraphicsGLRegisterImage)( 
					CUgraphicsResource *pCudaResource, unsigned int image, 
					int target, unsigned int Flags);
				CUresult (*cuGLSetBufferObjectMapFlags)(GLuint buffer, unsigned int flags);
			
			public:
				/*! \brief The constructor zeros out all of the pointers */
				Interface();
				
				/*! \brief The destructor closes dlls */
				~Interface();
				/*! \brief Load the cuda driver */
				void load();
				/*! \brief Has the driver been loaded? */
				bool loaded() const;
				/*! \brief unloads the driver */
				void unload();
		};

	public:
		/*! \brief Interface to the CUDA driver */
		static Interface _interface;
		
	public:
		/*********************************
		** Initialization
		*********************************/
		static CUresult cuInit(unsigned int Flags);

		/*********************************
		** Driver Version Query
		*********************************/
		static CUresult cuDriverGetVersion(int *driverVersion);

		/************************************
		**
		**    Device management
		**
		***********************************/

		static CUresult cuDeviceGet(CUdevice *device, int ordinal);
		static CUresult cuDeviceGetCount(int *count);
		static CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
		static CUresult cuDeviceComputeCapability(int *major, int *minor, 
			CUdevice dev);
		static CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);
		static CUresult cuDeviceGetProperties(CUdevprop *prop, 
			CUdevice dev);
		static CUresult cuDeviceGetAttribute(int *pi, 
			CUdevice_attribute attrib, CUdevice dev);

		/************************************
		**
		**    Context management
		**
		***********************************/

		static CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, 
			CUdevice dev );
		static CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
		static CUresult cuCtxGetLimit(size_t *, CUlimit);
		static CUresult cuCtxDestroy( CUcontext ctx );
		static CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags);
		static CUresult cuCtxDetach(CUcontext ctx);
		static CUresult cuCtxPushCurrent( CUcontext ctx );
		static CUresult cuCtxPopCurrent( CUcontext *pctx );
		static CUresult cuCtxGetDevice(CUdevice *device);
		static CUresult cuCtxSynchronize(void);

		/************************************
		**
		**    Module management
		**
		***********************************/

		static CUresult cuModuleLoad(CUmodule *module, const char *fname);
		static CUresult cuModuleLoadData(CUmodule *module, 
			const void *image);
		static CUresult cuModuleLoadDataEx(CUmodule *module, 
			const void *image, unsigned int numOptions, 
			CUjit_option *options, void **optionValues);
		static CUresult cuModuleLoadFatBinary(CUmodule *module, 
			const void *fatCubin);
		static CUresult cuModuleUnload(CUmodule hmod);
		static CUresult cuModuleGetFunction(CUfunction *hfunc, 
			CUmodule hmod, const char *name);
		static CUresult cuModuleGetGlobal(CUdeviceptr *dptr, 
			size_t *bytes, CUmodule hmod, const char *name);
		static CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, 
			const char *name);

		/************************************
		**
		**    Memory management
		**
		***********************************/

		static CUresult cuMemGetInfo(size_t *free, 
			size_t *total);

		static CUresult cuMemAlloc( CUdeviceptr *dptr, 
			unsigned int bytesize);
		static CUresult cuMemAllocPitch( CUdeviceptr *dptr, 
					          size_t *pPitch,
					          unsigned int WidthInBytes, 
					          unsigned int Height, 
					          unsigned int ElementSizeBytes
					         );
		static CUresult cuMemFree(CUdeviceptr dptr);
		static CUresult cuMemGetAddressRange( CUdeviceptr *pbase, 
			size_t *psize, CUdeviceptr dptr );

		static CUresult cuMemAllocHost(void **pp, unsigned int bytesize);
		static CUresult cuMemFreeHost(void *p);

		static CUresult cuMemHostAlloc(void **pp, 
			unsigned long long bytesize, unsigned int Flags );
		static CUresult cuMemHostRegister(void *pp, 
			unsigned long long bytesize, unsigned int Flags );
		static CUresult cuMemHostUnregister(void *pp);

		static CUresult cuMemHostGetDevicePointer( CUdeviceptr *pdptr, 
			void *p, unsigned int Flags );
		static CUresult cuMemHostGetFlags( unsigned int *pFlags, void *p );

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
		static CUresult cuMemcpyHtoD (CUdeviceptr dstDevice, 
			const void *srcHost, unsigned int ByteCount );
		static CUresult cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, 
			unsigned int ByteCount );

		// device <-> device memory
		static CUresult cuMemcpyDtoD (CUdeviceptr dstDevice, 
			CUdeviceptr srcDevice, unsigned int ByteCount );

		// device <-> array memory
		static CUresult cuMemcpyDtoA ( CUarray dstArray, 
			unsigned int dstIndex, CUdeviceptr srcDevice, 
			unsigned int ByteCount );
		static CUresult cuMemcpyAtoD ( CUdeviceptr dstDevice, 
			CUarray hSrc, unsigned int SrcIndex, unsigned int ByteCount );

		// system <-> array memory
		static CUresult cuMemcpyHtoA( CUarray dstArray, 
			unsigned int dstIndex, const void *pSrc, 
			unsigned int ByteCount );
		static CUresult cuMemcpyAtoH( void *dstHost, CUarray srcArray, 
			unsigned int srcIndex, unsigned int ByteCount );

		// array <-> array memory
		static CUresult cuMemcpyAtoA( CUarray dstArray, 
			unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, 
			unsigned int ByteCount );

		// 2D memcpy

		static CUresult cuMemcpy2D( const CUDA_MEMCPY2D *pCopy );
		static CUresult cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy );

		// 3D memcpy

		static CUresult cuMemcpy3D( const CUDA_MEMCPY3D *pCopy );

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
		static CUresult cuMemcpyHtoDAsync (CUdeviceptr dstDevice, 
		const void *srcHost, unsigned int ByteCount, CUstream hStream );
		static CUresult cuMemcpyDtoHAsync (void *dstHost, 
		CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );

		// system <-> array memory
		static CUresult cuMemcpyHtoAAsync( CUarray dstArray, 
			unsigned int dstIndex, const void *pSrc, 
			unsigned int ByteCount, CUstream hStream );
		static CUresult cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, 
			unsigned int srcIndex, unsigned int ByteCount, 
			CUstream hStream );

		// 2D memcpy
		static CUresult cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, 
			CUstream hStream );

		// 3D memcpy
		static CUresult cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, 
			CUstream hStream );

		/************************************
		**
		**    Memset
		**
		***********************************/
		static CUresult cuMemsetD8( CUdeviceptr dstDevice, 
			unsigned char uc, unsigned int N );
		static CUresult cuMemsetD16( CUdeviceptr dstDevice, 
			unsigned short us, unsigned int N );
		static CUresult cuMemsetD32( CUdeviceptr dstDevice, 
			unsigned int ui, unsigned int N );

		static CUresult cuMemsetD2D8( CUdeviceptr dstDevice,
			unsigned int dstPitch, unsigned char uc, unsigned int Width, 
			unsigned int Height );
		static CUresult cuMemsetD2D16( CUdeviceptr dstDevice, 
			unsigned int dstPitch, unsigned short us, unsigned int Width, 
			unsigned int Height );
		static CUresult cuMemsetD2D32( CUdeviceptr dstDevice, 
			unsigned int dstPitch, unsigned int ui, unsigned int Width, 
			unsigned int Height );

		/************************************
		**
		**    Function management
		**
		***********************************/


		static CUresult cuFuncSetBlockShape (CUfunction hfunc, int x, 
			int y, int z);
		static CUresult cuFuncSetSharedSize (CUfunction hfunc, 
			unsigned int bytes);
		static CUresult cuFuncGetAttribute (int *pi, 
			CUfunction_attribute attrib, CUfunction hfunc);
		static CUresult cuFuncSetCacheConfig(CUfunction hFunc, CUfunc_cache config);

		/************************************
		**
		**    Array management 
		**
		***********************************/

		static CUresult cuArrayCreate( CUarray *pHandle, 
			const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
		static CUresult cuArrayGetDescriptor( 
			CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
		static CUresult cuArrayDestroy( CUarray hArray );

		static CUresult cuArray3DCreate( CUarray *pHandle, 
			const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray );
		static CUresult cuArray3DGetDescriptor( 
			CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );


		/************************************
		**
		**    Texture reference management
		**
		***********************************/
		static CUresult cuTexRefCreate( CUtexref *pTexRef );
		static CUresult cuTexRefDestroy( CUtexref hTexRef );

		static CUresult cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, 
			unsigned int Flags );
		static CUresult cuTexRefSetAddress( size_t *ByteOffset, 
			CUtexref hTexRef, CUdeviceptr dptr, size_t bytes );
		static CUresult cuTexRefSetAddress2D( CUtexref hTexRef, 
			const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, 
			unsigned int Pitch);
		static CUresult cuTexRefSetFormat( CUtexref hTexRef, 
			CUarray_format fmt, int NumPackedComponents );
		static CUresult cuTexRefSetAddressMode( CUtexref hTexRef, int dim, 
			CUaddress_mode am );
		static CUresult cuTexRefSetFilterMode( CUtexref hTexRef, 
			CUfilter_mode fm );
		static CUresult cuTexRefSetFlags( CUtexref hTexRef, 
			unsigned int Flags );

		static CUresult cuTexRefGetAddress( CUdeviceptr *pdptr, 
			CUtexref hTexRef );
		static CUresult cuTexRefGetArray( CUarray *phArray, 
			CUtexref hTexRef );
		static CUresult cuTexRefGetAddressMode( CUaddress_mode *pam, 
			CUtexref hTexRef, int dim );
		static CUresult cuTexRefGetFilterMode( CUfilter_mode *pfm, 
			CUtexref hTexRef );
		static CUresult cuTexRefGetFormat( CUarray_format *pFormat, 
			int *pNumChannels, CUtexref hTexRef );
		static CUresult cuTexRefGetFlags( unsigned int *pFlags, 
			CUtexref hTexRef );

		/************************************
		**
		**    Parameter management
		**
		***********************************/

		static CUresult cuParamSetSize (CUfunction hfunc, 
			unsigned int numbytes);
		static CUresult cuParamSeti    (CUfunction hfunc, int offset, 
			unsigned int value);
		static CUresult cuParamSetf    (CUfunction hfunc, int offset, 
			float value);
		static CUresult cuParamSetv    (CUfunction hfunc, int offset, 
			void * ptr, unsigned int numbytes);
		static CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, 
			CUtexref hTexRef);

		/************************************
		**
		**    Launch functions
		**
		***********************************/

		static CUresult cuLaunch ( CUfunction f );
		static CUresult cuLaunchGrid (CUfunction f, int grid_width, 
			int grid_height);
		static CUresult cuLaunchGridAsync( CUfunction f, int grid_width, 
			int grid_height, CUstream hStream );

		/************************************
		**
		**    Events
		**
		***********************************/
		static CUresult cuEventCreate( CUevent *phEvent, 
			unsigned int Flags );
		static CUresult cuEventRecord( CUevent hEvent, CUstream hStream );
		static CUresult cuEventQuery( CUevent hEvent );
		static CUresult cuEventSynchronize( CUevent hEvent );
		static CUresult cuEventDestroy( CUevent hEvent );
		static CUresult cuEventElapsedTime( float *pMilliseconds, 
			CUevent hStart, CUevent hEnd );

		/************************************
		**
		**    Streams
		**
		***********************************/
		static CUresult cuStreamCreate( CUstream *phStream, 
			unsigned int Flags );
		static CUresult cuStreamQuery( CUstream hStream );
		static CUresult cuStreamSynchronize( CUstream hStream );
		static CUresult cuStreamDestroy( CUstream hStream );

		/************************************
		**
		**    Graphics
		**
		***********************************/
		static CUresult cuGraphicsUnregisterResource(
			CUgraphicsResource resource);
		static CUresult cuGraphicsSubResourceGetMappedArray(
			CUarray *pArray, CUgraphicsResource resource, 
			unsigned int arrayIndex, unsigned int mipLevel );
		static CUresult cuGraphicsResourceGetMappedPointer(
			CUdeviceptr *pDevPtr, size_t *pSize, 
			CUgraphicsResource resource );
		static CUresult cuGraphicsResourceSetMapFlags(
			CUgraphicsResource resource, unsigned int flags ); 
		static CUresult cuGraphicsMapResources(unsigned int count, 
			CUgraphicsResource *resources, CUstream hStream );
		static CUresult cuGraphicsUnmapResources(unsigned int count, 
			CUgraphicsResource *resources, CUstream hStream );

		/************************************
		**
		**    Export Table
		**
		***********************************/
		static CUresult cuGetExportTable(const void **ppExportTable,
			const CUuuid *pExportTableId);

		/************************************
		**
		**    OpenGL
		**
		***********************************/
		static CUresult cuGLInit();
		static CUresult cuGLCtxCreate(CUcontext *pCtx, 
			unsigned int Flags, CUdevice device);
		static CUresult cuGLRegisterBufferObject(GLuint bufferobj);
		static CUresult cuGraphicsGLRegisterBuffer( 
			CUgraphicsResource *pCudaResource, unsigned int buffer, 
			unsigned int Flags );
		static CUresult cuGraphicsGLRegisterImage( 
			CUgraphicsResource *pCudaResource, unsigned int image, 
			int target, unsigned int Flags);
		static CUresult cuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int flags);

		static std::string toString(CUresult result);

};

}

#endif

