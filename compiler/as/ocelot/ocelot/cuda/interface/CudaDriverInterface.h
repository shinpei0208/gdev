/*!
	\file CudaDriverInterface.h

	\author Andrew Kerr <arkerr@gatech.edu>
	\brief implements a CUDA Driver API front-end interface for GPU Ocelot
	\date Sept 16 2010
	\location somewhere over western Europe
*/

#ifndef OCELOT_CUDADRIVERINTERFACE_H_INCLUDED
#define OCELOT_CUDADRIVERINTERFACE_H_INCLUDED

// C++ includes
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef min
#endif

#include <GL/gl.h>

// Ocelot includes
#include <ocelot/cuda/interface/cuda_internal.h>

namespace cuda
{
	/*! \brief Dynamic interface to the cuda driver */
	class CudaDriverInterface {
		public:

			static CudaDriverInterface *get();

		public:
			/*********************************
			** Initialization
			*********************************/
			virtual CUresult cuInit(unsigned int Flags);

			/*********************************
			** Driver Version Query
			*********************************/
			virtual CUresult cuDriverGetVersion(int *driverVersion);
			virtual CUresult cuGetExportTable(const void **ppExportTable,
				const CUuuid *pExportTableId);

			/************************************
			**
			**    Device management
			**
			***********************************/

			virtual CUresult cuDeviceGet(CUdevice *device, int ordinal);
			virtual CUresult cuDeviceGetCount(int *count);
			virtual CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
			virtual CUresult cuDeviceComputeCapability(int *major, int *minor, 
				CUdevice dev);
			virtual CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev);
			virtual CUresult cuDeviceGetProperties(CUdevprop *prop, 
				CUdevice dev);
			virtual CUresult cuDeviceGetAttribute(int *pi, 
				CUdevice_attribute attrib, CUdevice dev);

			/************************************
			**
			**    Context management
			**
			***********************************/

			virtual CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, 
				CUdevice dev );
			virtual CUresult cuCtxDestroy( CUcontext ctx );
			virtual CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags);
			virtual CUresult cuCtxDetach(CUcontext ctx);
			virtual CUresult cuCtxPushCurrent( CUcontext ctx );
			virtual CUresult cuCtxPopCurrent( CUcontext *pctx );
			virtual CUresult cuCtxGetDevice(CUdevice *device);
			virtual CUresult cuCtxSynchronize(void);

			/************************************
			**
			**    Module management
			**
			***********************************/

			virtual CUresult cuModuleLoad(CUmodule *module, const char *fname);
			virtual CUresult cuModuleLoadData(CUmodule *module, 
				const void *image);
			virtual CUresult cuModuleLoadDataEx(CUmodule *module, 
				const void *image, unsigned int numOptions, 
				CUjit_option *options, void **optionValues);
			virtual CUresult cuModuleLoadFatBinary(CUmodule *module, 
				const void *fatCubin);
			virtual CUresult cuModuleUnload(CUmodule hmod);
			virtual CUresult cuModuleGetFunction(CUfunction *hfunc, 
				CUmodule hmod, const char *name);
			virtual CUresult cuModuleGetGlobal(CUdeviceptr *dptr, 
				size_t *bytes, CUmodule hmod, const char *name);
			virtual CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, 
				const char *name);

			/************************************
			**
			**    Memory management
			**
			***********************************/

			virtual CUresult cuMemGetInfo(size_t *free, 
				size_t *total);

			virtual CUresult cuMemAlloc( CUdeviceptr *dptr, 
				unsigned int bytesize);
			virtual CUresult cuMemAllocPitch( CUdeviceptr *dptr, 
						          size_t *pPitch,
						          unsigned int WidthInBytes, 
						          unsigned int Height, 
						          unsigned int ElementSizeBytes
						         );
			virtual CUresult cuMemFree(CUdeviceptr dptr);
			virtual CUresult cuMemGetAddressRange( CUdeviceptr *pbase, 
				size_t *psize, CUdeviceptr dptr );

			virtual CUresult cuMemAllocHost(void **pp, unsigned int bytesize);
			virtual CUresult cuMemFreeHost(void *p);

			virtual CUresult cuMemHostAlloc(void **pp, 
				unsigned long long bytesize, unsigned int Flags );

			virtual CUresult cuMemHostGetDevicePointer( CUdeviceptr *pdptr, 
				void *p, unsigned int Flags );
			virtual CUresult cuMemHostGetFlags( unsigned int *pFlags, void *p );

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
			virtual CUresult cuMemcpyHtoD (CUdeviceptr dstDevice, 
				const void *srcHost, unsigned int ByteCount );
			virtual CUresult cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, 
				unsigned int ByteCount );
			virtual CUresult cuMemcpyHtoH (void *dstHost, const void *srcHost, 
				unsigned int ByteCount );

			// device <-> device memory
			virtual CUresult cuMemcpyDtoD (CUdeviceptr dstDevice, 
				CUdeviceptr srcDevice, unsigned int ByteCount );

			// device <-> array memory
			virtual CUresult cuMemcpyDtoA ( CUarray dstArray, 
				unsigned int dstIndex, CUdeviceptr srcDevice, 
				unsigned int ByteCount );
			virtual CUresult cuMemcpyAtoD ( CUdeviceptr dstDevice, 
				CUarray hSrc, unsigned int SrcIndex, unsigned int ByteCount );

			// system <-> array memory
			virtual CUresult cuMemcpyHtoA( CUarray dstArray, 
				unsigned int dstIndex, const void *pSrc, 
				unsigned int ByteCount );
			virtual CUresult cuMemcpyAtoH( void *dstHost, CUarray srcArray, 
				unsigned int srcIndex, unsigned int ByteCount );

			// array <-> array memory
			virtual CUresult cuMemcpyAtoA( CUarray dstArray, 
				unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, 
				unsigned int ByteCount );

			// 2D memcpy

			virtual CUresult cuMemcpy2D( const CUDA_MEMCPY2D *pCopy );
			virtual CUresult cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy );

			// 3D memcpy

			virtual CUresult cuMemcpy3D( const CUDA_MEMCPY3D *pCopy );

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
			virtual CUresult cuMemcpyHtoDAsync (CUdeviceptr dstDevice, 
			const void *srcHost, unsigned int ByteCount, CUstream hStream );
			virtual CUresult cuMemcpyDtoHAsync (void *dstHost, 
			CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream );

			// system <-> array memory
			virtual CUresult cuMemcpyHtoAAsync( CUarray dstArray, 
				unsigned int dstIndex, const void *pSrc, 
				unsigned int ByteCount, CUstream hStream );
			virtual CUresult cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, 
				unsigned int srcIndex, unsigned int ByteCount, 
				CUstream hStream );

			// 2D memcpy
			virtual CUresult cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, 
				CUstream hStream );

			// 3D memcpy
			virtual CUresult cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, 
				CUstream hStream );

			/************************************
			**
			**    Memset
			**
			***********************************/
			virtual CUresult cuMemsetD8( CUdeviceptr dstDevice, 
				unsigned char uc, unsigned int N );
			virtual CUresult cuMemsetD16( CUdeviceptr dstDevice, 
				unsigned short us, unsigned int N );
			virtual CUresult cuMemsetD32( CUdeviceptr dstDevice, 
				unsigned int ui, unsigned int N );

			virtual CUresult cuMemsetD2D8( CUdeviceptr dstDevice,
				unsigned int dstPitch, unsigned char uc, unsigned int Width, 
				unsigned int Height );
			virtual CUresult cuMemsetD2D16( CUdeviceptr dstDevice, 
				unsigned int dstPitch, unsigned short us, unsigned int Width, 
				unsigned int Height );
			virtual CUresult cuMemsetD2D32( CUdeviceptr dstDevice, 
				unsigned int dstPitch, unsigned int ui, unsigned int Width, 
				unsigned int Height );

			/************************************
			**
			**    Function management
			**
			***********************************/


			virtual CUresult cuFuncSetBlockShape (CUfunction hfunc, int x, 
				int y, int z);
			virtual CUresult cuFuncSetSharedSize (CUfunction hfunc, 
				unsigned int bytes);
			virtual CUresult cuFuncGetAttribute (int *pi, 
				CUfunction_attribute attrib, CUfunction hfunc);
			virtual CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);

			/************************************
			**
			**    Array management 
			**
			***********************************/

			virtual CUresult cuArrayCreate( CUarray *pHandle, 
				const CUDA_ARRAY_DESCRIPTOR *pAllocateArray );
			virtual CUresult cuArrayGetDescriptor( 
				CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray );
			virtual CUresult cuArrayDestroy( CUarray hArray );

			virtual CUresult cuArray3DCreate( CUarray *pHandle, 
				const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray );
			virtual CUresult cuArray3DGetDescriptor( 
				CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray );


			/************************************
			**
			**    Texture reference management
			**
			***********************************/
			virtual CUresult cuTexRefCreate( CUtexref *pTexRef );
			virtual CUresult cuTexRefDestroy( CUtexref hTexRef );

			virtual CUresult cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, 
				unsigned int Flags );
			virtual CUresult cuTexRefSetAddress( size_t *ByteOffset, 
				CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes );
			virtual CUresult cuTexRefSetAddress2D( CUtexref hTexRef, 
				const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, 
				unsigned int Pitch);
			virtual CUresult cuTexRefSetFormat( CUtexref hTexRef, 
				CUarray_format fmt, int NumPackedComponents );
			virtual CUresult cuTexRefSetAddressMode( CUtexref hTexRef, int dim, 
				CUaddress_mode am );
			virtual CUresult cuTexRefSetFilterMode( CUtexref hTexRef, 
				CUfilter_mode fm );
			virtual CUresult cuTexRefSetFlags( CUtexref hTexRef, 
				unsigned int Flags );

			virtual CUresult cuTexRefGetAddress( CUdeviceptr *pdptr, 
				CUtexref hTexRef );
			virtual CUresult cuTexRefGetArray( CUarray *phArray, 
				CUtexref hTexRef );
			virtual CUresult cuTexRefGetAddressMode( CUaddress_mode *pam, 
				CUtexref hTexRef, int dim );
			virtual CUresult cuTexRefGetFilterMode( CUfilter_mode *pfm, 
				CUtexref hTexRef );
			virtual CUresult cuTexRefGetFormat( CUarray_format *pFormat, 
				int *pNumChannels, CUtexref hTexRef );
			virtual CUresult cuTexRefGetFlags( unsigned int *pFlags, 
				CUtexref hTexRef );

			/************************************
			**
			**    Parameter management
			**
			***********************************/

			virtual CUresult cuParamSetSize (CUfunction hfunc, 
				unsigned int numbytes);
			virtual CUresult cuParamSeti    (CUfunction hfunc, int offset, 
				unsigned int value);
			virtual CUresult cuParamSetf    (CUfunction hfunc, int offset, 
				float value);
			virtual CUresult cuParamSetv    (CUfunction hfunc, int offset, 
				void * ptr, unsigned int numbytes);
			virtual CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, 
				CUtexref hTexRef);

			/************************************
			**
			**    Launch functions
			**
			***********************************/

			virtual CUresult cuLaunch ( CUfunction f );
			virtual CUresult cuLaunchGrid (CUfunction f, int grid_width, 
				int grid_height);
			virtual CUresult cuLaunchGridAsync( CUfunction f, int grid_width, 
				int grid_height, CUstream hStream );

			/************************************
			**
			**    Events
			**
			***********************************/
			virtual CUresult cuEventCreate( CUevent *phEvent, 
				unsigned int Flags );
			virtual CUresult cuEventRecord( CUevent hEvent, CUstream hStream );
			virtual CUresult cuEventQuery( CUevent hEvent );
			virtual CUresult cuEventSynchronize( CUevent hEvent );
			virtual CUresult cuEventDestroy( CUevent hEvent );
			virtual CUresult cuEventElapsedTime( float *pMilliseconds, 
				CUevent hStart, CUevent hEnd );

			/************************************
			**
			**    Streams
			**
			***********************************/
			virtual CUresult cuStreamCreate( CUstream *phStream, 
				unsigned int Flags );
			virtual CUresult cuStreamQuery( CUstream hStream );
			virtual CUresult cuStreamSynchronize( CUstream hStream );
			virtual CUresult cuStreamDestroy( CUstream hStream );

			/************************************
			**
			**    Graphics
			**
			***********************************/
			virtual CUresult cuGraphicsUnregisterResource(
				CUgraphicsResource resource);
			virtual CUresult cuGraphicsSubResourceGetMappedArray(
				CUarray *pArray, CUgraphicsResource resource, 
				unsigned int arrayIndex, unsigned int mipLevel );
			virtual CUresult cuGraphicsResourceGetMappedPointer(
				CUdeviceptr *pDevPtr, size_t *pSize, 
				CUgraphicsResource resource );
			virtual CUresult cuGraphicsResourceSetMapFlags(
				CUgraphicsResource resource, unsigned int flags ); 
			virtual CUresult cuGraphicsMapResources(unsigned int count, 
				CUgraphicsResource *resources, CUstream hStream );
			virtual CUresult cuGraphicsUnmapResources(unsigned int count, 
				CUgraphicsResource *resources, CUstream hStream );

			/************************************
			**
			**    OpenGL
			**
			***********************************/
			virtual CUresult cuGLInit();
			virtual CUresult cuGLCtxCreate(CUcontext *pCtx, 
				unsigned int Flags, CUdevice device);
			virtual CUresult cuGraphicsGLRegisterBuffer( 
				CUgraphicsResource *pCudaResource, unsigned int buffer, 
				unsigned int Flags );
			virtual CUresult cuGraphicsGLRegisterImage( 
				CUgraphicsResource *pCudaResource, unsigned int image, 
				int target, unsigned int Flags);
			virtual CUresult cuGLRegisterBufferObject(GLuint bufferobj);
			virtual CUresult cuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int flags);

			std::string toString(CUresult result);

	};

}


#endif

