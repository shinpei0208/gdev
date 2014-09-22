/*! \file CudaDriver.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Friday March 26, 2010
	\brief The source file for the CudaDriver class.
*/
#ifndef CUDA_DRIVER_CPP_INCLUDED
#define CUDA_DRIVER_CPP_INCLUDED

// Ocelot includes
#include <ocelot/cuda/interface/CudaDriver.h>

// hydrazine includes
#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/debug.h>

// Linux system headers
#if __GNUC__
	#include <dlfcn.h>
#else 
	// TODO Add dynamic loading support on windows
	#define dlopen(a,b) 0
	#define dlclose(a) -1
	#define dlerror() "Unknown error"
	#define dlsym(a,b) 0
#endif

////////////////////////////////////////////////////////////////////////////////

// Macros
#define CHECK() {assertM(_interface.loaded(), __FUNCTION__ \
	<< " called without loading the driver.");\
	reportE(REPORT_ALL_CALLS, __FUNCTION__);}

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0
#define REPORT_ALL_CALLS 1

////////////////////////////////////////////////////////////////////////////////
// Dynamic linking

#define DynLink( function ) hydrazine::bit_cast( function, dlsym(_driver, #function))

// 64-bit functions have _v2 suffixes
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
#define DynLinkV( function ) hydrazine::bit_cast( function, dlsym(_driver, #function "_v2"))
#else
#define DynLinkV( function ) hydrazine::bit_cast( function, dlsym(_driver, #function))
#endif

////////////////////////////////////////////////////////////////////////////////

namespace cuda
{
	CudaDriver::Interface::Interface() : _driver( 0 )
	{
		_libname = "libcuda.so";
	}
	
	CudaDriver::Interface::~Interface()
	{
		unload();
	}
	
	/*! \brief unloads the driver */
	void CudaDriver::Interface::unload() {
	
		if( _driver )
		{
			report( "Closing " << _libname );
			#if __GNUC__
			dlclose( _driver );
			#else
			assertM(false, "CUDA Driver support not compiled into Ocelot.");
			#endif
			report("closed.");
		}
	}
	
	void CudaDriver::Interface::load()
	{
		if( _driver != 0 ) return;
		#if __GNUC__
		report( "Loading " << _libname );
		_driver = dlopen( _libname.c_str(), RTLD_LAZY );
		if( _driver == 0 )
		{
			report( "Failed to load cuda driver." );
			report( "  " << dlerror() );
			return;
		}
		
		DynLink(cuInit);
		DynLink(cuDriverGetVersion);
		DynLink(cuDeviceGet);
		DynLink(cuDeviceGetCount);
		DynLink(cuDeviceGetName);
		DynLink(cuDeviceComputeCapability);

		DynLinkV(cuDeviceTotalMem);
		
		DynLink(cuDeviceGetProperties);
		DynLink(cuDeviceGetAttribute);
		DynLink(cuCtxGetLimit);
		DynLink(cuCtxGetApiVersion);
		DynLinkV(cuCtxCreate);
		
		DynLink(cuCtxDestroy);
		DynLink(cuCtxAttach);
		DynLink(cuCtxDetach);
		DynLink(cuCtxPushCurrent);
		DynLink(cuCtxPopCurrent);
		DynLink(cuCtxGetDevice);
		DynLink(cuCtxSynchronize);
		DynLink(cuModuleLoad);
		DynLink(cuModuleLoadData);
		DynLink(cuModuleLoadDataEx);
		DynLink(cuModuleLoadFatBinary);
		DynLink(cuModuleUnload);
		DynLink(cuModuleGetFunction);
		
		DynLinkV(cuModuleGetGlobal);
		
		DynLink(cuModuleGetTexRef);
		
		DynLinkV(cuMemGetInfo);
		DynLinkV(cuMemAlloc);
		DynLinkV(cuMemAllocPitch);
		DynLinkV(cuMemFree);
		DynLinkV(cuMemGetAddressRange);

		DynLinkV(cuMemAllocHost);
		DynLinkV(cuMemHostRegister);
		DynLinkV(cuMemHostUnregister);
		
		DynLink(cuMemFreeHost);
		DynLink(cuMemHostAlloc);
		
		DynLinkV(cuMemHostGetDevicePointer);
		DynLink(cuMemHostGetFlags);
		DynLinkV(cuMemcpyHtoD);
		DynLinkV(cuMemcpyDtoH);
		DynLinkV(cuMemcpyDtoD);
		DynLinkV(cuMemcpyDtoA);
		DynLinkV(cuMemcpyAtoD);
		DynLinkV(cuMemcpyHtoA);
		DynLinkV(cuMemcpyAtoH);
		DynLinkV(cuMemcpyAtoA);
		DynLinkV(cuMemcpy2D);
		DynLinkV(cuMemcpy2DUnaligned);
		DynLinkV(cuMemcpy3D);
		DynLinkV(cuMemcpyHtoDAsync);
		DynLinkV(cuMemcpyDtoHAsync);
		DynLinkV(cuMemcpyHtoAAsync);
		DynLinkV(cuMemcpyAtoHAsync);
		DynLinkV(cuMemcpy2DAsync);
		DynLinkV(cuMemcpy3DAsync);
		DynLinkV(cuMemsetD8);
		DynLinkV(cuMemsetD16);
		DynLinkV(cuMemsetD32);
		DynLinkV(cuMemsetD2D8);
		DynLinkV(cuMemsetD2D16);
		DynLinkV(cuMemsetD2D32);
		
		DynLink(cuFuncSetBlockShape);
		DynLink(cuFuncSetSharedSize);
		DynLink(cuFuncGetAttribute);
		DynLink(cuFuncSetCacheConfig);
		
		DynLinkV(cuArrayCreate);
		DynLinkV(cuArrayGetDescriptor);
		DynLink(cuArrayDestroy);
		DynLinkV(cuArray3DCreate);
		DynLinkV(cuArray3DGetDescriptor);
		DynLink(cuTexRefCreate);
		DynLink(cuTexRefDestroy);
		DynLink(cuTexRefSetArray);
		DynLinkV(cuTexRefSetAddress);
		DynLinkV(cuTexRefSetAddress2D);
		DynLink(cuTexRefSetFormat);
		DynLink(cuTexRefSetAddressMode);
		DynLink(cuTexRefSetFilterMode);
		DynLink(cuTexRefSetFlags);
		DynLinkV(cuTexRefGetAddress);
		DynLink(cuTexRefGetArray);
		DynLink(cuTexRefGetAddressMode);
		DynLink(cuTexRefGetFilterMode);
		DynLink(cuTexRefGetFormat);
		DynLink(cuTexRefGetFlags);
		DynLink(cuParamSetSize);
		DynLink(cuParamSeti);
		DynLink(cuParamSetf);
		DynLink(cuParamSetv);
		DynLink(cuParamSetTexRef);
		DynLink(cuLaunch);
		DynLink(cuLaunchGrid);
		DynLink(cuLaunchGridAsync);
		DynLink(cuEventCreate);
		DynLink(cuEventRecord);
		DynLink(cuEventQuery);
		DynLink(cuEventSynchronize);
		DynLink(cuEventDestroy);
		DynLink(cuEventElapsedTime);
		DynLink(cuStreamCreate);
		DynLink(cuStreamQuery);
		DynLink(cuStreamSynchronize);
		DynLink(cuStreamDestroy);

		DynLink(cuGraphicsUnregisterResource);
		DynLink(cuGraphicsSubResourceGetMappedArray);
		DynLinkV(cuGraphicsResourceGetMappedPointer);
		DynLink(cuGraphicsResourceSetMapFlags);
		DynLink(cuGraphicsMapResources);
		DynLink(cuGraphicsUnmapResources);

		DynLink(cuGetExportTable);

		DynLink(cuGLInit);
		DynLinkV(cuGLCtxCreate);
		DynLink(cuGraphicsGLRegisterBuffer);
		DynLink(cuGraphicsGLRegisterImage);
		DynLink(cuGLRegisterBufferObject);
		DynLink(cuGLSetBufferObjectMapFlags);

		CUresult result = (*cuDriverGetVersion)(&_version);
		
		if (result == CUDA_SUCCESS) {
			report(" Driver version is: " << _version << " and was called successfully");
		}
		else {
			report("cuDriverGetVersion() returned " << result);
		}

		#else
		assertM(false, "CUDA Driver support not compiled into Ocelot.");
		#endif
	}

	bool CudaDriver::Interface::loaded() const
	{
		return _driver != 0;
	}

	CudaDriver::Interface CudaDriver::_interface;

	CUresult CudaDriver::cuInit(unsigned int Flags)
	{
		// Handle multiple calls
		if(_interface.loaded()) return CUDA_SUCCESS;
		_interface.load();
		if( !_interface.loaded() ) return CUDA_ERROR_NO_DEVICE;
		report("cuInit");
		return (*_interface.cuInit)(Flags);
	}

	CUresult CudaDriver::cuDriverGetVersion(int *driverVersion)
	{
		CHECK();
		return (*_interface.cuDriverGetVersion)(driverVersion);
	}

	CUresult CudaDriver::cuDeviceGet(CUdevice *device, int ordinal)
	{
		CHECK();
		return (*_interface.cuDeviceGet)(device, ordinal);
	}

	CUresult CudaDriver::cuDeviceGetCount(int *count)
	{
		if( !_interface.loaded() )
		{
			*count = 0;
			return CUDA_SUCCESS;
		}
		
		CUresult result= (*_interface.cuDeviceGetCount)(count);
		
		if(result != CUDA_SUCCESS)
		{
			*count = 0;
			return CUDA_SUCCESS;
		}
		
		return result;
	}

	CUresult CudaDriver::cuDeviceGetName(char *name, int len, CUdevice dev)
	{
		CHECK();
		return (*_interface.cuDeviceGetName)(name, len, dev);
	}

	CUresult CudaDriver::cuDeviceComputeCapability(int *major, int *minor, 
		CUdevice dev)
	{
		CHECK();
		return (*_interface.cuDeviceComputeCapability)(major, minor, dev);
	}

	CUresult CudaDriver::cuDeviceTotalMem(size_t *bytes, CUdevice dev)
	{
		CHECK();
		return (*_interface.cuDeviceTotalMem)(bytes, dev);
	}

	CUresult CudaDriver::cuDeviceGetProperties(CUdevprop *prop, CUdevice dev)
	{
		CHECK();
		return (*_interface.cuDeviceGetProperties)(prop, dev);
	}

	CUresult CudaDriver::cuDeviceGetAttribute(int *pi, 
		CUdevice_attribute attrib, CUdevice dev)
	{
		CHECK();
		return (*_interface.cuDeviceGetAttribute)(pi, attrib, dev);
	}

	CUresult CudaDriver::cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
		CHECK();
		return (*_interface.cuCtxGetApiVersion)(ctx, version);
	}
	CUresult CudaDriver::cuCtxCreate(CUcontext *pctx, unsigned int flags, 
		CUdevice dev )
	{
		CHECK();
		return (*_interface.cuCtxCreate)(pctx, flags, dev);
	}
	
	CUresult CudaDriver::cuCtxGetLimit(size_t *pval, CUlimit limit) 
	{
		CHECK();
		return (*_interface.cuCtxGetLimit)(pval, limit);
	}

	CUresult CudaDriver::cuCtxDestroy( CUcontext ctx )
	{
		CHECK();
		return (*_interface.cuCtxDestroy)(ctx);
	}

	CUresult CudaDriver::cuCtxAttach(CUcontext *pctx, unsigned int flags)
	{
		CHECK();
		return (*_interface.cuCtxAttach)(pctx, flags);
	}

	CUresult CudaDriver::cuCtxDetach(CUcontext ctx)
	{
		CHECK();
		return (*_interface.cuCtxDetach)(ctx);
	}

	CUresult CudaDriver::cuCtxPushCurrent( CUcontext ctx )
	{
		CHECK();
		return (*_interface.cuCtxPushCurrent)(ctx);
	}

	CUresult CudaDriver::cuCtxPopCurrent( CUcontext *pctx )
	{
		CHECK();
		return (*_interface.cuCtxPopCurrent)(pctx);
	}

	CUresult CudaDriver::cuCtxGetDevice(CUdevice *device)
	{
		CHECK();
		return (*_interface.cuCtxGetDevice)(device);
	}

	CUresult CudaDriver::cuCtxSynchronize(void)
	{
		CHECK();
		return (*_interface.cuCtxSynchronize)();
	}

	CUresult CudaDriver::cuModuleLoad(CUmodule *module, const char *fname)
	{
		CHECK();
		return (*_interface.cuModuleLoad)(module, fname);
	}

	CUresult CudaDriver::cuModuleLoadData(CUmodule *module, const void *image)
	{
		CHECK();
		return (*_interface.cuModuleLoadData)(module, image);
	}

	CUresult CudaDriver::cuModuleLoadDataEx(CUmodule *module, 
		const void *image, unsigned int numOptions, CUjit_option *options, 
		void **optionValues)
	{
		CHECK();
		return (*_interface.cuModuleLoadDataEx)(module, image, 
			numOptions, options, optionValues);
	}

	CUresult CudaDriver::cuModuleLoadFatBinary(CUmodule *module, 
		const void *fatCubin)
	{
		CHECK();
		return (*_interface.cuModuleLoadFatBinary)(module, fatCubin);
	}

	CUresult CudaDriver::cuModuleUnload(CUmodule hmod)
	{
		CHECK();
		return (*_interface.cuModuleUnload)(hmod);
	}

	CUresult CudaDriver::cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, 
		const char *name)
	{
		CHECK();
		return (*_interface.cuModuleGetFunction)(hfunc, hmod, name);
	}

	CUresult CudaDriver::cuModuleGetGlobal(CUdeviceptr *dptr, 
		size_t *bytes, CUmodule hmod, const char *name)
	{
		CHECK();
		return (*_interface.cuModuleGetGlobal)(dptr, bytes, hmod, name);
	}

	CUresult CudaDriver::cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, 
		const char *name)
	{
		CHECK();
		return (*_interface.cuModuleGetTexRef)(pTexRef, hmod, name);
	}


	CUresult CudaDriver::cuMemGetInfo(size_t *free, size_t *total)
	{
		CHECK();
		return (*_interface.cuMemGetInfo)(free, total);
	}


	CUresult CudaDriver::cuMemAlloc( CUdeviceptr *dptr, unsigned int bytesize)
	{
		CHECK();
		return (*_interface.cuMemAlloc)(dptr, bytesize);
	}

	CUresult CudaDriver::cuMemAllocPitch( CUdeviceptr *dptr, 
				          size_t *pPitch,
				          unsigned int WidthInBytes, 
				          unsigned int Height, 
				          unsigned int ElementSizeBytes
				         )
	{
		CHECK();
		return (*_interface.cuMemAllocPitch)(dptr, pPitch, WidthInBytes, 
			Height, ElementSizeBytes);
	}

	CUresult CudaDriver::cuMemFree(CUdeviceptr dptr)
	{
		CHECK();
		return (*_interface.cuMemFree)(dptr);
	}

	CUresult CudaDriver::cuMemGetAddressRange( CUdeviceptr *pbase, 
		size_t *psize, CUdeviceptr dptr )
	{
		CHECK();
		return (*_interface.cuMemGetAddressRange)(pbase, psize, dptr);
	}

	CUresult CudaDriver::cuMemAllocHost(void **pp, unsigned int bytesize)
	{
		CHECK();
		return (*_interface.cuMemAllocHost)(pp, bytesize);
	}

	CUresult CudaDriver::cuMemFreeHost(void *p)
	{
		CHECK();
		return (*_interface.cuMemFreeHost)(p);
	}

	CUresult CudaDriver::cuMemHostAlloc(void **pp, 
		unsigned long long bytesize, unsigned int Flags )
	{
		CHECK();
		return (*_interface.cuMemHostAlloc)(pp, bytesize, Flags);
	}

	CUresult CudaDriver::cuMemHostRegister(void *pp, 
		unsigned long long bytesize, unsigned int Flags )
	{
		CHECK();
		return (*_interface.cuMemHostRegister)(pp, bytesize, Flags);
	}

	CUresult CudaDriver::cuMemHostUnregister(void *pp)
	{
		CHECK();
		return (*_interface.cuMemHostUnregister)(pp);
	}

	CUresult CudaDriver::cuMemHostGetDevicePointer( CUdeviceptr *pdptr, 
		void *p, unsigned int Flags )
	{
		CHECK();
		return (*_interface.cuMemHostGetDevicePointer)(pdptr, p, Flags);
	}

	CUresult CudaDriver::cuMemHostGetFlags( unsigned int *pFlags, void *p )
	{
		CHECK();
		return (*_interface.cuMemHostGetFlags)(pFlags, p);
	}

	CUresult CudaDriver::cuMemcpyHtoD(CUdeviceptr dstDevice, 
		const void *srcHost, unsigned int ByteCount )
	{
		CHECK();
		return (*_interface.cuMemcpyHtoD)(dstDevice, srcHost, ByteCount);
	}

	CUresult CudaDriver::cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, 
		unsigned int ByteCount )
	{
		CHECK();
		return (*_interface.cuMemcpyDtoH)(dstHost, srcDevice, ByteCount);
	}

	CUresult CudaDriver::cuMemcpyDtoD(CUdeviceptr dstDevice, 
		CUdeviceptr srcDevice, unsigned int ByteCount )
	{
		CHECK();
		return (*_interface.cuMemcpyDtoD)(dstDevice, srcDevice, ByteCount);
	}

	CUresult CudaDriver::cuMemcpyDtoA( CUarray dstArray, 
		unsigned int dstIndex, CUdeviceptr srcDevice, unsigned int ByteCount )
	{
		CHECK();
		return (*_interface.cuMemcpyDtoA)(dstArray, dstIndex, srcDevice, 
			ByteCount);
	}

	CUresult CudaDriver::cuMemcpyAtoD( CUdeviceptr dstDevice, CUarray hSrc, 
		unsigned int SrcIndex, unsigned int ByteCount )
	{
		CHECK();
		return (*_interface.cuMemcpyAtoD)(dstDevice, hSrc, SrcIndex, 
			ByteCount);
	}

	CUresult CudaDriver::cuMemcpyHtoA( CUarray dstArray, 
		unsigned int dstIndex, const void *pSrc, unsigned int ByteCount )
	{
		CHECK();
		return (*_interface.cuMemcpyHtoA)(dstArray, dstIndex, pSrc, ByteCount);
	}

	CUresult CudaDriver::cuMemcpyAtoH( void *dstHost, CUarray srcArray, 
		unsigned int srcIndex, unsigned int ByteCount )
	{
		CHECK();
		return (*_interface.cuMemcpyAtoH)(dstHost, srcArray, srcIndex, 
			ByteCount);
	}

	CUresult CudaDriver::cuMemcpyAtoA( CUarray dstArray, 
		unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, 
		unsigned int ByteCount )
	{
		CHECK();
		return (*_interface.cuMemcpyAtoA)(dstArray, dstIndex, srcArray, 
			srcIndex, ByteCount);
	}

	CUresult CudaDriver::cuMemcpy2D( const CUDA_MEMCPY2D *pCopy )
	{
		CHECK();
		return (*_interface.cuMemcpy2D)(pCopy);
	}

	CUresult CudaDriver::cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy )
	{
		CHECK();
		return (*_interface.cuMemcpy2DUnaligned)(pCopy);
	}

	CUresult CudaDriver::cuMemcpy3D( const CUDA_MEMCPY3D *pCopy )
	{
		CHECK();
		return (*_interface.cuMemcpy3D)(pCopy);
	}
	
	CUresult CudaDriver::cuMemcpyHtoDAsync (CUdeviceptr dstDevice, 
		const void *srcHost, unsigned int ByteCount, CUstream hStream )
	{
		CHECK();
		return (*_interface.cuMemcpyHtoDAsync)(dstDevice, srcHost, 
			ByteCount, hStream);
	}

	CUresult CudaDriver::cuMemcpyDtoHAsync (void *dstHost, 
	CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream )
	{
		CHECK();
		return (*_interface.cuMemcpyDtoHAsync)(dstHost, srcDevice, 
			ByteCount, hStream);
	}

	CUresult CudaDriver::cuMemcpyHtoAAsync( CUarray dstArray, 
		unsigned int dstIndex, const void *pSrc, unsigned int ByteCount, 
		CUstream hStream )
	{
		CHECK();
		return (*_interface.cuMemcpyHtoAAsync)(dstArray, dstIndex, pSrc, 
			ByteCount, hStream);
	}

	CUresult CudaDriver::cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, 
		unsigned int srcIndex, unsigned int ByteCount, CUstream hStream )
	{
		CHECK();
		return (*_interface.cuMemcpyAtoHAsync)(dstHost, srcArray, srcIndex, 
			ByteCount, hStream);
	}

	CUresult CudaDriver::cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, 
		CUstream hStream )
	{
		CHECK();
		return (*_interface.cuMemcpy2DAsync)(pCopy, hStream);
	}

	CUresult CudaDriver::cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, 
		CUstream hStream )
	{
		CHECK();
		return (*_interface.cuMemcpy3DAsync)(pCopy, hStream);
	}

	CUresult CudaDriver::cuMemsetD8( CUdeviceptr dstDevice, unsigned char uc, 
		unsigned int N )
	{
		CHECK();
		return (*_interface.cuMemsetD8)(dstDevice, uc, N);
	}

	CUresult CudaDriver::cuMemsetD16( CUdeviceptr dstDevice, 
		unsigned short us, unsigned int N )
	{
		CHECK();
		return (*_interface.cuMemsetD16)(dstDevice, us, N);
	}

	CUresult CudaDriver::cuMemsetD32( CUdeviceptr dstDevice, unsigned int ui, 
		unsigned int N )
	{
		CHECK();
		return (*_interface.cuMemsetD32)(dstDevice, ui, N);
	}

	CUresult CudaDriver::cuMemsetD2D8( CUdeviceptr dstDevice, 
		unsigned int dstPitch, unsigned char uc, unsigned int Width, 
		unsigned int Height )
	{
		CHECK();
		return (*_interface.cuMemsetD2D8)(dstDevice, dstPitch, uc, 
			Width, Height);
	}

	CUresult CudaDriver::cuMemsetD2D16( CUdeviceptr dstDevice, 
		unsigned int dstPitch, unsigned short us, unsigned int Width, 
		unsigned int Height )
	{
		CHECK();
		return (*_interface.cuMemsetD2D16)(dstDevice, dstPitch, us, 
			Width, Height);
	}

	CUresult CudaDriver::cuMemsetD2D32( CUdeviceptr dstDevice, 
		unsigned int dstPitch, unsigned int ui, unsigned int Width, 
		unsigned int Height )
	{
		CHECK();
		return (*_interface.cuMemsetD2D32)(dstDevice, dstPitch, ui, 
			Width, Height);
	}


	CUresult CudaDriver::cuFuncSetBlockShape(CUfunction hfunc, int x, int y, 
		int z)
	{
		CHECK();
		return (*_interface.cuFuncSetBlockShape)(hfunc, x, y, z);
	}

	CUresult CudaDriver::cuFuncSetSharedSize(CUfunction hfunc, 
		unsigned int bytes)
	{
		CHECK();
		return (*_interface.cuFuncSetSharedSize)(hfunc, bytes);
	}

	CUresult CudaDriver::cuFuncGetAttribute(int *pi, 
		CUfunction_attribute attrib, CUfunction hfunc)
	{
		CHECK();
		return (*_interface.cuFuncGetAttribute)(pi, attrib, hfunc);
	}
	
	CUresult CudaDriver::cuFuncSetCacheConfig(CUfunction hFunc, CUfunc_cache config)
	{
		CHECK();
		return (*_interface.cuFuncSetCacheConfig)(hFunc, config);
	}

	CUresult CudaDriver::cuArrayCreate( CUarray *pHandle, 
		const CUDA_ARRAY_DESCRIPTOR *pAllocateArray )
	{
		CHECK();
		return (*_interface.cuArrayCreate)(pHandle, pAllocateArray);
	}

	CUresult CudaDriver::cuArrayGetDescriptor( 
		CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
	{
		CHECK();
		return (*_interface.cuArrayGetDescriptor)(pArrayDescriptor, hArray);
	}

	CUresult CudaDriver::cuArrayDestroy( CUarray hArray )
	{
		CHECK();
		return (*_interface.cuArrayDestroy)(hArray);
	}

	CUresult CudaDriver::cuArray3DCreate( CUarray *pHandle, 
		const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray )
	{
		CHECK();
		return (*_interface.cuArray3DCreate)(pHandle, pAllocateArray);
	}

	CUresult CudaDriver::cuArray3DGetDescriptor( 
		CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray )
	{
		CHECK();
		return (*_interface.cuArray3DGetDescriptor)(pArrayDescriptor, hArray);
	}

	CUresult CudaDriver::cuTexRefCreate( CUtexref *pTexRef )
	{
		CHECK();
		return (*_interface.cuTexRefCreate)(pTexRef);
	}

	CUresult CudaDriver::cuTexRefDestroy( CUtexref hTexRef )
	{
		CHECK();
		return (*_interface.cuTexRefDestroy)(hTexRef);
	}

	CUresult CudaDriver::cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, 
		unsigned int Flags )
	{
		CHECK();
		return (*_interface.cuTexRefSetArray)(hTexRef, hArray, Flags);
	}

	CUresult CudaDriver::cuTexRefSetAddress( size_t *ByteOffset, 
		CUtexref hTexRef, CUdeviceptr dptr, size_t bytes )
	{
		CHECK();
		return (*_interface.cuTexRefSetAddress)(ByteOffset, hTexRef, 
			dptr, bytes);
	}

	CUresult CudaDriver::cuTexRefSetAddress2D( CUtexref hTexRef, 
		const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, 
		unsigned int Pitch)
	{
		CHECK();
		return (*_interface.cuTexRefSetAddress2D)(hTexRef, desc, dptr, Pitch);
	}

	CUresult CudaDriver::cuTexRefSetFormat( CUtexref hTexRef, 
		CUarray_format fmt, int NumPackedComponents )
	{
		CHECK();
		return (*_interface.cuTexRefSetFormat)(hTexRef, fmt, 
			NumPackedComponents);
	}

	CUresult CudaDriver::cuTexRefSetAddressMode( CUtexref hTexRef, int dim, 
		CUaddress_mode am )
	{
		CHECK();
		return (*_interface.cuTexRefSetAddressMode)(hTexRef, dim, am);
	}

	CUresult CudaDriver::cuTexRefSetFilterMode( CUtexref hTexRef, 
		CUfilter_mode fm )
	{
		CHECK();
		return (*_interface.cuTexRefSetFilterMode)(hTexRef, fm);
	}

	CUresult CudaDriver::cuTexRefSetFlags( CUtexref hTexRef, 
		unsigned int Flags )
	{
		CHECK();
		return (*_interface.cuTexRefSetFlags)(hTexRef, Flags);
	}


	CUresult CudaDriver::cuTexRefGetAddress( CUdeviceptr *pdptr, 
		CUtexref hTexRef )
	{
		CHECK();
		return (*_interface.cuTexRefGetAddress)(pdptr, hTexRef);
	}

	CUresult CudaDriver::cuTexRefGetArray( CUarray *phArray, CUtexref hTexRef )
	{
		CHECK();
		return (*_interface.cuTexRefGetArray)(phArray, hTexRef);
	}

	CUresult CudaDriver::cuTexRefGetAddressMode( CUaddress_mode *pam, 
		CUtexref hTexRef, int dim )
	{
		CHECK();
		return (*_interface.cuTexRefGetAddressMode)(pam, hTexRef, dim);
	}

	CUresult CudaDriver::cuTexRefGetFilterMode( CUfilter_mode *pfm, 
		CUtexref hTexRef )
	{
		CHECK();
		return (*_interface.cuTexRefGetFilterMode)(pfm, hTexRef);
	}

	CUresult CudaDriver::cuTexRefGetFormat( CUarray_format *pFormat, 
		int *pNumChannels, CUtexref hTexRef )
	{
		CHECK();
		return (*_interface.cuTexRefGetFormat)(pFormat, pNumChannels, hTexRef);
	}

	CUresult CudaDriver::cuTexRefGetFlags( unsigned int *pFlags, 
		CUtexref hTexRef )
	{
		CHECK();
		return (*_interface.cuTexRefGetFlags)(pFlags, hTexRef);
	}


	CUresult CudaDriver::cuParamSetSize(CUfunction hfunc, 
		unsigned int numbytes)
	{
		CHECK();
		return (*_interface.cuParamSetSize)(hfunc, numbytes);
	}

	CUresult CudaDriver::cuParamSeti(CUfunction hfunc, int offset, 
		unsigned int value)
	{
		CHECK();
		return (*_interface.cuParamSeti)(hfunc, offset, value);
	}

	CUresult CudaDriver::cuParamSetf(CUfunction hfunc, int offset, float value)
	{
		CHECK();
		return (*_interface.cuParamSetf)(hfunc, offset, value);
	}

	CUresult CudaDriver::cuParamSetv(CUfunction hfunc, int offset, 
		void * ptr, unsigned int numbytes)
	{
		CHECK();
		return (*_interface.cuParamSetv)(hfunc, offset, ptr, numbytes);
	}

	CUresult CudaDriver::cuParamSetTexRef(CUfunction hfunc, int texunit, 
		CUtexref hTexRef)
	{
		CHECK();
		return (*_interface.cuParamSetTexRef)(hfunc, texunit, hTexRef);
	}


	CUresult CudaDriver::cuLaunch ( CUfunction f )
	{
		CHECK();
		return (*_interface.cuLaunch)(f);
	}

	CUresult CudaDriver::cuLaunchGrid (CUfunction f, int grid_width, 
		int grid_height)
	{
		CHECK();
		return (*_interface.cuLaunchGrid)(f, grid_width, grid_height);
	}

	CUresult CudaDriver::cuLaunchGridAsync( CUfunction f, int grid_width, 
		int grid_height, CUstream hStream )
	{
		CHECK();
		return (*_interface.cuLaunchGridAsync)(f, grid_width, grid_height, 
			hStream);
	}


	CUresult CudaDriver::cuEventCreate( CUevent *phEvent, unsigned int Flags )
	{
		CHECK();
		return (*_interface.cuEventCreate)(phEvent, Flags);
	}

	CUresult CudaDriver::cuEventRecord( CUevent hEvent, CUstream hStream )
	{
		CHECK();
		return (*_interface.cuEventRecord)(hEvent, hStream);
	}

	CUresult CudaDriver::cuEventQuery( CUevent hEvent )
	{
		CHECK();
		return (*_interface.cuEventQuery)(hEvent);
	}

	CUresult CudaDriver::cuEventSynchronize( CUevent hEvent )
	{
		CHECK();
		return (*_interface.cuEventSynchronize)(hEvent);
	}

	CUresult CudaDriver::cuEventDestroy( CUevent hEvent )
	{
		CHECK();
		return (*_interface.cuEventDestroy)(hEvent);
	}

	CUresult CudaDriver::cuEventElapsedTime( float *pMilliseconds, 
		CUevent hStart, CUevent hEnd )
	{
		CHECK();
		return (*_interface.cuEventElapsedTime)(pMilliseconds, hStart, hEnd);
	}


	CUresult CudaDriver::cuStreamCreate( CUstream *phStream, 
		unsigned int Flags )
	{
		CHECK();
		return (*_interface.cuStreamCreate)(phStream, Flags);
	}

	CUresult CudaDriver::cuStreamQuery( CUstream hStream )
	{
		CHECK();
		return (*_interface.cuStreamQuery)(hStream);
	}

	CUresult CudaDriver::cuStreamSynchronize( CUstream hStream )
	{
		CHECK();
		return (*_interface.cuStreamSynchronize)(hStream);
	}

	CUresult CudaDriver::cuStreamDestroy( CUstream hStream )
	{
		CHECK();
		return (*_interface.cuStreamDestroy)(hStream);
	}
	
	CUresult CudaDriver::cuGraphicsUnregisterResource(
		CUgraphicsResource resource)
	{
		CHECK();
		return (*_interface.cuGraphicsUnregisterResource)(resource);
	}
	
	CUresult CudaDriver::cuGraphicsSubResourceGetMappedArray(
		CUarray *pArray, CUgraphicsResource resource, 
		unsigned int arrayIndex, unsigned int mipLevel )
	{
		CHECK();
		return (*_interface.cuGraphicsSubResourceGetMappedArray)(pArray, 
			resource, arrayIndex, mipLevel);
	}

	CUresult CudaDriver::cuGraphicsResourceGetMappedPointer(
		CUdeviceptr *pDevPtr, size_t *pSize, 
		CUgraphicsResource resource)
	{
		CHECK();
		return (*_interface.cuGraphicsResourceGetMappedPointer)(pDevPtr, 
			pSize, resource);
	}
		
	CUresult CudaDriver::cuGraphicsResourceSetMapFlags(
		CUgraphicsResource resource, unsigned int flags ) 
	{
		CHECK();
		return (*_interface.cuGraphicsResourceSetMapFlags)(resource, flags);
	}

	CUresult CudaDriver::cuGraphicsMapResources(unsigned int count, 
		CUgraphicsResource *resources, CUstream hStream)
	{
		CHECK();
		return (*_interface.cuGraphicsMapResources)(count, resources, hStream);
	}

	CUresult CudaDriver::cuGraphicsUnmapResources(unsigned int count, 
		CUgraphicsResource *resources, CUstream hStream)
	{
		CHECK();
		return (*_interface.cuGraphicsUnmapResources)(count, 
			resources, hStream);
	}

	CUresult CudaDriver::cuGetExportTable(const void **ppExportTable,
		const CUuuid *pExportTableId)
	{
		CHECK();
		return (*_interface.cuGetExportTable)(ppExportTable, pExportTableId);
	}

	CUresult CudaDriver::cuGLInit()
	{
		CHECK();
		return (*_interface.cuGLInit)();
	}

	CUresult CudaDriver::cuGLCtxCreate(CUcontext *pCtx, 
		unsigned int Flags, CUdevice device)
	{
		CHECK();
		return (*_interface.cuGLCtxCreate)(pCtx, Flags, device);
	}

	CUresult CudaDriver::cuGraphicsGLRegisterBuffer( 
		CUgraphicsResource *pCudaResource, unsigned int buffer, 
		unsigned int Flags )
	{
		CHECK();
		return (*_interface.cuGraphicsGLRegisterBuffer)(pCudaResource, 
			buffer, Flags);
	}

	CUresult CudaDriver::cuGraphicsGLRegisterImage( 
		CUgraphicsResource *pCudaResource, unsigned int image, 
		int target, unsigned int Flags)
	{
		CHECK();
		return (*_interface.cuGraphicsGLRegisterImage)(pCudaResource, image, 
			target, Flags);
	}
	
	CUresult CudaDriver::cuGLRegisterBufferObject(GLuint bufferobj) {
		CHECK();
		return (*_interface.cuGLRegisterBufferObject)(bufferobj);
	}
	
	CUresult CudaDriver::cuGLSetBufferObjectMapFlags(GLuint buffer,
		unsigned int flags) {
		CHECK();
		return (*_interface.cuGLSetBufferObjectMapFlags)(buffer, flags);
	}

	std::string CudaDriver::toString(CUresult r)
	{
		switch( r )
		{
			case CUDA_SUCCESS: return "CUDA DRIVER - no errors";
			case CUDA_ERROR_INVALID_VALUE: return "CUDA DRIVER - invalid value";
			case CUDA_ERROR_OUT_OF_MEMORY: return "CUDA DRIVER - out of memory";
			case CUDA_ERROR_NOT_INITIALIZED:
				return "CUDA DRIVER - driver not initialized";
			case CUDA_ERROR_DEINITIALIZED: return "CUDA DRIVER - deinitialized";
			case CUDA_ERROR_NO_DEVICE: return "CUDA DRIVER - no device";
			case CUDA_ERROR_INVALID_DEVICE:
				return "CUDA DRIVER - invalid device";
			case CUDA_ERROR_INVALID_IMAGE:
				return "CUDA DRIVER - invalid kernel image";
			case CUDA_ERROR_INVALID_CONTEXT:
				return "CUDA DRIVER - invalid context";
			case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: 
				return "CUDA DRIVER - context already current";
			case CUDA_ERROR_MAP_FAILED: return "CUDA DRIVER - map failed";
			case CUDA_ERROR_UNMAP_FAILED: return "CUDA DRIVER - unmap failed";
			case CUDA_ERROR_ARRAY_IS_MAPPED:
				return "CUDA DRIVER - array is mapped";
			case CUDA_ERROR_ALREADY_MAPPED:
				return "CUDA DRIVER - already mapped";
			case CUDA_ERROR_NO_BINARY_FOR_GPU:
				return "CUDA DRIVER - no gpu binary";
			case CUDA_ERROR_ALREADY_ACQUIRED:
				return "CUDA DRIVER - already aquired";
			case CUDA_ERROR_NOT_MAPPED: return "CUDA DRIVER - not mapped";
			case CUDA_ERROR_INVALID_SOURCE:
				return "CUDA DRIVER - invalid source";
			case CUDA_ERROR_FILE_NOT_FOUND:
				return "CUDA DRIVER - file not found";
			case CUDA_ERROR_INVALID_HANDLE:
				return "CUDA DRIVER - invalid handle";
			case CUDA_ERROR_NOT_FOUND: return "CUDA DRIVER - not found";
			case CUDA_ERROR_NOT_READY: return "CUDA DRIVER - not ready";
			case CUDA_ERROR_LAUNCH_FAILED: return "CUDA DRIVER - launch failed";
			case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
				return "CUDA DRIVER - out of resources";
			case CUDA_ERROR_LAUNCH_TIMEOUT:
				return "CUDA DRIVER - launch timeout";
			case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: 
				return "CUDA DRIVER - incompatible texturing";
			case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "CUDA DRIVER - not mapped as pointer";
			case CUDA_ERROR_UNKNOWN: return "CUDA DRIVER - unknown error";
			default: break;
		}
		return "invalid_error";
	}
	
}

#endif

