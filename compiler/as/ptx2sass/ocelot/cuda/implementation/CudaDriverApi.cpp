/*!
	\file CudaDriverApi.cpp

	\author Andrew Kerr <arkerr@gatech.edu>
	\brief implements CUDA Driver API functions which call through to the implementation of the
		appropriate CUDA Driver API front-end implementation
	\date Sept 16 2010
	\location somewhere over western Europe
*/

// C++ includes
#include <string>

// Ocelot includes
#include <ocelot/cuda/interface/cuda_internal.h>
#include <ocelot/cuda/interface/CudaDriverFrontend.h>

// Hydrazine includes
#include <hydrazine/interface/debug.h>


#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////


// whether CUDA runtime catches exceptions thrown by Ocelot
#define CATCH_RUNTIME_EXCEPTIONS 0

// whether verbose error messages are printed
#define CUDA_VERBOSE 1

// whether debugging messages are printed
#define REPORT_BASE 1

//////////////////////////////////////////////////////////////////////////////////////////////////
//
// Error handling macros

#define Ocelot_Exception(x) { std::stringstream ss; ss << x; throw hydrazine::Exception(ss.str()) ); }

///////////////////////////////////////////////////////////////////////////////////////////////////

#if REPORT_BASE
#define trace() { std::cout << " - " << __FUNCTION__ << "() " << std::endl; }
#else
#define trace()
#endif

#if REPORT_BASE
#define RETURN(x) CUresult result = x; \
	if (result != CUDA_SUCCESS) { std::cout << "  error: " << (int)result << std::endl; } \
	return result;
#else
#define RETURN(x) return x 
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#define Function(f) CUDAAPI f
#define FunctionV(f) CUDAAPI f ## _v2

////////////////////////////////////////////////////////////////////////////////////////////////////

// use this to define either a pass-through driver API implementation for testing linkage with the
// shared object or to invoke Ocelot's Driver API frontend
typedef cuda::CudaDriverInterface CudaApi;

extern "C" {

/*********************************
** Initialization
*********************************/
CUresult Function(cuInit)(unsigned int Flags) {
	trace();
	RETURN( CudaApi::get()->cuInit(Flags) );
}


/*********************************
** Driver Version Query
*********************************/
CUresult Function(cuDriverGetVersion)(int *driverVersion) {
	trace();	
	RETURN( CudaApi::get()->cuDriverGetVersion(driverVersion) );
}


CUresult Function(cuGetExportTable)(const void **ppExportTable,
	const CUuuid *pExportTableId) {
	trace();
	return( CudaApi::get()->cuGetExportTable(ppExportTable, pExportTableId) );
}

/************************************
**
**    Device management
**
***********************************/

CUresult Function(cuDeviceGet)(CUdevice *device, int ordinal) {
	trace();	
	RETURN( CudaApi::get()->cuDeviceGet(device, ordinal) );
}

CUresult Function(cuDeviceGetCount)(int *count) {
	trace();	
	RETURN( CudaApi::get()->cuDeviceGetCount(count) );
}

CUresult Function(cuDeviceGetName)(char *name, int len, CUdevice dev) {
	trace();	
	RETURN( CudaApi::get()->cuDeviceGetName(name, len, dev) );
}

CUresult Function(cuDeviceComputeCapability)(int *major, int *minor, 
	CUdevice dev) {
	trace();	
	RETURN( CudaApi::get()->cuDeviceComputeCapability(major, minor, dev) );
}

CUresult Function(cuDeviceTotalMem)(size_t *bytes, CUdevice dev) {
	trace();	
	RETURN( CudaApi::get()->cuDeviceTotalMem(bytes, dev) );
}

CUresult FunctionV(cuDeviceTotalMem)(size_t *bytes, CUdevice dev) {
	trace();	
	RETURN( CudaApi::get()->cuDeviceTotalMem(bytes, dev) );
}

CUresult Function(cuDeviceGetProperties)(CUdevprop *prop, 
	CUdevice dev) {
	trace();	
	RETURN( CudaApi::get()->cuDeviceGetProperties(prop, dev) );
}

CUresult Function(cuDeviceGetAttribute)(int *pi, 
	CUdevice_attribute attrib, CUdevice dev) {
	trace();	
	RETURN( CudaApi::get()->cuDeviceGetAttribute(pi, attrib, dev) );
}


/************************************
**
**    Context management
**
***********************************/

CUresult Function(cuCtxCreate)(CUcontext *pctx, unsigned int flags, 
	CUdevice dev ) {
	trace();	
	RETURN( CudaApi::get()->cuCtxCreate(pctx, flags, dev) );
}

CUresult FunctionV(cuCtxCreate)(CUcontext *pctx, unsigned int flags, 
	CUdevice dev ) {
	trace();	
	RETURN( CudaApi::get()->cuCtxCreate(pctx, flags, dev) );
}

CUresult Function(cuCtxDestroy)( CUcontext ctx ) {
	trace();	
	RETURN( CudaApi::get()->cuCtxDestroy(ctx) );
}

CUresult Function(cuCtxAttach)(CUcontext *pctx, unsigned int flags) {
	trace();	
	RETURN( CudaApi::get()->cuCtxAttach(pctx, flags) );
}

CUresult Function(cuCtxDetach)(CUcontext ctx) {
	trace();	
	RETURN( CudaApi::get()->cuCtxDetach(ctx) );
}

CUresult Function(cuCtxPushCurrent)( CUcontext ctx ) {
	trace();	
	RETURN( CudaApi::get()->cuCtxPushCurrent(ctx) );
}

CUresult Function(cuCtxPopCurrent)( CUcontext *pctx ) {
	trace();	
	RETURN( CudaApi::get()->cuCtxPopCurrent(pctx) );
}

CUresult Function(cuCtxGetDevice)(CUdevice *device) {
	trace();	
	RETURN( CudaApi::get()->cuCtxGetDevice(device) );
}

CUresult Function(cuCtxSynchronize)(void) {
	trace();	
	RETURN( CudaApi::get()->cuCtxSynchronize() );
}


/************************************
**
**    Module management
**
***********************************/

CUresult Function(cuModuleLoad)(CUmodule *module, const char *fname) {
	trace();	
	RETURN( CudaApi::get()->cuModuleLoad(module, fname) );
}

CUresult Function(cuModuleLoadData)(CUmodule *module, 
	const void *image) {
	trace();	
	RETURN( CudaApi::get()->cuModuleLoadData(module, image) );
}

CUresult Function(cuModuleLoadDataEx)(CUmodule *module, 
	const void *image, unsigned int numOptions, 
	CUjit_option *options, void **optionValues) {
	trace();	
	RETURN( CudaApi::get()->cuModuleLoadDataEx(module, image, numOptions, options, optionValues) );
}

CUresult Function(cuModuleLoadFatBinary)(CUmodule *module, 
	const void *fatCubin) {
	trace();	
	RETURN( CudaApi::get()->cuModuleLoadFatBinary(module, fatCubin) );
}

CUresult Function(cuModuleUnload)(CUmodule hmod) {
	trace();	
	RETURN( CudaApi::get()->cuModuleUnload(hmod) );
}

CUresult Function(cuModuleGetFunction)(CUfunction *hfunc, 
	CUmodule hmod, const char *name) {
	trace();	
	RETURN( CudaApi::get()->cuModuleGetFunction(hfunc, hmod, name) );
}

CUresult FunctionV(cuModuleGetGlobal)(CUdeviceptr *dptr, 
	size_t *bytes, CUmodule hmod, const char *name) {
	trace();	
	RETURN( CudaApi::get()->cuModuleGetGlobal(dptr, bytes, hmod, name) );
}

CUresult Function(cuModuleGetTexRef)(CUtexref *pTexRef, CUmodule hmod, 
	const char *name) {
	trace();	
	RETURN( CudaApi::get()->cuModuleGetTexRef(pTexRef, hmod, name) );
}


/************************************
**
**    Memory management
**
***********************************/

CUresult FunctionV(cuMemGetInfo)(size_t *free, 
	size_t *total) {
	trace();	
	RETURN( CudaApi::get()->cuMemGetInfo(free, total) );
}


CUresult FunctionV(cuMemAlloc)( CUdeviceptr *dptr, 
	unsigned int bytesize) {
	trace();	
	RETURN( CudaApi::get()->cuMemAlloc(dptr, bytesize) );
}

CUresult FunctionV(cuMemAllocPitch)( CUdeviceptr *dptr, 
			          size_t *pPitch,
			          unsigned int WidthInBytes, 
			          unsigned int Height, 
			          unsigned int ElementSizeBytes
			         ) {
	trace();	
	RETURN( CudaApi::get()->cuMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes) );
}

CUresult FunctionV(cuMemFree)(CUdeviceptr dptr) {
	trace();	
	RETURN( CudaApi::get()->cuMemFree(dptr) );
}

CUresult FunctionV(cuMemGetAddressRange)( CUdeviceptr *pbase, 
	size_t *psize, CUdeviceptr dptr ) {
	trace();	
	RETURN( CudaApi::get()->cuMemGetAddressRange(pbase, psize, dptr) );
}


CUresult Function(cuMemAllocHost)(void **pp, size_t bytesize) {
	trace();	
	RETURN( CudaApi::get()->cuMemAllocHost(pp, bytesize) );
}

CUresult Function(cuMemFreeHost)(void *p) {
	trace();	
	RETURN( CudaApi::get()->cuMemFreeHost(p) );
}


CUresult Function(cuMemHostAlloc)(void **pp, 
	size_t bytesize, unsigned int Flags ) {
	trace();	
	RETURN( CudaApi::get()->cuMemHostAlloc(pp, bytesize, Flags) );
}


CUresult FunctionV(cuMemHostGetDevicePointer)( CUdeviceptr *pdptr, 
	void *p, unsigned int Flags ) {
	trace();	
	RETURN( CudaApi::get()->cuMemHostGetDevicePointer(pdptr, p, Flags) );
}

CUresult Function(cuMemHostGetFlags)( unsigned int *pFlags, void *p ) {
	trace();	
	RETURN( CudaApi::get()->cuMemHostGetFlags(pFlags, p) );
}


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
CUresult FunctionV(cuMemcpyHtoD )(CUdeviceptr dstDevice, 
	const void *srcHost, unsigned int ByteCount ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyHtoD(dstDevice, srcHost, ByteCount) );
}

CUresult FunctionV(cuMemcpyDtoH )(void *dstHost, CUdeviceptr srcDevice, 
	unsigned int ByteCount ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyDtoH(dstHost, srcDevice, ByteCount) );
}

// device <-> device memory
CUresult FunctionV(cuMemcpyDtoD )(CUdeviceptr dstDevice, 
	CUdeviceptr srcDevice, unsigned int ByteCount ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyDtoD(dstDevice, srcDevice, ByteCount) );
}

CUresult FunctionV(cuMemcpyHtoH )(void *dstHost, 
	const void *srcHost, unsigned int ByteCount ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyHtoH(dstHost, srcHost, ByteCount) );
}

// device <-> array memory
CUresult FunctionV(cuMemcpyDtoA )( CUarray dstArray, 
	unsigned int dstIndex, CUdeviceptr srcDevice, 
	unsigned int ByteCount ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyDtoA(dstArray, dstIndex, srcDevice, ByteCount) );
}

CUresult FunctionV(cuMemcpyAtoD )( CUdeviceptr dstDevice, 
	CUarray hSrc, unsigned int SrcIndex, unsigned int ByteCount ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyAtoD(dstDevice, hSrc, SrcIndex, ByteCount) );
}


// system <-> array memory
CUresult FunctionV(cuMemcpyHtoA)( CUarray dstArray, 
	unsigned int dstIndex, const void *pSrc, 
	unsigned int ByteCount ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyHtoA(dstArray, dstIndex, pSrc, ByteCount) );
}

CUresult FunctionV(cuMemcpyAtoH)( void *dstHost, CUarray srcArray, 
	unsigned int srcIndex, unsigned int ByteCount ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyAtoH(dstHost, srcArray, srcIndex, ByteCount) );
}


// array <-> array memory
CUresult FunctionV(cuMemcpyAtoA)( CUarray dstArray, 
	unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, 
	unsigned int ByteCount ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyAtoA(dstArray, dstIndex, srcArray, srcIndex, ByteCount) );
}


// 2D memcpy

CUresult FunctionV(cuMemcpy2D)( const CUDA_MEMCPY2D *pCopy ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpy2D(pCopy) );
}

CUresult FunctionV(cuMemcpy2DUnaligned)( const CUDA_MEMCPY2D *pCopy ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpy2DUnaligned(pCopy) );
}


// 3D memcpy

CUresult FunctionV(cuMemcpy3D)( const CUDA_MEMCPY3D *pCopy ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpy3D(pCopy) );
}


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
CUresult FunctionV(cuMemcpyHtoDAsync)(CUdeviceptr dstDevice, 
	const void *srcHost, unsigned int ByteCount, CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, hStream) );
}

CUresult FunctionV(cuMemcpyDtoHAsync)(void *dstHost, 
	CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, hStream) );
}


// system <-> array memory
CUresult FunctionV(cuMemcpyHtoAAsync)( CUarray dstArray, 
	unsigned int dstIndex, const void *pSrc, 
	unsigned int ByteCount, CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyHtoAAsync(dstArray, dstIndex, pSrc, ByteCount, hStream) );
}

CUresult FunctionV(cuMemcpyAtoHAsync)( void *dstHost, CUarray srcArray, 
	unsigned int srcIndex, unsigned int ByteCount, 
	CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpyAtoHAsync(dstHost, srcArray, srcIndex, ByteCount, hStream) );
}

// 2D memcpy
CUresult FunctionV(cuMemcpy2DAsync)( const CUDA_MEMCPY2D *pCopy, 
	CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpy2DAsync(pCopy, hStream) );
}


// 3D memcpy
CUresult FunctionV(cuMemcpy3DAsync)( const CUDA_MEMCPY3D *pCopy, 
	CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuMemcpy3DAsync(pCopy, hStream) );
}


/************************************
**
**    Memset
**
***********************************/
CUresult FunctionV(cuMemsetD8)( CUdeviceptr dstDevice, 
	unsigned char uc, unsigned int N ) {
	trace();	
	RETURN( CudaApi::get()->cuMemsetD8(dstDevice, uc, N) );
}

CUresult FunctionV(cuMemsetD16)( CUdeviceptr dstDevice, 
	unsigned short us, unsigned int N ) {
	trace();	
	RETURN( CudaApi::get()->cuMemsetD16(dstDevice, us, N) );
}

CUresult FunctionV(cuMemsetD32)( CUdeviceptr dstDevice, 
	unsigned int ui, unsigned int N ) {
	trace();	
	RETURN( CudaApi::get()->cuMemsetD32(dstDevice, ui, N) );
}


CUresult FunctionV(cuMemsetD2D8)( CUdeviceptr dstDevice,
	unsigned int dstPitch, unsigned char uc, unsigned int Width, 
	unsigned int Height ) {
	trace();	
	RETURN( CudaApi::get()->cuMemsetD2D8(dstDevice, dstPitch, uc, Width, Height) );
}

CUresult FunctionV(cuMemsetD2D16)( CUdeviceptr dstDevice, 
	unsigned int dstPitch, unsigned short us, unsigned int Width, 
	unsigned int Height ) {
	trace();	
	RETURN( CudaApi::get()->cuMemsetD2D16(dstDevice, dstPitch, us, Width, Height) );
}

CUresult FunctionV(cuMemsetD2D32)( CUdeviceptr dstDevice, 
	unsigned int dstPitch, unsigned int ui, unsigned int Width, 
	unsigned int Height ) {
	trace();	
	RETURN( CudaApi::get()->cuMemsetD2D32(dstDevice, dstPitch, ui, Width, Height) );
}


/************************************
**
**    Function management
**
***********************************/

CUresult Function(cuFuncSetBlockShape)(CUfunction hfunc, int x, 
	int y, int z) {
	trace();	
	RETURN( CudaApi::get()->cuFuncSetBlockShape(hfunc, x, y, z) );
}

CUresult Function(cuFuncSetSharedSize)(CUfunction hfunc, 
	unsigned int bytes) {
	trace();	
	RETURN( CudaApi::get()->cuFuncSetSharedSize(hfunc, bytes) );
}

CUresult Function(cuFuncGetAttribute)(int *pi, 
	CUfunction_attribute attrib, CUfunction hfunc) {
	trace();	
	RETURN( CudaApi::get()->cuFuncGetAttribute(pi, attrib, hfunc) );
}
/*
CUresult Function(cuFuncGetCacheConfig)(CUfunction hfunc, CUfunc_cache *pconfig) {
	trace();
	RETURN( CudaApi::get()->cuFuncGetCacheConfig(hfunc, pconfig) );
}
*/
CUresult Function(cuFuncSetCacheConfig)(CUfunction hfunc, CUfunc_cache config) {
	trace();
	RETURN( CudaApi::get()->cuFuncSetCacheConfig(hfunc, config) );
}

/************************************
**
**    Array management 
**
***********************************/

CUresult FunctionV(cuArrayCreate)( CUarray *pHandle, 
	const CUDA_ARRAY_DESCRIPTOR *pAllocateArray ) {
	trace();	
	RETURN( CudaApi::get()->cuArrayCreate(pHandle, pAllocateArray) );
}

CUresult FunctionV(cuArrayGetDescriptor)( 
	CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray ) {
	trace();	
	RETURN( CudaApi::get()->cuArrayGetDescriptor(pArrayDescriptor, hArray) );
}

CUresult Function(cuArrayDestroy)( CUarray hArray ) {
	trace();	
	RETURN( CudaApi::get()->cuArrayDestroy(hArray) );
}


CUresult FunctionV(cuArray3DCreate)( CUarray *pHandle, 
	const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray ) {
	trace();	
	RETURN( CudaApi::get()->cuArray3DCreate(pHandle, pAllocateArray) );
}

CUresult FunctionV(cuArray3DGetDescriptor)( 
	CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray ) {
	trace();	
	RETURN( CudaApi::get()->cuArray3DGetDescriptor(pArrayDescriptor, hArray) );
}



/************************************
**
**    Texture reference management
**
***********************************/
CUresult Function(cuTexRefCreate)( CUtexref *pTexRef ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefCreate(pTexRef) );
}

CUresult Function(cuTexRefDestroy)( CUtexref hTexRef ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefDestroy(hTexRef) );
}


CUresult Function(cuTexRefSetArray)( CUtexref hTexRef, CUarray hArray, 
	unsigned int Flags ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefSetArray(hTexRef, hArray, Flags) );
}

CUresult FunctionV(cuTexRefSetAddress)( size_t *ByteOffset, 
	CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefSetAddress(ByteOffset, hTexRef, dptr, bytes) );
}

CUresult FunctionV(cuTexRefSetAddress2D)( CUtexref hTexRef, 
	const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, 
	unsigned int Pitch) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefSetAddress2D(hTexRef, desc, dptr, Pitch) );
}

CUresult Function(cuTexRefSetFormat)( CUtexref hTexRef, 
	CUarray_format fmt, int NumPackedComponents ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents) );
}

CUresult Function(cuTexRefSetAddressMode)( CUtexref hTexRef, int dim, 
	CUaddress_mode am ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefSetAddressMode(hTexRef, dim, am) );
}

CUresult Function(cuTexRefSetFilterMode)( CUtexref hTexRef, 
	CUfilter_mode fm ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefSetFilterMode(hTexRef, fm) );
}

CUresult Function(cuTexRefSetFlags)( CUtexref hTexRef, 
	unsigned int Flags ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefSetFlags(hTexRef, Flags) );
}


CUresult FunctionV(cuTexRefGetAddress)( CUdeviceptr *pdptr, 
	CUtexref hTexRef ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefGetAddress(pdptr, hTexRef) );
}

CUresult Function(cuTexRefGetArray)( CUarray *phArray, 
	CUtexref hTexRef ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefGetArray(phArray, hTexRef) );
}

CUresult Function(cuTexRefGetAddressMode)( CUaddress_mode *pam, 
	CUtexref hTexRef, int dim ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefGetAddressMode(pam, hTexRef, dim) );
}

CUresult Function(cuTexRefGetFilterMode)( CUfilter_mode *pfm, 
	CUtexref hTexRef ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefGetFilterMode(pfm, hTexRef) );
}

CUresult Function(cuTexRefGetFormat)( CUarray_format *pFormat, 
	int *pNumChannels, CUtexref hTexRef ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefGetFormat(pFormat, pNumChannels, hTexRef) );
}

CUresult Function(cuTexRefGetFlags)( unsigned int *pFlags, 
	CUtexref hTexRef ) {
	trace();	
	RETURN( CudaApi::get()->cuTexRefGetFlags(pFlags, hTexRef) );
}


/************************************
**
**    Parameter management
**
***********************************/

CUresult Function(cuParamSetSize)(CUfunction hfunc, 
	unsigned int numbytes) {
	trace();	
	RETURN( CudaApi::get()->cuParamSetSize(hfunc, numbytes) );
}

CUresult Function(cuParamSeti)(CUfunction hfunc, int offset, 
	unsigned int value) {
	trace();	
	RETURN( CudaApi::get()->cuParamSeti(hfunc, offset, value) );
}

CUresult Function(cuParamSetf)(CUfunction hfunc, int offset, 
	float value) {
	trace();	
	RETURN( CudaApi::get()->cuParamSetf(hfunc, offset, value) );
}

CUresult Function(cuParamSetv)(CUfunction hfunc, int offset, 
	void * ptr, unsigned int numbytes) {
	trace();	
	RETURN( CudaApi::get()->cuParamSetv(hfunc, offset, ptr, numbytes) );
}

CUresult Function(cuParamSetTexRef)(CUfunction hfunc, int texunit, 
	CUtexref hTexRef) {
	trace();	
	RETURN( CudaApi::get()->cuParamSetTexRef(hfunc, texunit, hTexRef) );
}


/************************************
**
**    Launch functions
**
***********************************/

CUresult Function(cuLaunch)( CUfunction f ) {
	trace();	
	RETURN( CudaApi::get()->cuLaunch(f) );
}

CUresult Function(cuLaunchGrid)(CUfunction f, int grid_width, 
	int grid_height) {
	trace();	
	RETURN( CudaApi::get()->cuLaunchGrid(f, grid_width, grid_height) );
}

CUresult Function(cuLaunchGridAsync)( CUfunction f, int grid_width, 
	int grid_height, CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuLaunchGridAsync(f, grid_width, grid_height, hStream) );
}


/************************************
**
**    Events
**
***********************************/
CUresult Function(cuEventCreate)( CUevent *phEvent, 
	unsigned int Flags ) {
	trace();	
	RETURN( CudaApi::get()->cuEventCreate(phEvent, Flags) );
}

CUresult Function(cuEventRecord)( CUevent hEvent, CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuEventRecord(hEvent, hStream) );
}

CUresult Function(cuEventQuery)( CUevent hEvent ) {
	trace();	
	RETURN( CudaApi::get()->cuEventQuery(hEvent) );
}

CUresult Function(cuEventSynchronize)( CUevent hEvent ) {
	trace();	
	RETURN( CudaApi::get()->cuEventSynchronize(hEvent) );
}

CUresult Function(cuEventDestroy)( CUevent hEvent ) {
	trace();	
	RETURN( CudaApi::get()->cuEventDestroy(hEvent) );
}

CUresult Function(cuEventElapsedTime)( float *pMilliseconds, 
	CUevent hStart, CUevent hEnd ) {
	trace();	
	RETURN( CudaApi::get()->cuEventElapsedTime(pMilliseconds, hStart, hEnd) );
}


/************************************
**
**    Streams
**
***********************************/
CUresult Function(cuStreamCreate)( CUstream *phStream, 
	unsigned int Flags ) {
	trace();	
	RETURN( CudaApi::get()->cuStreamCreate(phStream, Flags) );
}

CUresult Function(cuStreamQuery)( CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuStreamQuery(hStream) );
}

CUresult Function(cuStreamSynchronize)( CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuStreamSynchronize(hStream) );
}

CUresult Function(cuStreamDestroy)( CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuStreamDestroy(hStream) );
}


/************************************
**
**    Graphics
**
***********************************/
CUresult Function(cuGraphicsUnregisterResource)(
	CUgraphicsResource resource) {
	trace();	
	RETURN( CudaApi::get()->cuGraphicsUnregisterResource(resource) );
}

CUresult Function(cuGraphicsSubResourceGetMappedArray)(
	CUarray *pArray, CUgraphicsResource resource, 
	unsigned int arrayIndex, unsigned int mipLevel ) {
	trace();	
	RETURN( CudaApi::get()->cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel) );
}

CUresult FunctionV(cuGraphicsResourceGetMappedPointer)(
	CUdeviceptr *pDevPtr, size_t *pSize, 
	CUgraphicsResource resource ) {
	trace();	
	RETURN( CudaApi::get()->cuGraphicsResourceGetMappedPointer(pDevPtr, pSize, resource) );
}

CUresult Function(cuGraphicsResourceSetMapFlags)(
	CUgraphicsResource resource, unsigned int flags ) {
	trace();	
	RETURN( CudaApi::get()->cuGraphicsResourceSetMapFlags(resource, flags) );
}
 
CUresult Function(cuGraphicsMapResources)(unsigned int count, 
	CUgraphicsResource *resources, CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuGraphicsMapResources(count, resources, hStream) );
}

CUresult Function(cuGraphicsUnmapResources)(unsigned int count, 
	CUgraphicsResource *resources, CUstream hStream ) {
	trace();	
	RETURN( CudaApi::get()->cuGraphicsUnmapResources(count, resources, hStream) );
}


/************************************
**
**    OpenGL
**
***********************************/
CUresult Function(cuGLInit)() {
	trace();	
	RETURN( CudaApi::get()->cuGLInit() );
}

CUresult Function(cuGLCtxCreate)(CUcontext *pCtx, 
	unsigned int Flags, CUdevice device) {
	trace();	
	RETURN( CudaApi::get()->cuGLCtxCreate(pCtx, Flags, device) );
}

CUresult Function(cuGLRegisterBufferObject)(GLuint bufferobj) {
	trace();
	RETURN ( CudaApi::get()->cuGLRegisterBufferObject(bufferobj) );
}

CUresult Function(cuGraphicsGLRegisterBuffer)( 
	CUgraphicsResource *pCudaResource, unsigned int buffer, 
	unsigned int Flags ) {
	trace();	
	RETURN( CudaApi::get()->cuGraphicsGLRegisterBuffer(pCudaResource, buffer, Flags) );
}

CUresult Function(cuGraphicsGLRegisterImage)( 
	CUgraphicsResource *pCudaResource, unsigned int image, 
	int target, unsigned int Flags) {
	trace();	
	RETURN( CudaApi::get()->cuGraphicsGLRegisterImage(pCudaResource, image, target, Flags) );
}

CUresult Function(cuGLSetBufferObjectMapFlags)(GLuint buffer, unsigned int flags) {
	trace();
	RETURN ( CudaApi::get()->cuGLSetBufferObjectMapFlags(buffer, flags) );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}

