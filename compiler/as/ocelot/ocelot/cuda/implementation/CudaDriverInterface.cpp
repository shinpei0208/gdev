/*!
	\file CudaDriverInterface.cpp

	\author Andrew Kerr <arkerr@gatech.edu>
	\brief implements an abstract base class for classes implementing the CUDA Driver API
	\date Sept 16 2010
	\location somewhere over western Europe
*/

#include <ocelot/cuda/interface/CudaDriverInterface.h>

///////////////////////////////////////////////////////////////////////////////////////////////////

			
/*********************************
** Initialization
*********************************/
CUresult cuda::CudaDriverInterface::cuInit(unsigned int Flags) {
	return CUDA_ERROR_NOT_FOUND;
}


/*********************************
** Driver Version Query
*********************************/
CUresult cuda::CudaDriverInterface::cuDriverGetVersion(int *driverVersion) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
	return CUDA_ERROR_NOT_FOUND;
}

/************************************
**
**    Device management
**
***********************************/

CUresult cuda::CudaDriverInterface::cuDeviceGet(CUdevice *device, int ordinal) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuDeviceGetCount(int *count) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuDeviceGetName(char *name, int len, CUdevice dev) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuDeviceComputeCapability(int *major, int *minor, 
	CUdevice dev) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuDeviceGetProperties(CUdevprop *prop, 
	CUdevice dev) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuDeviceGetAttribute(int *pi, 
	CUdevice_attribute attrib, CUdevice dev) {
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Context management
**
***********************************/

CUresult cuda::CudaDriverInterface::cuCtxCreate(CUcontext *pctx, unsigned int flags, 
	CUdevice dev ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuCtxDestroy( CUcontext ctx ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuCtxAttach(CUcontext *pctx, unsigned int flags) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuCtxDetach(CUcontext ctx) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuCtxPushCurrent( CUcontext ctx ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuCtxPopCurrent( CUcontext *pctx ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuCtxGetDevice(CUdevice *device) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuCtxSynchronize(void) {
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Module management
**
***********************************/

CUresult cuda::CudaDriverInterface::cuModuleLoad(CUmodule *module, const char *fname) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuModuleLoadData(CUmodule *module, 
	const void *image) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuModuleLoadDataEx(CUmodule *module, 
	const void *image, unsigned int numOptions, 
	CUjit_option *options, void **optionValues) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuModuleLoadFatBinary(CUmodule *module, 
	const void *fatCubin) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuModuleUnload(CUmodule hmod) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuModuleGetFunction(CUfunction *hfunc, 
	CUmodule hmod, const char *name) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuModuleGetGlobal(CUdeviceptr *dptr, 
	size_t *bytes, CUmodule hmod, const char *name) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, 
	const char *name) {
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Memory management
**
***********************************/

CUresult cuda::CudaDriverInterface::cuMemGetInfo(size_t *free, 
	size_t *total) {
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverInterface::cuMemAlloc( CUdeviceptr *dptr, 
	unsigned int bytesize) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemAllocPitch( CUdeviceptr *dptr, 
			          size_t *pPitch,
			          unsigned int WidthInBytes, 
			          unsigned int Height, 
			          unsigned int ElementSizeBytes
			         ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemFree(CUdeviceptr dptr) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemGetAddressRange( CUdeviceptr *pbase, 
	size_t *psize, CUdeviceptr dptr ) {
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverInterface::cuMemAllocHost(void **pp, unsigned int bytesize) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemFreeHost(void *p) {
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverInterface::cuMemHostAlloc(void **pp, 
	unsigned long long bytesize, unsigned int Flags ) {
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverInterface::cuMemHostGetDevicePointer( CUdeviceptr *pdptr, 
	void *p, unsigned int Flags ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemHostGetFlags( unsigned int *pFlags, void *p ) {
	return CUDA_ERROR_NOT_FOUND;
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
CUresult cuda::CudaDriverInterface::cuMemcpyHtoD (CUdeviceptr dstDevice, 
	const void *srcHost, unsigned int ByteCount ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, 
	unsigned int ByteCount ) {
	return CUDA_ERROR_NOT_FOUND;
}

// device <-> device memory
CUresult cuda::CudaDriverInterface::cuMemcpyDtoD (CUdeviceptr dstDevice, 
	CUdeviceptr srcDevice, unsigned int ByteCount ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemcpyHtoH (void *dstHost, const void *srcHost, 
	unsigned int ByteCount ) {
	return CUDA_ERROR_NOT_FOUND;
}

// device <-> array memory
CUresult cuda::CudaDriverInterface::cuMemcpyDtoA ( CUarray dstArray, 
	unsigned int dstIndex, CUdeviceptr srcDevice, 
	unsigned int ByteCount ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemcpyAtoD ( CUdeviceptr dstDevice, 
	CUarray hSrc, unsigned int SrcIndex, unsigned int ByteCount ) {
	return CUDA_ERROR_NOT_FOUND;
}


// system <-> array memory
CUresult cuda::CudaDriverInterface::cuMemcpyHtoA( CUarray dstArray, 
	unsigned int dstIndex, const void *pSrc, 
	unsigned int ByteCount ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemcpyAtoH( void *dstHost, CUarray srcArray, 
	unsigned int srcIndex, unsigned int ByteCount ) {
	return CUDA_ERROR_NOT_FOUND;
}


// array <-> array memory
CUresult cuda::CudaDriverInterface::cuMemcpyAtoA( CUarray dstArray, 
	unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, 
	unsigned int ByteCount ) {
	return CUDA_ERROR_NOT_FOUND;
}


// 2D memcpy

CUresult cuda::CudaDriverInterface::cuMemcpy2D( const CUDA_MEMCPY2D *pCopy ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy ) {
	return CUDA_ERROR_NOT_FOUND;
}


// 3D memcpy

CUresult cuda::CudaDriverInterface::cuMemcpy3D( const CUDA_MEMCPY3D *pCopy ) {
	return CUDA_ERROR_NOT_FOUND;
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
CUresult cuda::CudaDriverInterface::cuMemcpyHtoDAsync (CUdeviceptr dstDevice, 
const void *srcHost, unsigned int ByteCount, CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemcpyDtoHAsync (void *dstHost, 
CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}


// system <-> array memory
CUresult cuda::CudaDriverInterface::cuMemcpyHtoAAsync( CUarray dstArray, 
	unsigned int dstIndex, const void *pSrc, 
	unsigned int ByteCount, CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, 
	unsigned int srcIndex, unsigned int ByteCount, 
	CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}


// 2D memcpy
CUresult cuda::CudaDriverInterface::cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, 
	CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}


// 3D memcpy
CUresult cuda::CudaDriverInterface::cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, 
	CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Memset
**
***********************************/
CUresult cuda::CudaDriverInterface::cuMemsetD8( CUdeviceptr dstDevice, 
	unsigned char uc, unsigned int N ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemsetD16( CUdeviceptr dstDevice, 
	unsigned short us, unsigned int N ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemsetD32( CUdeviceptr dstDevice, 
	unsigned int ui, unsigned int N ) {
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverInterface::cuMemsetD2D8( CUdeviceptr dstDevice,
	unsigned int dstPitch, unsigned char uc, unsigned int Width, 
	unsigned int Height ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemsetD2D16( CUdeviceptr dstDevice, 
	unsigned int dstPitch, unsigned short us, unsigned int Width, 
	unsigned int Height ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuMemsetD2D32( CUdeviceptr dstDevice, 
	unsigned int dstPitch, unsigned int ui, unsigned int Width, 
	unsigned int Height ) {
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Function management
**
***********************************/


CUresult cuda::CudaDriverInterface::cuFuncSetBlockShape (CUfunction hfunc, int x, 
	int y, int z) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuFuncSetSharedSize (CUfunction hfunc, 
	unsigned int bytes) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuFuncGetAttribute (int *pi, 
	CUfunction_attribute attrib, CUfunction hfunc) {
	return CUDA_ERROR_NOT_FOUND;
}

/*
CUresult cuda::CudaDriverInterface::cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
	return CUDA_ERROR_NOT_FOUND;
}
*/

CUresult cuda::CudaDriverInterface::cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
	return CUDA_ERROR_NOT_FOUND;
}

/************************************
**
**    Array management 
**
***********************************/

CUresult cuda::CudaDriverInterface::cuArrayCreate( CUarray *pHandle, 
	const CUDA_ARRAY_DESCRIPTOR *pAllocateArray ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuArrayGetDescriptor( 
	CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuArrayDestroy( CUarray hArray ) {
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverInterface::cuArray3DCreate( CUarray *pHandle, 
	const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuArray3DGetDescriptor( 
	CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray ) {
	return CUDA_ERROR_NOT_FOUND;
}



/************************************
**
**    Texture reference management
**
***********************************/
CUresult cuda::CudaDriverInterface::cuTexRefCreate( CUtexref *pTexRef ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefDestroy( CUtexref hTexRef ) {
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverInterface::cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, 
	unsigned int Flags ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefSetAddress( size_t *ByteOffset, 
	CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefSetAddress2D( CUtexref hTexRef, 
	const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, 
	unsigned int Pitch) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefSetFormat( CUtexref hTexRef, 
	CUarray_format fmt, int NumPackedComponents ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefSetAddressMode( CUtexref hTexRef, int dim, 
	CUaddress_mode am ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefSetFilterMode( CUtexref hTexRef, 
	CUfilter_mode fm ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefSetFlags( CUtexref hTexRef, 
	unsigned int Flags ) {
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverInterface::cuTexRefGetAddress( CUdeviceptr *pdptr, 
	CUtexref hTexRef ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefGetArray( CUarray *phArray, 
	CUtexref hTexRef ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefGetAddressMode( CUaddress_mode *pam, 
	CUtexref hTexRef, int dim ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefGetFilterMode( CUfilter_mode *pfm, 
	CUtexref hTexRef ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefGetFormat( CUarray_format *pFormat, 
	int *pNumChannels, CUtexref hTexRef ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuTexRefGetFlags( unsigned int *pFlags, 
	CUtexref hTexRef ) {
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Parameter management
**
***********************************/

CUresult cuda::CudaDriverInterface::cuParamSetSize (CUfunction hfunc, 
	unsigned int numbytes) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuParamSeti    (CUfunction hfunc, int offset, 
	unsigned int value) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuParamSetf    (CUfunction hfunc, int offset, 
	float value) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuParamSetv    (CUfunction hfunc, int offset, 
	void * ptr, unsigned int numbytes) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuParamSetTexRef(CUfunction hfunc, int texunit, 
	CUtexref hTexRef) {
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Launch functions
**
***********************************/

CUresult cuda::CudaDriverInterface::cuLaunch ( CUfunction f ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuLaunchGrid (CUfunction f, int grid_width, 
	int grid_height) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuLaunchGridAsync( CUfunction f, int grid_width, 
	int grid_height, CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Events
**
***********************************/
CUresult cuda::CudaDriverInterface::cuEventCreate( CUevent *phEvent, 
	unsigned int Flags ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuEventRecord( CUevent hEvent, CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuEventQuery( CUevent hEvent ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuEventSynchronize( CUevent hEvent ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuEventDestroy( CUevent hEvent ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuEventElapsedTime( float *pMilliseconds, 
	CUevent hStart, CUevent hEnd ) {
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Streams
**
***********************************/
CUresult cuda::CudaDriverInterface::cuStreamCreate( CUstream *phStream, 
	unsigned int Flags ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuStreamQuery( CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuStreamSynchronize( CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuStreamDestroy( CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Graphics
**
***********************************/
CUresult cuda::CudaDriverInterface::cuGraphicsUnregisterResource(
	CUgraphicsResource resource) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuGraphicsSubResourceGetMappedArray(
	CUarray *pArray, CUgraphicsResource resource, 
	unsigned int arrayIndex, unsigned int mipLevel ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuGraphicsResourceGetMappedPointer(
	CUdeviceptr *pDevPtr, size_t *pSize, 
	CUgraphicsResource resource ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuGraphicsResourceSetMapFlags(
	CUgraphicsResource resource, unsigned int flags ) {
	return CUDA_ERROR_NOT_FOUND;
}
 
CUresult cuda::CudaDriverInterface::cuGraphicsMapResources(unsigned int count, 
	CUgraphicsResource *resources, CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuGraphicsUnmapResources(unsigned int count, 
	CUgraphicsResource *resources, CUstream hStream ) {
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    OpenGL
**
***********************************/
CUresult cuda::CudaDriverInterface::cuGLInit() {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuGLCtxCreate(CUcontext *pCtx, 
	unsigned int Flags, CUdevice device) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuGraphicsGLRegisterBuffer( 
	CUgraphicsResource *pCudaResource, unsigned int buffer, 
	unsigned int Flags ) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuGraphicsGLRegisterImage( 
	CUgraphicsResource *pCudaResource, unsigned int image, 
	int target, unsigned int Flags) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuGLRegisterBufferObject(GLuint bufferobj) {
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverInterface::cuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int flags) {
	return CUDA_ERROR_NOT_FOUND;
}

std::string cuda::CudaDriverInterface::toString(CUresult result) {
	return "CUresult - blah";
}

///////////////////////////////////////////////////////////////////////////////////////////////////

