/*!
	\file cuda_runtime.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief wraps CUDA Runtime API with calls to CudaRuntimeInterface methods
	\date 14 Dec 2009
*/

#include <configure.h>

#if EXCLUDE_CUDA_RUNTIME == 0

// C stdlib includes
#include <assert.h>

// Ocelot includes
#include <ocelot/cuda/interface/cuda_runtime.h>
#include <ocelot/cuda/interface/CudaRuntimeInterface.h>

/******************************************************************************/


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

void** __cudaRegisterFatBinary(void *fatCubin) {
	return cuda::CudaRuntimeInterface::get()->cudaRegisterFatBinary(fatCubin);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
	cuda::CudaRuntimeInterface::get()->cudaUnregisterFatBinary(fatCubinHandle);
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
	char *deviceAddress, const char *deviceName, int ext, int size,
	int constant, int global) {
	
	cuda::CudaRuntimeInterface::get()->cudaRegisterVar(fatCubinHandle,
		hostVar, deviceAddress,
		deviceName, ext, size, constant, global);
}

void __cudaRegisterTexture(
        void **fatCubinHandle,
  const struct textureReference *hostVar,
  const void **deviceAddress,
  const char *deviceName,
        int dim,
        int norm,
        int ext) {
	cuda::CudaRuntimeInterface::get()->cudaRegisterTexture(fatCubinHandle,
		hostVar, deviceAddress,
		deviceName, dim, norm, ext);
}

void __cudaRegisterShared(
  void **fatCubinHandle,
  void **devicePtr) {
	cuda::CudaRuntimeInterface::get()->cudaRegisterShared(fatCubinHandle,
		devicePtr);
}

void __cudaRegisterSharedVar(
  void **fatCubinHandle,
  void **devicePtr,
  size_t size,
  size_t alignment,
  int storage) {
  
	cuda::CudaRuntimeInterface::get()->cudaRegisterSharedVar(fatCubinHandle,
		devicePtr, size,
		alignment, storage);
}

void __cudaRegisterFunction(
        void **fatCubinHandle,
  const char *hostFun,
        char *deviceFun,
  const char *deviceName,
        int thread_limit,
        uint3 *tid,
        uint3 *bid,
        dim3 *bDim,
        dim3 *gDim,
        int *wSize) {
	cuda::CudaRuntimeInterface::get()->cudaRegisterFunction(fatCubinHandle,
		hostFun, deviceFun,	
		deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent) {
	return cuda::CudaRuntimeInterface::get()->cudaMalloc3D(pitchedDevPtr, extent);
}

cudaError_t  cudaMalloc3DArray(struct cudaArray** arrayPtr, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent) {
	return cuda::CudaRuntimeInterface::get()->cudaMalloc3DArray(arrayPtr, desc, extent);
}

cudaError_t  cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent) {
	return cuda::CudaRuntimeInterface::get()->cudaMemset3D(pitchedDevPtr, value, extent);
}

cudaError_t  cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpy3D(p);
}

cudaError_t  cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpy3DAsync(p, stream);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaMalloc(void **devPtr, size_t size) {
	return cuda::CudaRuntimeInterface::get()->cudaMalloc(devPtr, size);
}

cudaError_t  cudaMallocHost(void **ptr, size_t size) {
	return cuda::CudaRuntimeInterface::get()->cudaMallocHost(ptr, size);
}

cudaError_t  cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
	return cuda::CudaRuntimeInterface::get()->cudaMallocPitch(devPtr, pitch, width, height);
}

cudaError_t  cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height) {
	return cuda::CudaRuntimeInterface::get()->cudaMallocArray(array, desc, width, height);
}

cudaError_t  cudaFree(void *devPtr) {
	return cuda::CudaRuntimeInterface::get()->cudaFree(devPtr);
}

cudaError_t  cudaFreeHost(void *ptr) {
	return cuda::CudaRuntimeInterface::get()->cudaFreeHost(ptr);
}

cudaError_t  cudaFreeArray(struct cudaArray *array) {
	return cuda::CudaRuntimeInterface::get()->cudaFreeArray(array);
}


cudaError_t  cudaHostAlloc(void **pHost, size_t bytes, unsigned int flags) {
	return cuda::CudaRuntimeInterface::get()->cudaHostAlloc(pHost, bytes, flags);
}

cudaError_t  cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
	return cuda::CudaRuntimeInterface::get()->cudaHostGetDevicePointer(pDevice, pHost, flags);
}

cudaError_t  cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
	return cuda::CudaRuntimeInterface::get()->cudaHostGetFlags(pFlags, pHost);
}

cudaError_t cudaHostRegister(void *pHost, size_t bytes, unsigned int flags) {
	return cuda::CudaRuntimeInterface::get()->cudaHostRegister(pHost, bytes, flags);
}

cudaError_t cudaHostUnregister(void *pHost) {
	return cuda::CudaRuntimeInterface::get()->cudaHostUnregister(pHost);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpy(dst, src, count, kind);
}

cudaError_t  cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
}

cudaError_t  cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
}

cudaError_t  cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
}

cudaError_t  cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t  cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
}

cudaError_t  cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
}

cudaError_t  cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind);
}

cudaError_t  cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t  cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t  cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream);
}

cudaError_t  cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream);
}

cudaError_t  cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}

cudaError_t  cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream);
}

cudaError_t  cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream);
}

cudaError_t  cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
}

cudaError_t  cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaMemset(void *devPtr, int value, size_t count) {
	return cuda::CudaRuntimeInterface::get()->cudaMemset(devPtr, value, count);
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
	return cudaMemset(devPtr, value, count);
}

cudaError_t  cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height) {
	return cuda::CudaRuntimeInterface::get()->cudaMemset2D(devPtr, pitch, value, width, height);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaGetSymbolAddress(void **devPtr, const char *symbol) {
	return cuda::CudaRuntimeInterface::get()->cudaGetSymbolAddress(devPtr, symbol);
}

cudaError_t  cudaGetSymbolSize(size_t *size, const char *symbol) {
	return cuda::CudaRuntimeInterface::get()->cudaGetSymbolSize(size, symbol);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaGetDeviceCount(int *count) {
	return cuda::CudaRuntimeInterface::get()->cudaGetDeviceCount(count);
}

cudaError_t  cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
	return cuda::CudaRuntimeInterface::get()->cudaGetDeviceProperties(prop, device);
}

cudaError_t  cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
	return cuda::CudaRuntimeInterface::get()->cudaChooseDevice(device, prop);
}

cudaError_t  cudaSetDevice(int device) {
	return cuda::CudaRuntimeInterface::get()->cudaSetDevice(device);
}

cudaError_t  cudaGetDevice(int *device) {
	return cuda::CudaRuntimeInterface::get()->cudaGetDevice(device);
}

cudaError_t  cudaSetValidDevices(int *device_arr, int len) {
	return cuda::CudaRuntimeInterface::get()->cudaSetValidDevices(device_arr, len);
}

cudaError_t  cudaSetDeviceFlags( int flags ) {
	return cuda::CudaRuntimeInterface::get()->cudaSetDeviceFlags(flags);
}

cudaError_t cudaDeviceGetAttribute( int* value, cudaDeviceAttr attribute,
	int device ) {
	return cuda::CudaRuntimeInterface::get()->cudaDeviceGetAttribute(
		value, attribute, device);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size) {
	return cuda::CudaRuntimeInterface::get()->cudaBindTexture(offset, texref, devPtr, desc, size);
}

cudaError_t  cudaBindTexture2D(size_t *offset,const struct textureReference *texref,const void *devPtr, const struct cudaChannelFormatDesc *desc,size_t width, size_t height, size_t pitch) {
	return cuda::CudaRuntimeInterface::get()->cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch);
}

cudaError_t  cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc) {
	return cuda::CudaRuntimeInterface::get()->cudaBindTextureToArray(texref, array, desc);
}

cudaError_t  cudaUnbindTexture(const struct textureReference *texref) {
	return cuda::CudaRuntimeInterface::get()->cudaUnbindTexture(texref);
}

cudaError_t  cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref) {
	return cuda::CudaRuntimeInterface::get()->cudaGetTextureAlignmentOffset(offset, texref);
}

cudaError_t  cudaGetTextureReference(const struct textureReference **texref, const char *symbol) {
	return cuda::CudaRuntimeInterface::get()->cudaGetTextureReference(texref, symbol);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array) {
	return cuda::CudaRuntimeInterface::get()->cudaGetChannelDesc(desc, array);
}

struct cudaChannelFormatDesc  cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f) {
	struct cudaChannelFormatDesc desc = {x, y, z, w, f};
	return desc;
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaGetLastError(void) {
	return cuda::CudaRuntimeInterface::get()->cudaGetLastError();
}

cudaError_t cudaPeekAtLastError() {
	return cuda::CudaRuntimeInterface::get()->cudaPeekAtLastError();
}

#define _CASE(x) case x: return #x;

const char*  cudaGetErrorString(cudaError_t error) {
	switch (error) {
	_CASE(cudaSuccess)
	_CASE(cudaErrorMissingConfiguration)
	_CASE(cudaErrorMemoryAllocation)
	_CASE(cudaErrorInitializationError)
	_CASE(cudaErrorLaunchFailure)
	_CASE(cudaErrorPriorLaunchFailure)
	_CASE(cudaErrorLaunchTimeout)
	_CASE(cudaErrorLaunchOutOfResources)
	_CASE(cudaErrorInvalidDeviceFunction)
	_CASE(cudaErrorInvalidConfiguration)
	_CASE(cudaErrorInvalidDevice)
	_CASE(cudaErrorInvalidValue)
	_CASE(cudaErrorInvalidPitchValue)
	_CASE(cudaErrorInvalidSymbol)
	_CASE(cudaErrorMapBufferObjectFailed)
	_CASE(cudaErrorUnmapBufferObjectFailed)
	_CASE(cudaErrorInvalidHostPointer)
	_CASE(cudaErrorInvalidDevicePointer)
	_CASE(cudaErrorInvalidTexture)
	_CASE(cudaErrorInvalidTextureBinding)
	_CASE(cudaErrorInvalidChannelDescriptor)
	_CASE(cudaErrorInvalidMemcpyDirection)
	_CASE(cudaErrorAddressOfConstant)
	_CASE(cudaErrorTextureFetchFailed)
	_CASE(cudaErrorTextureNotBound)
	_CASE(cudaErrorSynchronizationError)
	_CASE(cudaErrorInvalidFilterSetting)
	_CASE(cudaErrorInvalidNormSetting)
	_CASE(cudaErrorMixedDeviceExecution)
	_CASE(cudaErrorCudartUnloading)
	_CASE(cudaErrorUnknown)
	_CASE(cudaErrorNotYetImplemented)
	_CASE(cudaErrorMemoryValueTooLarge)
	_CASE(cudaErrorInvalidResourceHandle)
	_CASE(cudaErrorNotReady)
	_CASE(cudaErrorInsufficientDriver)
	_CASE(cudaErrorSetOnActiveProcess)
	_CASE(cudaErrorNoDevice)
	_CASE(cudaErrorStartupFailure)
	_CASE(cudaErrorApiFailureBase)
		default:
		break;
	}
	return "unimplemented";
}

#undef _CASE

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

cudaError_t  cudaSetupArgument(const void *arg, size_t size, size_t offset) {
	return cuda::CudaRuntimeInterface::get()->cudaSetupArgument(arg, size, offset);
}

cudaError_t  cudaLaunch(const char *entry) {
	return cuda::CudaRuntimeInterface::get()->cudaLaunch(entry);
}

cudaError_t  cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func) {
	return cuda::CudaRuntimeInterface::get()->cudaFuncGetAttributes(attr, func);
}

cudaError_t cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache cacheConfig) {
	return cuda::CudaRuntimeInterface::get()->cudaFuncSetCacheConfig(func, cacheConfig);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaStreamCreate(cudaStream_t *pStream) {
	return cuda::CudaRuntimeInterface::get()->cudaStreamCreate(pStream);
}

cudaError_t  cudaStreamDestroy(cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaStreamDestroy(stream);
}

cudaError_t  cudaStreamSynchronize(cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaStreamSynchronize(stream);
}

cudaError_t  cudaStreamQuery(cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaStreamQuery(stream);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
	return cuda::CudaRuntimeInterface::get()->cudaStreamWaitEvent(stream, event, flags);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaEventCreate(cudaEvent_t *event) {
	return cuda::CudaRuntimeInterface::get()->cudaEventCreate(event);
}

cudaError_t  cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
	return cuda::CudaRuntimeInterface::get()->cudaEventCreateWithFlags(event, flags);
}

cudaError_t  cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaEventRecord(event, stream);
}

cudaError_t  cudaEventQuery(cudaEvent_t event) {
	return cuda::CudaRuntimeInterface::get()->cudaEventQuery(event);
}

cudaError_t  cudaEventSynchronize(cudaEvent_t event) {
	return cuda::CudaRuntimeInterface::get()->cudaEventSynchronize(event);
}

cudaError_t  cudaEventDestroy(cudaEvent_t event) {
	return cuda::CudaRuntimeInterface::get()->cudaEventDestroy(event);
}

cudaError_t  cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
	return cuda::CudaRuntimeInterface::get()->cudaEventElapsedTime(ms, start, end);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaSetDoubleForDevice(double *d) {
	return cuda::CudaRuntimeInterface::get()->cudaSetDoubleForDevice(d);
}

cudaError_t  cudaSetDoubleForHost(double *d) {
	return cuda::CudaRuntimeInterface::get()->cudaSetDoubleForHost(d);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t cudaGLMapBufferObject(void **devPtr, GLuint bufObj) {
	return cuda::CudaRuntimeInterface::get()->cudaGLMapBufferObject(devPtr, bufObj);
}

cudaError_t cudaGLMapBufferObjectAsync(void **devPtr, GLuint bufObj, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaGLMapBufferObjectAsync(devPtr, bufObj, stream);
}

cudaError_t cudaGLRegisterBufferObject(GLuint bufObj) {
	return cuda::CudaRuntimeInterface::get()->cudaGLRegisterBufferObject(bufObj);
}

cudaError_t cudaGLSetBufferObjectMapFlags(GLuint bufObj, unsigned int flags) {
	return cuda::CudaRuntimeInterface::get()->cudaGLSetBufferObjectMapFlags(bufObj, flags);
}

cudaError_t cudaGLSetGLDevice(int device) {
	return cuda::CudaRuntimeInterface::get()->cudaGLSetGLDevice(device);
}

cudaError_t cudaGLUnmapBufferObject(GLuint bufObj) {
	return cuda::CudaRuntimeInterface::get()->cudaGLUnmapBufferObject(bufObj);
}

cudaError_t cudaGLUnmapBufferObjectAsync(GLuint bufObj, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaGLUnmapBufferObjectAsync(
		bufObj, stream);
}

cudaError_t cudaGLUnregisterBufferObject(GLuint bufObj) {
	return cuda::CudaRuntimeInterface::get()->cudaGLUnregisterBufferObject(
		bufObj);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t cudaGraphicsGLRegisterBuffer(struct cudaGraphicsResource **resource,
	GLuint buffer, unsigned int flags) {
	return cuda::CudaRuntimeInterface::get()->cudaGraphicsGLRegisterBuffer(
		resource, buffer, flags);
}

cudaError_t cudaGraphicsGLRegisterImage(struct cudaGraphicsResource **resource,
	GLuint image, int target, unsigned int flags) {
	return cuda::CudaRuntimeInterface::get()->cudaGraphicsGLRegisterImage(
		resource, image, target, flags);
}

cudaError_t cudaGraphicsUnregisterResource(
	struct cudaGraphicsResource *resource) {
	return cuda::CudaRuntimeInterface::get()->cudaGraphicsUnregisterResource(
		resource);
}

cudaError_t cudaGraphicsResourceSetMapFlags(
	struct cudaGraphicsResource *resource, unsigned int flags) {
	return cuda::CudaRuntimeInterface::get()->cudaGraphicsResourceSetMapFlags(
		resource, flags);
}

cudaError_t cudaGraphicsMapResources(int count, 
	struct cudaGraphicsResource **resources, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaGraphicsMapResources(
		count, resources, stream);
}

cudaError_t cudaGraphicsUnmapResources(int count, 
	struct cudaGraphicsResource **resources, cudaStream_t stream) {
	return cuda::CudaRuntimeInterface::get()->cudaGraphicsUnmapResources(
		count, resources, stream);
}

cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, 
	size_t *size, struct cudaGraphicsResource *resource) {
	return cuda::CudaRuntimeInterface::get(
		)->cudaGraphicsResourceGetMappedPointer(devPtr, size, resource);
}

cudaError_t cudaGraphicsSubResourceGetMappedArray(
	struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, 
	unsigned int arrayIndex, unsigned int mipLevel) {
	return cuda::CudaRuntimeInterface::get(
		)->cudaGraphicsSubResourceGetMappedArray(arrayPtr, 
		resource, arrayIndex, mipLevel);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t cudaDeviceReset(void) {
	return cuda::CudaRuntimeInterface::get()->cudaDeviceReset();
}

cudaError_t cudaDeviceSynchronize(void) {
	return cuda::CudaRuntimeInterface::get()->cudaDeviceSynchronize();
}

cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value) {
	return cuda::CudaRuntimeInterface::get()->cudaDeviceSetLimit(limit, value);
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit) {
	return cuda::CudaRuntimeInterface::get()->cudaDeviceGetLimit(pValue, limit);
}

cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache *c) {
	return cuda::CudaRuntimeInterface::get()->cudaDeviceGetCacheConfig(c);
}

cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache c) {
	return cuda::CudaRuntimeInterface::get()->cudaDeviceSetCacheConfig(c);
}

cudaError_t  cudaThreadExit(void) {
	return cuda::CudaRuntimeInterface::get()->cudaThreadExit();
}

cudaError_t  cudaThreadSynchronize(void) {
	return cuda::CudaRuntimeInterface::get()->cudaThreadSynchronize();
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

cudaError_t  cudaDriverGetVersion(int *driverVersion) {
	return cuda::CudaRuntimeInterface::get()->cudaDriverGetVersion(
		driverVersion);
}

cudaError_t  cudaRuntimeGetVersion(int *runtimeVersion) {
	return cuda::CudaRuntimeInterface::get()->cudaRuntimeGetVersion(
		runtimeVersion);
}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
cudaError_t cudaGetExportTable(const void **ppExportTable,
	const cudaUUID_t *pExportTableId) {
	return cuda::CudaRuntimeInterface::get()->cudaGetExportTable(ppExportTable,
		pExportTableId);
}


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
extern "C" {

void __cudaMutexOperation(int lock) {
	return cuda::CudaRuntimeInterface::get()->cudaMutexOperation(lock);
}

int __cudaSynchronizeThreads(void** one, void* two) {
	return cuda::CudaRuntimeInterface::get()->cudaSynchronizeThreads(one, two);
}

void __cudaTextureFetch(const void* tex, void* index, int integer, void* val) {
	return cuda::CudaRuntimeInterface::get()->cudaTextureFetch(tex, 
		index, integer, val);
}

}

#endif // EXCLUDE_CUDA_RUNTIME

