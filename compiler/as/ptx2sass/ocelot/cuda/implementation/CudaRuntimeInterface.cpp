/*!
	\file CudaRuntimeInterface.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief defines the CudaRuntimeInterface interface
	\date 11 Dec 2009
*/

// C std lib includes
#include <assert.h>

// C++ std lib includes

// Ocelot includes
#include <ocelot/cuda/interface/CudaRuntimeInterface.h>
#include <ocelot/cuda/interface/CudaRuntime.h>

// Hydrazine includes
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

////////////////////////////////////////////////////////////////////////////////

cuda::CudaRuntimeInterface * cuda::CudaRuntimeInterface::instance = 0;

static void destroyInstance() {
	if (cuda::CudaRuntimeInterface::instance) {
		delete cuda::CudaRuntimeInterface::instance;
		cuda::CudaRuntimeInterface::instance = 0;
	}
}

////////////////////////////////////////////////////////////////////////////////

cuda::CudaRuntimeInterface * cuda::CudaRuntimeInterface::get() {
	if (!cuda::CudaRuntimeInterface::instance) {
		if (api::OcelotConfiguration::get().cuda.implementation
			== "CudaRuntime") {
			cuda::CudaRuntimeInterface::instance = new CudaRuntime;
			cuda::CudaRuntimeInterface::instance->ocelotRuntime.configure(
				api::OcelotConfiguration::get());
			std::atexit(destroyInstance);
		}
		else {
			assertM(false,"no CUDA runtime implementation "
				"matches what is requested");
		}
	}
	return cuda::CudaRuntimeInterface::instance;
}

cuda::CudaRuntimeInterface::CudaRuntimeInterface() {

}

cuda::CudaRuntimeInterface::~CudaRuntimeInterface() {

}

////////////////////////////////////////////////////////////////////////////////

void cuda::CudaRuntimeInterface::addTraceGenerator( trace::TraceGenerator& gen, 
	bool persistent ) {
	assert(0 && "unimplemented");
}
			
void cuda::CudaRuntimeInterface::clearTraceGenerators() {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::addPTXPass(transforms::Pass &pass) {
	assert(0 && "unimplemented");
}
void cuda::CudaRuntimeInterface::removePTXPass(transforms::Pass &pass) {
	assert(0 && "unimplemented");
}
void cuda::CudaRuntimeInterface::clearPTXPasses() {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::limitWorkerThreads( unsigned int limit ) {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::registerPTXModule(std::istream& stream, 
	const std::string& name) {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::registerTexture(
	const void* texref,
	const std::string& moduleName,
	const std::string& textureName, bool normalize){
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::clearErrors() {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::reset() {
	assert(0 && "unimplemented");
}

ocelot::PointerMap cuda::CudaRuntimeInterface::contextSwitch( 
	unsigned int destinationDevice, unsigned int sourceDevice ) {
	assert(0 && "unimplemented");
	return ocelot::PointerMap();
}

void cuda::CudaRuntimeInterface::unregisterModule( const std::string& name ) {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::launch(const std::string& moduleName, const 
	std::string& kernelName) {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::setOptimizationLevel(
	translator::Translator::OptimizationLevel l) {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::registerExternalFunction(
	const std::string& name, void* function) {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::removeExternalFunction(
	const std::string& name) {
	assert(0 && "unimplemented");
}

bool cuda::CudaRuntimeInterface::isExternalFunction(
	const std::string& name) {
	assert(0 && "unimplemented");
	
	return false;
}

void cuda::CudaRuntimeInterface::getDeviceProperties(
	executive::DeviceProperties &, int deviceIndex) {
	assert(0 && "unimplemented");
}

////////////////////////////////////////////////////////////////////////////////

void** cuda::CudaRuntimeInterface::cudaRegisterFatBinary(void *fatCubin) {
	assert(0 && "unimplemented");
	return 0;
}

void cuda::CudaRuntimeInterface::cudaUnregisterFatBinary(
	void **fatCubinHandle) {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::cudaRegisterVar(void **fatCubinHandle,
	char *hostVar, char *deviceAddress, const char *deviceName, int ext,
	int size, int constant, int global) {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::cudaRegisterTexture(
	void **fatCubinHandle,
	const struct textureReference *hostVar,
	const void **deviceAddress,
	const char *deviceName,
	int dim,
	int norm,
	int ext) {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::cudaRegisterShared(
	void **fatCubinHandle,
	void **devicePtr) {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::cudaRegisterSharedVar(
	void **fatCubinHandle,
	void **devicePtr,
	size_t size,
	size_t alignment,
	int storage) {
	assert(0 && "unimplemented");
}

void cuda::CudaRuntimeInterface::cudaRegisterFunction(
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
	assert(0 && "unimplemented");
}

////////////////////////////////////////////////////////////////////////////////

cudaError_t cuda::CudaRuntimeInterface::cudaMalloc3D(
	struct cudaPitchedPtr* pitchedDevPtr, 
	struct cudaExtent extent) {
	
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMalloc3DArray(
	struct cudaArray** arrayPtr, 
	const struct cudaChannelFormatDesc* desc, struct cudaExtent extent) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemset3D(
	struct cudaPitchedPtr pitchedDevPtr, int value, 
	struct cudaExtent extent) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpy3D(
	const struct cudaMemcpy3DParms *p) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpy3DAsync(
	const struct cudaMemcpy3DParms *p, 
	cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}



cudaError_t cuda::CudaRuntimeInterface::cudaMalloc(void **devPtr, size_t size) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMallocHost(
	void **ptr, size_t size) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMallocPitch(void **devPtr,
	size_t *pitch, size_t width, 
	size_t height) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMallocArray(
	struct cudaArray **array, 
	const struct cudaChannelFormatDesc *desc, size_t width, size_t height) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaFree(void *devPtr) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaFreeHost(void *ptr) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaFreeArray(struct cudaArray *array) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaHostRegister(void *pHost,
	size_t bytes, unsigned int flags) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaHostUnregister(void *pHost) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaHostAlloc(void **pHost,
	size_t bytes, unsigned int flags) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaHostGetDevicePointer(
	void **pDevice, void *pHost, 
	unsigned int flags) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaHostGetFlags(
	unsigned int *pFlags, void *pHost) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}



cudaError_t cuda::CudaRuntimeInterface::cudaMemcpy(void *dst,
	const void *src, size_t count, 
	enum cudaMemcpyKind kind) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpyToArray(
	struct cudaArray *dst, size_t wOffset, 
	size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpyFromArray(
	void *dst, const struct cudaArray *src, 
	size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpyArrayToArray(
	struct cudaArray *dst, size_t wOffsetDst, 
	size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc,
	size_t hOffsetSrc, 
	size_t count, enum cudaMemcpyKind kind) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpy2D(
	void *dst, size_t dpitch, const void *src, 
	size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpy2DToArray(
	struct cudaArray *dst, size_t wOffset, 
	size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, 
	enum cudaMemcpyKind kind) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpy2DFromArray(
	void *dst, size_t dpitch, 	
	const struct cudaArray *src, size_t wOffset, size_t hOffset,
	size_t width, size_t height, 
	enum cudaMemcpyKind kind) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpy2DArrayToArray(
	struct cudaArray *dst, size_t wOffsetDst, 
	size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc,
	size_t hOffsetSrc, 
	size_t width, size_t height, enum cudaMemcpyKind kind) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpyToSymbol(const char *symbol,
	const void *src, 
	size_t count, size_t offset, enum cudaMemcpyKind kind) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpyFromSymbol(void *dst, 
	const char *symbol, size_t count, 
	size_t offset, enum cudaMemcpyKind kind) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}



cudaError_t cuda::CudaRuntimeInterface::cudaMemcpyAsync(void *dst,
	const void *src, size_t count, 
	enum cudaMemcpyKind kind, cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpyToArrayAsync(
	struct cudaArray *dst, size_t wOffset, 
	size_t hOffset, const void *src, size_t count,
	enum cudaMemcpyKind kind, cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpyFromArrayAsync(void *dst,
	const struct cudaArray *src, 
	size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind,
	cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpy2DAsync(
	void *dst, size_t dpitch, const void *src, 
	size_t spitch, size_t width, size_t height,
	enum cudaMemcpyKind kind, cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpy2DToArrayAsync(
	struct cudaArray *dst, size_t wOffset, 
	size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, 
	enum cudaMemcpyKind kind, cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpy2DFromArrayAsync(
	void *dst, size_t dpitch, 
	const struct cudaArray *src, size_t wOffset, size_t hOffset,
	size_t width, size_t height, 
	enum cudaMemcpyKind kind, cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpyToSymbolAsync(
	const char *symbol, const void *src, 
	size_t count, size_t offset, enum cudaMemcpyKind kind,
	cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemcpyFromSymbolAsync(
	void *dst, const char *symbol, 
	size_t count, size_t offset, enum cudaMemcpyKind kind,
	cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}



cudaError_t cuda::CudaRuntimeInterface::cudaMemset(
	void *devPtr, int value, size_t count) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaMemset2D(
	void *devPtr, size_t pitch, int value, size_t width, 
	size_t height) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}



cudaError_t cuda::CudaRuntimeInterface::cudaGetSymbolAddress(
	void **devPtr, const char *symbol) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGetSymbolSize(
	size_t *size, const char *symbol) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}


cudaError_t cuda::CudaRuntimeInterface::cudaGetDeviceCount(int *count) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGetDeviceProperties(
	struct cudaDeviceProp *prop, int device) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaChooseDevice(
	int *device, const struct cudaDeviceProp *prop) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaSetDevice(int device) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGetDevice(int *device) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaSetValidDevices(
	int *device_arr, int len) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaSetDeviceFlags( int flags ) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}


cudaError_t cuda::CudaRuntimeInterface::cudaDeviceGetAttribute( int* value,
	cudaDeviceAttr attrbute, int device ) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaBindTexture(size_t *offset, 
	const struct textureReference *texref, const void *devPtr, 
	const struct cudaChannelFormatDesc *desc, size_t size) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaBindTexture2D(size_t *offset,
	const struct textureReference *texref,const void *devPtr, 
	const struct cudaChannelFormatDesc *desc,size_t width, size_t height,
	size_t pitch) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaBindTextureToArray(
	const struct textureReference *texref, 
	const struct cudaArray *array, const struct cudaChannelFormatDesc *desc) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaUnbindTexture(
	const struct textureReference *texref) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGetTextureAlignmentOffset(
	size_t *offset, 
	const struct textureReference *texref) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGetTextureReference(
	const struct textureReference **texref, 
	const char *symbol) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}



cudaError_t cuda::CudaRuntimeInterface::cudaGetChannelDesc(
	struct cudaChannelFormatDesc *desc, 
	const struct cudaArray *array) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

struct cudaChannelFormatDesc cuda::CudaRuntimeInterface::cudaCreateChannelDesc(
	int x, int y, int z, int w, 
	enum cudaChannelFormatKind f) {
	struct cudaChannelFormatDesc desc = {x, y, z, w, f};
	return desc;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGetLastError(void) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaPeekAtLastError(void) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaConfigureCall(dim3 gridDim,
	dim3 blockDim, 
	size_t sharedMem, cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaSetupArgument(const void *arg,
	size_t size, size_t offset) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaLaunch(const char *entry) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaFuncGetAttributes(
	struct cudaFuncAttributes *attr, const char *func) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaFuncSetCacheConfig(const char *func, 
	enum cudaFuncCache cacheConfig) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}


cudaError_t cuda::CudaRuntimeInterface::cudaStreamCreate(cudaStream_t *pStream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaStreamDestroy(cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaStreamSynchronize(
	cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaStreamQuery(cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, 
	unsigned int flags) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaEventCreate(cudaEvent_t *event) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaEventCreateWithFlags(
	cudaEvent_t *event, int flags) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaEventRecord(
	cudaEvent_t event, cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaEventQuery(cudaEvent_t event) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaEventSynchronize(
	cudaEvent_t event) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaEventDestroy(cudaEvent_t event) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaEventElapsedTime(
	float *ms, cudaEvent_t start, cudaEvent_t end) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}


cudaError_t cuda::CudaRuntimeInterface::cudaGLMapBufferObject(
	void **devPtr, GLuint bufObj) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGLMapBufferObjectAsync(
	void **devPtr, GLuint bufObj, cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGLRegisterBufferObject(
	GLuint bufObj) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGLSetBufferObjectMapFlags(
	GLuint bufObj, unsigned int flags) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGLSetGLDevice(int device) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGLUnmapBufferObject(GLuint bufObj) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGLUnmapBufferObjectAsync(
	GLuint bufObj, cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGLUnregisterBufferObject(
	GLuint bufObj) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGraphicsGLRegisterBuffer(
	struct cudaGraphicsResource **resource, GLuint buffer, unsigned int flags) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGraphicsGLRegisterImage(
	struct cudaGraphicsResource **resource, GLuint image, int target, 
	unsigned int flags) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGraphicsUnregisterResource(
	struct cudaGraphicsResource* resource) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGraphicsResourceSetMapFlags(
	struct cudaGraphicsResource *resource, unsigned int flags) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGraphicsMapResources(int count, 
	struct cudaGraphicsResource **resources, cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGraphicsUnmapResources(int count, 
	struct cudaGraphicsResource **resources, cudaStream_t stream) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGraphicsResourceGetMappedPointer(
	void **devPtr, size_t *size, struct cudaGraphicsResource *resource) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGraphicsSubResourceGetMappedArray(
	struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, 
	unsigned int arrayIndex, unsigned int mipLevel) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaSetDoubleForDevice(double *d) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaSetDoubleForHost(double *d) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaDeviceReset(void) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaDeviceSynchronize(void) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaDeviceSetLimit(enum cudaLimit limit,
	size_t value) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaDeviceGetLimit(size_t *pValue,
	enum cudaLimit limit) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaDeviceGetCacheConfig(
	enum cudaFuncCache *pCacheConfig) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaDeviceSetCacheConfig(
	enum cudaFuncCache cacheConfig) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaThreadExit(void) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaThreadSynchronize(void) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaThreadSetLimit(enum cudaLimit limit,
	size_t value) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaDriverGetVersion(
	int *driverVersion) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaRuntimeGetVersion(
	int *runtimeVersion) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}

cudaError_t cuda::CudaRuntimeInterface::cudaGetExportTable(
	const void **ppExportTable, const cudaUUID_t *pExportTableId) {
	assert(0 && "unimplemented");
	return cudaErrorNotYetImplemented;
}


void cuda::CudaRuntimeInterface::cudaMutexOperation(int lock) {
	assert(0 && "unimplemented");
}

int cuda::CudaRuntimeInterface::cudaSynchronizeThreads(void** one, void* two) {
	assert(0 && "unimplemented");
	return 0;
}

void cuda::CudaRuntimeInterface::cudaTextureFetch(const void* tex, 
	void* index, int integer, void* val) {
	assert(0 && "unimplemented");

}


