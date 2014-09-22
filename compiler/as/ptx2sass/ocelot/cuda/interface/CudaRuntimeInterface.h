/*! \file CudaRuntime.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief defines the CudaRuntime interface
	\date 11 Dec 2009
*/

#ifndef OCELOT_CUDA_RUNTIME_INTERFACE_H_INCLUDED
#define OCELOT_CUDA_RUNTIME_INTERFACE_H_INCLUDED

#include <ocelot/api/interface/ocelot.h>

#include <ocelot/api/interface/OcelotConfiguration.h>
#include <ocelot/api/interface/OcelotRuntime.h>

#include <ocelot/cuda/interface/cuda_runtime.h>
#include <ocelot/trace/interface/TraceGenerator.h>
#include <ocelot/translator/interface/Translator.h>

namespace cuda {
	/*!
		Singleton object called directly by CUDA Runtime API wrapper 
			- on instantiation, selects appropriate CUDA Runtime 
			implementation and dispatches calls
	*/
	class CudaRuntimeInterface {
	public:
		/* singleton accessors */
		static CudaRuntimeInterface *instance;
		
		static CudaRuntimeInterface *get();
		
		CudaRuntimeInterface();
		
		virtual ~CudaRuntimeInterface();

	public:
		//! \brief gets the Ocelot runtime object
		const ocelot::OcelotRuntime & ocelot() const;

	protected:
		//! \brief Ocelot runtime object containing state related to Ocelot
		ocelot::OcelotRuntime ocelotRuntime;
		
	public:
		/*
			Registration
		*/

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
	
		/*
			Memory - 3D
		*/
		virtual cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, 
			struct cudaExtent extent);
		virtual cudaError_t cudaMalloc3DArray(struct cudaArray** arrayPtr, 
			const struct cudaChannelFormatDesc* desc, struct cudaExtent extent);
		virtual cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, 
			int value, struct cudaExtent extent);
		virtual cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
		virtual cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, 
			cudaStream_t stream);

		/*
			Memory - linear
		*/

		virtual cudaError_t cudaMalloc(void **devPtr, size_t size);
		virtual cudaError_t cudaMallocHost(void **ptr, size_t size);
		virtual cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, 
			size_t width, size_t height);
		virtual cudaError_t cudaMallocArray(struct cudaArray **array, 
			const struct cudaChannelFormatDesc *desc, size_t width, 
			size_t height = 1);
		virtual cudaError_t cudaFree(void *devPtr);
		virtual cudaError_t cudaFreeHost(void *ptr);
		virtual cudaError_t cudaFreeArray(struct cudaArray *array);

		/*
			Memory - host allocations
		*/

		virtual cudaError_t cudaHostRegister(void *pHost, size_t bytes, 
			unsigned int flags);
		virtual cudaError_t cudaHostUnregister(void *pHost);
		virtual cudaError_t cudaHostAlloc(void **pHost, size_t bytes, 
			unsigned int flags);
		virtual cudaError_t cudaHostGetDevicePointer(void **pDevice, 
			void *pHost, unsigned int flags);
		virtual cudaError_t cudaHostGetFlags(unsigned int *pFlags, 
			void *pHost);


		/*
			Memcpy
		*/

		virtual cudaError_t cudaMemcpy(void *dst, const void *src, 
			size_t count, enum cudaMemcpyKind kind);
		virtual cudaError_t cudaMemcpyToArray(struct cudaArray *dst, 
			size_t wOffset, size_t hOffset, const void *src, 
			size_t count, enum cudaMemcpyKind kind);
		virtual cudaError_t cudaMemcpyFromArray(void *dst, 
			const struct cudaArray *src, size_t wOffset, size_t hOffset, 
			size_t count, enum cudaMemcpyKind kind);
		virtual cudaError_t cudaMemcpyArrayToArray(struct cudaArray *dst, 
			size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, 
			size_t wOffsetSrc, size_t hOffsetSrc, size_t count, 
			enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice);
		virtual cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, 
			const void *src, size_t spitch, size_t width, size_t height, 
			enum cudaMemcpyKind kind);
		virtual cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst, 
			size_t wOffset, size_t hOffset, const void *src, size_t spitch, 
			size_t width, size_t height, enum cudaMemcpyKind kind);
		virtual cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, 
			const struct cudaArray *src, size_t wOffset, size_t hOffset, 
			size_t width, size_t height, enum cudaMemcpyKind kind);
		virtual cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *dst, 
			size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, 
			size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, 
			enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice);
		virtual cudaError_t cudaMemcpyToSymbol(const char *symbol, 
			const void *src, size_t count, size_t offset = 0, 
			enum cudaMemcpyKind kind = cudaMemcpyHostToDevice);
		virtual cudaError_t cudaMemcpyFromSymbol(void *dst, 
			const char *symbol, size_t count, size_t offset = 0, 
			enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost);

		/*
			Memcpy - async
		*/

		virtual cudaError_t cudaMemcpyAsync(void *dst, const void *src, 
			size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
		virtual cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *dst, 
			size_t wOffset, size_t hOffset, const void *src, size_t count, 
			enum cudaMemcpyKind kind, cudaStream_t stream);
		virtual cudaError_t cudaMemcpyFromArrayAsync(void *dst, 
			const struct cudaArray *src, size_t wOffset, size_t hOffset, 
			size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
		virtual cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, 
			const void *src, size_t spitch, size_t width, size_t height, 
			enum cudaMemcpyKind kind, cudaStream_t stream);
		virtual cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *dst, 
			size_t wOffset, size_t hOffset, const void *src, size_t spitch, 
			size_t width, size_t height, enum cudaMemcpyKind kind, 
			cudaStream_t stream);
		virtual cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, 
			size_t dpitch, const struct cudaArray *src, size_t wOffset, 
				size_t hOffset, size_t width, size_t height, 
				enum cudaMemcpyKind kind, cudaStream_t stream);
		virtual cudaError_t cudaMemcpyToSymbolAsync(const char *symbol, 
			const void *src, size_t count, size_t offset, 
			enum cudaMemcpyKind kind, cudaStream_t stream);
		virtual cudaError_t cudaMemcpyFromSymbolAsync(void *dst, 
			const char *symbol, size_t count, size_t offset, 
			enum cudaMemcpyKind kind, cudaStream_t stream);

		/*
			Memset
		*/

		virtual cudaError_t cudaMemset(void *devPtr, int value, size_t count);
		virtual cudaError_t cudaMemset2D(void *devPtr, size_t pitch, 
			int value, size_t width, size_t height);

		/*
			Symbols
		*/

		virtual cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol);
		virtual cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol);

		/*
			Device enumeration and count
		*/

		virtual cudaError_t cudaGetDeviceCount(int *count);
		virtual cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
		virtual cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
		virtual cudaError_t cudaSetDevice(int device);
		virtual cudaError_t cudaGetDevice(int *device);
		virtual cudaError_t cudaSetValidDevices(int *device_arr, int len);
		virtual cudaError_t cudaSetDeviceFlags( int flags );
		virtual cudaError_t cudaDeviceGetAttribute( int* value, cudaDeviceAttr attrbute,
			int device );

		/*
			Texture binding
		*/

		virtual cudaError_t cudaBindTexture(size_t *offset, 
			const struct textureReference *texref, const void *devPtr, 
			const struct cudaChannelFormatDesc *desc, size_t size = UINT_MAX);
		virtual cudaError_t cudaBindTexture2D(size_t *offset,
			const struct textureReference *texref,const void *devPtr, 
			const struct cudaChannelFormatDesc *desc,size_t width, 
			size_t height, size_t pitch);
		virtual cudaError_t cudaBindTextureToArray(
			const struct textureReference *texref, const struct cudaArray *array, 
			const struct cudaChannelFormatDesc *desc);
		virtual cudaError_t cudaUnbindTexture(const struct textureReference *texref);
		virtual cudaError_t cudaGetTextureAlignmentOffset(size_t *offset, 
			const struct textureReference *texref);
		virtual cudaError_t cudaGetTextureReference(const struct textureReference **texref, 
			const char *symbol);

		/*
			Channel creation
		*/

		virtual cudaError_t cudaGetChannelDesc(
			struct cudaChannelFormatDesc *desc, const struct cudaArray *array);
		virtual struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, 
			int y, int z, int w, enum cudaChannelFormatKind f);

		/*
			Error enumeration
		*/

		virtual cudaError_t cudaGetLastError(void);
		virtual cudaError_t cudaPeekAtLastError(void);

		/*
			Kernel launch
		*/

		virtual cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, 
			size_t sharedMem = 0, cudaStream_t stream = 0);
		virtual cudaError_t cudaSetupArgument(const void *arg, size_t size, 
			size_t offset);
		virtual cudaError_t cudaLaunch(const char *entry);
		virtual cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, 
			const char *func);
		virtual cudaError_t cudaFuncSetCacheConfig(const char *func, 
			enum cudaFuncCache cacheConfig);


		/*
			Stream creation
		*/

		virtual cudaError_t cudaStreamCreate(cudaStream_t *pStream);
		virtual cudaError_t cudaStreamDestroy(cudaStream_t stream);
		virtual cudaError_t cudaStreamSynchronize(cudaStream_t stream);
		virtual cudaError_t cudaStreamQuery(cudaStream_t stream);
		virtual cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);

		/*
			Event creation
		*/

		virtual cudaError_t cudaEventCreate(cudaEvent_t *event);
		virtual cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, int flags);
		virtual cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
		virtual cudaError_t cudaEventQuery(cudaEvent_t event);
		virtual cudaError_t cudaEventSynchronize(cudaEvent_t event);
		virtual cudaError_t cudaEventDestroy(cudaEvent_t event);
		virtual cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, 
			cudaEvent_t end);

		/* 
			OpenGl
		*/
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

		/*
			Graphics interface
		*/
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
			
		/*
			double precision
		*/

		virtual cudaError_t cudaSetDoubleForDevice(double *d);
		virtual cudaError_t cudaSetDoubleForHost(double *d);

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

		/*
			Thread synchronization
		*/

		virtual cudaError_t cudaThreadExit(void);
		virtual cudaError_t cudaThreadSynchronize(void);
		virtual cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value);

		/*
			Version accessors
		*/

		virtual cudaError_t cudaDriverGetVersion(int *driverVersion);
		virtual cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);

		/*
			Symbol Tables
		*/
		virtual cudaError_t cudaGetExportTable(const void **ppExportTable, 
			const cudaUUID_t *pExportTableId);

		/*
			Runtime Synchronization
		*/
		virtual void cudaMutexOperation( int lock );
		virtual int cudaSynchronizeThreads( void** one, void* two );
		
		/*	Texture emulation */
		virtual void cudaTextureFetch(const void* tex, void* index, 
			int integer, void* val);
	public:
	
		/*! \brief Adds a trace generator for the next kernel invocation 
	
			\param gen A reference to the generator being added, it must not
				be destroyed until the next kernel is executed.
			\param persistent The trace generator will be associated with all
				subsequent kernels until clear is called, otherwise it will
				only be associated with the next kernel.
		*/
		virtual void addTraceGenerator( trace::TraceGenerator& gen, 
			bool persistent = false );
		/*! \brief Clear all trace generators */
		virtual void clearTraceGenerators();
		/*! \brief Adds a PTX->PTX pass for the next *Module load* */
		virtual void addPTXPass(transforms::Pass &pass);
		/*!	\brief removes the specified pass */
		virtual void removePTXPass(transforms::Pass &pass);
		/*! \brief clears all PTX->PTX passes */
		virtual void clearPTXPasses();
		
		/*! \brief Sets a limit on the number of host worker threads to launch
			when executing a CUDA kernel on a Multi-Core CPU.
			\param limit The max number of worker threads to launch per kernel.
		*/
		virtual void limitWorkerThreads( unsigned int limit = 1024 );
		/*! \brief Register an istream containing a PTX module.
		
			\param stream An input stream containing a PTX module
			\param The name of the module being registered. Must be Unique.
		*/
		virtual void registerPTXModule(std::istream& stream, 
			const std::string& name);
		/*! \brief Register a texture with the cuda runtime */
		virtual void registerTexture(const void* texref,
			const std::string& moduleName,
			const std::string& textureName, bool normalize);
		/*! \brief Clear all errors in the Cuda Runtime */
		virtual void clearErrors();
		/*! \brief Reset all CUDA runtime state */
		virtual void reset();
		/*! \brief Perform a device context switch */
		virtual ocelot::PointerMap contextSwitch(unsigned int destinationDevice, 
			unsigned int sourceDevice);
		/*! \brief Unregister a module, either PTX or LLVM, not a fatbinary */
		virtual void unregisterModule( const std::string& name );
		/*! \brief Launch a cuda kernel by name */
		virtual void launch(const std::string& moduleName, 
			const std::string& kernelName);
		/*! \brief Set the optimization level */
		virtual void setOptimizationLevel(
			translator::Translator::OptimizationLevel l);
		/*! \brief Register a callable host function with Ocelot 

			This function will be callable as a PTX function.
		*/
		virtual void registerExternalFunction(const std::string& name,
			void* function);
		/*! \brief Remove a previously registered host function */
		virtual void removeExternalFunction(const std::string& name);
		/*! \brief Is a named function already registered? */
		virtual bool isExternalFunction(const std::string& name);
	
		virtual void getDeviceProperties(executive::DeviceProperties &, int deviceIndex = -1);
	};

}

#endif

