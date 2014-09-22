/*! \file TraceGeneratingCudaRuntime.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Saturday September 5, 2009
	\brief The header file for the TraceGeneratingCudaRuntime class.
*/

#ifndef TRACE_GENERATING_CUDA_RUNTIME_H_INCLUDED
#define TRACE_GENERATING_CUDA_RUNTIME_H_INCLUDED

#include <ocelot/cuda/interface/CudaRuntimeBase.h>
#include <fstream>

namespace cuda
{
	/*! \brief The main high performance implementation of the CUDA API */
	class TraceGeneratingCudaRuntime : public CudaRuntimeInterface, 
		public hydrazine::Configurable
	{
		private:
			/*! \brief Forwarding the actual calls here */
			CudaRuntimeBase _runtime;
			
			/*! \brief The file used to write the trace to */
			std::ofstream _trace;
			
			/*! \brief The timer used to determine the call latency */
			hydrazine::Timer _timer;
	
		public:
			/*! \brief Create the trace file */
			TraceGeneratingCudaRuntime();
			/*! \brief Destroy the trace file */
			~TraceGeneratingCudaRuntime();
	
		public:
			cudaError_t cudaMalloc3D( cudaPitchedPtr* pitchedDevPtr, 
				cudaExtent extent );
			cudaError_t cudaMalloc3DArray( cudaArray** arrayPtr, 
				const cudaChannelFormatDesc* desc, cudaExtent extent );
			cudaError_t cudaMemset3D( cudaPitchedPtr pitchedDevPtr, 
				int value, cudaExtent extent );
			cudaError_t cudaMemcpy3D( const cudaMemcpy3DParms* p );
			cudaError_t cudaMemcpy3DAsync( const cudaMemcpy3DParms* p, 
				cudaStream_t stream );

		public:
			cudaError_t cudaMalloc( void** devPtr, size_t size );
			cudaError_t cudaMallocHost( void** ptr, size_t size );
			cudaError_t cudaMallocPitch( void** devPtr, 
				size_t* pitch, size_t width, size_t height );
			cudaError_t cudaMallocArray( cudaArray** array, 
				const cudaChannelFormatDesc* desc, size_t width, 
				size_t height = 1 );
			cudaError_t cudaFree( void* devPtr );
			cudaError_t cudaFreeHost( void* ptr );
			cudaError_t cudaFreeArray( cudaArray* array );

			cudaError_t cudaHostAlloc( void** pHost, size_t bytes, 
				unsigned int flags );
			cudaError_t cudaHostGetDevicePointer( void** pDevice, 
				void* pHost, unsigned int flags );


		public:
			cudaError_t cudaMemcpy( void* dst, const void* src, 
				size_t count, cudaMemcpyKind kind );
			cudaError_t cudaMemcpyToArray( cudaArray* dst, 
				size_t wOffset, size_t hOffset, const void* src, 
				size_t count, cudaMemcpyKind kind );
			cudaError_t cudaMemcpyFromArray( void* dst, 
				const cudaArray* src, size_t wOffset, size_t hOffset, 
				size_t count, cudaMemcpyKind kind );
			cudaError_t cudaMemcpyArrayToArray( cudaArray* dst, 
				size_t wOffsetDst, size_t hOffsetDst, 
				const cudaArray* src, size_t wOffsetSrc, 
				size_t hOffsetSrc, size_t count, 
					cudaMemcpyKind kind = cudaMemcpyDeviceToDevice );
			cudaError_t cudaMemcpy2D( void* dst, size_t dpitch, 
				const void* src, size_t spitch, size_t width, size_t height, 
				cudaMemcpyKind kind );
			cudaError_t cudaMemcpy2DToArray( cudaArray* dst, 
				size_t wOffset, size_t hOffset, const void* src, 
				size_t spitch, size_t width, size_t height, 
				cudaMemcpyKind kind );
			cudaError_t cudaMemcpy2DFromArray( void* dst, 
				size_t dpitch, const cudaArray* src, size_t wOffset, 
				size_t hOffset, size_t width, size_t height, 
				cudaMemcpyKind kind );
			cudaError_t cudaMemcpy2DArrayToArray( cudaArray* dst, 
				size_t wOffsetDst, size_t hOffsetDst, const cudaArray* src, 
				size_t wOffsetSrc, size_t hOffsetSrc, size_t width, 
				size_t height, 
				cudaMemcpyKind kind = cudaMemcpyDeviceToDevice );
			cudaError_t cudaMemcpyToSymbol( const char* symbol, 
				const void* src, size_t count, size_t offset , 
				cudaMemcpyKind kind = cudaMemcpyHostToDevice );
			cudaError_t cudaMemcpyFromSymbol( void* dst, 
				const char* symbol, size_t count, size_t offset, 
				cudaMemcpyKind kind = cudaMemcpyDeviceToHost );

		public:
			cudaError_t cudaMemcpyAsync( void* dst, const void* src, 
				size_t count, cudaMemcpyKind kind, cudaStream_t stream );
			cudaError_t cudaMemcpyToArrayAsync( cudaArray* dst, 
				size_t wOffset, size_t hOffset, const void* src, size_t count, 
				cudaMemcpyKind kind, cudaStream_t stream );
			cudaError_t cudaMemcpyFromArrayAsync( void* dst, 
				const cudaArray* src, size_t wOffset, size_t hOffset, 
				size_t count, cudaMemcpyKind kind, cudaStream_t stream );
			cudaError_t cudaMemcpy2DAsync( void* dst, size_t dpitch, 
				const void* src, size_t spitch, size_t width, size_t height, 
				cudaMemcpyKind kind, cudaStream_t stream );
			cudaError_t cudaMemcpy2DToArrayAsync( cudaArray* dst, 
				size_t wOffset, size_t hOffset, const void* src, 
				size_t spitch, size_t width, size_t height, 
				cudaMemcpyKind kind, cudaStream_t stream );
			cudaError_t cudaMemcpy2DFromArrayAsync( void* dst, 
				size_t dpitch, const cudaArray* src, size_t wOffset, 
				size_t hOffset, size_t width, size_t height, 
				cudaMemcpyKind kind, cudaStream_t stream );
			cudaError_t cudaMemcpyToSymbolAsync( const char* symbol, 
				const void* src, size_t count, size_t offset, 
				cudaMemcpyKind kind, cudaStream_t stream );
			cudaError_t cudaMemcpyFromSymbolAsync( void* dst, 
				const char* symbol, size_t count, size_t offset, 
				cudaMemcpyKind kind, cudaStream_t stream );

		public:
			cudaError_t cudaMemset( void* devPtr, int value, 
				size_t count );
			cudaError_t cudaMemset2D( void* devPtr, size_t pitch, 
				int value, size_t width, size_t height );

		public:
			cudaError_t cudaGetSymbolAddress( void** devPtr, 
				const char* symbol );
			cudaError_t cudaGetSymbolSize( size_t* size, 
				const char* symbol );

		public:
			cudaError_t cudaGetDeviceCount( int* count );
			cudaError_t cudaGetDeviceProperties( cudaDeviceProp* prop, 
				int device );
			cudaError_t cudaChooseDevice( int* device, 
				const cudaDeviceProp* prop );
			cudaError_t cudaSetDevice( int device );
			cudaError_t cudaGetDevice( int* device );
			cudaError_t cudaSetValidDevices( int* device_arr, 
				int len );
			cudaError_t cudaSetDeviceFlags( int flags );

		public:
			cudaError_t cudaBindTexture( size_t* offset, 
				const textureReference* texref, const void* devPtr, 
				const cudaChannelFormatDesc* desc, size_t size = UINT_MAX );
			cudaError_t cudaBindTexture2D( size_t* offset, 
				const textureReference* texref, const void* devPtr, 
				const cudaChannelFormatDesc* desc, size_t width, 
				size_t height, size_t pitch );
			cudaError_t cudaBindTextureToArray( 
				const textureReference* texref, const cudaArray* array, 
				const cudaChannelFormatDesc* desc );
			cudaError_t cudaUnbindTexture( 
				const textureReference* texref );
			cudaError_t cudaGetTextureAlignmentOffset( size_t* offset, 
				const textureReference* texref );
			cudaError_t cudaGetTextureReference( 
				const textureReference** texref, const char* symbol );

		public:
			cudaError_t cudaGetChannelDesc( cudaChannelFormatDesc* desc, 
				const cudaArray* array );
			cudaChannelFormatDesc cudaCreateChannelDesc( 
				int x, int y, int z, int w, cudaChannelFormatKind f );

		public:
			cudaError_t cudaGetLastError( void );
			const char* cudaGetErrorString( cudaError_t error );

		public:
			cudaError_t cudaConfigureCall( dim3 gridDim, dim3 blockDim, 
				size_t sharedMem, cudaStream_t stream );
			cudaError_t cudaSetupArgument( const void* arg, 
				size_t size, size_t offset );
			cudaError_t cudaLaunch( const char* entry );
			cudaError_t cudaFuncGetAttributes( 
				cudaFuncAttributes* attr, const char* func );

		public:
			cudaError_t cudaStreamCreate( cudaStream_t* pStream );
			cudaError_t cudaStreamDestroy( cudaStream_t stream );
			cudaError_t cudaStreamSynchronize( cudaStream_t stream );
			cudaError_t cudaStreamQuery( cudaStream_t stream );

		public:
			cudaError_t cudaEventCreate( cudaEvent_t* event );
			cudaError_t cudaEventCreateWithFlags( 
				cudaEvent_t* event, int flags );
			cudaError_t cudaEventRecord( cudaEvent_t event, 
				cudaStream_t stream );
			cudaError_t cudaEventQuery( cudaEvent_t event );
			cudaError_t cudaEventSynchronize( cudaEvent_t event );
			cudaError_t cudaEventDestroy( cudaEvent_t event );
			cudaError_t cudaEventElapsedTime( float* ms, 
				cudaEvent_t start, cudaEvent_t end );

		public:
			cudaError_t cudaSetDoubleForDevice( double* d );
			cudaError_t cudaSetDoubleForHost( double* d );

		public:
			cudaError_t cudaThreadExit( void );
			cudaError_t cudaThreadSynchronize( void );

		public:
			cudaError_t cudaDriverGetVersion( int* driverVersion );
			cudaError_t cudaRuntimeGetVersion( int* runtimeVersion );
			
		public:
			cudaError_t cudaGLSetGLDevice( int device );
			cudaError_t cudaGLRegisterBufferObject( GLuint bufObj );
			cudaError_t cudaGLMapBufferObject( void **devPtr, GLuint bufObj );
			cudaError_t cudaGLUnmapBufferObject( GLuint bufObj );
			cudaError_t cudaGLUnregisterBufferObject( GLuint bufObj );
			void cudaTextureFetch( const void *tex, void *index, 
				int integer, void *val );

		public:
			void** cudaRegisterFatBinary( void* fatCubin );
			void cudaUnregisterFatBinary( void** fatCubinHandle );
			void cudaRegisterVar( void** fatCubinHandle, char* hostVar, 
				char* deviceAddress, const char* deviceName, int ext, int size, 
				int constant, int global );
			void cudaRegisterTexture( void** fatCubinHandle, 
				const struct textureReference* hostVar, 
				const void** deviceAddress, const char* deviceName, int dim, 
				int norm, int ext );
			void cudaRegisterShared( void** fatCubinHandle, void** devicePtr );
			void cudaRegisterSharedVar( void** fatCubinHandle, 
				void** devicePtr, size_t size, size_t alignment, int storage );
			void cudaRegisterFunction( void** fatCubinHandle, 
				const char* hostFun, char* deviceFun, const char* deviceName, 
				int thread_limit, uint3* tid, uint3* bid, dim3* bDim, 
				dim3* gDim, int* wSize );
			void cudaMutexOperation( int lock );
			int cudaSynchronizeThreads( void**, void* );
		
		public:
			void addTraceGenerator( trace::TraceGenerator& gen, 
				bool persistent, bool safe );
			void clearTraceGenerators( bool safe );
			void limitWorkerThreads( unsigned int limit );
			void registerPTXModule(std::istream& stream, 
				const std::string& name);
			const char* getKernelPointer(const std::string& name, 
				const std::string& module);
			void** getFatBinaryHandle(const std::string& name);
			void clearErrors();
			
		public:
			void configure( const Configuration& c );
	};

}

#endif

