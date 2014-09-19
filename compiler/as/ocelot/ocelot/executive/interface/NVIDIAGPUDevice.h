/*! \file NVIDIAGPUDevice.h
	\author Gregory Diamos
	\date April 7, 2010
	\brief The header file for the NVIDIAGPUDevice class.
*/

#ifndef NVIDIA_GPU_DEVICE_H_INCLUDED
#define NVIDIA_GPU_DEVICE_H_INCLUDED

// ocelot includes
#include <ocelot/executive/interface/Device.h>
#include <ocelot/cuda/interface/cuda_internal.h>

namespace executive
{
	class NVIDIAExecutableKernel;
}

namespace executive
{
	/*! Interface that should be bound to a single nvidia gpu */
	class NVIDIAGPUDevice : public Device
	{
		public:
			/*! \brief An interface to a memory allocation on the cuda driver */
			class MemoryAllocation : public Device::MemoryAllocation
			{
				private:
					/*! \brief The flags for page-locked host memory */
					unsigned int _flags;
					/*! \brief The size of the allocation in bytes */
					size_t _size;
					/*! \brief The pointer to the base of the allocation */
					CUdeviceptr _devicePointer;
					/*! \brief Host pointer to mapped/page-locked allocation */
					void* _hostPointer;
					/*! \brief Is the allocation external? */
					bool _external;
				
				public:
					/*! \brief Generic Construct */
					MemoryAllocation();
					/*! \brief Construct a device allocation */
					MemoryAllocation(size_t size);
					/*! \brief Construct a host allocation */
					MemoryAllocation(size_t size, unsigned int flags);
					/*! \brief Construct a global allocation */
					MemoryAllocation(CUmodule module, const ir::Global& global);
					/*! \brief Construct an external allocaton */
					MemoryAllocation(void* pointer, size_t size);
					/*! \brief Construct an external host allocaton */
					MemoryAllocation(void* pointer, size_t size,
						unsigned int flags);
					/*! \brief Desructor */
					~MemoryAllocation();

				public:
					/*! \brief Copy constructor */
					MemoryAllocation(const MemoryAllocation& a);
					/*! \brief Move constructor */
					MemoryAllocation(MemoryAllocation&& a);
					
					/*! \brief Assignment operator */
					MemoryAllocation& operator=(const MemoryAllocation& a);
					/*! \brief Move operator */
					MemoryAllocation& operator=(MemoryAllocation&& a);
			
				public:
					/*! \brief Get the flags if this is a host pointer */
					unsigned int flags() const;
					/*! \brief Get a host pointer if for a host allocation */
					void* mappedPointer() const;
					/*! \brief Get a device pointer to the base */
					void* pointer() const;
					/*! \brief The size of the allocation */
					size_t size() const;
					/*! \brief Copy from an external host pointer */
					void copy(size_t offset, const void* host, size_t size );
					/*! \brief Copy to an external host pointer */
					void copy(void* host, size_t offset, size_t size) const;
					/*! \brief Memset the allocation */
					void memset(size_t offset, int value, size_t size);
					/*! \brief Copy to another allocation */
					void copy(Device::MemoryAllocation* allocation, 
						size_t toOffset, size_t fromOffset, size_t size) const;
			};

		private:
			/*! \brief A class for holding the state associated with a module */
			class Module
			{
				private:
					/*! \brief A handle to the module or 0*/
					CUmodule _handle;

					/*! \brief a handler to the Device*/
					NVIDIAGPUDevice * _device;

				public:
					/*! \brief This is a map from a global name to pointer */
					typedef std::unordered_map<std::string, void*> GlobalMap;
					/*! \brief A map from a kernel name to its translation */
					typedef std::unordered_map<std::string, 
						NVIDIAExecutableKernel*> KernelMap;
					/*! \brief A vector of memory allocations */
					typedef std::vector<MemoryAllocation*> AllocationVector;
			
				public:
					/*! \brief The ir representation of a module */
					const ir::Module* ir;
					/*! \brief The set of global allocations in the module */
					GlobalMap globals;
					/*! \brief The set of translated kernels */
					KernelMap kernels;
					
				public:
					/*! \brief Construct this based on a module */
					Module(NVIDIAGPUDevice * device, const ir::Module* m = 0);
					/*! \brief Copy constructor */
					Module(const Module& m);
					/*! \brief Clean up all translated kernels */
					~Module();
					
				public:
					/*! \brief Load the module using the driver */
					void load();
					/*! \brief Has this module been loaded? */
					bool loaded() const;
					/*! \brief Translate all kernels in the module */
					void translate();
					/*! \brief Has this module been translated? */
					bool translated() const;
					/*! \brief Load all of the globals for this module */
					AllocationVector loadGlobals();
					/*! \brief Get a specific kernel or 0 */
					NVIDIAExecutableKernel* getKernel(const std::string& name);
					/*! \brief Get an opaque pointer to the texture or 0 */
					void* getTexture(const std::string& name);
			};

			/*! \brief A map of registered modules */
			typedef std::unordered_map<std::string, Module> ModuleMap;

			/*! \brief A map of memory allocations */
			typedef std::map<void*, MemoryAllocation*> AllocationMap;
			
			/*! \brief A map of registered streams */
			typedef std::unordered_map<unsigned int, CUstream> StreamMap;
			
			/*! \brief A map of registered events */
			typedef std::unordered_map<unsigned int, CUevent> EventMap;
			
			/*! \brief A map of registered graphics resources */
			typedef std::unordered_map<unsigned int, 
				CUgraphicsResource> GraphicsMap;
			
			/*! \brief A 3d array description */
			class Array3D
			{
				public:
					/*! \brief The array or 0*/
					CUarray array;
					/*! \brief The device pointer that the array is mirroring */
					CUdeviceptr ptr;
					/*! \brief The dimensions of the array */
					ir::Dim3 size;
					/*! \brief Bytes per element in the array */
					unsigned int bytesPerElement;
					
				public:
					/*! \brief Create and allocate a new array */
					Array3D(const cudaChannelFormatDesc& desc, 
						const ir::Dim3& size, CUdeviceptr d);
					/*! \brief Create a new blank array */
					Array3D();
					/*! \brief Destroy the array */
					~Array3D();
			
				public:
					/*! \brief Propagate updates from the device ptr to array*/
					void update();
			};
			
			/*! \brief A map of 3D array descriptors */
			typedef std::unordered_map<std::string, Array3D*> ArrayMap;
			
		private:
			/*! \brief A map of memory allocations in device space */
			AllocationMap _allocations;
			
			/*! \brief A map of allocations in host space */
			AllocationMap _hostAllocations;
			
			/*! \brief The modules that have been loaded */
			ModuleMap _modules;
			
			/*! \brief Registered streams */
			StreamMap _streams;
			
			/*! \brief Registered events */
			EventMap _events;
			
			/*! \brief Registered graphics resources */
			GraphicsMap _graphics;
		
			/*! \brief The driver context for this device. */
			CUcontext _context;
			
			/*! \brief Has this device been selected? */
			bool _selected;
			
			/*! \brief The next handle to assign to an event, stream, etc */
			unsigned int _next;
		
			/*! \brief The currently selected stream */
			unsigned int _selectedStream;
		
			/*! \brief Is opengl enabled? */
			bool _opengl;
			
			/*! \brief Bound 3d arrays */
			ArrayMap _arrays;
				
		private:
			/*! \brief Has the cuda driver been initialized? */
			static bool _cudaDriverInitialized;
			/*! \brief The last error code */
			static CUresult _lastError;
			
		public:
			/*! \brief Allocate a new device for each CUDA capable GPU */
			static DeviceVector createDevices(unsigned int flags,
				int computeCapability);
			/*! \brief Determine the number of CUDA GPUs in the system */
			static unsigned int deviceCount(int computeCapability);
		
		public:
			/*! \brief Sets the device properties, bind this to the cuda id */
			NVIDIAGPUDevice(int id = 0, unsigned int flags = 0);
			/*! \brief Clears all state */
			~NVIDIAGPUDevice();
			
		public:
			Device::MemoryAllocation* getMemoryAllocation(const void* address, 
				AllocationType type) const;
			/*! \brief Get the address of a global by stream */
			Device::MemoryAllocation* getGlobalAllocation(
				const std::string& module, const std::string& name);
			/*! \brief Allocate some new dynamic memory on this device */
			Device::MemoryAllocation* allocate(size_t size);
			/*! \brief Make this a host memory allocation */
			Device::MemoryAllocation* allocateHost(size_t size, 
				unsigned int flags);
			/*! \brief Register a host memory allocation */
			Device::MemoryAllocation* registerHost(void* pointer, size_t size, 
				unsigned int flags);
			/*! \brief Free an existing non-global allocation */
			void free(void* pointer);
			/*! \brief Get nearby allocations to a pointer */
			MemoryAllocationVector getNearbyAllocations(void* pointer) const;
			/*! \brief Get all allocations, host, global, and device */
			Device::MemoryAllocationVector getAllAllocations() const;
			/*! \brief Wipe all memory allocations, but keep modules */
			void clearMemory();
		
		public:
			/*! \brief Registers an opengl buffer with a resource */
			void* glRegisterBuffer(unsigned int buffer, 
				unsigned int flags);
			/*! \brief Registers an opengl image with a resource */
			void* glRegisterImage(unsigned int image, 
				unsigned int target, unsigned int flags);
			/*! \brief Unregister a resource */
			void unRegisterGraphicsResource(void* resource);
			/*! \brief Map a graphics resource for use with this device */
			void mapGraphicsResource(void** resource, int count, 
				unsigned int stream);
			/*! \brief Get a pointer to a mapped resource along with its size */
			void* getPointerToMappedGraphicsResource(size_t& size, 
				void* resource);
			/*! \brief Change the flags of a mapped resource */
			void setGraphicsResourceFlags(void* resource, 
				unsigned int flags);
			/*! \brief Unmap a mapped resource */
			void unmapGraphicsResource(void** resource, int count, unsigned int streamID);

		public:
			/*! \brief Load a module, must have a unique name */
			void load(const ir::Module* module);
			/*! \brief Unload a module by name */
			void unload(const std::string& name);
			/*! \brief Get a translated kernel from the device */
			ExecutableKernel* getKernel(const std::string& module, 
				const std::string& kernel);

		public:
			/*! \brief Create a new event */
			unsigned int createEvent(int flags);
			/*! \brief Destroy an existing event */
			void destroyEvent(unsigned int event);
			/*! \brief Query to see if an event has been recorded (yes/no) */
			bool queryEvent(unsigned int event);
			/*! \brief Record something happening on an event */
			void recordEvent(unsigned int event, unsigned int stream);
			/*! \brief Synchronize on an event */
			void synchronizeEvent(unsigned int event);
			/*! \brief Get the elapsed time in ms between two recorded events */
			float getEventTime(unsigned int start, unsigned int end);
		
		public:
			/*! \brief Create a new stream */
			unsigned int createStream();
			/*! \brief Destroy an existing stream */
			void destroyStream(unsigned int stream);
			/*! \brief Query the status of an existing stream (ready/not) */
			bool queryStream(unsigned int stream);
			/*! \brief Synchronize a particular stream */
			void synchronizeStream(unsigned int stream);
			/*! \brief Sets the current stream */
			void setStream(unsigned int stream);
			
		public:
			/*! \brief Select this device as the current device. 
				Only one device is allowed to be selected at any time. */
			void select();
			/*! \brief Deselect this device. */
			void unselect();
		
		public:
			/*! \brief Binds a texture to a memory allocation at a pointer */
			void bindTexture(void* pointer, 
				const std::string& moduleName, const std::string& textureName, 
				const textureReference& ref, const cudaChannelFormatDesc& desc, 
				const ir::Dim3& size);
			/*! \brief unbinds anything bound to a particular texture */
			void unbindTexture(const std::string& moduleName, 
				const std::string& textureName);
			/*! \brief Get's a reference to an internal texture */
			void* getTextureReference(const std::string& moduleName,
				const std::string& textureName);

		public:
			/*! \brief helper function for launching a kernel
				\param module module name
				\param kernel kernel name
				\param grid grid dimensions
				\param block block dimensions
				\param sharedMemory shared memory size
				\param argumentBlock array of bytes for parameter memory
				\param argumentBlockSize number of bytes in parameter memory
				\param traceGenerators vector of trace generators to add 
					and remove from kernel
			*/
			void launch(const std::string& module, 
				const std::string& kernel, const ir::Dim3& grid, 
				const ir::Dim3& block, size_t sharedMemory, 
				const void* argumentBlock, size_t argumentBlockSize, 
				const trace::TraceGeneratorVector& 
				traceGenerators = trace::TraceGeneratorVector(),
				const ir::ExternalFunctionSet* externals = 0);
			/*! \brief Get the function attributes of a specific kernel */
			cudaFuncAttributes getAttributes(const std::string& module, 
				const std::string& kernel);
			/*! \brief Get the last error from this device */
			unsigned int getLastError();
			/*! \brief Wait until all asynchronous operations have completed */
			void synchronize();
			
		public:
			/*! \brief Limit the worker threads used by this device */
			void limitWorkerThreads(unsigned int threads);			
			/*! \brief Set the optimization level for kernels in this device */
			void setOptimizationLevel(translator::Translator::OptimizationLevel 
				level);

	};
}

#endif
