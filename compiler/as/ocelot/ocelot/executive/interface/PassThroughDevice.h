/*!
	\file PassThroughDevice.h
	\author Andrew Kerr
	\date 15 February 2011
	\brief Defines a wrapper for Ocelot devices enabling monitoring,
		serialization, and decoupling
*/

#ifndef OCELOT_PASSTHROUGHDEVICE_H_INCLUDED
#define OCELOT_PASSTHROUGHDEVICE_H_INCLUDED

// Ocelot includes
#include <ocelot/executive/interface/Device.h>
#include <ocelot/util/interface/ExtractedDeviceState.h>

namespace executive
{
	class ExecutableKernel;
}

namespace executive 
{
	/*! Interface for controlling an Ocelot device */
	class PassThroughDevice: public Device 
	{
		public:
			/*! \brief Check a memory access against all allocations */
			bool checkMemoryAccess(const void* pointer, 
				size_t size) const;
			/*! \brief Get the allocation containing a pointer or 0 */
			Device::MemoryAllocation* getMemoryAllocation(const void* address, 
				AllocationType type = DeviceAllocation) const;
			/*! \brief Get the address of a global by name */
			Device::MemoryAllocation* getGlobalAllocation(
				const std::string& module, const std::string& name);
			/*! \brief Allocate some new dynamic memory on this device */
			Device::MemoryAllocation* allocate(size_t size);
			/*! \brief Make this a host memory allocation */
			Device::MemoryAllocation* allocateHost(size_t size, 
				unsigned int flags = 0);
			/*! \brief Register a host memory allocation */
			Device::MemoryAllocation* registerHost(void* pointer, size_t size, 
				unsigned int flags);
			/*! \brief Free an existing non-global allocation */
			void free(void* pointer);
			/*! \brief Get nearby allocations to a pointer */
			Device::MemoryAllocationVector getNearbyAllocations(
				void* pointer) const;
			/*! \brief Get all allocations, host, global, and device */
			Device::MemoryAllocationVector getAllAllocations() const;
			/*! \brief Get a string representation of nearby allocations */
			std::string nearbyAllocationsToString(void* pointer) const;
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
			void unmapGraphicsResource(void** resource, int count,
				unsigned int stream);

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
			void recordEvent(unsigned int event, 
				unsigned int stream);
			/*! \brief Synchronize on an event */
			void synchronizeEvent(unsigned int event);
			/*! \brief Get the elapsed time in ms between two recorded events */
			float getEventTime(unsigned int start, 
				unsigned int end);
		
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
			/*! \brief is this device selected? */
			bool selected();
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
				\param argumentBlock array of bytes for argument memory
				\param argumentBlockSize number of bytes in argument memory
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
			void limitWorkerThreads(unsigned int threads);
			void setOptimizationLevel(
				translator::Translator::OptimizationLevel level);
		
		public:
			PassThroughDevice(Device *target, unsigned int flags = 0,
				const std::string& filter = "");
			~PassThroughDevice();

		private:
			/*! \brief A vector of module pointers */
			typedef std::vector<const ir::Module*> ModuleVector;

		private:
			/*! \brief Record all of the state that could affect the execution
				of the kernel */
			void _recordStatePreExecution();
			
			/*! \brief Record information representing the kernel launch */
			void _recordKernelLaunch(const std::string& module, 
				const std::string& kernel, 
				const ir::Dim3& grid, 
				const ir::Dim3& block, 
				size_t sharedMemory, 
				const void* argumentBlock, 
				size_t argumentBlockSize);

			/*! \brief Record the memory state after the kernel launch for
				comparison, serialize it to disk */
			void _recordStatePostExecution();

			/*! \brief Get the amount of static shared memory needed by a
				specific kernel */
			unsigned int _getStaticSharedMemorySize(const std::string& module, 
				const std::string& kernel);
			
		private:
			/*! \brief a counter for the number of kernels launched */
			unsigned int _kernelCount;
		
			/*! \brief maintains a snapshot of state when the kernel executes */
			util::ExtractedDeviceState _state;
			
			/*! \brief function calls target this device*/
			Device* _target;
			
			/*! \brief The list of all modules */
			ModuleVector _modules;
			
			/*! \brief The filter of kernel names to instrument */
			std::string _kernelFilter;	
	};
}

#endif

