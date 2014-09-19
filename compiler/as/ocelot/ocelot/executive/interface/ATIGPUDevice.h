/*! \file ATIGPUDevice.h
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 7, 2010
 *  \brief The header file for the ATI GPU Device class.
 */

#ifndef ATIGPUDEVICE_H_INCLUDED
#define ATIGPUDEVICE_H_INCLUDED

// Ocelot includes
#include <ocelot/executive/interface/Device.h>
#include <ocelot/cal/interface/CalDriver.h>

namespace executive
{
	/*! \brief ATI GPU Device */
	class ATIGPUDevice : public Device 
	{
		public:
			/*! \brief ATI memory allocation */
			class MemoryAllocation : public Device::MemoryAllocation
			{
				public:
					/*! \brief Construct an allocation for a particular 
					 * resource */
					MemoryAllocation(CALresource *resource, 
							const CALdeviceptr basePtr, size_t size);

					/*! \brief Get the flags if this is a host pointer */
					unsigned int flags() const;
					/*! \brief Get a host pointer if for a host allocation */
					void *mappedPointer() const;
					/*! \brief Get a device pointer to the base */
					void *pointer() const;
					/*! \brief The size of the allocation */
					size_t size() const;
					/*! \brief Copy from an external host pointer */
					void copy(size_t offset, const void *host, size_t size );
					/*! \brief Copy to an external host pointer */
					void copy(void *host, size_t offset, size_t size) const;
					/*! \brief Memset the allocation */
					void memset(size_t offset, int value, size_t size);
					/*! \brief Copy to another allocation */
					void copy(Device::MemoryAllocation *a, 
						size_t toOffset, size_t fromOffset, size_t size) const;

				private:
					/*! \brief Resource where the allocation lives */
					const CALresource *const _resource;
					/*! \brief Base pointer of the allocation */
					const CALdeviceptr _basePtr;
					/*! \brief Size of the allocation */
					const size_t _size;
			};

		private:
			/*! \brief A class for holding the state associated with a module */
			class Module
			{
				public:
					/*! \brief This is a map from a global name to pointer */
					typedef std::unordered_map<std::string, void*> GlobalMap;
					/*! \brief A map from a kernel name to its translation */
					typedef std::unordered_map<std::string, 
						ExecutableKernel*> KernelMap;
					/*! \brief A vector of memory allocations */
					typedef std::vector<MemoryAllocation*> AllocationVector;

				public:
					/*! \brief The ir representation of a module */
					const ir::Module* ir;
					/*! \brief The device associated with this module */
					ATIGPUDevice* device;
					/*! \brief The set of global allocations in the module */
					GlobalMap globals;
					/*! \brief The set of translated kernels */
					KernelMap kernels;

				public:
					/*! \brief Construct this based on a module */
					Module(const ir::Module* m = 0, ATIGPUDevice* d = 0);
					/*! \brief Clean up all translated kernels */
					virtual ~Module();

				public:
					/*! \brief Load all of the globals for this module */
					AllocationVector loadGlobals();
					/*! \brief Get a specific kernel or 0 */
					virtual ExecutableKernel* getKernel(
						const std::string& name);
			};

		public:
			/*! \brief Allocate a new device for each CAL capable GPU */
			static DeviceVector createDevices(unsigned int flags,
				int computeCapability);
		
			/*! \brief Get the total number of CAL devices in the system */
			static unsigned int deviceCount(int computeCapability); 
		
			/*! \brief Constructor */
			ATIGPUDevice();
			/*! \brief Destructor */
			~ATIGPUDevice();

			/*! \brief Get the allocation containing a pointer or 0 */
			Device::MemoryAllocation *getMemoryAllocation(const void *address, 
				AllocationType type) const;
			/*! \brief Get the address of a global by name */
			Device::MemoryAllocation *getGlobalAllocation(
				const std::string& module, const std::string& name);
			/*! \brief Allocate some new dynamic memory on this device */
			Device::MemoryAllocation *allocate(size_t size);
			/*! \brief Make this a host memory allocation */
			Device::MemoryAllocation *allocateHost(size_t size, 
				unsigned int flags);
			/*! \brief Register a host memory allocation */
			Device::MemoryAllocation *registerHost(void* pointer, size_t size, 
				unsigned int flags);
			/*! \brief Free an existing non-global allocation */
			void free(void *pointer);
			/*! \brief Get nearby allocations to a pointer */
			Device::MemoryAllocationVector getNearbyAllocations(
				void *pointer) const;
			/*! \brief Get all allocations, host, global, and device */
			Device::MemoryAllocationVector getAllAllocations() const;
			/*! \brief Wipe all memory allocations, but keep modules */
			void clearMemory();
		
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
                    unsigned int streamID);

			/*! \brief Load a module, must have a unique name */
			void load(const ir::Module *irModule);
			/*! \brief Unload a module by name */
			void unload(const std::string& name);
			/*! \brief Get a translated kernel from the device */
			ExecutableKernel* getKernel(const std::string& module, 
				const std::string& kernel);

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
			
			/*! \brief Binds a texture to a memory allocation at a pointer */
			void bindTexture(void* pointer, const std::string& moduleName,
				const std::string& textureName, 
				const textureReference& ref, const cudaChannelFormatDesc& desc, 
				const ir::Dim3& size);
			/*! \brief unbinds anything bound to a particular texture */
			void unbindTexture(const std::string& moduleName, 
				const std::string& textureName);
			/*! \brief Get's a reference to an internal texture */
			void* getTextureReference(const std::string& moduleName, 
				const std::string& textureName);
			
			/*! \brief helper function for launching a kernel
			 *  \param module module name
			 *  \param kernel kernel name
			 *  \param grid grid dimensions
			 *  \param block block dimensions
			 *  \param sharedMemory shared memory size
			 *  \param argumentBlock array of bytes for argument memory
			 *  \param argumentBlockSize number of bytes in argument memory
			 *  \param traceGenerators vector of trace generators to add and 
			 	remove from kernel
			 */
			void launch(const std::string& module, 
					const std::string& kernel, const ir::Dim3& grid, 
					const ir::Dim3& block, size_t sharedMemory, 
					const void *argumentBlock, size_t argumentBlockSize, 
					const trace::TraceGeneratorVector& 
					traceGenerators = trace::TraceGeneratorVector(),
					const ir::ExternalFunctionSet* externals = 0);
			/*! \brief Get the function attributes of a specific kernel */
			cudaFuncAttributes getAttributes(const std::string& path, 
				const std::string& kernelName);
			/*! \brief Get the last error from this device */
			unsigned int getLastError();
			/*! \brief Wait until all asynchronous operations have completed */
			void synchronize();

			/*! \brief Limit the worker threads used by this device */
			void limitWorkerThreads(unsigned int threads);
			/*! \brief Set the optimization level for kernels in this device */
			void setOptimizationLevel(translator::Translator::OptimizationLevel 
				level);

			/*! \brief uav0 size (150 MB) */
			static const size_t Uav0Size = 150000000;

		private:
			/*! \brief Run PTX optimization on the kernel k in the module m */
			void _optimizePTX(Module* m, const std::string& k);

			/*! \brief A map of registered modules */
			typedef std::unordered_map<std::string, Module*> ModuleMap;

			/*! \brief A map of memory allocations */
			typedef std::map<void*, MemoryAllocation*> AllocationMap;

			/********************************************************//**
			 * \name uav0 Memory Manager
			 *
			 * uav0 acts as the global memory. The allocation policy is 
			 * very simple. It allocates chuncks of memory sequentially 
			 * and never reallocates a chunck that has been freed.
			 ***********************************************************/
			//@{
			/*! \brief A map of memory allocations in device space */
			AllocationMap _allocations;
			/*! \brief Pointer to the next chunck of allocatable memory */
			CALdeviceptr _uav0AllocPtr;
			/*! \brief CAL uav0 resource */
			CALresource _uav0Resource;
			//@}

			/*! \brief Maximum constant buffer size (in vectors) */
			static const CALuint cbMaxSize = 1024;

			/********************************************************//**
			 * \name cb0 Memory Manager (ABI data)
			 ***********************************************************/
			//@{
			/*! \brief CAL cb0 resource */
			CALresource _cb0Resource;
			//@}

			/********************************************************//**
			 * \name cb1 Memory Manager (param)
			 ***********************************************************/
			//@{
			/*! \brief CAL cb1 resource */
			CALresource _cb1Resource;
			//@}

			/*! \brief CAL Device */
			CALdevice _device;
			/*! \brief CAL Device Status */
			CALdevicestatus _status;
			/*! \brief CAL Device Attributes */
			CALdeviceattribs _attribs;
			/*! \brief CAL Device Info */
			CALdeviceinfo _info;
			/*! \brief CAL Context. Multiple contexts per device is 
				not supported yet */
			CALcontext _context;
			/*! \brief CAL Event */
			CALevent _event;

			/*! \brief The modules that have been loaded */
			ModuleMap _modules;

			/*! \brief Returns a pointer to an instance to the 
				CalDriver singleton */
			static const cal::CalDriver *CalDriver();
	};

	/*! \brief Align a to nearest higher multiple of b */
	size_t AlignUp(size_t a, size_t b);
}

#endif
