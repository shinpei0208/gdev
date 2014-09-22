/*! \file Device.h
	\author Gregory Diamos
	\date April 1, 2010
	\brief The header file for the Device class.
*/

#ifndef DEVICE_H_INCLUDED
#define DEVICE_H_INCLUDED
// C++ standard library includes
#include <fstream>
#include <string>
#include <vector>

// Ocelot includes
#include <ocelot/ir/interface/Module.h>
#include <ocelot/trace/interface/TraceGenerator.h>
#include <ocelot/translator/interface/Translator.h>
#include <ocelot/executive/interface/DeviceProperties.h>

// Boost includes
#include <boost/thread/thread.hpp>

// forward declarations
struct cudaChannelFormatDesc;
struct cudaFuncAttributes;
struct textureReference;

namespace ir { class ExternalFunctionSet; }

namespace executive 
{
	/*! Interface for controlling an Ocelot device */
	class Device 
	{
		public:
			
			/*! \brief properties of a specific device */
			class Properties : public executive::DeviceProperties
			{
				public:
					/*! \brief constructor sets the default values */
					Properties(const DeviceProperties& = DeviceProperties());
			
				public:
					/*! Write attributes of the device to an output stream */
					std::ostream& write(std::ostream &out) const;
			};

			/*! \brief An interface to a memory allocation */
			class MemoryAllocation
			{
				protected:
					/*! \brief Is this memory allocation a global variable? */
					bool _global;
					/*! \brief Is this memory allocation a host allocation? */
					bool _host;

				public:
					/*! \brief Generic constructor */
					MemoryAllocation(bool global = false, bool host = false);
					/*! \brief Virtual desructor */
					virtual ~MemoryAllocation();
			
				public:
					/*! \brief Is this a mapped host pointer? */
					virtual bool host() const;
					/*! \brief Is this a mapping for a global variable? */
					virtual bool global() const;
					/*! \brief Get the flags if this is a host pointer */
					virtual unsigned int flags() const = 0;
					/*! \brief Get a host pointer if for a host allocation */
					virtual void* mappedPointer() const = 0;
					/*! \brief Get a device pointer to the base */
					virtual void* pointer() const = 0;
					/*! \brief The size of the allocation */
					virtual size_t size() const = 0;
					/*! \brief Copy from an external host pointer */
					virtual void copy(size_t offset, const void* host, 
						size_t size ) = 0;
					/*! \brief Copy to an external host pointer */
					virtual void copy(void* host, size_t offset, 
						size_t size) const = 0;
					/*! \brief Memset the allocation */
					virtual void memset(size_t offset, int value, 
						size_t size) = 0;
					/*! \brief Copy to another allocation */
					virtual void copy(MemoryAllocation* allocation, 
						size_t toOffset, size_t fromOffset, 
						size_t size) const = 0;
			};

			/*! \brief Vector of memory allocations */
			typedef std::vector< MemoryAllocation* > MemoryAllocationVector;
			
			/*! \brief Vector of devices */
			typedef std::vector< Device* > DeviceVector;

			/*! \brief Possible types of allocations */
			enum AllocationType
			{
				HostAllocation,
				DeviceAllocation,
				AnyAllocation,
				InvalidAllocation
			};

		protected:
			/*! \brief The status of each thread that is
				connected to this device */
			typedef std::map<boost::thread::id, bool> ThreadMap;

		protected:
			/*! \brief The properties of this device */
			Properties _properties;
			/*! \brief The driver version */
			int _driverVersion;
			/*! \brief The runtime version */
			int _runtimeVersion;
			/*! \brief Device flags */
			unsigned int _flags;
			
		protected:
			/*! \brief Threads that have selected this device */
			ThreadMap _selected;
			/*! \brief Locking object for updating selected threads */
			boost::mutex _mutex;
			
		public:
			/*! \brief Create devices with the selected isa */
			static DeviceVector createDevices(ir::Instruction::Architecture isa,
				unsigned int flags, int computeCapability);
			/*! \brief Get the total number of devices of a given ISA */
			static unsigned int deviceCount(ir::Instruction::Architecture isa,
				int computeCapability);

		public:
			/*! \brief Check a memory access against all allocations */
			virtual bool checkMemoryAccess(const void* pointer, 
				size_t size) const;
			/*! \brief Get the allocation containing a pointer or 0 */
			virtual MemoryAllocation* getMemoryAllocation(const void* address, 
				AllocationType type = DeviceAllocation) const = 0;
			/*! \brief Get the address of a global by name */
			virtual MemoryAllocation* getGlobalAllocation(
				const std::string& module, const std::string& name) = 0;
			/*! \brief Allocate some new dynamic memory on this device */
			virtual MemoryAllocation* allocate(size_t size) = 0;
			/*! \brief Register some host memory */
			virtual MemoryAllocation* registerHost(void* p, size_t size, 
				unsigned int flags = 0) = 0;
			/*! \brief Make this a host memory allocation */
			virtual MemoryAllocation* allocateHost(size_t size, 
				unsigned int flags = 0) = 0;
			/*! \brief Free an existing non-global allocation */
			virtual void free(void* pointer) = 0;
			/*! \brief Get nearby allocations to a pointer */
			virtual MemoryAllocationVector getNearbyAllocations(
				void* pointer) const = 0;
			/*! \brief Get all allocations, host, global, and device */
			virtual MemoryAllocationVector getAllAllocations() const = 0;
			/*! \brief Get a string representation of nearby allocations */
			virtual std::string nearbyAllocationsToString(void* pointer) const;
			/*! \brief Wipe all memory allocations, but keep modules */
			virtual void clearMemory() = 0;
		
		public:
			/*! \brief Registers an opengl buffer with a resource */
			virtual void* glRegisterBuffer(unsigned int buffer, 
				unsigned int flags) = 0;
			/*! \brief Registers an opengl image with a resource */
			virtual void* glRegisterImage(unsigned int image, 
				unsigned int target, unsigned int flags) = 0;
			/*! \brief Unregister a resource */
			virtual void unRegisterGraphicsResource(void* resource) = 0;
			/*! \brief Map a graphics resource for use with this device */
			virtual void mapGraphicsResource(void** resource, int count, 
				unsigned int stream) = 0;
			/*! \brief Get a pointer to a mapped resource along with its size */
			virtual void* getPointerToMappedGraphicsResource(size_t& size, 
				void* resource) = 0;
			/*! \brief Change the flags of a mapped resource */
			virtual void setGraphicsResourceFlags(void* resource, 
				unsigned int flags) = 0;
			/*! \brief Unmap a mapped resource */
			virtual void unmapGraphicsResource(void** resource, int count,
				unsigned int stream) = 0;

		public:
			/*! \brief Load a module, must have a unique name */
			virtual void load(const ir::Module* module) = 0;
			/*! \brief Unload a module by name */
			virtual void unload(const std::string& name) = 0;
			/*! \brief Get a translated kernel from the device */
			virtual ExecutableKernel* getKernel(const std::string& module, 
				const std::string& kernel) = 0;

		public:
			/*! \brief Get the device properties */
			const Properties& properties() const;

		public:
			/*! \brief Create a new event */
			virtual unsigned int createEvent(int flags) = 0;
			/*! \brief Destroy an existing event */
			virtual void destroyEvent(unsigned int event) = 0;
			/*! \brief Query to see if an event has been recorded (yes/no) */
			virtual bool queryEvent(unsigned int event) = 0;
			/*! \brief Record something happening on an event */
			virtual void recordEvent(unsigned int event, 
				unsigned int stream) = 0;
			/*! \brief Synchronize on an event */
			virtual void synchronizeEvent(unsigned int event) = 0;
			/*! \brief Get the elapsed time in ms between two recorded events */
			virtual float getEventTime(unsigned int start, 
				unsigned int end) = 0;
		
		public:
			/*! \brief Create a new stream */
			virtual unsigned int createStream() = 0;
			/*! \brief Destroy an existing stream */
			virtual void destroyStream(unsigned int stream) = 0;
			/*! \brief Query the status of an existing stream (ready/not) */
			virtual bool queryStream(unsigned int stream) = 0;
			/*! \brief Synchronize a particular stream */
			virtual void synchronizeStream(unsigned int stream) = 0;
			/*! \brief Sets the current stream */
			virtual void setStream(unsigned int stream) = 0;
			
		public:
			/*! \brief Select this device as the current device. 
				Only one device is allowed to be selected at any time. */
			virtual void select();
			/*! \brief is this device selected? */
			virtual bool selected();
			/*! \brief Deselect this device. */
			virtual void unselect();
		
		public:
			/*! \brief Binds a texture to a memory allocation at a pointer */
			virtual void bindTexture(void* pointer, 
				const std::string& moduleName, const std::string& textureName,
				const textureReference& ref, const cudaChannelFormatDesc& desc, 
				const ir::Dim3& size) = 0;
			/*! \brief unbinds anything bound to a particular texture */
			virtual void unbindTexture(const std::string& moduleName, 
				const std::string& textureName) = 0;
			/*! \brief Get's a reference to an internal texture */
			virtual void* getTextureReference(const std::string& moduleName, 
				const std::string& textureName) = 0;
		
		public:
			/*! \brief Get the driver version */
			int driverVersion() const;
			/*! \brief Get the runtime version */
			int runtimeVersion() const;

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
			virtual void launch(const std::string& module, 
				const std::string& kernel, const ir::Dim3& grid, 
				const ir::Dim3& block, size_t sharedMemory, 
				const void* argumentBlock, size_t argumentBlockSize, 
				const trace::TraceGeneratorVector& 
				traceGenerators = trace::TraceGeneratorVector(),
				const ir::ExternalFunctionSet* externals = 0) = 0;
			/*! \brief Get the function attributes of a specific kernel */
			virtual cudaFuncAttributes getAttributes(const std::string& module, 
				const std::string& kernel) = 0;
			/*! \brief Get the last error from this device */
			virtual unsigned int getLastError() = 0;
			/*! \brief Wait until all asynchronous operations have completed */
			virtual void synchronize() = 0;
			
		public:
			/*! \brief Limit the worker threads used by this device */
			virtual void limitWorkerThreads(unsigned int threads) = 0;
			/*! \brief Set the optimization level for kernels in this device */
			virtual void setOptimizationLevel(
				translator::Translator::OptimizationLevel level) = 0;
			
		public:
			/*! \brief Sets the device properties */
			Device(unsigned int flags = 0);
			/*! \brief Virtual destructor is required */
			virtual ~Device();
			
	};
	
	typedef Device::DeviceVector DeviceVector;
}

#endif

