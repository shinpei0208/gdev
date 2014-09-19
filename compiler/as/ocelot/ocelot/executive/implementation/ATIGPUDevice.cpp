/*! \file ATIGPUDevice.cpp
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 7, 2010
 *  \brief The source file for the ATI GPU Device class.
 */

// C standard library includes
#include <string.h>

// Ocelot includes
#include <ocelot/executive/interface/ATIGPUDevice.h>
#include <ocelot/executive/interface/ATIExecutableKernel.h>
#include <ocelot/transforms/interface/PassManager.h>

// Hydrazine includes
#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>

// TODO Temporarily. Shouldn't be here
#include <ocelot/cuda/interface/cuda_runtime.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

#define Throw(x) {std::stringstream s; s << x; \
	throw hydrazine::Exception(s.str()); }

namespace executive
{
    ATIGPUDevice::ATIGPUDevice() 
		: 
			_allocations(),
			_uav0AllocPtr(0),
			_uav0Resource(0),
			_cb0Resource(0),
			_cb1Resource(0),
			_device(0), 
			_status(),
			_attribs(),
			_context(0), 
			_event(0)
    {
		report("Creating new ATIGPUDevice");

		CalDriver()->calDeviceOpen(&_device, 0);

		_attribs.struct_size = sizeof(CALdeviceattribs);
		CalDriver()->calDeviceGetAttribs(&_attribs, 0);

		_status.struct_size = sizeof(CALdevicestatus);
		CalDriver()->calDeviceGetStatus(&_status, _device);

		CalDriver()->calDeviceGetInfo(&_info, 0);

		report("Setting device properties");
		_properties.ISA = ir::Instruction::CAL;
		std::strcpy(_properties.name, "CAL Device");
		_properties.multiprocessorCount = _attribs.numberOfSIMD;
		_properties.maxThreadsPerBlock = 256;
		_properties.sharedMemPerBlock = 32768;
		_properties.SIMDWidth = _attribs.wavefrontSize;
		_properties.regsPerBlock = 16384;
		_properties.major = 2;
		_properties.minor = 0;

        // Multiple contexts per device is not supported yet
        // only one context per device so we can create it in the constructor
		report("Creating device context");
		CalDriver()->calCtxCreate(&_context, _device);

		// Allocate uav0 resource
		report("Allocating global memory (uav0)");
		CALuint width = Uav0Size;
		CALuint flags = CAL_RESALLOC_GLOBAL_BUFFER;
		CalDriver()->calResAllocLocal1D(
				&_uav0Resource, 
				_device, 
				width,
				CAL_FORMAT_UNSIGNED_INT32_1,
				flags);

		// Allocate cb0 resource
		report("Allocating ABI memory (cb0)");
		flags = 0;
		CalDriver()->calResAllocLocal1D(
				&_cb0Resource, 
				_device, 
				cbMaxSize,
				CAL_FORMAT_INT_4,
				flags);

		// Allocate cb1 resource
		flags = 0;
		report("Allocating param memory (cb1)");
		CalDriver()->calResAllocLocal1D(
				&_cb1Resource, 
				_device, 
				cbMaxSize,
				CAL_FORMAT_INT_1,
				flags);
    }

    ATIGPUDevice::~ATIGPUDevice() 
    {
		report("Destroying ATIGPUDevice");

		for(AllocationMap::iterator allocation = _allocations.begin(); 
			allocation != _allocations.end(); ++allocation)
		{
			delete allocation->second;
		}
		
		for(ModuleMap::iterator module = _modules.begin(); 
			module != _modules.end(); ++module)
		{
			delete module->second;
		}
		_modules.clear();

		CalDriver()->calResFree(_uav0Resource);
		CalDriver()->calResFree(_cb0Resource);
		CalDriver()->calResFree(_cb1Resource);

        CalDriver()->calCtxDestroy(_context);
        CalDriver()->calDeviceClose(_device);
    }

	DeviceVector ATIGPUDevice::createDevices(unsigned int flags,
		int computeCapability)
	{
		DeviceVector devices;

		if(deviceCount(computeCapability) == 0) return devices;

		try {
			// Multiple devices is not supported yet
			devices.push_back(new ATIGPUDevice());
		} catch (hydrazine::Exception he) {
			// Swallow the exception and return empty device vector
			report(he.what());
		}

		return devices;
	}
	
	unsigned int ATIGPUDevice::deviceCount(int computeCapability)
	{
		CALuint count = 0;

		try {
			CalDriver()->calDeviceGetCount(&count);

			// Multiple devices is not supported yet
			if (count > 1) {
				assertM(false, "Multiple devices is not supported yet");
			}
		} catch (hydrazine::Exception he) {
			// Swallow the exception and return 0 devices
			report(he.what());
		}

		return count;
	}	

    void ATIGPUDevice::load(const ir::Module *ir)
    {
		report("Loading Module...");

		if (_modules.count(ir->path()) != 0)
		{
			Throw("Duplicate module - " << ir->path());
		}
		_modules.insert(std::make_pair(ir->path(), new Module(ir, this)));
    }

    void ATIGPUDevice::unload(const std::string& name)
    {
		ModuleMap::iterator module = _modules.find(name);
		if(module == _modules.end())
		{
			Throw("Cannot unload unknown module - " << name);
		}
		
		for(Module::GlobalMap::iterator global = module->second->globals.begin();
			global != module->second->globals.end(); ++global)
		{
			AllocationMap::iterator allocation = 
				_allocations.find(global->second);
			assert(allocation != _allocations.end());
			delete allocation->second;
			_allocations.erase(allocation);
		}
		
		delete module->second;
		
		_modules.erase(module);
    }

    ExecutableKernel* ATIGPUDevice::getKernel(const std::string& moduleName, 
		const std::string& kernelName)
    {
		assertM(false, "Not implemented.");
		return 0;
    }

	Device::MemoryAllocation *ATIGPUDevice::getMemoryAllocation(
			const void *address, AllocationType type) const
	{
		MemoryAllocation *allocation = 0;

		if (type == HostAllocation) {
			assertM(false, "Not implemented yet");
		} else {
			if (!_allocations.empty()) {
				AllocationMap::const_iterator alloc = 
					_allocations.upper_bound((void*)address);
				if(alloc != _allocations.begin()) --alloc;
				if(alloc != _allocations.end())
				{
					if(!alloc->second->host()
					 	&& (char*)address >= (char*)alloc->second->pointer())
					{
						allocation = alloc->second;
						return allocation;
					}
				}
			}
		}

		return allocation;
	}
	
	Device::MemoryAllocation *ATIGPUDevice::getGlobalAllocation(
			const std::string &moduleName, const std::string &name)
	{
		report("Getting global allocation: " << "module = [" << moduleName 
				<< "] " << "global = [" << name << "]");

		if (moduleName.empty())
		{
			// try a brute force search over all modules
			for (ModuleMap::iterator module = _modules.begin(); 
				module != _modules.end(); ++module)
			{
				if (module->second->globals.empty())
				{
					Module::AllocationVector allocations = std::move(
						module->second->loadGlobals());
					for(Module::AllocationVector::iterator 
						allocation = allocations.begin(); 
						allocation != allocations.end(); ++allocation)
					{
						_allocations.insert(std::make_pair(
							(*allocation)->pointer(), *allocation));
					}
				}

				Module::GlobalMap::iterator global = 
					module->second->globals.find(name);
				if (global != module->second->globals.end())
				{
					return getMemoryAllocation(global->second,
						DeviceAllocation);
				}
			}
			return 0;
		}

		ModuleMap::iterator module = _modules.find(moduleName);
		if (module == _modules.end()) return 0;

		if (module->second->globals.empty())
		{
			Module::AllocationVector allocations = std::move(
					module->second->loadGlobals());
			for (Module::AllocationVector::iterator
					allocation = allocations.begin();
					allocation != allocations.end(); ++allocation)
			{
				_allocations.insert(std::make_pair((*allocation)->pointer(),
									*allocation));
			}
		}

		Module::GlobalMap::iterator global = module->second->globals.find(name);
		if (global == module->second->globals.end()) return 0;

		return getMemoryAllocation(global->second, DeviceAllocation);
	}

	Device::MemoryAllocation *ATIGPUDevice::allocate(size_t size)
	{
		// uav0 accesses should be aligned to 4
		size_t aSize = AlignUp(size, 4);

		// Check uav0 size limits
		assertM(_uav0AllocPtr + aSize < Uav0Size,
				"Out of global memory: " << _uav0AllocPtr
				<< " + " << aSize
				<< " greater than " << Uav0Size);

		MemoryAllocation *allocation = 
			new MemoryAllocation(&_uav0Resource, _uav0AllocPtr, size);
		_allocations.insert(
				std::make_pair(allocation->pointer(), allocation));

		_uav0AllocPtr += aSize;

		report("New allocation of " << size << " bytes at " 
				<< std::hex << allocation->pointer());

		return allocation;
	}

	Device::MemoryAllocation *ATIGPUDevice::allocateHost(size_t size, 
			unsigned int flags)
	{
		assertM(false, "Not implemented yet");
        return 0;
	}

	Device::MemoryAllocation *ATIGPUDevice::registerHost(void* pointer,
		size_t size, unsigned int flags)
	{
		assertM(false, "Not implemented yet");
		return 0;
	}

	void ATIGPUDevice::free(void *pointer)
	{
		if(pointer == 0) return;
		
		AllocationMap::iterator allocation = _allocations.find(pointer);
		if(allocation != _allocations.end())
		{
			if(allocation->second->global())
			{
				Throw("Cannot free global pointer - " << pointer);
			}
			delete allocation->second;
			_allocations.erase(allocation);
		}
		else
		{
			Throw("Tried to free invalid pointer - " << pointer);
		}
	}

	Device::MemoryAllocationVector ATIGPUDevice::getNearbyAllocations(
		void *pointer) const
	{
		assertM(false, "Not implemented yet");
		return Device::MemoryAllocationVector();
	}

	Device::MemoryAllocationVector ATIGPUDevice::getAllAllocations() const
	{
		assertM(false, "Not implemented yet");
		return Device::MemoryAllocationVector();
	}

	void ATIGPUDevice::clearMemory()
	{
		assertM(false, "Not implemented yet");
	}

	void *ATIGPUDevice::glRegisterBuffer(unsigned int buffer, 
			unsigned int flags)
	{
		assertM(false, "Not implemented yet");
        return 0;
	}

	void *ATIGPUDevice::glRegisterImage(unsigned int image, unsigned int target, 
			unsigned int flags)
	{
		assertM(false, "Not implemented yet");
        return 0;
	}

	void ATIGPUDevice::unRegisterGraphicsResource(void* resource)
	{
		assertM(false, "Not implemented yet");
	}

	void ATIGPUDevice::mapGraphicsResource(void** resource, int count, 
			unsigned int stream)
	{
		assertM(false, "Not implemented yet");
	}

	void *ATIGPUDevice::getPointerToMappedGraphicsResource(size_t& size, 
			void* resource)
	{
		assertM(false, "Not implemented yet");
        return 0;
	}

	void ATIGPUDevice::setGraphicsResourceFlags(void* resource, 
		unsigned int flags)
	{
		assertM(false, "Not implemented yet");
	}

	void ATIGPUDevice::unmapGraphicsResource(void** resource, int count, unsigned int streamID)
	{
		assertM(false, "Not implemented yet");
	}

	unsigned int ATIGPUDevice::createEvent(int flags)
	{
		// silently ignore
		return 0;
	}

	void ATIGPUDevice::destroyEvent(unsigned int event)
	{
		// silently ignore
	}

	bool ATIGPUDevice::queryEvent(unsigned int event)
	{
		assertM(false, "Not implemented yet");
        return 0;
	}

	void ATIGPUDevice::recordEvent(unsigned int event, unsigned int stream)
	{
		// silently ignore
	}

	void ATIGPUDevice::synchronizeEvent(unsigned int event)
	{
		// silently ignore
	}

	float ATIGPUDevice::getEventTime(unsigned int start, unsigned int end)
	{
		// silently ignore
		return 0.0;
	}

	unsigned int ATIGPUDevice::createStream()
	{
		assertM(false, "Not implemented yet");
        return 0;
	}

	void ATIGPUDevice::destroyStream(unsigned int stream)
	{
		assertM(false, "Not implemented yet");
	}

	bool ATIGPUDevice::queryStream(unsigned int stream)
	{
		assertM(false, "Not implemented yet");
        return 0;
	}

	void ATIGPUDevice::synchronizeStream(unsigned int stream)
	{
		assertM(false, "Not implemented yet");
	}

	void ATIGPUDevice::setStream(unsigned int stream)
	{
		assertM(false, "Not implemented yet");
	}

	void ATIGPUDevice::bindTexture(void* pointer, const std::string& moduleName, 
		const std::string& textureName, const textureReference& ref, 
		const cudaChannelFormatDesc& desc, const ir::Dim3& size)
	{
		assertM(false, "Not implemented yet");
	}

	void ATIGPUDevice::unbindTexture(const std::string& moduleName, 
		const std::string& textureName)
	{
		assertM(false, "Not implemented yet");
	}

	void* ATIGPUDevice::getTextureReference(const std::string& moduleName, 
		const std::string& textureName)
	{
		assertM(false, "Not implemented yet");
        return 0;
	}

	void ATIGPUDevice::_optimizePTX(Module* m, const std::string& k)
	{
		using namespace transforms;

		PassManager manager(const_cast<ir::Module*>(m->ir));

		manager.runOnKernel(k);
	}
	
	void ATIGPUDevice::launch(
			const std::string& moduleName,
			const std::string& kernelName, 
			const ir::Dim3& grid, 
			const ir::Dim3& block, 
			size_t sharedMemory, 
			const void *argumentBlock, 
			size_t argumentBlockSize, 
			const trace::TraceGeneratorVector& traceGenerators,
			const ir::ExternalFunctionSet* externals)
	{
		ModuleMap::iterator module = _modules.find(moduleName);

		if (module == _modules.end())
		{
			Throw("Unknown module - " << moduleName);
		}

		ATIExecutableKernel* kernel = 
			static_cast<ATIExecutableKernel*>(module->second->getKernel(kernelName));
		
		if(kernel == 0)
		{
			Throw("Unknown kernel - " << kernelName 
				<< " in module " << moduleName);
		}
	
		report("Launching " << moduleName << ":" << kernelName);

		if(kernel->sharedMemorySize() + sharedMemory > 
			(size_t)properties().sharedMemPerBlock)
		{
			Throw("Out of shared memory for kernel \""
				<< kernel->name << "\" : \n\tpreallocated "
				<< kernel->sharedMemorySize() << " + requested " 
				<< sharedMemory << " is greater than available " 
				<< properties().sharedMemPerBlock << " for device " 
				<< properties().name);
		}
		
		kernel->setKernelShape(block.x, block.y, block.z);
		kernel->setArgumentBlock((const unsigned char *)argumentBlock, 
				argumentBlockSize);
		kernel->updateArgumentMemory();
		kernel->updateMemory();
		kernel->setExternSharedMemorySize(sharedMemory);
		kernel->setVoteMemorySize(_properties.maxThreadsPerBlock / 32 * 4); 
		kernel->launchGrid(grid.x, grid.y, grid.z);
	}

	cudaFuncAttributes ATIGPUDevice::getAttributes(const std::string& path, 
			const std::string& kernelName)
	{
		ModuleMap::iterator module = _modules.find(path);
		
		if(module == _modules.end())
		{
			Throw("Unknown module - " << path);
		}
		
		ExecutableKernel* kernel = module->second->getKernel(kernelName);
		
		if(kernel == 0)
		{
			Throw("Unknown kernel - " << kernelName 
				<< " in module " << path);
		}
		
		cudaFuncAttributes attributes;

		memset(&attributes, 0, sizeof(cudaFuncAttributes));
		attributes.sharedSizeBytes = kernel->sharedMemorySize();
		attributes.constSizeBytes = kernel->constMemorySize();
		attributes.localSizeBytes = kernel->localMemorySize();
		attributes.maxThreadsPerBlock = kernel->maxThreadsPerBlock();
		attributes.numRegs = kernel->registerCount();
		
		return std::move(attributes);
	}

	unsigned int ATIGPUDevice::getLastError()
	{
		assertM(false, "Not implemented yet");
        return 0;
	}

	void ATIGPUDevice::synchronize()
	{
		while(_event && !CalDriver()->calCtxIsEventDone(_context, _event));
	}

	void ATIGPUDevice::limitWorkerThreads(unsigned int threads)
	{
		assertM(false, "Not implemented yet");
	}		

	void ATIGPUDevice::setOptimizationLevel(
			translator::Translator::OptimizationLevel l)
	{
	}		

	inline const cal::CalDriver *ATIGPUDevice::CalDriver()
	{
		return cal::CalDriver::Instance();
	}

	ATIGPUDevice::MemoryAllocation::MemoryAllocation(CALresource *resource, 
			const CALdeviceptr basePtr, size_t size) 
		: _resource(resource), _basePtr(basePtr), _size(size)
	{
		assertM(resource, "Invalid resource");
		assertM(size, "Invalid size");
	}

	unsigned int ATIGPUDevice::MemoryAllocation::flags() const
	{
		assertM(false, "Not implemented yet");
        return 0;
	}

	void *ATIGPUDevice::MemoryAllocation::mappedPointer() const
	{
		assertM(false, "Not implemented yet");
        return 0;
	}

	void *ATIGPUDevice::MemoryAllocation::pointer() const
	{
        return hydrazine::bit_cast<void *>(_basePtr);
	}

	size_t ATIGPUDevice::MemoryAllocation::size() const
	{
		return _size;
	}

	/*! \brief Copy from an external host pointer */
	void ATIGPUDevice::MemoryAllocation::copy(size_t offset, const void *host, 
			size_t size)
	{
		assertM(host != 0, "Invalid host pointer");
		assertM(offset + size <= _size, "Invalid copy size");
		
		CALvoid *data = NULL;
		CALuint pitch = 0;
		CALuint flags = 0;

		CalDriver()->calResMap(&data, &pitch, *_resource, flags);

		CALdeviceptr addr = _basePtr + offset;
		std::memcpy((char *)data + addr, host, size);

		report("MemoryAllocation::copy("
				<< "offset = " << std::dec << offset
				<< ", host = " << std::hex << std::showbase << host
				<< ", size = " << std::dec << size
				<< ")");
		
		CalDriver()->calResUnmap(*_resource);
	}

	/*! \brief Copy to an external host pointer */
	void ATIGPUDevice::MemoryAllocation::copy(void *host, size_t offset, 
			size_t size) const
	{
		assertM(host != 0, "Invalid host pointer");
		assertM(offset + size <= _size, "Invalid copy size");

		CALvoid *data = NULL;
		CALuint pitch = 0;
		CALuint flags = 0;

		CalDriver()->calResMap(&data, &pitch, *_resource, flags);

		CALdeviceptr addr = _basePtr + offset;
		std::memcpy(host, (char *)data + addr, size);
		report("MemoryAllocation::copy("
				<< "host = " << std::hex << std::showbase << host
				<< ", offset = " << std::dec << offset
				<< ", size = " << std::dec << size
				<< ")");
		
		CalDriver()->calResUnmap(*_resource);
	}

	void ATIGPUDevice::MemoryAllocation::memset(size_t offset, int value, 
			size_t size)
	{
		assertM(offset + size <= _size, "Invalid memset size");
		
		CALvoid *data = NULL;
		CALuint pitch = 0;
		CALuint flags = 0;

		CalDriver()->calResMap(&data, &pitch, *_resource, flags);

		CALdeviceptr addr = _basePtr + offset;
		std::memset((char *)data + addr, value, size);

		report("MemoryAllocation::memset("
				<< "offset = " << std::dec << offset
				<< ", value = " << std::dec << value
				<< ", size = " << std::dec << size
				<< ")");
		
		CalDriver()->calResUnmap(*_resource);
	}

	/*! \brief Copy to another allocation */
	void ATIGPUDevice::MemoryAllocation::copy(Device::MemoryAllocation *a,
			size_t toOffset, size_t fromOffset, size_t size) const
	{
		MemoryAllocation* allocation = static_cast<MemoryAllocation*>(a);

		assertM(_resource == allocation->_resource, "Invalid copy resources");
		assert(fromOffset + size <= _size);
		assert(toOffset + size <= allocation->_size);

		CALvoid *data = NULL;
		CALuint pitch = 0;
		CALuint flags = 0;

		CalDriver()->calResMap(&data, &pitch, *_resource, flags);

		CALdeviceptr baseFromAddr = _basePtr;
		CALdeviceptr fromAddr = baseFromAddr + fromOffset;

		CALdeviceptr baseToAddr = allocation->_basePtr;
		CALdeviceptr toAddr = baseToAddr + toOffset;

		std::memcpy((char*)data + toAddr, (char *)data + fromAddr, size);
		report("MemoryAllocation::copy("
				<< "dev = " << std::hex << std::showbase << baseFromAddr
				<< ", offset = " << std::dec << fromOffset
				<< " dev = " << std::hex << std::showbase << baseToAddr
				<< ", offset = " << std::dec << toOffset
				<< ", size = " << std::dec << size
				<< ")");
		
		CalDriver()->calResUnmap(*_resource);
	}

	ATIGPUDevice::Module::Module(const ir::Module* m, ATIGPUDevice* d) 
		: ir(m), device(d)
	{
	}

	ATIGPUDevice::Module::~Module()
	{
		for(KernelMap::iterator kernel = kernels.begin(); 
			kernel != kernels.end(); ++kernel)
		{
			delete kernel->second;
		}
	}

	ATIGPUDevice::Module::AllocationVector ATIGPUDevice::Module::loadGlobals()
	{
		assert(globals.empty());

		report("Loading module globals");

		AllocationVector allocations;

		for (ir::Module::GlobalMap::const_iterator
				global = ir->globals().begin();
				global != ir->globals().end(); ++global)
		{
			// Skip external globals
			if (global->second.statement.attribute == ir::PTXStatement::Extern)
				continue;

			report("Loading global '" << global->first << "' of size "
					<< global->second.statement.bytes());

			size_t size = global->second.statement.bytes();
			size_t aSize = AlignUp(size, 4);

			// Check uav0 size limits
			assertM(
				device->_uav0AllocPtr + aSize 
				< ATIGPUDevice::Uav0Size, "Out of global memory: " 
				<< device->_uav0AllocPtr
				<< " + " << aSize
				<< " greater than " << ATIGPUDevice::Uav0Size);

			MemoryAllocation* allocation = new MemoryAllocation(
					&(device->_uav0Resource), device->_uav0AllocPtr, size);

			// copy initial data
			void* src = std::malloc(size);
			global->second.statement.copy(src);
			allocation->copy(0, src, size);
			std::free(src);

			globals.insert(std::make_pair(global->first, 
						allocation->pointer()));
			allocations.push_back(allocation);

			// uav0 accesses should be aligned to 4
			device->_uav0AllocPtr += aSize;
		}

		return allocations;
	}

	ExecutableKernel* ATIGPUDevice::Module::getKernel(const std::string& name)
	{
		KernelMap::iterator kernel = kernels.find(name);
		if(kernel != kernels.end())
		{
			return kernel->second;
		}
		
		ir::Module::KernelMap::const_iterator ptxKernel = 
			ir->kernels().find(name);
		if(ptxKernel != ir->kernels().end())
		{
			kernel = kernels.insert(std::make_pair(name, 
				new ATIExecutableKernel(*ptxKernel->second, &device->_context, 
					&device->_event, &device->_uav0Resource, &device->_cb0Resource,
					&device->_cb1Resource, device))).first;

			return kernel->second;
		}
		
		return 0;
	}
	
	size_t AlignUp(size_t a, size_t b)
	{
		return (a % b != 0) ? (a - a % b + b) : a;
	}
}
