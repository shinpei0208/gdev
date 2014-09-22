/*! \file NVIDIAGPUDevice.cpp
	\author Gregory Diamos
	\date April 7, 2010
	\brief The source file for the NVIDIAGPUDevice class.
*/

#ifndef NVIDIA_GPU_DEVICE_CPP_INCLUDED
#define NVIDIA_GPU_DEVICE_CPP_INCLUDED

// C++ includes
#include <iomanip>

// ocelot includes
#include <ocelot/executive/interface/NVIDIAGPUDevice.h>
#include <ocelot/executive/interface/NVIDIAExecutableKernel.h>
#include <ocelot/cuda/interface/CudaDriver.h>
#include <ocelot/cuda/interface/cuda_runtime.h>
#include <ocelot/transforms/interface/PassManager.h>
#include <ocelot/transforms/interface/SharedPtrAttribute.h>

// hydrazine includes
#include <hydrazine/interface/SystemCompatibility.h>
#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/string.h>

// standard library includes
#include <cstring>

////////////////////////////////////////////////////////////////////////////////

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define checkError(x) if((_lastError = x) != CUDA_SUCCESS) { \
	report("exception"); \
	throw hydrazine::Exception("Cuda Driver Error - " #x + \
		driver::toString(_lastError)); }
#define Throw(x) {std::stringstream s; s << x; report(s.str()); \
	throw hydrazine::Exception(s.str()); }

////////////////////////////////////////////////////////////////////////////////

// Turn on report messages
#define REPORT_BASE 0

// Print out the full ptx for each module as it is loaded
#define REPORT_PTX 0

// if 1, adds line numbers to reported PTX
#define REPORT_PTX_WITH_LINENUMBERS 0

// if 1, overrides REPORT_PTX in the event of a JIT compilation error
#define REPORT_PTX_ON_ERROR 1

// if 1, turns on error reporting for PTX JIT error even when REPORT_BASE is 0
#define OVERRIDE_REPORT_BASE_ON_PTX_ERROR 1

////////////////////////////////////////////////////////////////////////////////

typedef cuda::CudaDriver driver;

namespace executive 
{
	NVIDIAGPUDevice::MemoryAllocation::MemoryAllocation() : _flags(0), _size(0),
		_devicePointer(0), _hostPointer(0), _external(false)
	{
		
	}

	NVIDIAGPUDevice::MemoryAllocation::MemoryAllocation(size_t size) : 
		_flags(0), _size(size), _hostPointer(0), _external(false)
	{
		checkError(driver::cuMemAlloc(&_devicePointer, size));
		report("MemoryAllocation::MemoryAllocation() - allocated " << _size 
			<< " bytes of host-allocated memory");
		report("  device pointer: "	<< (const void *)_devicePointer);

	}
	
	NVIDIAGPUDevice::MemoryAllocation::MemoryAllocation(size_t size, 
		unsigned int flags) : Device::MemoryAllocation(false, true), 
		_flags(flags), _size(size), _external(false)
	{
		checkError(driver::cuMemHostAlloc(&_hostPointer, size, _flags));
		if(CUDA_SUCCESS != driver::cuMemHostGetDevicePointer(&_devicePointer, 
			_hostPointer, 0)) 
		{
			_devicePointer = 0;
		}
		report("MemoryAllocation::MemoryAllocation() - allocated " << _size 
			<< " bytes of host-allocated memory");
		report("  host: " << (const void *)_hostPointer << ", device pointer: "
			<< (const void *)_devicePointer);

	}
	
	NVIDIAGPUDevice::MemoryAllocation::MemoryAllocation(CUmodule module, 
		const ir::Global& g) : Device::MemoryAllocation(true, false), _flags(0),
		_size(g.statement.bytes()), _hostPointer(0), _external(false)
	{
		size_t bytes;
		checkError(driver::cuModuleGetGlobal(&_devicePointer, &bytes, module, 
			g.statement.name.c_str()));
		if(bytes != _size)
		{
			Throw("Global variable - " << g.statement.name 
				<< " - declared with " << _size << " bytes in Ocelot, but " 
				<< bytes << " in the CUDA driver.");
		}
	}
	
	NVIDIAGPUDevice::MemoryAllocation::MemoryAllocation(
		void* pointer, size_t size) : Device::MemoryAllocation(false, false), 
		_flags(0), _size(size), 
		_devicePointer(hydrazine::bit_cast<CUdeviceptr>(pointer)), 
		_hostPointer(0), _external(true)
	{
	
	}

	NVIDIAGPUDevice::MemoryAllocation::MemoryAllocation(
		void* pointer, size_t size, unsigned int flags) :
		Device::MemoryAllocation(false, true), 
		_flags(flags), _size(size), 
		_devicePointer(0), 
		_hostPointer(pointer), _external(true)
	{
		checkError(driver::cuMemHostRegister(_hostPointer, size, _flags));
		if(CUDA_SUCCESS != driver::cuMemHostGetDevicePointer(&_devicePointer, 
			_hostPointer, 0)) 
		{
			_devicePointer = 0;
		}
		report("MemoryAllocation::MemoryAllocation() - registered " << _size 
			<< " bytes of host-allocated memory");
		report("  host: " << (const void *)_hostPointer << ", device pointer: "
			<< (const void *)_devicePointer);
	}
	
	NVIDIAGPUDevice::MemoryAllocation::~MemoryAllocation()
	{
		report("MemoryAllocation::~MemoryAllocation() : _external = "
			<< _external << ", host() = " << host());
		if(!_external)
		{
			if(host())
			{
				checkError(driver::cuMemFreeHost(_hostPointer));
			}
			else if(!global())
			{
				checkError(driver::cuMemFree(_devicePointer));
			}
		}
	}

	NVIDIAGPUDevice::MemoryAllocation::MemoryAllocation(
		const MemoryAllocation& a) : Device::MemoryAllocation(
		a.global(), a.host()), _flags(a.flags()), _size(a.size()), 
		_devicePointer(0), _hostPointer(0), _external(a._external)
	{
		if(host())
		{
			report("MemoryAllocation::MemoryAllocation() - allocated "
				<< _size << " bytes of host-allocated memory");
			checkError(driver::cuMemHostAlloc(&_hostPointer, _size, _flags));
			checkError(driver::cuMemHostGetDevicePointer(&_devicePointer,
				_hostPointer, 0));
			memcpy(_hostPointer, a._hostPointer, _size);
		}
		else if(global() || _external)
		{
			_devicePointer = a._devicePointer;
		}
		else if(!_external)
		{
			checkError(driver::cuMemAlloc(&_devicePointer, _size));
			checkError(driver::cuMemcpyDtoD(_devicePointer, 
				a._devicePointer, _size));
		}
	}
	
	NVIDIAGPUDevice::MemoryAllocation::MemoryAllocation(MemoryAllocation&& a) :
		_flags(0), _size(0), _devicePointer(0), _hostPointer(0)
	{
		std::swap(_host, a._host);
		std::swap(_global, a._global);
		std::swap(_flags, a._flags);
		std::swap(_size, a._size);
		std::swap(_devicePointer, a._devicePointer);
		std::swap(_hostPointer, a._hostPointer);
		std::swap(_external, a._external);
	}

	NVIDIAGPUDevice::MemoryAllocation& 
		NVIDIAGPUDevice::MemoryAllocation::operator=(const MemoryAllocation& a)
	{
		if(this == &a) return *this;

		if(host())
		{
			checkError(driver::cuMemFreeHost(_hostPointer));
		}
		else if(!global() && !_external)
		{
			checkError(driver::cuMemFree(_devicePointer));
		}
		
		_host = a._host;
		_global = a._global;
		_flags = a._flags;
		_size = a._size;
		_hostPointer = 0;
		_devicePointer = 0;
		_external = a._external;
		
		if(host())
		{
			checkError(driver::cuMemHostAlloc(&_hostPointer, _size, _flags));
			checkError(driver::cuMemHostGetDevicePointer(&_devicePointer, 
				_hostPointer, 0));
			memcpy(_hostPointer, a._hostPointer, _size);
		}
		else if(global() || _external)
		{
			_devicePointer = a._devicePointer;
		}
		else
		{
			checkError(driver::cuMemAlloc(&_devicePointer, _size));
			checkError(driver::cuMemcpyDtoD(_devicePointer, 
				a._devicePointer, _size));
		}
		
		return *this;
	}
	
	NVIDIAGPUDevice::MemoryAllocation& 
		NVIDIAGPUDevice::MemoryAllocation::operator=(MemoryAllocation&& a)
	{
		if(this == &a) return *this;

		std::swap(_host, a._host);
		std::swap(_global, a._global);
		std::swap(_flags, a._flags);
		std::swap(_size, a._size);
		std::swap(_devicePointer, a._devicePointer);
		std::swap(_hostPointer, a._hostPointer);
		std::swap(_external, a._external);

		return *this;		
	}

	unsigned int NVIDIAGPUDevice::MemoryAllocation::flags() const
	{
		return _flags;
	}

	void* NVIDIAGPUDevice::MemoryAllocation::mappedPointer() const
	{
		return _hostPointer;
	}

	void* NVIDIAGPUDevice::MemoryAllocation::pointer() const
	{
		return (void*)_devicePointer;
	}

	size_t NVIDIAGPUDevice::MemoryAllocation::size() const
	{
		return _size;
	}

	void NVIDIAGPUDevice::MemoryAllocation::copy(size_t offset, 
		const void* src, size_t s)
	{
		report("NVIDIAGPUDevice::..::copy() 1");
		assert(offset + s <= size());
		if(host())
		{
			memcpy((char*)_hostPointer + offset, src, s);
		}
		else
		{
			CUdeviceptr dst = _devicePointer + offset;
			checkError(driver::cuMemcpyHtoD(dst, src, s));
		}
	}
	
	void NVIDIAGPUDevice::MemoryAllocation::copy(void* dst, 
		size_t offset, size_t s) const
	{
		report("NVIDIAGPUDevice::..::copy() 2 - is host? " << host() << ", "
			<< s << " bytes");
		assert(offset + s <= size());
		if(host())
		{
			std::memcpy(dst, (char*)_hostPointer + offset, s);
		}
		else
		{
			CUdeviceptr src = _devicePointer + offset;
			checkError(driver::cuMemcpyDtoH(dst, src, s));
		}
	}
	
	void NVIDIAGPUDevice::MemoryAllocation::memset(size_t offset, 
		int value, size_t s)
	{
		assert(s + offset <= size());
		if(host())
		{
			std::memset((char*)_hostPointer + offset, value, s);
		}
		else
		{
			CUdeviceptr dst = _devicePointer + offset;
			checkError(driver::cuMemsetD8(dst, value, s));
		}
	}
	
	void NVIDIAGPUDevice::MemoryAllocation::copy(Device::MemoryAllocation* a, 
		size_t toOffset, size_t fromOffset, size_t s) const
	{
		report("NVIDIAGPUDevice::..::copy() 3");
		MemoryAllocation* allocation = static_cast<MemoryAllocation*>(a);
		assert(fromOffset + s <= size());
		assert(toOffset + s <= allocation->size());
		
		
		if(host())
		{
			if(allocation->host())
			{
				void* src = (char*)_hostPointer + fromOffset;
				void* dst = (char*)allocation->_hostPointer + toOffset;
				
				memcpy(dst, src, s);
			}
			else
			{
				void* src = (char*)_hostPointer + fromOffset;
				CUdeviceptr dst = allocation->_devicePointer + toOffset;

				checkError(driver::cuMemcpyHtoD(dst, src, s));
			}
		}
		else
		{
			if(allocation->host())
			{
				CUdeviceptr src = _devicePointer + fromOffset;
				void* dst = (char*)allocation->_hostPointer + toOffset;

				checkError(driver::cuMemcpyDtoH(dst, src, s));
			}
			else
			{
				CUdeviceptr src = _devicePointer + fromOffset;
				CUdeviceptr dst = allocation->_devicePointer + toOffset;

				checkError(driver::cuMemcpyDtoD(dst, src, s));
			}
		}
	}

	NVIDIAGPUDevice::Module::Module(NVIDIAGPUDevice * device, const ir::Module* m) : 
	_handle(0), _device(device), ir(m)
	{
	
	}
	
	NVIDIAGPUDevice::Module::Module(const Module& m) : _handle(0), _device(m._device), ir(m.ir)
	{
		
	}
	
	NVIDIAGPUDevice::Module::~Module()
	{
		for(KernelMap::iterator kernel = kernels.begin(); 
			kernel != kernels.end(); ++kernel)
		{
			delete kernel->second;
		}
		
		if(_handle != 0)
		{
			assert(driver::cuModuleUnload(_handle) == CUDA_SUCCESS );
		}
	}
	
	void NVIDIAGPUDevice::Module::load()
	{
		report("Loading module - " << ir->path() << " on NVIDIA GPU.");
		
		// deal with .ptr.shared kernel parameter attributes
		const ir::Module *module = ir;
		ir::Module *copyModule = 0;
		if (transforms::SharedPtrAttribute::testModule(*module)) {
			transforms::SharedPtrAttribute ptrAttributePass;
			copyModule = new ir::Module(*ir);
			transforms::PassManager manager(copyModule);
			manager.addPass(&ptrAttributePass);
			manager.runOnModule();
			manager.releasePasses();

			module = copyModule;
		}
		
		assert(!loaded());
		std::stringstream stream;
		
		module->writeIR(stream, ir::PTXEmitter::Target_NVIDIA_PTX30);

#if REPORT_PTX_WITH_LINENUMBERS == 1		
		reportE(REPORT_PTX, " Binary is:\n" 
			<< hydrazine::addLineNumbers(stream.str()));
#else
		reportE(REPORT_PTX, stream.str());
#endif
		
		CUjit_option options[] = {
			CU_JIT_TARGET,
			CU_JIT_ERROR_LOG_BUFFER, 
			CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, 
		};
		
		const uint32_t errorLogSize       = 2048;
		uint32_t       errorLogActualSize = errorLogSize - 1;

		uint8_t errorLogBuffer[errorLogSize];

		std::memset(errorLogBuffer, 0, errorLogSize);

		void* optionValues[3] = {
			(void*)CU_TARGET_COMPUTE_20,
			(void*)errorLogBuffer, 
			hydrazine::bit_cast<void*>(errorLogActualSize), 
		};
		
		if (_device->properties().major == 3) {
			optionValues[0] = (void *)CU_TARGET_COMPUTE_30;
		}
		
		std::string ptxModule = stream.str();

		CUresult result = driver::cuModuleLoadDataEx(&_handle, 
			stream.str().c_str(), 3, options, optionValues);
		
		if(result != CUDA_SUCCESS)
		{
#if OVERRIDE_REPORT_BASE_ON_PTX_ERROR
#undef REPORT_BASE
#define REPORT_BASE 1
#endif
#if REPORT_PTX_WITH_LINENUMBERS == 1
		reportE(REPORT_PTX_ON_ERROR, " Binary is:\n" 
			<< hydrazine::addLineNumbers(stream.str()));
#else
		reportE(REPORT_PTX_ON_ERROR, stream.str());
#endif
#if OVERRIDE_REPORT_BASE_ON_PTX_ERROR
#undef REPORT_BASE
#define REPORT_BASE 0
#endif

			Throw("cuModuleLoadDataEx() - returned " << result 
				<< ". Failed to JIT module - " << ir->path() 
				<< " using NVIDIA JIT with error:\n" << errorLogBuffer);
		}
		
		report(" Module loaded successfully.");
		
		for(ir::Module::TextureMap::const_iterator 
			texture = module->textures().begin(); 
			texture != module->textures().end(); ++texture)
		{
			unsigned int flags = texture->second.normalizedFloat 
				? 0 : CU_TRSF_READ_AS_INTEGER;
			CUtexref reference;
			checkError(driver::cuModuleGetTexRef(&reference, _handle, 
				texture->first.c_str()));
			checkError(driver::cuTexRefSetFlags(reference, flags));
		}
		
		if (copyModule) {
			delete copyModule;
		}
	}

	bool NVIDIAGPUDevice::Module::loaded() const
	{
		return _handle != 0;	
	}	
	
	void NVIDIAGPUDevice::Module::translate()
	{
		if(!loaded()) load();
		
		report("Creating NVIDIA kernels for module - " << ir->path());
		for(ir::Module::KernelMap::const_iterator 
			kernel = ir->kernels().begin(); 
			kernel != ir->kernels().end(); ++kernel)
		{
			if (!kernel->second->function()) {
				CUfunction function;
				report(" - " << kernel->first);
				checkError(driver::cuModuleGetFunction(&function, _handle, 
					kernel->first.c_str()));
				kernels.insert(std::make_pair(kernel->first, 
					new NVIDIAExecutableKernel(*kernel->second, function)));
			}
		}
	}
	
	NVIDIAGPUDevice::Module::AllocationVector 
		NVIDIAGPUDevice::Module::loadGlobals()
	{
		if(!loaded()) load();
		assert(globals.empty());
		
		AllocationVector allocations;
		report("Loading globals in module - " << ir->path());
		for(ir::Module::GlobalMap::const_iterator 
			global = ir->globals().begin(); 
			global != ir->globals().end(); ++global)
		{
			if(global->second.statement.directive 
				== ir::PTXStatement::Shared) continue;
			report(" " << global->first);
			MemoryAllocation* allocation = new MemoryAllocation(_handle, 
				global->second);
			report("  pointer - " << allocation->pointer());
			
			globals.insert(std::make_pair(global->first, 
				allocation->pointer()));
			allocations.push_back(allocation);
		}
		
		return allocations;
	}
	
	bool NVIDIAGPUDevice::Module::translated() const
	{
		return kernels.size() == ir->kernels().size();
	}
	
	NVIDIAExecutableKernel* NVIDIAGPUDevice::Module::getKernel(
		const std::string& name)
	{
		if(!translated()) translate();
		
		NVIDIAExecutableKernel* kernel = 0;
		KernelMap::iterator k = kernels.find(name);
		if(k != kernels.end())
		{
			kernel = k->second;
		}
		return kernel;
	}

	void* NVIDIAGPUDevice::Module::getTexture(const std::string& name)
	{
		if(!loaded()) load();
		
		CUtexref texture;
		checkError(driver::cuModuleGetTexRef(&texture, _handle, name.c_str()));
		return texture;
	}

	static unsigned int channels(const cudaChannelFormatDesc& desc)
	{
		unsigned int channels = 0;
		
		channels = (desc.x > 0) ? channels + 1 : channels;
		channels = (desc.y > 0) ? channels + 1 : channels;
		channels = (desc.z > 0) ? channels + 1 : channels;
		channels = (desc.w > 0) ? channels + 1 : channels;
	
		return channels;
	}
	
	static CUarray_format_enum format(const cudaChannelFormatDesc& desc)
	{
		switch(desc.f)
		{
			case cudaChannelFormatKindSigned:
			{
				if(desc.x == 8)
				{
					return CU_AD_FORMAT_SIGNED_INT8;
				}
				else if(desc.y == 16)
				{
					return CU_AD_FORMAT_SIGNED_INT16;
				}
				else
				{
					return CU_AD_FORMAT_SIGNED_INT32;
				}			
			}
			case cudaChannelFormatKindUnsigned:
			{
				if(desc.x == 8)
				{
					return CU_AD_FORMAT_UNSIGNED_INT8;
				}
				else if(desc.y == 16)
				{
					return CU_AD_FORMAT_UNSIGNED_INT16;
				}
				else
				{
					return CU_AD_FORMAT_UNSIGNED_INT32;
				}
			}
			case cudaChannelFormatKindFloat:
			{
				if(desc.x == 16)
				{
					return CU_AD_FORMAT_HALF;
				}
				else
				{
					return CU_AD_FORMAT_FLOAT;
				}
			}
			case cudaChannelFormatKindNone: break;
		}
		return CU_AD_FORMAT_UNSIGNED_INT8;
	}
	
	static CUaddress_mode_enum convert(cudaTextureAddressMode mode)
	{
		// Note that the cuda runtime does not expose CU_TR_ADDRESS_MODE_MIRROR 
		if(mode == cudaAddressModeWrap) return CU_TR_ADDRESS_MODE_WRAP;
		return CU_TR_ADDRESS_MODE_CLAMP;
	}

	NVIDIAGPUDevice::Array3D::Array3D(const cudaChannelFormatDesc& desc, 
		const ir::Dim3& size, CUdeviceptr d) : ptr(d), size(size)
	{
		CUDA_ARRAY3D_DESCRIPTOR descriptor;
		descriptor.Width = size.x;
		descriptor.Height = size.y;
		descriptor.Depth = size.z;
		descriptor.NumChannels = channels(desc);
		descriptor.Format = format(desc);
		descriptor.Flags = 0;
		
		bytesPerElement = ((desc.x + desc.y + desc.z + desc.w) / 8);
		
		checkError(driver::cuArray3DCreate(&array, &descriptor));
	}

	NVIDIAGPUDevice::Array3D::Array3D() : array(0)



	{
		
	}

	NVIDIAGPUDevice::Array3D::~Array3D()
	{
		checkError(driver::cuArrayDestroy(array));
	}

	void NVIDIAGPUDevice::Array3D::update()
	{
		report("Updating texture.");
	
		CUDA_MEMCPY3D memcpy;
		
		memcpy.srcLOD = 0;
		memcpy.dstLOD = 0;
		
		memcpy.WidthInBytes = bytesPerElement * size.x;
		memcpy.Height = size.y;
		memcpy.Depth = size.z;
		
		memcpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		memcpy.srcDevice = ptr;
		memcpy.srcPitch = 0;
		memcpy.srcHeight = 0;
		
		memcpy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		memcpy.dstArray = array;
		
		memcpy.srcXInBytes = 0;
		memcpy.srcY = 0;
		memcpy.srcZ = 0;
		
		memcpy.dstXInBytes = 0;
		memcpy.dstY = 0;
		memcpy.dstZ = 0;
		
		checkError(driver::cuMemcpy3D(&memcpy));
	}

	bool NVIDIAGPUDevice::_cudaDriverInitialized = false;

	CUresult NVIDIAGPUDevice::_lastError = CUDA_SUCCESS;

	DeviceVector NVIDIAGPUDevice::createDevices(unsigned int flags,
		int computeCapability)
	{
		report("NVIDIAGPUDevice::createDevices()");
		if(!_cudaDriverInitialized)
		{
			driver::cuInit(0);
			_cudaDriverInitialized = true;
			
			report("driver::cuInit(0) called");
		}

		DeviceVector devices;
		int count;
		checkError(driver::cuDeviceGetCount(&count));
		
		for(int i = 0; i != count; ++i)
		{
			int major = 0;
			int minor = 0;
			checkError(driver::cuDeviceComputeCapability(&major, &minor, i));
			
			if(major < computeCapability)
			{
				char name[256];
				checkError(driver::cuDeviceGetName(name, 255, i));

				std::cerr << "==Ocelot== WARNING - This version of Ocelot only "
					"supports compute capability " << computeCapability
					<< ".0 and higher, ignoring device: '" << name << "'\n";				
			}
			else
			{
				devices.push_back(new NVIDIAGPUDevice(i, flags));
			}
		}
		
		return devices;
	}

	unsigned int NVIDIAGPUDevice::deviceCount(int computeCapability)
	{
		if(!_cudaDriverInitialized)
		{
			driver::cuInit(0);
			_cudaDriverInitialized = true;
		}
		
		int count;

		checkError(driver::cuDeviceGetCount(&count));

		int ignored = 0;
		for(int i = 0; i < count; ++i)
		{
			int major = 0;
			int minor = 0;
			checkError(driver::cuDeviceComputeCapability(&major, &minor, i));
			
			if(major < computeCapability) ++ignored;
		}
		
		return count - ignored;
	}

	NVIDIAGPUDevice::NVIDIAGPUDevice(int id, unsigned int flags) : 
		Device(flags), _selected(false), _next(0), _selectedStream(0), 
		_opengl(false)
	{
		if(!_cudaDriverInitialized)
		{
			checkError(driver::cuInit(0));
			_cudaDriverInitialized = true;
		}
		
		_runtimeVersion = 3020;
		checkError(driver::cuDriverGetVersion(&_driverVersion));
		
		CUdevice device;
		checkError(driver::cuDeviceGet(&device, id));
		
		_opengl = hydrazine::isAnOpenGLContextAvailable();

		report(" creating context");
		if(_opengl)
		{
			report(" creating GL context - flags: " << flags << ", device: " << device);
			checkError(driver::cuGLCtxCreate(&_context, flags, device));
		}
		else
		{
			report(" creating context - flags: " << flags << ", device: " << device);
			checkError(driver::cuCtxCreate(&_context, flags, device));
		}
		
		report("NVIDIAGPUDevice::NVIDIAGPUDevice() - created context."
			"_opengl = " << _opengl);		
		checkError(driver::cuCtxPopCurrent(&_context));
				
		checkError(driver::cuDeviceGetName(_properties.name, 255, device));
		
		size_t total;
		checkError(driver::cuDeviceTotalMem(&total, device));
		_properties.totalMemory = total;
		_properties.ISA = ir::Instruction::SASS;
		
		checkError(driver::cuDeviceGetAttribute(
			(int*)&_properties.multiprocessorCount,
			CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.memcpyOverlap, 
			CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.maxThreadsPerBlock,
			CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
		
		checkError(driver::cuDeviceGetAttribute(&_properties.maxThreadsDim[0], 
			CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.maxThreadsDim[1], 
			CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.maxThreadsDim[2], 
			CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device));
		
		checkError(driver::cuDeviceGetAttribute(&_properties.maxGridSize[0], 
			CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.maxGridSize[1], 
			CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.maxGridSize[2], 
			CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device));

		checkError(driver::cuDeviceGetAttribute(&_properties.sharedMemPerBlock, 
			CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
		checkError(driver::cuDeviceGetAttribute(
			&_properties.totalConstantMemory, 
			CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.SIMDWidth, 
			CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.memPitch, 
			CU_DEVICE_ATTRIBUTE_MAX_PITCH, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.regsPerBlock, 
			CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.clockRate, 
			CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.textureAlign, 
			CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, device));
		
		checkError(driver::cuDeviceGetAttribute(&_properties.integrated, 
			CU_DEVICE_ATTRIBUTE_INTEGRATED, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.concurrentKernels, 
			CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device));
		checkError(driver::cuDeviceComputeCapability(&_properties.major, 
			&_properties.minor, device));
		
		int unifiedAddressing = false;
		checkError(driver::cuDeviceGetAttribute(&unifiedAddressing, 
			CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
			
		_properties.unifiedAddressing = unifiedAddressing;
		
		checkError(driver::cuDeviceGetAttribute(&_properties.memoryClockRate, 
			CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.memoryBusWidth, 
			CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));
		checkError(driver::cuDeviceGetAttribute(&_properties.l2CacheSize, 
			CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device));
		checkError(driver::cuDeviceGetAttribute(
			&_properties.maxThreadsPerMultiProcessor, 
			CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device));
	}

	NVIDIAGPUDevice::~NVIDIAGPUDevice()
	{
		assert(!selected());
		select();
		_modules.clear();
		checkError(driver::cuCtxDestroy(_context));
	}
	
	Device::MemoryAllocation* NVIDIAGPUDevice::getMemoryAllocation(
		const void* address, AllocationType type) const
	{
		MemoryAllocation* allocation = 0;
		
		if(type == DeviceAllocation || type == AnyAllocation)
		{
			if(!_allocations.empty())
			{
				AllocationMap::const_iterator alloc = _allocations.upper_bound(
					(void*)address);
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

		if(type == HostAllocation || type == AnyAllocation)
		{
			for(AllocationMap::const_iterator alloc = _hostAllocations.begin(); 
				alloc != _hostAllocations.end(); ++alloc)
			{
				if(alloc->second->host())
				{
					if((char*)address >= alloc->second->mappedPointer()
						&& (char*)address
						< (char*)alloc->second->mappedPointer()
						+ alloc->second->size())
					{
						allocation = alloc->second;
						break;
					}
				}
			}
		}
		
		return allocation;		
	}

	Device::MemoryAllocation* NVIDIAGPUDevice::getGlobalAllocation(
		const std::string& moduleName, const std::string& name)
	{
		if(moduleName.empty())
		{
			// try a brute force search over all modules
			for(ModuleMap::iterator module = _modules.begin(); 
				module != _modules.end(); ++module)
			{
				if(module->second.globals.empty())
				{
					Module::AllocationVector allocations = std::move(
						module->second.loadGlobals());
					for(Module::AllocationVector::iterator 
						allocation = allocations.begin(); 
						allocation != allocations.end(); ++allocation)
					{
						_allocations.insert(std::make_pair(
							(*allocation)->pointer(), *allocation));
					}
				}

				Module::GlobalMap::iterator global = 
					module->second.globals.find(name);
				if(global != module->second.globals.end())
				{
					return getMemoryAllocation(global->second, 
						DeviceAllocation);
				}
			}
			return 0;
		}

		ModuleMap::iterator module = _modules.find(moduleName);
		if(module == _modules.end()) return 0;
		
		if(module->second.globals.empty())
		{
			Module::AllocationVector allocations = std::move(
				module->second.loadGlobals());
			for(Module::AllocationVector::iterator 
				allocation = allocations.begin(); 
				allocation != allocations.end(); ++allocation)
			{
				_allocations.insert(std::make_pair((*allocation)->pointer(), 
					*allocation));
			}
		}
		
		Module::GlobalMap::iterator global = module->second.globals.find(name);
		if(global == module->second.globals.end()) return 0;
		
		return getMemoryAllocation(global->second, DeviceAllocation);
	}

	Device::MemoryAllocation* NVIDIAGPUDevice::allocate(size_t size)
	{
		MemoryAllocation* allocation = new MemoryAllocation(size);
		_allocations.insert(std::make_pair(allocation->pointer(), allocation));
		return allocation;
	}

	Device::MemoryAllocation* NVIDIAGPUDevice::allocateHost(
		size_t size, unsigned int flags)
	{
		MemoryAllocation* allocation = new MemoryAllocation(size, flags);
		_hostAllocations.insert(std::make_pair(allocation->mappedPointer(),
			allocation));

		report("NVIDIAGPUDevice::allocateHost() - adding key "
			<< allocation->mappedPointer());

		return allocation;
	}

	Device::MemoryAllocation* NVIDIAGPUDevice::registerHost(void* pointer,
		size_t size, unsigned int flags)
	{
		MemoryAllocation* allocation =
			new MemoryAllocation(pointer, size, flags);
		_hostAllocations.insert(std::make_pair(allocation->pointer(),
			allocation));

		report("NVIDIAGPUDevice::registerHost() - adding key "
			<< allocation->pointer());

		return allocation;
	}

	void NVIDIAGPUDevice::free(void* pointer)
	{
		if(pointer == 0) return;
		
		AllocationMap::iterator allocation = _allocations.find(pointer);
		if(allocation != _allocations.end())
		{
			if(allocation->second->global())
			{
				report("cannot free global pointer");
				Throw("Cannot free global pointer - " << pointer);
			}
			delete allocation->second;
			_allocations.erase(allocation);
		}
		else
		{
			allocation = _hostAllocations.find(pointer);
			if(allocation != _hostAllocations.end())
			{
				delete allocation->second;
				_hostAllocations.erase(allocation);
			}
			else
			{
				Throw("Tried to free invalid pointer - " << pointer);
			}
		}
	}
	
	Device::MemoryAllocationVector NVIDIAGPUDevice::getNearbyAllocations(
		void* pointer) const
	{
		MemoryAllocationVector allocations;
		for(AllocationMap::const_iterator allocation = _allocations.begin(); 
			allocation != _allocations.end(); ++allocation)
		{
			allocations.push_back(allocation->second);
		}
		return std::move(allocations);
	}

	Device::MemoryAllocationVector NVIDIAGPUDevice::getAllAllocations() const
	{
		MemoryAllocationVector allocations;
		for(AllocationMap::const_iterator allocation = _allocations.begin(); 
			allocation != _allocations.end(); ++allocation)
		{
			allocations.push_back(allocation->second);
		}
		
		for(AllocationMap::const_iterator allocation = _hostAllocations.begin();
			allocation != _hostAllocations.end(); ++allocation)
		{
			allocations.push_back(allocation->second);
		}

		return allocations;
	}

	void NVIDIAGPUDevice::clearMemory()
	{
		for(AllocationMap::iterator allocation = _allocations.begin(); 
			allocation != _allocations.end();)
		{
			if(allocation->second->global())
			{
				++allocation;		
			}
			else
			{
				delete allocation->second;
				_allocations.erase(allocation++);
			}
		}
		
		for(AllocationMap::iterator allocation = _hostAllocations.begin();
			allocation != _hostAllocations.end(); ++allocation)
		{
			delete allocation->second;
		}

		_hostAllocations.clear();
	}

	void* NVIDIAGPUDevice::glRegisterBuffer(unsigned int buffer, 
		unsigned int flags)
	{
		report("Regstering open gl buffer " << buffer);

		if(!_opengl) Throw("No active opengl contexts.");

		CUgraphicsResource resource;
		checkError(driver::cuGraphicsGLRegisterBuffer(&resource,
			buffer, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE));

		report(" to resource - " << resource);
		return hydrazine::bit_cast<void*>(resource);
	}
	
	void* NVIDIAGPUDevice::glRegisterImage(unsigned int image, 
		unsigned int target, unsigned int flags)
	{
		report("Regstering open gl image " << image << ", target " << target);

		if(!_opengl) Throw("No active opengl contexts.");

		CUgraphicsResource resource;
		checkError(driver::cuGraphicsGLRegisterImage(&resource, image, 
			target, flags));
		report(" to resource " << resource);
		return resource;
	}
	
	void NVIDIAGPUDevice::unRegisterGraphicsResource(void* resource)
	{
		report("Unregistering graphics resource - " << resource);

		if(!_opengl) Throw("No active opengl contexts.");

		checkError(driver::cuGraphicsUnregisterResource(
			hydrazine::bit_cast<CUgraphicsResource>(resource)));
	}

	void NVIDIAGPUDevice::mapGraphicsResource(void** resourceVoidPtr, int count, 
		unsigned int streamId)
	{
		CUstream id = 0;
		if(streamId != 0)
		{
			StreamMap::iterator stream = _streams.find(streamId);

			if(stream == _streams.end())
			{
				Throw("Invalid stream - " << streamId);
			}

			id = stream->second;
		}
		CUgraphicsResource * graphicsResources =
			(CUgraphicsResource *)resourceVoidPtr;

		if(!_opengl) Throw("No active opengl contexts.");

		report("NVIDIAGPUDevice::mapGraphicsResource() - count = " << count );
		CUresult result = driver::cuGraphicsMapResources(count,
			graphicsResources, id);
		report("driver::cuGraphicsMapresources() - " << result << ", " 
			<< cuda::CudaDriver::toString(result));
		
		checkError(result);
	}
	
	void* NVIDIAGPUDevice::getPointerToMappedGraphicsResource(size_t& size, 
		void* resource)
	{
		CUdeviceptr pointer;
		size_t bytes = 0;
		report("Getting pointer to mapped resource " << resource);

		if(!_opengl) Throw("No active opengl contexts.");

		CUresult result = driver::cuGraphicsResourceGetMappedPointer(
			&pointer, &bytes, (CUgraphicsResource)resource);
		report("  cuGraphicsResourceGetMappedPointer() returned " << result)
		checkError(result);
			
		void* p = (void*)pointer;
		if (_allocations.find(p) == _allocations.end()) {
			_allocations.insert(std::make_pair(p, 
				new MemoryAllocation(p, bytes)));
		}

		size = bytes;
		report(" size - " << size << ", pointer - " << pointer);
		return hydrazine::bit_cast<void*>(pointer);
	}

	void NVIDIAGPUDevice::setGraphicsResourceFlags(void* resource, 
		unsigned int flags)
	{
		report("Setting flags to " << flags 
			<< " for graphics resource " << resource);

		if(!_opengl) Throw("No active opengl contexts.");

		checkError(driver::cuGraphicsResourceSetMapFlags(
			(CUgraphicsResource)resource, flags));
	}

	void NVIDIAGPUDevice::unmapGraphicsResource(void** resourceVoidPtr, 
		int count, unsigned int streamID)
	{
		CUstream id = 0;
		
		if(_selectedStream != 0)
		{
			StreamMap::iterator stream = _streams.find(_selectedStream);
			assert(stream != _streams.end());
			id = stream->second;
		}
		
		if(!_opengl) Throw("No active opengl contexts.");

		CUdeviceptr pointer;
		size_t bytes = 0;

		CUgraphicsResource * graphicsResources =
			(CUgraphicsResource *)resourceVoidPtr;
		
		checkError(driver::cuGraphicsResourceGetMappedPointer(&pointer,
			&bytes, graphicsResources[0]));

		AllocationMap::iterator allocation = _allocations.find(
			hydrazine::bit_cast<void*>(pointer));
		assert(allocation != _allocations.end());
		
		delete allocation->second;
		_allocations.erase(allocation);

		checkError(driver::cuGraphicsUnmapResources(1, graphicsResources, id));
	}

	void NVIDIAGPUDevice::load(const ir::Module* module)
	{
		assert(selected());
	
		if(_modules.count(module->path()) != 0)
		{
			Throw("Duplicate module - " << module->path());
		}
		_modules.insert(std::make_pair(module->path(), 
			Module(this, module)));
	}
	
	void NVIDIAGPUDevice::unload(const std::string& name)
	{
		assert(selected());
	
		ModuleMap::iterator module = _modules.find(name);
		if(module == _modules.end())
		{
			Throw("Cannot unload unknown module - " << name);
		}
		
		for(Module::GlobalMap::iterator global = module->second.globals.begin();
			global != module->second.globals.end(); ++global)
		{
			AllocationMap::iterator allocation = 
				_allocations.find(global->second);
			assert(allocation != _allocations.end());
			delete allocation->second;
			_allocations.erase(allocation);
		}
		
		_modules.erase(module);
	}

	ExecutableKernel* NVIDIAGPUDevice::getKernel(const std::string& moduleName, 
		const std::string& kernelName)
	{
		ModuleMap::iterator module = _modules.find(moduleName);
		
		if(module == _modules.end()) return 0;
		
		return module->second.getKernel(kernelName);	
	}
	
	unsigned int NVIDIAGPUDevice::createEvent(int flags)
	{
		CUevent event;
		checkError(driver::cuEventCreate(&event, flags));
		
		unsigned int handle = _next++;
		_events.insert(std::make_pair(handle, event));
		
		return handle;
	}

	void NVIDIAGPUDevice::destroyEvent(unsigned int handle)
	{
		EventMap::iterator event = _events.find(handle);
		if(event == _events.end())
		{
			Throw("Invalid event - " << handle);
		}
		

		checkError(driver::cuEventDestroy(event->second));
		_events.erase(event);
	}

	bool NVIDIAGPUDevice::queryEvent(unsigned int handle)
	{
		EventMap::const_iterator event = _events.find(handle);
		if(event == _events.end())
		{
			Throw("Invalid event - " << handle);
		}

		CUresult result = driver::cuEventQuery(event->second);
		if(result == CUDA_SUCCESS) return true;
		return false;
	}
	
	void NVIDIAGPUDevice::recordEvent(unsigned int handle, unsigned int sHandle)
	{
		EventMap::const_iterator event = _events.find(handle);
		if(event == _events.end())
		{
			Throw("Invalid event - " << handle);
		}

		if(sHandle != 0)
		{
			StreamMap::const_iterator stream = _streams.find(sHandle);
			if(stream == _streams.end())
			{
				Throw("Invalid stream - " << sHandle);
			}
			checkError(driver::cuEventRecord(event->second, stream->second));
		}
		else
		{
			checkError(driver::cuEventRecord(event->second, 0));
		}
	}

	void NVIDIAGPUDevice::synchronizeEvent(unsigned int handle)
	{
		EventMap::const_iterator event = _events.find(handle);
		if(event == _events.end())
		{
			Throw("Invalid event - " << handle);
		}

		checkError(driver::cuEventSynchronize(event->second));		
	}

	float NVIDIAGPUDevice::getEventTime(unsigned int startHandle, 
		unsigned int endHandle)
	{
		EventMap::const_iterator start = _events.find(startHandle);
		if(start == _events.end())
		{
			report("invalid start event");
			Throw("Invalid start event - " << startHandle);
		}
		
		EventMap::const_iterator end = _events.find(endHandle);
		if(end == _events.end())
		{
			report("invalid end event");
			Throw("Invalid end event - " << endHandle);
		}
		
		float time = 0.0f;
		checkError(driver::cuEventElapsedTime(&time,
			start->second, end->second));
		
		return time;
	}
	
	unsigned int NVIDIAGPUDevice::createStream()
	{
		CUstream stream;
		checkError(driver::cuStreamCreate(&stream, 0));
		
		unsigned int handle = _next++;
		_streams.insert(std::make_pair(handle, stream));
		
		return handle;
	}
	
	void NVIDIAGPUDevice::destroyStream(unsigned int handle)
	{
		StreamMap::iterator stream = _streams.find(handle);
		if(stream == _streams.end())
		{
			Throw("Invalid stream - " << handle);
		}
		
		checkError(driver::cuStreamDestroy(stream->second));
		_streams.erase(stream);
	}
	
	bool NVIDIAGPUDevice::queryStream(unsigned int handle)
	{
		StreamMap::const_iterator stream = _streams.find(handle);
		if(stream == _streams.end())
		{
			Throw("Invalid stream - " << handle);
		}
	
		CUresult result = driver::cuStreamQuery(stream->second);
		if(result == CUDA_SUCCESS) return true;
		return false;
	}

	void NVIDIAGPUDevice::synchronizeStream(unsigned int handle)
	{
		StreamMap::iterator stream = _streams.find(handle);
		if(stream == _streams.end())
		{
			Throw("Invalid stream - " << handle);
		}
		
		checkError(driver::cuStreamSynchronize(stream->second));
	}

	void NVIDIAGPUDevice::setStream(unsigned int stream)
	{
		_selectedStream = stream;
	}
			
	void NVIDIAGPUDevice::select()
	{
		Device::select();
		
		report("NVIDIAGPUDevice::select()");
		checkError(driver::cuCtxPushCurrent(_context));
	}
	
	void NVIDIAGPUDevice::unselect()
	{
		Device::unselect();
		
		checkError(driver::cuCtxPopCurrent(&_context));
		report("NVIDIAGPUDevice::unselect()");
	}
		
	void NVIDIAGPUDevice::bindTexture(void* pointer, 
		const std::string& moduleName, const std::string& textureName, 
		const textureReference& texref, const cudaChannelFormatDesc& desc, 
		const ir::Dim3& size)
	{
		ModuleMap::iterator module = _modules.find(moduleName);
		if(module == _modules.end())
		{
			Throw("Invalid Module - " << moduleName);
		}
		
		void* tex = module->second.getTexture(textureName);
		if(tex == 0)
		{
			Throw("Invalid Texture - " << textureName 
				<< " in Module - " << moduleName);
		}

		if(_arrays.count(textureName) != 0)
		{
			unbindTexture(moduleName, textureName);
		}

		report("Binding texture " << textureName << " in module " << moduleName
			<< " to pointer " << pointer << " with dimensions (" << size.x
			<< "," << size.y << "," << size.z << ")");

		CUtexref ref = hydrazine::bit_cast<CUtexref>(tex);
		CUdeviceptr ptr = hydrazine::bit_cast<CUdeviceptr>(pointer);
		size_t offset = 0;
		unsigned int bytesPerElement = ((desc.x + desc.y 
			+ desc.z + desc.w) / 8);
		unsigned int pitch = bytesPerElement * size.x;
		
		if(size.z > 1)
		{
			Array3D* array = new Array3D(desc, size, ptr);
			_arrays[textureName] = array;
			
			checkError(driver::cuTexRefSetArray(ref,
				array->array, CU_TRSA_OVERRIDE_FORMAT));
		}
		else if(size.y > 1)
		{
			CUDA_ARRAY_DESCRIPTOR descriptor;
			descriptor.Width = size.x;
			descriptor.Height = size.y;
			descriptor.NumChannels = channels(desc);
			descriptor.Format = format(desc);
			
			checkError(driver::cuTexRefSetAddress2D(ref, &descriptor, 
				ptr, pitch));

			_arrays[textureName] = 0;
		}
		else
		{
			checkError(driver::cuTexRefSetAddress(&offset, ref, ptr, pitch));
			_arrays[textureName] = 0;
		}
		
		if(texref.filterMode == cudaFilterModeLinear)
		{
			checkError(driver::cuTexRefSetFilterMode(ref, 
				CU_TR_FILTER_MODE_LINEAR));
		}
		else
		{
			checkError(driver::cuTexRefSetFilterMode(ref, 
				CU_TR_FILTER_MODE_POINT));
		}

		checkError(driver::cuTexRefSetAddressMode(ref, 0,
			convert(texref.addressMode[0])));
		checkError(driver::cuTexRefSetAddressMode(ref, 1,
			convert(texref.addressMode[1])));
		checkError(driver::cuTexRefSetAddressMode(ref, 2,
			convert(texref.addressMode[2])));
		
		unsigned int flags = 0;
		checkError(driver::cuTexRefGetFlags(&flags, ref));
		flags &= CU_TRSF_READ_AS_INTEGER;
		flags |= (texref.normalized) ? CU_TRSF_NORMALIZED_COORDINATES : 0;		
		checkError(driver::cuTexRefSetFlags(ref, flags));
	}
	
	void NVIDIAGPUDevice::unbindTexture(const std::string& moduleName, 
		const std::string& textureName)
	{
		// this is a nop, textures cannot be unbound
		ModuleMap::iterator module = _modules.find(moduleName);
		if(module == _modules.end())
		{


			Throw("Invalid Module - " << moduleName);
		}
		
		void* tex = module->second.getTexture(textureName);
		if(tex == 0)
		{
			Throw("Invalid Texture - " << textureName 
				<< " in Module - " << moduleName);
		}
		
		report("Unbinding texture " << textureName 
			<< " in module " << moduleName);
		
		ArrayMap::iterator array = _arrays.find(textureName);
		if (array != _arrays.end()) {
			assert(array != _arrays.end());
			delete array->second;
			_arrays.erase(array);
		}
	}
	
	void* NVIDIAGPUDevice::getTextureReference(const std::string& moduleName, 
		const std::string& textureName)
	{
		ModuleMap::iterator module = _modules.find(moduleName);
		
		if(module == _modules.end()) return 0;
		return module->second.getTexture(textureName);
	}

	void NVIDIAGPUDevice::launch(const std::string& moduleName, 
		const std::string& kernelName, const ir::Dim3& grid, 
		const ir::Dim3& block, size_t sharedMemory,
		const void* argumentBlock, size_t argumentBlockSize,
		const trace::TraceGeneratorVector& traceGenerators,
		const ir::ExternalFunctionSet* externals)
	{
		ModuleMap::iterator module = _modules.find(moduleName);
		
		if(module == _modules.end())
		{
			Throw("Unknown module - " << moduleName);
		}
		
		NVIDIAExecutableKernel* kernel = module->second.getKernel(kernelName);
		
		if(kernel == 0)
		{
			Throw("Unknown kernel - " << kernelName 
				<< " in module " << moduleName);
		}
		
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
		
		if(kernel->constMemorySize() > (size_t)properties().totalConstantMemory)
		{
			Throw("Out of constant memory for kernel \""
				<< kernel->name << "\" : \n\tpreallocated "
				<< kernel->constMemorySize() << " is greater than available " 
				<< properties().totalConstantMemory << " for device " 

				<< properties().name);
		}
		
		kernel->device = this;
		kernel->setKernelShape(block.x, block.y, block.z);
		kernel->setArgumentBlock((const unsigned char*)argumentBlock, 
			argumentBlockSize);
		kernel->updateArgumentMemory();
		kernel->updateMemory();
		kernel->setExternSharedMemorySize(sharedMemory);
		kernel->setTraceGenerators(traceGenerators);
		
		for(ArrayMap::iterator array = _arrays.begin(); 
			array != _arrays.end(); ++array)
		{
			if(array->second != 0) array->second->update();
		}
		
		for(auto generator = traceGenerators.begin();
			generator != traceGenerators.end(); ++generator)
		{
			(*generator)->initialize(*kernel);
		}
		
		kernel->launchGrid(grid.x, grid.y, grid.z);
		synchronize();

		for(auto generator = traceGenerators.begin();
			generator != traceGenerators.end(); ++generator)
		{
			(*generator)->finish();
		}
		
	}

	cudaFuncAttributes NVIDIAGPUDevice::getAttributes(const std::string& path, 
		const std::string& kernelName)
	{
		ModuleMap::iterator module = _modules.find(path);
		
		if(module == _modules.end())
		{
			Throw("Unknown module - " << path);
		}
		
		NVIDIAExecutableKernel* kernel = module->second.getKernel(kernelName);
		
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
		attributes.ptxVersion = 21;
		attributes.binaryVersion = 21;
		
		return std::move(attributes);
	}

	static unsigned int translateError(CUresult error)
	{
		switch(error)
		{	
			case CUDA_SUCCESS: return cudaSuccess;
			case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
			case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
			case CUDA_ERROR_NOT_INITIALIZED: 
				return cudaErrorInitializationError;
			case CUDA_ERROR_DEINITIALIZED: return cudaErrorInitializationError;
			case CUDA_ERROR_NO_DEVICE: return cudaErrorNoDevice;
			case CUDA_ERROR_INVALID_DEVICE: return cudaErrorInvalidDevice;
			case CUDA_ERROR_INVALID_IMAGE: 
				return cudaErrorInvalidDeviceFunction;
			case CUDA_ERROR_INVALID_CONTEXT: return cudaErrorApiFailureBase;
			case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: 
				return cudaErrorApiFailureBase;
			case CUDA_ERROR_MAP_FAILED: return cudaErrorMapBufferObjectFailed;
			case CUDA_ERROR_UNMAP_FAILED: 
				return cudaErrorUnmapBufferObjectFailed;
			case CUDA_ERROR_ARRAY_IS_MAPPED: 
				return cudaErrorMapBufferObjectFailed;
			case CUDA_ERROR_ALREADY_MAPPED: 
				return cudaErrorMapBufferObjectFailed;
			case CUDA_ERROR_NO_BINARY_FOR_GPU: 
				return cudaErrorInvalidDeviceFunction;
			case CUDA_ERROR_ALREADY_ACQUIRED: 
				return cudaErrorSetOnActiveProcess;
			case CUDA_ERROR_NOT_MAPPED: return cudaErrorUnmapBufferObjectFailed;
			case CUDA_ERROR_INVALID_SOURCE: return cudaErrorInvalidValue;
			case CUDA_ERROR_FILE_NOT_FOUND: return cudaErrorInvalidValue;
			case CUDA_ERROR_INVALID_HANDLE: return cudaErrorInvalidValue;
			case CUDA_ERROR_NOT_FOUND: return cudaErrorInvalidValue;
			case CUDA_ERROR_NOT_READY: return cudaErrorNotReady;
			case CUDA_ERROR_LAUNCH_FAILED: return cudaErrorLaunchFailure;
			case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: 
				return cudaErrorLaunchOutOfResources;
			case CUDA_ERROR_LAUNCH_TIMEOUT: return cudaErrorLaunchTimeout;
			case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: 
				return cudaErrorInvalidTexture;
			case CUDA_ERROR_UNKNOWN: return cudaErrorUnknown;
			default: break;
		}
		return cudaErrorUnknown;
	}

	unsigned int NVIDIAGPUDevice::getLastError()
	{
		return translateError(_lastError);
	}
	
	void NVIDIAGPUDevice::synchronize()
	{
		checkError(driver::cuCtxSynchronize());
	}
			
	void NVIDIAGPUDevice::limitWorkerThreads(unsigned int threads)
	{
		// this is a nop here
	}

	void NVIDIAGPUDevice::setOptimizationLevel(
		translator::Translator::OptimizationLevel level)
	{
		// TODO work in something with the PTX JIT optimization level here
	}
}

////////////////////////////////////////////////////////////////////////////////

#endif

