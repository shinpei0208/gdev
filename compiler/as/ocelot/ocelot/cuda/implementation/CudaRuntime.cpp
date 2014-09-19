/*! \file CudaRuntime.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief implements the CUDA Runtime API for Ocelot
*/

// C standard library includes
#include <assert.h>

// C++ standard library includes
#include <sstream>
#include <algorithm>

// Ocelot includes
#include <ocelot/cuda/interface/CudaRuntime.h>
#include <ocelot/cuda/interface/CudaDriver.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/executive/interface/RuntimeException.h>
#include <ocelot/executive/interface/ExecutableKernel.h>
#include <ocelot/transforms/interface/PassManager.h>

// Hydrazine includes
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/string.h>
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

////////////////////////////////////////////////////////////////////////////////

// whether CUDA runtime catches exceptions thrown by Ocelot
#define CATCH_RUNTIME_EXCEPTIONS 0

// whether verbose error messages are printed
#define CUDA_VERBOSE 1

// whether debugging messages are printed
#define REPORT_BASE 0

// report all ptx modules
#define REPORT_ALL_PTX 0

////////////////////////////////////////////////////////////////////////////////
//
// Error handling macros

#define Ocelot_Exception(x) { std::stringstream ss; ss << x; \
	throw hydrazine::Exception(ss.str()); }

////////////////////////////////////////////////////////////////////////////////

typedef api::OcelotConfiguration config;

////////////////////////////////////////////////////////////////////////////////

cuda::HostThreadContext::HostThreadContext(): selectedDevice(0),
	lastError(cudaSuccess), parameterBlock(0), parameterBlockSize(1<<13) {
	parameterBlock = (unsigned char *)malloc(parameterBlockSize);
}

cuda::HostThreadContext::~HostThreadContext() {
	::free(parameterBlock);
}

cuda::HostThreadContext::HostThreadContext(const HostThreadContext& c): 
	selectedDevice(c.selectedDevice),
	validDevices(c.validDevices),
	launchConfigurations(c.launchConfigurations),
	lastError(c.lastError),
	parameterBlock((unsigned char *)malloc(c.parameterBlockSize)),
	parameterBlockSize(c.parameterBlockSize),
	parameterIndices(c.parameterIndices),
	parameterSizes(c.parameterSizes)
{
	memcpy(parameterBlock, c.parameterBlock, parameterBlockSize);
}

cuda::HostThreadContext& cuda::HostThreadContext::operator=(
	const HostThreadContext& c) {
	if(&c == this) return *this;
	selectedDevice = c.selectedDevice;
	validDevices = c.validDevices;
	lastError = c.lastError;
	launchConfigurations = c.launchConfigurations;
	parameterIndices = c.parameterIndices;
	parameterSizes = c.parameterSizes;
	memcpy(parameterBlock, c.parameterBlock, parameterBlockSize);
	return *this;
}

cuda::HostThreadContext::HostThreadContext(HostThreadContext&& c): 
	selectedDevice(0), parameterBlock(0), parameterBlockSize(1<<13) {
	*this = std::move(c);
}

cuda::HostThreadContext& cuda::HostThreadContext::operator=(
	HostThreadContext&& c) {
	if (this == &c) return *this;
	std::swap(selectedDevice, c.selectedDevice);
	std::swap(validDevices, c.validDevices);
	std::swap(lastError, c.lastError);
	std::swap(parameterBlock, c.parameterBlock);
	std::swap(launchConfigurations, c.launchConfigurations);
	std::swap(parameterIndices, c.parameterIndices);
	std::swap(parameterSizes, c.parameterSizes);
	return *this;
}

void cuda::HostThreadContext::clearParameters() {
	parameterIndices.clear();
	parameterSizes.clear();
}

void cuda::HostThreadContext::clear() {
	validDevices.clear();
	launchConfigurations.clear();
	clearParameters();
}

unsigned int cuda::HostThreadContext::mapParameters(const ir::Kernel* kernel) {
	unsigned int dst = 0;

	if (kernel->arguments.size() == parameterIndices.size()) {
		IndexVector::iterator offset = parameterIndices.begin();
		SizeVector::iterator size = parameterSizes.begin();
		unsigned char* temp = (unsigned char*)malloc(parameterBlockSize);
		for (ir::Kernel::ParameterVector::const_iterator 
			parameter = kernel->arguments.begin(); 
			parameter != kernel->arguments.end();
			++parameter, ++offset, ++size) {
			unsigned int misalignment = dst % parameter->getAlignment();
			unsigned int alignmentOffset = misalignment == 0 
				? 0 : parameter->getAlignment() - misalignment;
			dst += alignmentOffset;
		
			memset(temp + dst, 0, parameter->getSize());
			memcpy(temp + dst, parameterBlock + *offset, *size);
			report( "Mapping parameter at offset " << *offset << " of size " 
				<< *size << " to offset " << dst << " of size " 
				<< parameter->getSize() << "\n   data = " 
				<< hydrazine::dataToString(temp + dst, parameter->getSize()));
			dst += parameter->getSize();
		}
		free(parameterBlock);
		parameterBlock = temp;
		clearParameters();
	}
	else if (parameterIndices.size() == 1
		&& parameterIndices[0] == 0 && parameterSizes[0]) {
		
		parameterBlockSize = parameterSizes[0];
		
		unsigned char *temp = (unsigned char *)malloc(parameterBlockSize);
		memcpy(temp, parameterBlock, parameterBlockSize);
		free(parameterBlock);
		parameterBlock = temp;
		
		report("parameter block formatted by client: offset "
			<< parameterIndices[0] << ", " 
			<< parameterSizes[0] << " bytes");
		clearParameters();

		dst = parameterBlockSize;
	}
	else {
		report("Parameter ERROR: offset " << parameterIndices[0] << ", "
			<< parameterSizes[0] << " bytes. Expected parameter sizes of "
			<< parameterBlockSize);
		assert((kernel->arguments.size() == parameterIndices.size()) && 
			"unaccepted argument formatting");
	}

	return dst;
}

////////////////////////////////////////////////////////////////////////////////
cuda::RegisteredKernel::RegisteredKernel(size_t h, const std::string& m, 
	const std::string& k) : handle(h), module(m), kernel(k) {
}

cuda::RegisteredTexture::RegisteredTexture(const std::string& m, 
	const std::string& t, bool n) : module(m), texture(t), norm(n) {
	
}

cuda::RegisteredGlobal::RegisteredGlobal(const std::string& m, 
	const std::string& g) : module(m), global(g) {

}

cuda::Dimension::Dimension(int _x, int _y, int _z, 
	const cudaChannelFormatDesc& _f) : x(_x), y(_y), z(_z), format(_f) {

}

size_t cuda::Dimension::pitch() const {
	return ((format.x + format.y + format.z + format.w) / 8) * x;
}

////////////////////////////////////////////////////////////////////////////////

void cuda::CudaRuntime::_memcpy(void* dst, const void* src, size_t count, 
	enum cudaMemcpyKind kind) {
	
	switch(kind) {
		case cudaMemcpyHostToHost: {
			report("  _memcpy(" << (void *)dst << ", " << src
				<< ", " << count << " bytes) h-to-h");
			memcpy(dst, src, count);
		}
		break;
		case cudaMemcpyDeviceToHost: {
			report("  _memcpy(" << (void *)dst << ", " << src
				<< ", " << count << " bytes) d-to-h");
			if (!_getDevice().checkMemoryAccess(src, count)) {
				_release();
				_memoryError(src, count, "cudaMemcpy");
			}
			
			executive::Device::MemoryAllocation* allocation = 
				_getDevice().getMemoryAllocation(src);
			size_t offset = (char*)src - (char*)allocation->pointer();
			allocation->copy(dst, offset, count);
		}
		break;
		case cudaMemcpyDeviceToDevice: {
			report("  _memcpy(" << (void *)dst << ", "
				<< src << ", " << count << " bytes) d-to-d");
			if (!_getDevice().checkMemoryAccess(src, count)) {
				_release();
				_memoryError(src, count, "cudaMemcpy");
			}
			if (!_getDevice().checkMemoryAccess(dst, count)) {
				_release();
				_memoryError(dst, count, "cudaMemcpy");
			}
				
			executive::Device::MemoryAllocation* fromAllocation = 
				_getDevice().getMemoryAllocation(src);
			executive::Device::MemoryAllocation* toAllocation =
				_getDevice().getMemoryAllocation(dst);
			size_t fromOffset = (char*)src 
				- (char*)fromAllocation->pointer();
			size_t toOffset = (char*)dst - (char*)toAllocation->pointer();
			fromAllocation->copy(toAllocation, toOffset, fromOffset, count);
		}
		break;
		case cudaMemcpyHostToDevice: {
			report("  _memcpy(" << (void *)dst << ", "
				<< src << ", " << count << " bytes) h-to-d");
			if (!_getDevice().checkMemoryAccess(dst, count)) {
				_release();
				_memoryError(dst, count, "cudaMemcpy");
			}
			
			executive::Device::MemoryAllocation* allocation = 
				_getDevice().getMemoryAllocation(dst);
			size_t offset = (char*)dst - (char*)allocation->pointer();
			allocation->copy(offset, src, count);
		}
		break;
	}
}

void cuda::CudaRuntime::_memoryError(const void* address, size_t count, 
	const std::string& function) {
	std::stringstream stream;
	stream << "In function - " << function << " - invalid memory access at " 
		<< address << "(" << count << " bytes)"; 
	throw hydrazine::Exception(stream.str());
}

void cuda::CudaRuntime::_enumerateDevices() {
	if(_devicesLoaded) return;
	report("Creating devices.");
	if(config::get().executive.enableNVIDIA) {
		executive::DeviceVector d = 
			executive::Device::createDevices(ir::Instruction::SASS, _flags,
				_computeCapability);
		report(" - Added " << d.size() << " nvidia gpu devices." );
		_devices.insert(_devices.end(), d.begin(), d.end());
	}
	if(config::get().executive.enableEmulated) {
		executive::DeviceVector d = 
			executive::Device::createDevices(ir::Instruction::Emulated, _flags,
				_computeCapability);
		report(" - Added " << d.size() << " emulator devices." );
		_devices.insert(_devices.end(), d.begin(), d.end());
	}
	if(config::get().executive.enableLLVM) {
		executive::DeviceVector d = 
			executive::Device::createDevices(ir::Instruction::LLVM, _flags,
				_computeCapability);
		report(" - Added " << d.size() << " llvm-cpu devices." );
		_devices.insert(_devices.end(), d.begin(), d.end());
		
		if (config::get().executive.workerThreadLimit > 0) {
			for (executive::DeviceVector::iterator d_it = d.begin();
				d_it != d.end(); ++d_it) {
				(*d_it)->limitWorkerThreads(
					config::get().executive.workerThreadLimit);
			}
		}
	}
	if(config::get().executive.enableAMD) {
		executive::DeviceVector d =
			executive::Device::createDevices(ir::Instruction::CAL, _flags,
				_computeCapability);
		report(" - Added " << d.size() << " amd gpu devices." );
		_devices.insert(_devices.end(), d.begin(), d.end());
	}
	if(config::get().executive.enableRemote) {
		executive::DeviceVector d =
			executive::Device::createDevices(ir::Instruction::Remote, _flags,
				_computeCapability);
		report(" - Added " << d.size() << " remote devices." );
		_devices.insert(_devices.end(), d.begin(), d.end());
	}
	
	_devicesLoaded = true;
	
	if(_devices.empty())
	{
		std::cerr << "==Ocelot== WARNING - No CUDA devices found or all " 
			<< "devices disabled!\n";
		std::cerr << "==Ocelot==  Consider enabling the emulator in " 
			<< "configure.ocelot.\n";
	}
	else
	{
		// Register modules that have already been loaded
		for(ModuleMap::iterator module = _modules.begin(); 
			module != _modules.end(); ++module) {
			if(!module->second.loaded()) continue;
			for(DeviceVector::iterator device = _devices.begin(); 
				device != _devices.end(); ++device) {
				(*device)->select();
				(*device)->load(&module->second);
				(*device)->setOptimizationLevel(_optimization);
				(*device)->unselect();
			}
		}
		
		// Create worker threads for each device
		_workers.resize(_devices.size());

		ThreadVector::iterator worker = _workers.begin();
		for(DeviceVector::iterator device = _devices.begin(); 
			device != _devices.end(); ++device, ++worker) {
			worker->setDevice(*device);
			worker->start();
		}
	}
}

//! acquires mutex and locks the runtime
void cuda::CudaRuntime::_lock() {
	_mutex.lock();
}

//! releases mutex
void cuda::CudaRuntime::_unlock() {
	_mutex.unlock();
}

//! sets the last error state for the CudaRuntime object
cudaError_t cuda::CudaRuntime::_setLastError(cudaError_t result) {
	HostThreadContext& thread = _getCurrentThread();
	thread.lastError = result;
	return result;
}

cuda::HostThreadContext& cuda::CudaRuntime::_bind() {
	_enumerateDevices();

	HostThreadContext& thread = _getCurrentThread();

	if (_devices.empty()) return thread;
	
	_selectedDevice = thread.selectedDevice;
	executive::Device& device = _getDevice();

	assert(!device.selected());
	device.select();
	
	return thread;
}

void cuda::CudaRuntime::_unbind() {
	executive::Device& device = _getDevice();
	assert(_getCurrentThread().selectedDevice == _selectedDevice);
	
	_selectedDevice = -1;
	assert(device.selected());
	device.unselect();
}

void cuda::CudaRuntime::_acquire() {
	_lock();
	_bind();
	if (_devices.empty()) _unlock();
}

void cuda::CudaRuntime::_release() {
	_unbind();
	_unlock();
}

void cuda::CudaRuntime::_wait() {
	for(ThreadVector::iterator worker = _workers.begin(); 
		worker != _workers.end(); ++worker) {
		worker->wait();
	}
}

executive::Device& cuda::CudaRuntime::_getDevice() {
	assert(_selectedDevice >= 0);
	assert(_selectedDevice < (int)_devices.size());
	return *_devices[_selectedDevice];
}

cuda::CudaWorkerThread& cuda::CudaRuntime::_getWorkerThread() {
	assert(_selectedDevice >= 0);
	assert(_selectedDevice < (int)_workers.size());
	return _workers[_selectedDevice];
}

std::string cuda::CudaRuntime::_formatError( const std::string& message ) {
	std::string result = "==Ocelot== ";
	for(std::string::const_iterator mi = message.begin(); 
		mi != message.end(); ++mi) {
		result.push_back(*mi);
		if(*mi == '\n') {
			result.append("==Ocelot== ");
		}
	}
	return result;
}

cuda::HostThreadContext& cuda::CudaRuntime::_getCurrentThread() {
	HostThreadContextMap::iterator t = _threads.find(
		boost::this_thread::get_id());
	if (t == _threads.end()) {
		report("Creating new context for thread "
			<< boost::this_thread::get_id());
		t = _threads.insert(std::make_pair(boost::this_thread::get_id(), 
			HostThreadContext())).first;
	}
	return t->second;
}

void cuda::CudaRuntime::_registerModule(ModuleMap::iterator module) {
	if(module->second.loaded()) return;
	
	_wait();

	module->second.loadNow();
	
	for(RegisteredTextureMap::iterator texture = _textures.begin(); 
		texture != _textures.end(); ++texture) {
		if(texture->second.module != module->first) continue;
		ir::Texture* tex = module->second.getTexture(texture->second.texture);
		assert(tex != 0);
		tex->normalizedFloat = texture->second.norm;
	}
	
	transforms::PassManager manager(&module->second);
	
	for(PassSet::iterator pass = _passes.begin(); pass != _passes.end(); ++pass)
	{
		manager.addPass(*pass);
	}
	
	manager.runOnModule();
	manager.releasePasses();	
	
	for(DeviceVector::iterator device = _devices.begin(); 
		device != _devices.end(); ++device) {
		(*device)->select();
		(*device)->load(&module->second);
		(*device)->setOptimizationLevel(_optimization);
		(*device)->unselect();
	}
}

void cuda::CudaRuntime::_registerModule(const std::string& name) {
	ModuleMap::iterator module = _modules.find(name);
	if(module != _modules.end()) {
		_registerModule(module);
	}
}

void cuda::CudaRuntime::_registerAllModules() {
	for(ModuleMap::iterator module = _modules.begin(); 
		module != _modules.end(); ++module) {
		_registerModule(module);
	}
}

////////////////////////////////////////////////////////////////////////////////

cuda::CudaRuntime::CudaRuntime() :
	_deviceCount(0), _devicesLoaded(false), 
	_selectedDevice(-1), _nextSymbol(1), _computeCapability(2), _flags(0), 
	_optimization((translator::Translator::OptimizationLevel)
		config::get().executive.optimizationLevel) {

	// get device count
	if(config::get().executive.enableNVIDIA) {
		_deviceCount += executive::Device::deviceCount(
			ir::Instruction::SASS, _computeCapability);
	}
	if(config::get().executive.enableEmulated) {
		_deviceCount += executive::Device::deviceCount(
			ir::Instruction::Emulated, _computeCapability);
	}
	if(config::get().executive.enableLLVM) {
		_deviceCount += executive::Device::deviceCount(
			ir::Instruction::LLVM, _computeCapability);
	}
	if(config::get().executive.enableAMD) {
		_deviceCount += executive::Device::deviceCount(
			ir::Instruction::CAL, _computeCapability);
	}
	if(config::get().executive.enableRemote) {
		_deviceCount += executive::Device::deviceCount(
			ir::Instruction::Remote, _computeCapability);
	}
}

cuda::CudaRuntime::~CudaRuntime() {
	// wait for all devices to finish
	_wait();

	//
	// free things that need freeing
	//
	// devices
	for (DeviceVector::iterator device = _devices.begin(); 
		device != _devices.end(); ++device) {
		delete *device;
	}
	
	// mutex

	// thread contexts
	
	// textures
	
	// kernels
	
	// fat binaries
	
	// config
	config::destroy();
	
	// globals
}


////////////////////////////////////////////////////////////////////////////////

/*!
	registers a CUDA fatbinary and returns a handle
	for referencing the fat binary
*/

void** cuda::CudaRuntime::cudaRegisterFatBinary(void *fatCubin) {
	size_t handle = 0;
	_lock();

	handle = _fatBinaries.size();
	
	FatBinaryMap::const_iterator fatbin = _fatBinaries.insert(std::make_pair(
		handle,	FatBinaryContext(fatCubin))).first;
	
	for (FatBinaryMap::const_iterator it = _fatBinaries.begin();
		it != _fatBinaries.end(); ++it) {
		if(fatbin == it) continue;
		
		if (std::string(it->second.name()) == fatbin->second.name()) {
			_unlock();
	
			assert(0 && "binary already exists");		
			return 0;
		}	
	}

	// register associated PTX
	ModuleMap::iterator module = _modules.insert(
		std::make_pair(fatbin->second.name(), ir::Module())).first;
	module->second.lazyLoad(fatbin->second.ptx(), fatbin->second.name());
	
	report("Loading module (fatbin) - " << module->first);
	reportE(REPORT_ALL_PTX, " with PTX\n" << fatbin->second.ptx());
		
	_unlock();
	
	return (void **)handle;
}

/*!
	unregister a cuda fat binary
*/
void cuda::CudaRuntime::cudaUnregisterFatBinary(void **fatCubinHandle) {
	// Andy: do we really care?
	// Greg: For most cuda applications, probably not.  The only use would be 
	// to remove and reinsert a fat binary.  Let's use a different interface
	//  for that
	size_t handle = (size_t)fatCubinHandle;
	if (handle >= _fatBinaries.size()) {
		Ocelot_Exception("cudaUnregisterFatBinary(" << handle 
			<< ") - invalid handle.");
	}
}

/*!
	\brief register a CUDA global variable
	\param fatCubinHandle
	\param hostVar
	\param deviceAddress
	\param deviceName
	\param ext
	\param size
	\param constant
	\param global
*/
void cuda::CudaRuntime::cudaRegisterVar(void **fatCubinHandle, char *hostVar, 
	char *deviceAddress, const char *deviceName, int ext, int size, 
	int constant, int global) {

	report("cuda::CudaRuntime::cudaRegisterVar() - host var: "
		<< (void *)hostVar << ", deviceName: " << deviceName << ", size: "
		<< size << " bytes," << (constant ? " constant" : " ")
		<< (global ? " global" : " "));

	size_t handle = (size_t)fatCubinHandle;
	_lock();

	std::string moduleName = _fatBinaries[handle].name();
	
	_globals[(void *)hostVar] = RegisteredGlobal(moduleName, deviceName);
	
	_unlock();
}

/*!
	\brief registers a CUDA texture reference
	\param fatCubinHandle
	\param hostVar
	\param deviceAddress
	\param deviceName
	\param dim
	\param norm
	\param ext
*/
void cuda::CudaRuntime::cudaRegisterTexture(
	void **fatCubinHandle,
	const struct textureReference* hostVar,
	const void **deviceAddress,
	const char *deviceName,
	int dim,
	int norm,
	int ext) {
	
	size_t handle = (size_t)fatCubinHandle;
	
	_lock();
	
	std::string moduleName = _fatBinaries[handle].name();
	
	report("cudaRegisterTexture('" << deviceName << ", dim: " << dim 
		<< ", norm: " << norm << ", ext: " << ext);

	_textures[(void*)hostVar] = RegisteredTexture(moduleName, deviceName, norm);
	
	_unlock();
}

void cuda::CudaRuntime::cudaRegisterShared(
	void **fatCubinHandle,
	void **devicePtr) {
	
	_lock();
	
	report("cudaRegisterShared() - module " 
		<< _fatBinaries[(size_t)fatCubinHandle].name()
		<< ", devicePtr: " << devicePtr << " named "
		<< (const char *)devicePtr);
	
	report(" Ignoring this variable.");
	
	_unlock();
}

void cuda::CudaRuntime::cudaRegisterSharedVar(
	void **fatCubinHandle,
	void **devicePtr,
	size_t size,
	size_t alignment,
	int storage) {

	_lock();
	
	report("cudaRegisterSharedVar() - module " 
		<< _fatBinaries[(size_t)fatCubinHandle].name() 
		<< ", devicePtr: " << devicePtr << " named " << (const char *)devicePtr 
		<< ", size: " << size << ", alignment: " << alignment << ", storage: " 
		<< storage);
	
	report(" Ignoring this variable.");
		
	_unlock();
}

void cuda::CudaRuntime::cudaRegisterFunction(
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

	size_t handle = (size_t)fatCubinHandle;
	
	_lock();

	void *symbol = (void *)hostFun;
	std::string kernelName = deviceFun;
	std::string moduleName = _fatBinaries[handle].name();
	
	report("Registered kernel - " << kernelName 
		<< " in module '" << moduleName << "'");
	_kernels[symbol] = RegisteredKernel(handle, moduleName, kernelName);
	
	_unlock();
}

cudaError_t cuda::CudaRuntime::cudaGetExportTable(const void **ppExportTable,
	const cudaUUID_t *pExportTableId) {

	std::cout << "cudaGetExportTable() GUID: ";
	for (size_t i = 0; i < 16; i++) {
		std::cout << std::hex << " 0x" << ((unsigned int)((const char *)pExportTableId)[i] & 0x0ff);
	}
	std::cout << std::endl;


	assertM(false, "cudaGetExportTable is actually a backdoor to the NVIDIA "
		"driver.  Ocelot cannot support this because it requires the NVIDIA "
			"driver to be installed.  If you want to run an application that "
			"depends on this hack, please complain to NVIDIA.");

	return cudaSuccess;
}

#undef DEBUG_MODE

////////////////////////////////////////////////////////////////////////////////
//
// memory allocation

cudaError_t cuda::CudaRuntime::cudaMalloc(void **devPtr, size_t size) {
	cudaError_t result = cudaErrorMemoryAllocation;

	_acquire();
	if (_devices.empty()) {
		return _setLastError(cudaErrorNoDevice);
	}
		
	try {
		executive::Device::MemoryAllocation* 
			allocation = _getDevice().allocate(size);
		*devPtr = allocation->pointer();
		result = cudaSuccess;
	}
	catch(hydrazine::Exception&) {
		
	}
	
	report("cudaMalloc( *devPtr = " << (void *)*devPtr 
	<< ", size = " << size << ")");

	_release();
	
	return _setLastError(result);
}

/*!
	constructs a host-side allocation, returns pointer to mapped region -
	this allocation is referenced by the mappedPointer()
*/
cudaError_t cuda::CudaRuntime::cudaMallocHost(void **ptr, size_t size) {
	cudaError_t result = cudaErrorMemoryAllocation;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	try {
		executive::Device::MemoryAllocation* 
			allocation = _getDevice().allocateHost(size);
		*ptr = allocation->mappedPointer();
		result = cudaSuccess;
	}
	catch(hydrazine::Exception &exp) {
		report("  cudaMallocHost() - error:\n" << exp.what());
	}

	report("cudaMallocHost( *pPtr = " << (void *)*ptr 
		<< ", size = " << size << ")");
		
	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMallocPitch(void **devPtr, size_t *pitch, 
	size_t width, size_t height) {
	cudaError_t result = cudaErrorMemoryAllocation;
	
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	*pitch = width;

	try {
		executive::Device::MemoryAllocation* 
			allocation = _getDevice().allocate(width * height);
		_dimensions[allocation->pointer()] = Dimension(width, height, 1);
		*devPtr = allocation->pointer();
		result = cudaSuccess;
	}
	catch(hydrazine::Exception&) {
	
	}
	
	report("cudaMallocPitch( *devPtr = " << (void *)*devPtr 
		<< ", pitch = " << *pitch << ")");

	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMallocArray(struct cudaArray **array, 
	const struct cudaChannelFormatDesc *desc, size_t width, size_t height) {
	cudaError_t result = cudaErrorMemoryAllocation;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	unsigned int x = desc->x == 0 ? 1 : desc->x;
	unsigned int y = desc->y == 0 ? 1 : desc->y;
	unsigned int z = desc->z == 0 ? 1 : desc->z;
	unsigned int w = desc->w == 0 ? 1 : desc->w;
	
	height = height == 0 ? 1 : height;
	
	size_t size = width * height * ( x + y + z + w ) / 8;
	
	try {
		executive::Device::MemoryAllocation* 
			allocation = _getDevice().allocate(size);
		_dimensions[allocation->pointer()] = Dimension(width, height, 1, *desc);
		*array = (struct cudaArray*)allocation->pointer();
		result = cudaSuccess;
	}
	catch(hydrazine::Exception&) {
	
	}
	
	report("cudaMallocArray( *array = " << (void *)*array << ")");

	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaFree(void *devPtr) {
	cudaError_t result = cudaErrorMemoryAllocation;

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	report("cudaFree(" << devPtr << ")");
	
	try {
		if (devPtr) {
			_getDevice().free(devPtr);
		}
		result = cudaSuccess;
	}
	catch(hydrazine::Exception&) {
		
	}

	_release();

	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaFreeHost(void *ptr) {
	cudaError_t result = cudaErrorMemoryAllocation;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaFreeHost(" << ptr << ")");	
	
	try {
		if (ptr) {
			_getDevice().free(ptr);
		}
		result = cudaSuccess;
	}
	catch(hydrazine::Exception&) {
		
	}
	
	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaFreeArray(struct cudaArray *array) {
	cudaError_t result = cudaErrorMemoryAllocation;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	try {
		if (array) {
			_getDevice().free(array);
		}
		result = cudaSuccess;
	}
	catch(hydrazine::Exception&) {
		
	}

	report("cudaFreeArray() array = " << (void *)array);

	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMalloc3D(struct cudaPitchedPtr* devPtr, 
	struct cudaExtent extent) {
	cudaError_t result = cudaErrorMemoryAllocation;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaMalloc3D() extent - width " << extent.width 
		<< " - height " << extent.height << " - depth " << extent.depth );
	size_t padding = (extent.width % 256) ? 256 - extent.width % 256 : 0;
	devPtr->pitch = extent.width + padding;
	devPtr->xsize = extent.width;
	devPtr->ysize = extent.height; //* extent.depth;
	
	size_t size = devPtr->pitch * extent.height * extent.depth;

	try {
		executive::Device::MemoryAllocation* 
			allocation = _getDevice().allocate(size);
		devPtr->ptr = allocation->pointer();
		_dimensions[allocation->pointer()] = Dimension(devPtr->pitch, 
			extent.height, extent.depth);
		result = cudaSuccess;
	}
	catch(hydrazine::Exception&) {
	
	}
	
	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMalloc3DArray(struct cudaArray** arrayPtr, 
	const struct cudaChannelFormatDesc* desc, struct cudaExtent extent) {

	cudaError_t result = cudaErrorMemoryAllocation;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	size_t size = extent.width * extent.height * extent.depth * (
		desc->x + desc->y + desc->z + desc->w ) / 8;

	try {
		executive::Device::MemoryAllocation* 
			allocation = _getDevice().allocate(size);
		*arrayPtr = (struct cudaArray*)allocation->pointer();
		_dimensions[allocation->pointer()] = Dimension(extent.width, extent.height, extent.depth, *desc);
		result = cudaSuccess;
	}
	catch(hydrazine::Exception&) {
	
	}

	report("cudaMalloc3DArray() - *arrayPtr = " << (void *)(*arrayPtr));
	
	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaHostAlloc(void **pHost, size_t bytes, 
	unsigned int flags) {
	cudaError_t result = cudaErrorMemoryAllocation;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	try {
		executive::Device::MemoryAllocation* 
			allocation = _getDevice().allocateHost(bytes, flags);
		*pHost = allocation->mappedPointer();
		result = cudaSuccess;
	}
	catch(hydrazine::Exception&) {
		
	}

	report("cudaHostAlloc() - *pHost = " << (void *)(*pHost) 
		<< ", bytes " << bytes << ", " << flags);
	
	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaHostGetDevicePointer(void **pDevice, 
	void *pHost, unsigned int flags) {

	cudaError_t result = cudaErrorInvalidHostPointer;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	executive::Device::MemoryAllocation *allocation =
		_getDevice().getMemoryAllocation(pHost, 
			executive::Device::HostAllocation);

	if (allocation != 0) {	
		if (allocation->host()) {
			size_t offset = (char*)pHost - (char*)allocation->mappedPointer();
			*pDevice = (char*)allocation->pointer() + offset;
			result = cudaSuccess;
		}
	}
	else {
		report("cudaHostGetDevicePointer() - failed to get device pointer "
			"from host allocation at " << (const void *)pHost);
	}

	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaHostGetFlags(unsigned int *pFlags, 
	void *pHost) {

	cudaError_t result = cudaErrorInvalidValue;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	executive::Device::MemoryAllocation* 
		allocation = _getDevice().getMemoryAllocation(pHost, 
			executive::Device::HostAllocation);
	
	if (allocation != 0) {
		result = cudaSuccess;
		*pFlags = allocation->flags();
	}
	
	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaHostRegister(void *pHost, size_t bytes, 
	unsigned int flags)
{
	cudaError_t result = cudaErrorInvalidValue;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	executive::Device::MemoryAllocation *allocation =
		_getDevice().registerHost(pHost, bytes, flags);

	if (allocation != 0) {	
		assert(allocation->host());
		result = cudaSuccess;
	}
	else {
		report("cudaHostRegister() - failed to register host allocation at "
			<< (const void *)pHost);
	}

	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaHostUnregister(void *pHost)
{
	cudaError_t result = cudaErrorInvalidValue;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	try {
		if (pHost) {
			_getDevice().free(pHost);
		}
		result = cudaSuccess;
	}
	catch(hydrazine::Exception&) {
		
	}

	_release();
	
	return _setLastError(result);
}

////////////////////////////////////////////////////////////////////////////////
//
// memory copying

cudaError_t cuda::CudaRuntime::cudaMemcpy(void *dst, const void *src, 
	size_t count, enum cudaMemcpyKind kind) {
	cudaError_t result = cudaErrorInvalidDevicePointer;
	if (kind >= 0 && kind <= 3) {
		_wait();
		_acquire();
		if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

		report("cudaMemcpy(" << dst << ", " << src << ", " << count << ")");
		_memcpy(dst, src, count, kind);
		result = cudaSuccess;

		_release();
	}
	else {
		result = cudaErrorInvalidMemcpyDirection;
	}

	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMemcpyToSymbol(const char *symbol, 
	const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) {
	report("cuda::CudaRuntime::cudaMemcpyToSymbol('" << symbol << "' - " 
		<< (void *)symbol << " - value " 
		<< hydrazine::dataToString(src, count));
	
	if (kind != cudaMemcpyHostToDevice) {
		return _setLastError(cudaErrorInvalidMemcpyDirection);
	}

	cudaError_t result = cudaErrorInvalidDevicePointer;
	_wait();
	_lock();

	_enumerateDevices();
	if (_devices.empty()) {
		_unlock();
		return _setLastError(cudaErrorNoDevice);
	}
	
	RegisteredGlobalMap::iterator global = _globals.find((void*)symbol);
	std::string name;
	std::string module;
	
	if (global == _globals.end()) {
		name = symbol;
		_registerAllModules();
	}
	else {
		name = global->second.global;
		module = global->second.module;
		try {
			_registerModule(module);
		}
		catch(...) {
			_unlock();
			throw;
		}
	}

	_bind();

	executive::Device::MemoryAllocation* allocation = 
		_getDevice().getGlobalAllocation(module, name);

	if (allocation != 0) {
		if (!_getDevice().checkMemoryAccess((char*)allocation->pointer() 
			+ offset, count)) {
			_release();
			_memoryError((char*)allocation->pointer() + offset, 
				count, "cudaMemcpyToSymbol");
		}
	
		allocation->copy(offset, src, count);
		result = cudaSuccess;
	}
	
	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMemcpyFromSymbol(void *dst, 
	const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
	report("cuda::CudaRuntime::cudaMemcpyFromSymbol('" << symbol << "' - " 
		<< (void *)symbol << " - value " 
		<< hydrazine::dataToString(dst, count));

	if (kind != cudaMemcpyDeviceToHost) {
		return _setLastError(cudaErrorInvalidMemcpyDirection);
	}

	cudaError_t result = cudaErrorInvalidDevicePointer;
	_wait();
	_lock();

	_enumerateDevices();
	if (_devices.empty()) {
		_unlock();
		return _setLastError(cudaErrorNoDevice);
	}
	
	RegisteredGlobalMap::iterator global = _globals.find((void*)symbol);
	std::string name;
	std::string module;
	
	if (global == _globals.end()) {
		name = symbol;	
		_registerAllModules();
	}
	else {
		name = global->second.global;
		module = global->second.module;
		try {
			_registerModule(module);
		}
		catch(...) {
			_unlock();
			throw;
		}
	}
	
	_bind();

	executive::Device::MemoryAllocation* allocation = 
		_getDevice().getGlobalAllocation(module, name);

	if (allocation != 0) {
		if (!_getDevice().checkMemoryAccess((char*)allocation->pointer() 
			+ offset, count)) {
			_release();
			_memoryError((char*)allocation->pointer() + offset, 
				count, "cudaMemcpyFromSymbol");
		}
	
		allocation->copy(dst, offset, count);
		result = cudaSuccess;
	}
	
	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMemcpyAsync(void *dst, const void *src, 
	size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
	cudaError_t result = cudaErrorInvalidDevicePointer;
	if (kind >= 0 && kind <= 3) {
		_acquire();
		if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

		report("cudaMemcpyAsync(" << dst << ", " << src
			<< ", " << count << ")");
		_memcpy(dst, src, count, kind);
		result = cudaSuccess;

		_release();
	}
	else {
		result = cudaErrorInvalidMemcpyDirection;
	}

	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMemcpyToArray(struct cudaArray *dst, 
	size_t wOffset, size_t hOffset, const void *src, size_t count, 
	enum cudaMemcpyKind kind) {

	cudaError_t result = cudaErrorInvalidValue ;

	_wait();

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaMemcpyToArray("<< dst << ", " << src << ", " << wOffset 
		<< ", " << hOffset << ", " << count << ")");

	if (kind == cudaMemcpyHostToDevice) {
		executive::Device::MemoryAllocation* 
			allocation = _getDevice().getMemoryAllocation(dst);

		if (allocation != 0) {
			Dimension& dimension = _dimensions[allocation->pointer()];
			size_t offset = wOffset + hOffset * dimension.pitch();
			void* address = (char*)allocation->pointer() + offset;
			if (!_getDevice().checkMemoryAccess(address, count)) {
				_release();
				_memoryError(address, count, "cudaMemcpyToArray");
			}
			allocation->copy(offset, src, count);
			result = cudaSuccess;
		}
		else
		{
			_release();
			_memoryError(dst, count, "cudaMemcpyToArray");
		}
	}
	else if (kind == cudaMemcpyDeviceToDevice) {
		executive::Device::MemoryAllocation* 
			destination = _getDevice().getMemoryAllocation(dst);
		executive::Device::MemoryAllocation* 
			source = _getDevice().getMemoryAllocation(src);
		if (destination != 0 && source != 0) {
			Dimension& dimension = _dimensions[destination->pointer()];
			size_t offset = wOffset + hOffset * dimension.pitch();
			void* address = (char*)destination->pointer() + offset;
			if (!_getDevice().checkMemoryAccess(address, count)) {
				_release();
				_memoryError(address, count, "cudaMemcpyToArray");
			}
			if (!_getDevice().checkMemoryAccess(src, count)) {
				_release();
				_memoryError(src, count, "cudaMemcpyToArray");
			}
			size_t sourceOffset = (char*)src - (char*)source->pointer();
			source->copy(destination, offset, sourceOffset, count);
			result = cudaSuccess;
		}
		else
		{
			_release();
			if(destination == 0)
			{
				_memoryError(dst, count, "cudaMemcpyToArray");
			}
			else
			{
				_memoryError(src, count, "cudaMemcpyToArray");
			}
		}
	}

	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMemcpyFromArray(void *dst, 
	const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, 
	enum cudaMemcpyKind kind) {

	cudaError_t result = cudaErrorInvalidValue ;

	_wait();
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaMemcpyFromArray("<< dst << ", " << src << ", " << wOffset 
		<< ", " << hOffset << ", " << count << ")");

	if (kind == cudaMemcpyDeviceToHost) {
		executive::Device::MemoryAllocation* 
			allocation = _getDevice().getMemoryAllocation(src);

		if (allocation != 0) {
			Dimension& dimension = _dimensions[allocation->pointer()];
			assert(_dimensions.count(allocation->pointer()) != 0);
			size_t offset = wOffset + hOffset * dimension.pitch();
			void* address = (char*)allocation->pointer() + offset;
			if (!_getDevice().checkMemoryAccess(address, count)) {
				_release();
				_memoryError(address, count, "cudaMemcpyFromArray");
			}
			allocation->copy(dst, offset, count);
			result = cudaSuccess;
		}
		else
		{
			_release();
			_memoryError(src, count, "cudaMemcpyFromArray");
		}
	}
	else if (kind == cudaMemcpyDeviceToDevice) {
		executive::Device::MemoryAllocation* 
			destination = _getDevice().getMemoryAllocation(dst);
		executive::Device::MemoryAllocation* 
			source = _getDevice().getMemoryAllocation(src);
		if (destination != 0 && source != 0) {
			Dimension& dimension = _dimensions[source->pointer()];
			size_t offset = wOffset + hOffset * dimension.pitch();
			void* address = (char*)destination->pointer() + offset;
			if (!_getDevice().checkMemoryAccess(address, count)) {
				_release();
				_memoryError(address, count, "cudaMemcpyFromArray");
			}
			if (!_getDevice().checkMemoryAccess(dst, count)) {
				_release();
				_memoryError(src, count, "cudaMemcpyFromArray");
			}
			size_t destinationOffset = (char*)dst 
				- (char*)destination->pointer();
			source->copy(destination, destinationOffset, offset, count);
			result = cudaSuccess;
		}
		else
		{
			_release();
			 if(destination == 0)
			 {
			 	_memoryError(dst, count, "cudaMemcpyFromArray");
			 }
			 else
			 {
			 	_memoryError(src, count, "cudaMemcpyFromArray");			 
			 }
		}
	}

	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMemcpyArrayToArray(struct cudaArray *dst, 
	size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, 
	size_t wOffsetSrc, size_t hOffsetSrc, size_t count, 
	enum cudaMemcpyKind kind) {

	cudaError_t result = cudaErrorInvalidValue;
	report("cudaMemcpyArrayToArray()");

	_wait();
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	if (kind == cudaMemcpyDeviceToDevice) {
		executive::Device::MemoryAllocation* 
			destination = _getDevice().getMemoryAllocation(dst);
		if (destination->pointer() != dst) {
			_release();
			_memoryError(dst, count, "cudaMemcpyArrayToArray");
		}
		executive::Device::MemoryAllocation* 
			source = _getDevice().getMemoryAllocation(src);
		if (source->pointer() != src) {
			_release();
			_memoryError(src, count, "cudaMemcpyArrayToArray");
		}
		if (destination != 0 && source != 0) {
			assert(_dimensions.count(source->pointer()) != 0);
			assert(_dimensions.count(destination->pointer()) != 0);
			Dimension& sourceDimension = _dimensions[source->pointer()];
			Dimension& destinationDimension = 
				_dimensions[destination->pointer()];
			size_t sourceOffset = wOffsetSrc 
				+ hOffsetSrc * sourceDimension.pitch();
			size_t destinationOffset = wOffsetDst 
				+ hOffsetDst * destinationDimension.pitch();
			void* sourceAddress = (char*)source->pointer() + sourceOffset;
			void* destinationAddress = (char*)destination->pointer() 
				+ destinationOffset;
			if (!_getDevice().checkMemoryAccess(sourceAddress, count)) {
				_release();
				_memoryError(sourceAddress, count, "cudaMemcpyArrayToArray");
			}
			if (!_getDevice().checkMemoryAccess(destinationAddress, count)) {
				_release();
				_memoryError(destinationAddress, 
					count, "cudaMemcpyArrayToArray");
			}
			source->copy(destination, destinationOffset, sourceOffset, count);
			result = cudaSuccess;
		}	
	}
	
	_release();
	return _setLastError(result);	
}

/*!
	\brief perform a 2D memcpy from a dense buffer
*/
cudaError_t cuda::CudaRuntime::cudaMemcpy2D(void *dst, size_t dpitch, 
	const void *src, size_t spitch, size_t width, size_t height, 
	enum cudaMemcpyKind kind) {

	_wait();

	cudaError_t result = cudaErrorInvalidValue;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaMemcpy2D()");	

	switch (kind) {
		case cudaMemcpyHostToHost:
		{
			for (size_t row = 0; row < height; row++) {
				char* dstPtr = (char*)dst + dpitch * row;
				char* srcPtr = (char*)src + spitch * row;
				std::memcpy(dstPtr, srcPtr, width);
			}
			result = cudaSuccess;
		}
		break;
		case cudaMemcpyDeviceToHost:
		{
			executive::Device::MemoryAllocation* source = 
				_getDevice().getMemoryAllocation(src);
			if (source != 0) {
				for (size_t row = 0; row < height; row++) {
					void* dstPtr = (char *)dst + dpitch * row;
					size_t srcOffset = spitch * row;

					if (!_getDevice().checkMemoryAccess(
						(char*)source->pointer() + srcOffset, width)) {
						_release();
						_memoryError((char*)source->pointer() + srcOffset, 
							width, "cudaMemcpy2D");
					}
					
					source->copy(dstPtr, srcOffset, width);
				}
				result = cudaSuccess;
			}
			else {
				_release();
				_memoryError(src, width * height, "cudaMemcpy2D");
			}
		}
		break;		
		case cudaMemcpyHostToDevice:
		{
			executive::Device::MemoryAllocation* destination = 
				_getDevice().getMemoryAllocation(dst);
			size_t dstPitch = _dimensions[destination->pointer()].pitch();
			if (destination != 0) {
				for (size_t row = 0; row < height; row++) {
					void* srcPtr = (char *)src + spitch * row;
					size_t dstOffset = dstPitch * row;

					if (!_getDevice().checkMemoryAccess(
						(char*)destination->pointer() + dstOffset, width)) {
						_release();
						_memoryError((char*)destination->pointer() + dstOffset, 
							width, "cudaMemcpy2D");
					}
					
					destination->copy(dstOffset, srcPtr, width);
				}
				result = cudaSuccess;
			}
			else {
				_release();
				_memoryError(dst, width * height, "cudaMemcpy2D");
			}
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			executive::Device::MemoryAllocation* destination = 
				_getDevice().getMemoryAllocation(dst);
			executive::Device::MemoryAllocation* source = 
				_getDevice().getMemoryAllocation(src);
			if (destination != 0 && source != 0) {
				for (size_t row = 0; row < height; row++) {
					size_t srcOffset = spitch * row;
					size_t dstOffset = dpitch * row;

					if (!_getDevice().checkMemoryAccess(
						(char*)destination->pointer() + dstOffset, width)) {
						_release();
						_memoryError((char*)destination->pointer() + dstOffset, 
							width, "cudaMemcpy2D");
					}
					if (!_getDevice().checkMemoryAccess(
						(char*)source->pointer() + srcOffset, width)) {
						_release();
						_memoryError((char*)source->pointer() + srcOffset, 
							width, "cudaMemcpy2D");
					}
					
					source->copy(destination, dstOffset, srcOffset, width);
				}
				result = cudaSuccess;
			}
			else {
				_release();
				if (destination == 0) {
					_memoryError(dst, width * height, "cudaMemcpy2D");
				}
				else {
					_memoryError(src, width * height, "cudaMemcpy2D");
				}
			}
		}
		break;
		default: break;
	}
	
	_release();
	return _setLastError(result);	
}

/*!
	\brief perform a 2D memcpy to an array
*/
cudaError_t cuda::CudaRuntime::cudaMemcpy2DToArray(struct cudaArray *dst, 
	size_t wOffset, size_t hOffset, const void *src, size_t spitch, 
	size_t width, size_t height, enum cudaMemcpyKind kind) {

	_wait();

	cudaError_t result = cudaErrorInvalidValue;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaMemcpy2DtoArray(dst = " << (void *)dst 
		<< ", src = " << (void *)src);
	
	if (kind == cudaMemcpyHostToDevice) {
		executive::Device::MemoryAllocation* destination = 
			_getDevice().getMemoryAllocation(dst);
		
		if (destination == 0) {
			_release();
			_memoryError(dst, width * height, "cudaMemcpy2DtoArray");
		}
		
		size_t dstPitch = _dimensions[destination->pointer()].pitch();

		for (size_t row = 0; row < height; ++row) {
			void* srcPtr = (char*)src + row * spitch;
			size_t dstOffset = (row + hOffset) * dstPitch + wOffset;
			
			if (!_getDevice().checkMemoryAccess((char*)destination->pointer() 
				+ dstOffset, width)) {
				_release();
				_memoryError((char*)destination->pointer() + dstOffset, 
					width, "cudaMemcpy2DtoArray");
			}
			
			destination->copy(dstOffset, srcPtr, width);
		}
		
		result = cudaSuccess;
	}
	else if (kind == cudaMemcpyDeviceToDevice) {
		executive::Device::MemoryAllocation* destination = 
			_getDevice().getMemoryAllocation(dst);
		executive::Device::MemoryAllocation* source = 
			_getDevice().getMemoryAllocation(src);
		
		if (destination == 0) {
			_release();
			_memoryError(dst, width * height, "cudaMemcpy2DtoArray");
		}

		if (source == 0) {
			_release();
			_memoryError(src, width * height, "cudaMemcpy2DtoArray");
		}
		
		size_t dstPitch = _dimensions[destination->pointer()].pitch();
		
		for (size_t row = 0; row < height; ++row) {
			size_t srcOffset = row * spitch;
			size_t dstOffset = (row + hOffset) * dstPitch + wOffset;
			
			if (!_getDevice().checkMemoryAccess((char*)destination->pointer() 
				+ dstOffset, width)) {
				_release();
				_memoryError((char*)destination->pointer() 
					+ dstOffset, width, "cudaMemcpy2DtoArray");
			}

			if (!_getDevice().checkMemoryAccess((char*)source->pointer() 
				+ srcOffset, width)) {
				_release();
				_memoryError((char*)source->pointer() + srcOffset, 
					width, "cudaMemcpy2DtoArray");
			}
			
			source->copy(destination, dstOffset, srcOffset, width);
		}
		
		result = cudaSuccess;
	}
	
	_release();
	return _setLastError(result);
}

/*!
	\brief perform a 2D memcpy from an array
*/
cudaError_t cuda::CudaRuntime::cudaMemcpy2DFromArray(void *dst, size_t dpitch, 
	const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, 
	size_t height, enum cudaMemcpyKind kind) {

	_wait();

	cudaError_t result = cudaErrorInvalidValue;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaMemcpy2DfromArray(dst = " << (void *)dst 
		<< ", src = " << (void *)src);
	
	if (kind == cudaMemcpyDeviceToHost) {
		executive::Device::MemoryAllocation* source = 
			_getDevice().getMemoryAllocation(dst);
		
		if (source == 0) {
			_release();
			_memoryError(src, width * height, "cudaMemcpy2DfromArray");
		}
		
		assert(_dimensions.count(source->pointer()) != 0);
		size_t srcPitch = _dimensions[source->pointer()].pitch();

		for (size_t row = 0; row < height; ++row) {
			void* dstPtr = (char*)dst + row * dpitch;
			size_t srcOffset = (row + hOffset) * srcPitch + wOffset;
			
			if (!_getDevice().checkMemoryAccess((char*)source->pointer() 
				+ srcOffset, width)) {
				_release();
				_memoryError((char*)source->pointer() 
					+ srcOffset, width, "cudaMemcpy2DfromArray");
			}
			
			source->copy(dstPtr, srcOffset, width);
		}
		
		result = cudaSuccess;
	}
	else if (kind == cudaMemcpyDeviceToDevice) {
		executive::Device::MemoryAllocation* destination = 
			_getDevice().getMemoryAllocation(dst);
		executive::Device::MemoryAllocation* source = 
			_getDevice().getMemoryAllocation(src);
		
		if (destination == 0) {
			_release();
			_memoryError(dst, width * height, "cudaMemcpy2DfromArray");
		}

		if (source == 0) {
			_release();
			_memoryError(src, width * height, "cudaMemcpy2DfromArray");
		}
		
		size_t srcPitch = _dimensions[source->pointer()].pitch();
		
		for (size_t row = 0; row < height; ++row) {
			size_t dstOffset = row * dpitch;
			size_t srcOffset = (row + hOffset) * srcPitch + wOffset;
			
			if (!_getDevice().checkMemoryAccess((char*)destination->pointer() 
				+ dstOffset, width)) {
				_release();
				_memoryError((char*)destination->pointer() 
					+ dstOffset, width, "cudaMemcpy2DfromArray");
			}

			if (!_getDevice().checkMemoryAccess((char*)source->pointer() 
				+ srcOffset, width)) {
				_release();
				_memoryError((char*)source->pointer() + srcOffset, width, 
					"cudaMemcpy2DfromArray");
			}
			
			destination->copy(source, srcOffset, dstOffset, width);
		}
		
		result = cudaSuccess;
	}
	
	_release();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {

	_wait();

	cudaError_t result = cudaErrorInvalidValue;

	cudaPitchedPtr dst;
	cudaPitchedPtr src;
	const cudaExtent& extent = p->extent;

	if (p->dstArray) {
		dst.ptr = (void *)p->dstArray;
		dst.pitch = _dimensions[p->dstArray].pitch();
		dst.xsize = _dimensions[p->dstArray].x;
		dst.ysize = _dimensions[p->dstArray].y;		
	}
	else {
		dst = p->dstPtr;
	}

	if (p->srcArray) {
		src.ptr = (void *)p->srcArray;
		src.pitch = _dimensions[p->srcArray].pitch();
		src.xsize = _dimensions[p->srcArray].x;
		src.ysize = _dimensions[p->srcArray].y;
	}
	else {
		src = p->srcPtr;
	}

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaMemcpy3D() - dstPtr = (" << (void *)dst.ptr << ", " 
		<< dst.xsize << ", " << dst.ysize << ") - srcPtr = (" 
		<< (void *)src.ptr << ", " << src.xsize << ", " 
		<< src.ysize << ")");

	switch(p->kind) {
		case cudaMemcpyHostToHost:
		{
			for (size_t z = 0; z < extent.depth; ++z) {
				for (size_t y = 0; y < extent.height; ++y) {
					void* dstPtr = (char*)dst.ptr + p->dstPos.x + dst.xsize 
						* ((p->dstPos.y+y) + (z+p->dstPos.z) * dst.ysize);
					void* srcPtr = (char*)src.ptr + p->srcPos.x + src.xsize 
						* ((p->srcPos.y+y) + (z+p->srcPos.z) * src.ysize);

					std::memcpy(dstPtr, srcPtr, extent.width);
				}
			}
			result = cudaSuccess;
		}
		break;
		case cudaMemcpyHostToDevice:
		{
			executive::Device::MemoryAllocation* destination = 
				_getDevice().getMemoryAllocation(dst.ptr);
			
			if (destination == 0) {
				_release();
				_memoryError(dst.ptr, extent.width 
					* extent.height * extent.depth, "cudaMemcpy3D");
			}
			
			for (size_t z = 0; z < extent.depth; ++z) {
				for (size_t y = 0; y < extent.height; ++y) {
					size_t dstPtr = p->dstPos.x + dst.xsize 
						* ((p->dstPos.y+y) + (z+p->dstPos.z) * dst.ysize);
					void* srcPtr = (char*)src.ptr + p->srcPos.x + src.xsize 
						* ((p->srcPos.y+y) + (z+p->srcPos.z) * src.ysize);

					if (!_getDevice().checkMemoryAccess(
						(char*)destination->pointer() + dstPtr, extent.width)) {
						_release();
						_memoryError((char*)destination->pointer() + dstPtr, 
							extent.width, "cudaMemcpy3D");
					}

					destination->copy(dstPtr, srcPtr, extent.width);
				}
			}
			result = cudaSuccess;
		}
		break;
		case cudaMemcpyDeviceToHost:
		{
			executive::Device::MemoryAllocation* source = 
				_getDevice().getMemoryAllocation(src.ptr);
			
			if (source == 0) {
				_release();
				_memoryError(src.ptr, extent.width 
					* extent.height * extent.depth, "cudaMemcpy3D");
			}
			
			for (size_t z = 0; z < extent.depth; ++z) {
				for (size_t y = 0; y < extent.height; ++y) {
					void* dstPtr = (char*)dst.ptr + p->dstPos.x + dst.xsize 
						* ((p->dstPos.y+y) + (z+p->dstPos.z) * dst.ysize);
					size_t srcPtr = p->srcPos.x + src.xsize 
						* ((p->srcPos.y+y) + (z+p->srcPos.z) * src.ysize);

					if (!_getDevice().checkMemoryAccess(
						(char*)source->pointer() + srcPtr, extent.width)) {
						_release();
						_memoryError((char*)source->pointer() + srcPtr, 
							extent.width, "cudaMemcpy3D");
					}

					source->copy(dstPtr, srcPtr, extent.width);
				}
			}
			result = cudaSuccess;
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			executive::Device::MemoryAllocation* source = 
				_getDevice().getMemoryAllocation(src.ptr);
			
			if (source == 0) {
				_release();
				_memoryError(src.ptr, extent.width 
					* extent.height * extent.depth, "cudaMemcpy3D");
			}

			executive::Device::MemoryAllocation* destination = 
				_getDevice().getMemoryAllocation(dst.ptr);
			
			if (destination == 0) {
				_release();
				_memoryError(dst.ptr, extent.width 
					* extent.height * extent.depth, "cudaMemcpy3D");
			}
			
			for (size_t z = 0; z < extent.depth; ++z) {
				for (size_t y = 0; y < extent.height; ++y) {
					size_t dstPtr = p->dstPos.x + dst.xsize 
						* ((p->dstPos.y+y) + (z+p->dstPos.z) * dst.ysize);
					size_t srcPtr = p->srcPos.x + src.xsize 
						* ((p->srcPos.y+y) + (z+p->srcPos.z) * src.ysize);

					if (!_getDevice().checkMemoryAccess(
						(char*)source->pointer() + srcPtr, extent.width)) {
						_release();
						_memoryError((char*)source->pointer() + srcPtr,
							extent.width, "cudaMemcpy3D");
					}

					if (!_getDevice().checkMemoryAccess(
						(char*)destination->pointer() + dstPtr, extent.width)) {
						_release();
						_memoryError((char*)destination->pointer() + dstPtr, 
							extent.width, "cudaMemcpy3D");
					}

					source->copy(destination, dstPtr, srcPtr, extent.width);
				}
			}
			result = cudaSuccess;
		}
		break;
	}
	
	_release();
	return _setLastError(result);	
}

cudaError_t cuda::CudaRuntime::cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, 
	cudaStream_t stream) {
	return cudaMemcpy3D(p);
}

////////////////////////////////////////////////////////////////////////////////
//
// memset
//

cudaError_t cuda::CudaRuntime::cudaMemset(void *devPtr, int value, size_t count) {
	cudaError_t result = cudaErrorInvalidDevicePointer;

	_wait();

	_acquire();
	if (_devices.empty()) {
		return _setLastError(cudaErrorNoDevice);
	}
	
	if (!_getDevice().checkMemoryAccess(devPtr, count)) {
		_release();
		_memoryError(devPtr, count, "cudaMemset");
	}
	
	executive::Device::MemoryAllocation* allocation = 
		_getDevice().getMemoryAllocation(devPtr);
	
	size_t offset = (char*)devPtr - (char*)allocation->pointer();
	
	allocation->memset(offset, value, count);
	result = cudaSuccess;
	
	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaMemset2D(void *devPtr, size_t pitch, 
	int value, size_t width, size_t height) {

	_wait();

	cudaError_t result = cudaErrorInvalidValue;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	executive::Device::MemoryAllocation* allocation = 
		_getDevice().getMemoryAllocation(devPtr);
	
	if (allocation == 0) {
		_release();
		_memoryError(devPtr, width * height, "cudaMemset2D");
	}
		
	size_t offset = (char*)devPtr - (char*)allocation->pointer();
	
	if (pitch == width) {
		if (!_getDevice().checkMemoryAccess(devPtr, width * height)) {
			_release();
			_memoryError(devPtr, width * height, "cudaMemset2D");
		}
		
		allocation->memset(offset, value, pitch * height);
	}
	else {
		for (size_t i = 0; i < height; i++) {
			size_t ptr = offset + pitch * i;
			void* address = (char*)allocation->pointer() + ptr;
			if (!_getDevice().checkMemoryAccess(address, width)) {
				_release();
				_memoryError(address, width, "cudaMemset2D");
			}
		
			allocation->memset(ptr, value, width);
		}
	}

	result = cudaSuccess;
	
	_release();
	return _setLastError(result);	
}

//does not support the use of cudaPitchedPtr.pos!!
cudaError_t cuda::CudaRuntime::cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr,
	int value, struct cudaExtent extent) {

	cudaError_t result = cudaErrorInvalidValue;

	size_t pitch = pitchedDevPtr.pitch;
	//size_t xsize = pitchedDevPtr.xsize;
	size_t ysize = pitchedDevPtr.ysize;
	size_t width = extent.width;
	size_t height = extent.height;
	size_t depth = extent.depth;
	void *ptr;
	for (size_t i = 0; i < depth; i++) {
		ptr = (char*)pitchedDevPtr.ptr + i * pitch * ysize;
		result = cudaMemset2D(ptr, pitch, value, width, height);
	}
	
	return _setLastError(result);	
}

////////////////////////////////////////////////////////////////////////////////
//
// memory allocation
//

cudaError_t cuda::CudaRuntime::cudaGetSymbolAddress(void **devPtr, 
	const char *symbol) {

	_wait();

	report("cuda::CudaRuntime::cudaGetSymbolAddress(" << devPtr << ", " 
		<< (void*)symbol << ")");
	cudaError_t result = cudaSuccess;
	_lock();
	
	_enumerateDevices();
	if (_devices.empty()) {
		_unlock();
		return _setLastError(cudaErrorNoDevice);
	}
	
	RegisteredGlobalMap::iterator global = _globals.find((void*)symbol);
	
	std::string name;
	std::string module;
	
	if (global != _globals.end()) {
		name = global->second.global;
		module = global->second.module;
		
		try {
			_registerModule(module);
		}
		catch(...) {
			_unlock();
			throw;
		}
	}
	else {
		name = symbol;
		_registerAllModules();
	}
	
	_bind();
	
	executive::Device::MemoryAllocation* 
		allocation = _getDevice().getGlobalAllocation(module, name);
	assertM(allocation != 0, "Invalid global name " << name 
		<< " in module " << module);

	*devPtr = allocation->pointer();
	report("devPtr: " << *devPtr);	
	
	_release();
	return _setLastError(result);	
}

cudaError_t cuda::CudaRuntime::cudaGetSymbolSize(size_t *size,
	const char *symbol) {

	_wait();

	cudaError_t result = cudaSuccess;
	report("cuda::CudaRuntime::cudaGetSymbolSize(" << size << ", " 
		<< (void*) symbol << ")");
	
	_lock();
	
	_enumerateDevices();
	if (_devices.empty()) {
		_unlock();
		return _setLastError(cudaErrorNoDevice);
	}
	
	RegisteredGlobalMap::iterator global = _globals.find((void*)symbol);
	
	std::string name;
	std::string module;
	
	if (global != _globals.end()) {
		name = global->second.global;
		module = global->second.module;
		try {
			_registerModule(module);
		}
		catch(...) {
			_unlock();
			throw;
		}
	}
	else {
		name = symbol;
		_registerAllModules();
	}
	
	_bind();
	
	executive::Device::MemoryAllocation* 
		allocation = _getDevice().getGlobalAllocation(module, name);
	assertM(allocation != 0, "Invalid global name " << name 
		<< " in module " << module);

	*size = allocation->size();
	report("size: " << *size);	
	
	_release();
	return _setLastError(result);	
}

////////////////////////////////////////////////////////////////////////////////

cudaError_t cuda::CudaRuntime::cudaGetDeviceCount(int *count) {
	cudaError_t result = cudaSuccess;
	*count = _deviceCount;	
	return _setLastError(result);
}

#define minimum(x, y) ((x) > (y) ? (y) : (x))

cudaError_t cuda::CudaRuntime::cudaGetDeviceProperties(
	struct cudaDeviceProp *prop, int dev) {
	cudaError_t result = cudaSuccess;

	_lock();	
	bool notLoaded = !_devicesLoaded;
	_enumerateDevices();

	if (dev < (int)_devices.size() && dev >= 0) {
		executive::Device& device = *_devices[dev];
		const executive::Device::Properties& properties = device.properties();
	
		report("cuda::CudaRuntime::cudaGetDeviceProperties(dev = " << dev 
			<< ") - major: " << properties.major 
			<< ", minor: " << properties.minor);

		memset(prop, 0, sizeof(struct cudaDeviceProp));
		hydrazine::strlcpy( prop->name, properties.name, 256 );
		prop->canMapHostMemory = 1;
		prop->clockRate = properties.clockRate;
		prop->computeMode = cudaComputeModeDefault;
		prop->deviceOverlap = properties.memcpyOverlap;
		prop->integrated = 0;
		prop->kernelExecTimeoutEnabled = 0;
		prop->major = properties.major;
		prop->minor = properties.minor;
		prop->maxGridSize[0] = properties.maxGridSize[0];
		prop->maxGridSize[1] = properties.maxGridSize[1];
		prop->maxGridSize[2] = properties.maxGridSize[2];
		prop->maxThreadsDim[0] = properties.maxThreadsDim[0];
		prop->maxThreadsDim[1] = properties.maxThreadsDim[1];
		prop->maxThreadsDim[2] = properties.maxThreadsDim[2];
		prop->maxThreadsPerBlock = properties.maxThreadsPerBlock;
		prop->memPitch = properties.memPitch;
		prop->multiProcessorCount = properties.multiprocessorCount;
		prop->regsPerBlock = properties.regsPerBlock;
		prop->sharedMemPerBlock = properties.sharedMemPerBlock;
		prop->textureAlignment = properties.textureAlign;
		prop->totalConstMem = properties.totalConstantMemory;
		prop->totalGlobalMem = properties.totalMemory;
		prop->warpSize = properties.SIMDWidth;
		prop->concurrentKernels = properties.concurrentKernels;

		prop->integrated = properties.integrated;
		prop->unifiedAddressing = properties.unifiedAddressing;
		prop->memoryClockRate = properties.memoryClockRate;
		prop->memoryBusWidth = properties.memoryBusWidth;
		prop->l2CacheSize = properties.l2CacheSize;
		prop->maxThreadsPerMultiProcessor =
			properties.maxThreadsPerMultiProcessor;
		
		report("  returning: prop->major = " << prop->major 
			<< ", prop->minor = " << prop->minor);
		
		result = cudaSuccess;
	}
	else {
		result = cudaErrorInvalidDevice;
	}
	
	// this is a horrible hack needed because cudaGetDeviceProperties can be 
	// called before setflags
	if (notLoaded) {
		_devicesLoaded = false;
		_workers.clear();

		for (DeviceVector::iterator device = _devices.begin(); 
			device != _devices.end(); ++device) {
			delete *device;
		}
		
		_devices.clear();
	}
	
	_unlock();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaChooseDevice(int *device, 
	const struct cudaDeviceProp *prop) {
	cudaError_t result = cudaSuccess;
	_lock();
	_enumerateDevices();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	*device = 0;
	_unlock();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaSetDevice(int device) {
	cudaError_t result = cudaErrorInvalidDevice;
	
	_lock();

	if ((int)_deviceCount > device && device >= 0) {
		HostThreadContext& thread = _getCurrentThread();
		thread.selectedDevice = device;
		report("Setting device for thread " 
			<< boost::this_thread::get_id() << " to " 
			<< device);
		result = cudaSuccess;
	}

	_unlock();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGetDevice(int *device) {
	cudaError_t result = cudaSuccess;
	
	_lock();
	HostThreadContext& thread = _getCurrentThread();
	*device = thread.selectedDevice;
	_unlock();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaSetValidDevices(int *device_arr, int len) {
	cudaError_t result = cudaSuccess;
	_lock();
	HostThreadContext& thread = _getCurrentThread();
	thread.validDevices.resize(len);
	for (int i = 0 ; i < len; i++) {
		thread.validDevices[i] = device_arr[i];
	}
	_unlock();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaSetDeviceFlags(int f) {
	cudaError_t result = cudaSuccess;
	_lock();
	if(!_devicesLoaded)
	{
		_flags = f;
	}
	else
	{
		result = cudaErrorSetOnActiveProcess;
	}
	_unlock();
	return _setLastError(result);
}

////////////////////////////////////////////////////////////////////////////////

//! binds a texture to a reference and a CUDA memory block
cudaError_t cuda::CudaRuntime::cudaBindTexture(size_t *offset, 
	const struct textureReference *texref, const void *devPtr, 
	const struct cudaChannelFormatDesc *desc, size_t size) {

	_wait();

	cudaError_t result = cudaErrorInvalidValue;
		
	_lock();
	
	_enumerateDevices();
	if (_devices.empty()) {
		_unlock();
		return _setLastError(cudaErrorNoDevice);
	}

	RegisteredTextureMap::iterator texture = _textures.find((void*)texref);
	if(texture != _textures.end()) {
		try {
			_registerModule(texture->second.module);
		}
		catch(...) {
			_unlock();
			throw;
		}
		
		_bind();
		try {
			_getDevice().bindTexture((void*)devPtr, texture->second.module, 
				texture->second.texture, *texref, *desc, ir::Dim3(size, 1, 1));
			if(offset != 0) *offset = 0;
			result = cudaSuccess;
		}
		catch(hydrazine::Exception&) {
		
		}
		
		_unbind();
	}
	
	report("cudaBindTexture(ref = " << texref 
		<< ", devPtr = " << devPtr << ", size = " << size << ")");

	_unlock();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaBindTexture2D(size_t *offset,
	const struct textureReference *texref, const void *devPtr, 
	const struct cudaChannelFormatDesc *desc, size_t width, 
	size_t height, size_t pitch) {

	_wait();

	cudaError_t result = cudaErrorInvalidValue;
	assert(pitch != 0);

	_lock();
	
	_enumerateDevices();
	if (_devices.empty()) {
		_unlock();
		return _setLastError(cudaErrorNoDevice);
	}

	RegisteredTextureMap::iterator texture = _textures.find((void*)texref);

	if(texture != _textures.end()) {
		try {
			_registerModule(texture->second.module);
		}
		catch(...) {
			_unlock();
			throw;
		}
		_bind();
		try {
			_getDevice().bindTexture((void*)devPtr, texture->second.module, 
				texture->second.texture, *texref, *desc, 
				ir::Dim3(width, height, 1));
			if(offset != 0) *offset = 0;
			result = cudaSuccess;
		}
		catch(hydrazine::Exception&) {

		}
		_unbind();
	}
	
	report("cudaBindTexture2D(ref = " << texref 
		<< ", devPtr = " << devPtr << ", width = " << width << ", height = " 
		<< height << ", pitch = " << pitch << ")");

	_unlock();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaBindTextureToArray(
	const struct textureReference *texref, const struct cudaArray *array, 
	const struct cudaChannelFormatDesc *desc) {
	cudaError_t result = cudaErrorInvalidValue;

	_wait();
	
	_lock();
	
	_enumerateDevices();
	if (_devices.empty()) {
		_unlock();
		return _setLastError(cudaErrorNoDevice);
	}

	report("cudaBindTextureToArray() - texref = '" << texref << "', array = " 
		<< (void *)array << " - format: " << desc->f << " (" << desc->x 
		<< ", " << desc->y << ", " << desc->z << ", " << desc->w << ")");

	DimensionMap::iterator dimension = _dimensions.find((void*)array);
	assert(dimension != _dimensions.end());

	ir::Dim3 size(dimension->second.x, dimension->second.y, 
		dimension->second.z);

	RegisteredTextureMap::iterator texture = _textures.find((void*)texref);
	if(texture != _textures.end()) {
		try {
			_registerModule(texture->second.module);
		}
		catch(...) {
			_unlock();
			throw;
		}
		_bind();
		try {
			_getDevice().bindTexture((void*)array, texture->second.module, 
				texture->second.texture, *texref, *desc, size);
			result = cudaSuccess;
		}
		catch(hydrazine::Exception&) {

		}
		_unbind();
	}
	
	_unlock();
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaUnbindTexture(
	const struct textureReference *texref) {
	cudaError_t result = cudaErrorInvalidValue;

	_wait();
	
	_lock();
	
	_enumerateDevices();
	if (_devices.empty()) {
		_unlock();
		return _setLastError(cudaErrorNoDevice);
	}

	RegisteredTextureMap::iterator texture = _textures.find((void*)texref);
	if(texture != _textures.end()) {
		try {
			_registerModule(texture->second.module);
		}
		catch(...) {
			_unlock();
			throw;
		}
		_bind();
		try {
			_getDevice().unbindTexture(texture->second.module, 
				texture->second.texture);
			result = cudaSuccess;
		}
		catch(hydrazine::Exception&) {

		}
		_unbind();
	}
	
	_unlock();

	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGetTextureAlignmentOffset(size_t *offset, 
	const struct textureReference *texref) {
	*offset = 0;
	return _setLastError(cudaSuccess);
}

cudaError_t cuda::CudaRuntime::cudaGetTextureReference(
	const struct textureReference **texref, const char *symbol) {
	cudaError_t result = cudaErrorInvalidTexture;

	_lock();
	
	std::string name = symbol;
	RegisteredTextureMap::iterator matchedTexture = _textures.end();
	
	for(RegisteredTextureMap::iterator texture = _textures.begin(); 
		texture != _textures.end(); ++texture) {
		if(texture->second.texture == name) {
			if(matchedTexture != _textures.end()) {
				_unlock();
				Ocelot_Exception("==Ocelot== Ambiguous texture '" << name 
					<< "' declared in at least two modules ('" 
					<< texture->second.module << "' and '" 
					<< matchedTexture->second.module << "')");
			}
			matchedTexture = texture;
		}
	}
	
	if(matchedTexture != _textures.end()) {
		*texref = (const struct textureReference*)matchedTexture->first;
	}
	
	_unlock();
	return _setLastError(result);
}

////////////////////////////////////////////////////////////////////////////////

cudaError_t cuda::CudaRuntime::cudaGetChannelDesc(
	struct cudaChannelFormatDesc *desc, const struct cudaArray *array) {
	cudaError_t result = cudaErrorInvalidValue;

	_lock();

	DimensionMap::iterator dimension = _dimensions.find((void*)array);
	
	if (dimension != _dimensions.end()) {
		*desc = dimension->second.format;
		result = cudaSuccess;
	}
	
	_unlock();
	return _setLastError(result);
}

struct cudaChannelFormatDesc cuda::CudaRuntime::cudaCreateChannelDesc(int x, 
	int y, int z, int w, enum cudaChannelFormatKind f) {

	struct cudaChannelFormatDesc desc;
	desc.w = w; desc.x = x; desc.y = y; desc.z = z; desc.f = f;
	return desc;
}

////////////////////////////////////////////////////////////////////////////////

cudaError_t cuda::CudaRuntime::cudaGetLastError(void) {
	HostThreadContext& thread = _getCurrentThread();
	cudaError_t lastError = thread.lastError;
	thread.lastError = cudaSuccess;
	return lastError;
}

cudaError_t cuda::CudaRuntime::cudaPeekAtLastError(void) {
	HostThreadContext& thread = _getCurrentThread();
	return thread.lastError;
}

////////////////////////////////////////////////////////////////////////////////
	
cudaError_t cuda::CudaRuntime::cudaConfigureCall(dim3 gridDim, dim3 blockDim, 
	size_t sharedMem, cudaStream_t stream) {

	_lock();
	report("cudaConfigureCall()");
	
	cudaError_t result = cudaErrorInvalidConfiguration;
	HostThreadContext &thread = _getCurrentThread();
	
	KernelLaunchConfiguration launch(gridDim, blockDim, sharedMem, stream);
	thread.launchConfigurations.push_back(launch);
	result = cudaSuccess;
	
	_unlock();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaSetupArgument(const void *arg, size_t size, 
	size_t offset) {
	cudaError_t result = cudaSuccess;
	
	_lock();
	
	HostThreadContext &thread = _getCurrentThread();

	report("cudaSetupArgument() - offset " << offset << ", size " << size);
	
	memcpy(thread.parameterBlock + offset, arg, size);
	
	thread.parameterIndices.push_back(offset);
	thread.parameterSizes.push_back(size);
	
	_unlock();
	
	return _setLastError(result);
}

static ir::Dim3 convert(const dim3& d) {
	return std::move(ir::Dim3(d.x, d.y, d.z));
}

cudaError_t cuda::CudaRuntime::_launchKernel(const std::string& moduleName, 
	const std::string& kernelName )
{
	_lock();

	_enumerateDevices();
	if (_devices.empty()) {
		_unlock();
		return _setLastError(cudaErrorNoDevice);
	}
	
	ModuleMap::iterator module = _modules.find(moduleName);
	assert(module != _modules.end());

	try {
		_registerModule(module);
	}
	catch(...) {
		_unlock();
		throw;
	}

	ir::Kernel* k = module->second.getKernel(kernelName);
	assert(k != 0);

	_bind();

	HostThreadContext& thread = _getCurrentThread();
	cudaError_t result = cudaSuccess;
	
	assert(thread.launchConfigurations.size());
	
	KernelLaunchConfiguration launch(thread.launchConfigurations.back());
	thread.launchConfigurations.pop_back();
	
	unsigned int paramSize = thread.mapParameters(k);
	
	report("kernel launch (" << kernelName 
		<< ") on thread " << boost::this_thread::get_id());
	
	try {
		trace::TraceGeneratorVector traceGens;

		traceGens = _persistentTraceGenerators;
		traceGens.insert(traceGens.end(),
			_nextTraceGenerators.begin(), 
			_nextTraceGenerators.end());

		if (config::get().executive.asynchronousKernelLaunch) {
			_getWorkerThread().launch(moduleName, kernelName,
				convert(launch.gridDim), 
				convert(launch.blockDim), launch.sharedMemory, 
				thread.parameterBlock, paramSize, traceGens, &_externals);
		}
		else {
			_getDevice().launch(moduleName, kernelName, convert(launch.gridDim),
				convert(launch.blockDim), launch.sharedMemory,
				thread.parameterBlock, paramSize, traceGens, &_externals);
		}
		report(" launch completed successfully");	
	}
	catch( const executive::RuntimeException& e ) {
		std::cerr << "==Ocelot== PTX Emulator failed to run kernel \"" 
			<< kernelName 
			<< "\" with exception: \n";
		std::cerr << _formatError( e.toString() ) 
			<< "\n" << std::flush;
		thread.lastError = cudaErrorLaunchFailure;
		_release();
		throw;
	}
	catch( const std::exception& e ) {
		std::cerr << "==Ocelot== " << _getDevice().properties().name
			<< " failed to run kernel \""
			<< kernelName
			<< "\" with exception: \n";
		std::cerr << _formatError( e.what() )
			<< "\n" << std::flush;
		thread.lastError = cudaErrorLaunchFailure;
		_release();
		throw;
	}
	catch(...) {
		thread.lastError = cudaErrorLaunchFailure;
		_release();
		throw;
	}
	_release();
	
	_wait();
	
	return result;
}

cudaError_t cuda::CudaRuntime::cudaLaunch(const char *entry) {
	_lock();
	
	RegisteredKernelMap::iterator kernel = _kernels.find((void*)entry);
	assert(kernel != _kernels.end());
	std::string moduleName = kernel->second.module;
	std::string kernelName = kernel->second.kernel;

	_unlock();

	cudaError_t result = _launchKernel(moduleName, kernelName);
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaFuncGetAttributes(
	struct cudaFuncAttributes *attr, const char *symbol) {
	cudaError_t result = cudaErrorInvalidDeviceFunction;		

	_wait();
	_lock();

	_enumerateDevices();
	if (_devices.empty()) {
		_unlock();
		return _setLastError(cudaErrorNoDevice);
	}
	
	//
	// go find the kernel and fill out its attributes
	//
	RegisteredKernelMap::iterator kernel = _kernels.find((void*)symbol);
	if (kernel != _kernels.end()) {
		try {
			_registerModule(kernel->second.module);
		}
		catch(...) {
			_unlock();
			throw;
		}
		
		_bind();

		*attr = _getDevice().getAttributes(kernel->second.module, 
			kernel->second.kernel);
		result = cudaSuccess;

		_release();
	}
	else {
		_unlock();
	}
	
	return _setLastError(result);
}

static executive::ExecutableKernel::CacheConfiguration
	_translateCacheConfiguration(enum cudaFuncCache config) {
	switch (config) {
		case cudaFuncCachePreferShared:
			return executive::ExecutableKernel::CachePreferShared;
		case cudaFuncCachePreferL1:
			return executive::ExecutableKernel::CachePreferL1;
		default:
			break;
	}
	return executive::ExecutableKernel::CacheConfigurationDefault;
}

cudaError_t cuda::CudaRuntime::cudaFuncSetCacheConfig(const char *entry, 
	enum cudaFuncCache cacheConfig)
{
	cudaError_t result = cudaSuccess;
    _wait();
    _lock();
	_enumerateDevices();
	if (_devices.empty()) {
		_unlock();
		return _setLastError(cudaErrorInitializationError);
	}

	RegisteredKernelMap::iterator kernel = _kernels.find((void*)entry);
	if (kernel != _kernels.end()) {
		try {
			_registerModule(kernel->second.module);
		}
		catch(...) {
			_unlock();
			throw;
		}

	    _bind();

		executive::ExecutableKernel *executableKernel =
			_getDevice().getKernel(kernel->second.module,
				kernel->second.kernel);
		executableKernel->setCacheConfiguration(
			_translateCacheConfiguration(cacheConfig));
		
		result = cudaSuccess;

		_release();
	}
	else
		_unlock();

	return _setLastError(result);
}

////////////////////////////////////////////////////////////////////////////////
//
// CUDA event creation

cudaError_t cuda::CudaRuntime::cudaEventCreate(cudaEvent_t *event) {
	cudaError_t result = cudaSuccess;
	
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	*event = _getDevice().createEvent( 0 );
	
	_release();

	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventCreateWithFlags(cudaEvent_t *event, 
	int flags) {
	cudaError_t result = cudaSuccess;
	
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	*event = _getDevice().createEvent(flags);
	
	_release();

	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventRecord(cudaEvent_t event, 
	cudaStream_t stream) {
	cudaError_t result = cudaErrorInvalidValue;
	
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	try {
		_getDevice().recordEvent(event, stream);
		result = cudaSuccess;
	}
	catch(...) {
	
	}
	
	_release();	
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventQuery(cudaEvent_t event) {
	cudaError_t result = cudaErrorInvalidValue;

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	try {
		if (_getDevice().queryEvent(event)) {
		
			if(!config::get().executive.asynchronousKernelLaunch ||
				!_getWorkerThread().areAnyKernelsRunning())
			{
				result = cudaSuccess;
			}
			else
			{
				result = cudaErrorNotReady;
			}
		}
		else {
			result = cudaErrorNotReady;
		}
	}
	catch(...) {
	
	}
	
	_release();

	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventSynchronize(cudaEvent_t event) {
	cudaError_t result = cudaErrorInvalidValue;
	
	_wait();
	
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	try {
		_getDevice().synchronizeEvent(event);
		result = cudaSuccess;
	}
	catch(...) {
	
	}
	
	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventDestroy(cudaEvent_t event) {
	cudaError_t result = cudaErrorInvalidValue;
	
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	try {
		_getDevice().destroyEvent(event);
		result = cudaSuccess;
	}
	catch(...) {
	
	}	
	
	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaEventElapsedTime(float *ms, 
	cudaEvent_t start, cudaEvent_t end) {
	cudaError_t result = cudaErrorInvalidValue;
	
	_wait();
	
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	try {
		*ms = _getDevice().getEventTime(start, end);
		result = cudaSuccess;
	}
	catch(...) {

	}

	_release();
	
	return _setLastError(result);
}

////////////////////////////////////////////////////////////////////////////////
//
// CUDA streams
//

cudaError_t cuda::CudaRuntime::cudaStreamCreate(cudaStream_t *pStream) {
	cudaError_t result = cudaErrorInvalidValue;
	
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	try {
		*pStream = _getDevice().createStream();
		result = cudaSuccess;
	}
	catch (...) {
	
	}
	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaStreamDestroy(cudaStream_t stream) {
	cudaError_t result = cudaErrorInvalidValue;
	
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	try {
		_getDevice().destroyStream(stream);
		result = cudaSuccess;
	}
	catch(...) {
	
	}
	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaStreamSynchronize(cudaStream_t stream) {
	cudaError_t result = cudaErrorInvalidValue;

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	try {
		_getDevice().synchronizeStream(stream);
		result = cudaSuccess;
	}
	catch(...) {
	
	}
	
	_release();

	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaStreamQuery(cudaStream_t stream) {
	cudaError_t result = cudaErrorInvalidValue;

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	try {
		_getDevice().queryStream(stream);
		result = cudaSuccess;
	}
	catch(...) {
	
	}
	
	_release();
	
	return _setLastError(result);
}

////////////////////////////////////////////////////////////////////////////////
cudaError_t cuda::CudaRuntime::cudaDriverGetVersion(int *driverVersion) {
	cudaError_t result = cudaSuccess;

	_lock();	
	bool notLoaded = !_devicesLoaded;
	_enumerateDevices();

	if (_devices.empty()) { 
		result = cudaErrorNoDevice;
	}
	else {
		*driverVersion = _devices[0]->driverVersion();
	}
	
	// this is a horrible hack needed because this can be 
	// called before setflags
	if (notLoaded) {
		_devicesLoaded = false;
		_workers.clear();
		for (DeviceVector::iterator device = _devices.begin(); 
			device != _devices.end(); ++device) {
			delete *device;
		}
		_devices.clear();
	}
	
	_unlock();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaRuntimeGetVersion(int *runtimeVersion) {
	cudaError_t result = cudaSuccess;

	_lock();	
	bool notLoaded = !_devicesLoaded;
	_enumerateDevices();

	if (_devices.empty()) { 
		result = cudaErrorNoDevice;
	}
	else {
		*runtimeVersion = _devices[0]->runtimeVersion();
	}
	
	// this is a horrible hack needed because this can be 
	// called before setflags, it creates the devices, gets their attributes
	// then deletes them at the end of the function so that there is still
	// no set device
	if (notLoaded) {
		_devicesLoaded = false;
		_workers.clear();
		for (DeviceVector::iterator device = _devices.begin(); 
			device != _devices.end(); ++device) {
			delete *device;
		}
		_devices.clear();
	}
	
	_unlock();
	
	return _setLastError(result);
}


////////////////////////////////////////////////////////////////////////////////
cudaError_t cuda::CudaRuntime::cudaDeviceReset(void) {

    _lock();
   
    _devicesLoaded = false;
    _workers.clear(); 
	for (DeviceVector::iterator device = _devices.begin(); 
		device != _devices.end(); ++device) {
		delete *device;
	}
    _devices.clear();

    _unlock();
    return _setLastError(cudaSuccess);
}

cudaError_t cuda::CudaRuntime::cudaDeviceSynchronize(void) {
	return cudaThreadSynchronize();
}

cudaError_t cuda::CudaRuntime::cudaDeviceSetLimit(enum cudaLimit limit,
	size_t value) {
	return CudaRuntimeInterface::cudaDeviceSetLimit(limit, value);
}

cudaError_t cuda::CudaRuntime::cudaDeviceGetLimit(size_t *pValue,
	enum cudaLimit limit) {
	return CudaRuntimeInterface::cudaDeviceGetLimit(pValue, limit);
}

cudaError_t cuda::CudaRuntime::cudaDeviceGetCacheConfig(
	enum cudaFuncCache *pCacheConfig) {
	return CudaRuntimeInterface::cudaDeviceGetCacheConfig(pCacheConfig);
}

cudaError_t cuda::CudaRuntime::cudaDeviceSetCacheConfig(
	enum cudaFuncCache cacheConfig) {
	return CudaRuntimeInterface::cudaDeviceSetCacheConfig(cacheConfig);
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////

cudaError_t cuda::CudaRuntime::cudaThreadExit(void) {
	cudaError_t result = cudaSuccess;

	_wait();
	
	_lock();

	_workers.clear();

	report("Destroying " << _devices.size() << " devices");
	for (DeviceVector::iterator device = _devices.begin(); 
		device != _devices.end(); ++device) {
		report( " Destroying - " << (*device)->properties().name);
		delete *device;
	}
	_devices.clear();
	
	_devicesLoaded = false;
	_unlock();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaThreadSynchronize(void) {
	cudaError_t result = cudaSuccess;
	_wait();
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	_getDevice().synchronize();

	_release();
	
	return _setLastError(result);
}

////////////////////////////////////////////////////////////////////////////////

cudaError_t cuda::CudaRuntime::cudaGLMapBufferObject(void **devPtr, 
	GLuint bufObj) {
	return cudaGLMapBufferObjectAsync(devPtr, bufObj, 0);
}

cudaError_t cuda::CudaRuntime::cudaGLMapBufferObjectAsync(void **devPtr, 
	GLuint bufObj, cudaStream_t stream) {
	cudaError_t result = cudaSuccess;
	
	_wait();

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaGLMapBufferObjectAsync(" << bufObj << ", " << stream << ")");
	GLBufferMap::iterator buffer = _buffers.find(bufObj);
	if (buffer != _buffers.end()) {

		_getDevice().mapGraphicsResource(& buffer->second, 1, stream);
		
		size_t bytes = 0;
		
		// semantics of this are questionable
		*devPtr = _getDevice().getPointerToMappedGraphicsResource(
			bytes, buffer->second);
		result = cudaSuccess;
	}
	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGLRegisterBufferObject(GLuint bufObj) {
	cudaError_t result = cudaErrorInvalidValue;

	_wait();

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaGLRegisterBufferObject(" << bufObj << ")");	

	if (_buffers.count(bufObj) == 0) {
		void* graphic = _getDevice().glRegisterBuffer(bufObj, 0);
		_buffers.insert(std::make_pair(bufObj, graphic));
		result = cudaSuccess;
	}
	
	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGLSetBufferObjectMapFlags(GLuint bufObj, 
	unsigned int flags) {
	cudaError_t result = cudaErrorInvalidValue;

	_wait();

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaGLRegisterBufferObjectMapFlags(" << bufObj << ")");	

	if (_buffers.count(bufObj) == 0) {
		void* graphic = _getDevice().glRegisterBuffer(bufObj, flags);
		_buffers.insert(std::make_pair(bufObj, graphic));
		result = cudaSuccess;
	}
	
	_release();
	
	return _setLastError(result);

}

cudaError_t cuda::CudaRuntime::cudaGLSetGLDevice(int device) {
	report("cudaGLSetGLDevice(" << device << ")");
	return cudaSetDevice(device);
}

cudaError_t cuda::CudaRuntime::cudaGLUnmapBufferObject(GLuint bufObj) {
	cudaError_t result = cudaErrorInvalidValue;

	_wait();

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaGLUnmapBufferObject(" << bufObj << ")");
	
	GLBufferMap::iterator buffer = _buffers.find(bufObj);
	if (buffer != _buffers.end()) {
		_getDevice().unmapGraphicsResource(& buffer->second, 1, 0);
		result = cudaSuccess;
	}
	
	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGLUnmapBufferObjectAsync(GLuint bufObj, 
	cudaStream_t stream) {
	return cudaGLUnmapBufferObject(bufObj);
}

cudaError_t cuda::CudaRuntime::cudaGLUnregisterBufferObject(GLuint bufObj) {

	_wait();

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	cudaError_t result = cudaErrorInvalidValue;

	report("cudaGLUnregisterBufferObject");
	
	GLBufferMap::iterator buffer = _buffers.find(bufObj);
	if (buffer != _buffers.end()) {
		_getDevice().unRegisterGraphicsResource(buffer->second);	
		_buffers.erase(buffer);
		result = cudaSuccess;
	}
	
	_release();
	
	return _setLastError(result);
}

////////////////////////////////////////////////////////////////////////////////

cudaError_t cuda::CudaRuntime::cudaGraphicsGLRegisterBuffer(
	struct cudaGraphicsResource **resource, GLuint buffer, unsigned int flags) {
	cudaError_t result = cudaSuccess;
	
	_wait();

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	report("cudaGraphicsGLRegisterBuffer");

	*resource = (struct cudaGraphicsResource*)_getDevice().glRegisterBuffer(
		buffer, flags);

	_release();
	
	return _setLastError(result);	
}

cudaError_t cuda::CudaRuntime::cudaGraphicsGLRegisterImage(
	struct cudaGraphicsResource **resource, GLuint image, int target, 
	unsigned int flags) {
	cudaError_t result = cudaSuccess;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	report("cudaGraphicsGLRegisterImage");
	
	*resource = (struct cudaGraphicsResource*)_getDevice().glRegisterImage(
		image, target, flags);

	_release();
	
	return _setLastError(result);	
}

cudaError_t cuda::CudaRuntime::cudaGraphicsUnregisterResource(
	struct cudaGraphicsResource *resource) {
	cudaError_t result = cudaSuccess;
	
	_wait();

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaGraphicsUnregisterResource");
	
	_getDevice().unRegisterGraphicsResource(resource);	

	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGraphicsResourceSetMapFlags(
	struct cudaGraphicsResource *resource, unsigned int flags) {
	cudaError_t result = cudaSuccess;
	
	_wait();

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaGraphicsResourceSetMapFlags");
	
	_getDevice().setGraphicsResourceFlags(resource, flags);	

	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGraphicsMapResources(int count, 
	struct cudaGraphicsResource **resource, cudaStream_t stream) {
	cudaError_t result = cudaSuccess;
	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);
	
	report("mapGraphicsResource");
	
	_getDevice().mapGraphicsResource((void **)resource, count, stream);	

	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGraphicsUnmapResources(int count, 
	struct cudaGraphicsResource **resources, cudaStream_t stream) {
	cudaError_t result = cudaSuccess;
	
	_wait();

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaGraphicsUnmapResources");
	
	_getDevice().unmapGraphicsResource((void **)resources, count, stream);	

	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGraphicsResourceGetMappedPointer(
	void **devPtr, size_t *size, struct cudaGraphicsResource *resource) {
	cudaError_t result = cudaSuccess;
	
	_wait();

	_acquire();
	if (_devices.empty()) return _setLastError(cudaErrorNoDevice);

	report("cudaGraphicsResourceGetMappedPointer");
	
	*devPtr = _getDevice().getPointerToMappedGraphicsResource(*size, resource);	

	_release();
	
	return _setLastError(result);
}

cudaError_t cuda::CudaRuntime::cudaGraphicsSubResourceGetMappedArray(
	struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, 
	unsigned int arrayIndex, unsigned int mipLevel) {
	assertM(false, "Not implemented.");
	return cudaSuccess;
}


////////////////////////////////////////////////////////////////////////////////

void cuda::CudaRuntime::addTraceGenerator( trace::TraceGenerator& gen,
	bool persistent ) {
	_lock();

	if (persistent) {
		_persistentTraceGenerators.push_back(&gen);
	}
	else {
		_nextTraceGenerators.push_back(&gen);
	}

	_unlock();
}

void cuda::CudaRuntime::clearTraceGenerators() {
	_lock();
	
	_persistentTraceGenerators.clear();
	_nextTraceGenerators.clear();
	
	_unlock();
}

void cuda::CudaRuntime::addPTXPass(transforms::Pass &pass) {
	_lock();
	_passes.insert(&pass);
	_unlock();
}

void cuda::CudaRuntime::removePTXPass(transforms::Pass &pass) {
	_lock();

	assert(_passes.count(&pass) != 0);
	_passes.erase(&pass);

	_unlock();
}

void cuda::CudaRuntime::clearPTXPasses() {
	_lock();
	_passes.clear();
	_unlock();
}

void cuda::CudaRuntime::limitWorkerThreads(unsigned int limit) {
	_wait();

	_acquire();

	for (DeviceVector::iterator device = _devices.begin(); 
		device != _devices.end(); ++device) {
		(*device)->limitWorkerThreads(limit);
	}
	_release();
}

void cuda::CudaRuntime::registerPTXModule(std::istream& ptx, 
	const std::string& name) {
	_wait();
	_lock();
	report("Loading module (ptx) - " << name);
	assert(_modules.count(name) == 0);
	
	ModuleMap::iterator module = _modules.insert(
		std::make_pair(name, ir::Module())).first;
	
	std::string temp;
	
	ptx.seekg(0, std::ios::end);
	size_t size = ptx.tellg();
	ptx.seekg(0, std::ios::beg);
	
	temp.resize(size);
	ptx.read((char*)temp.data(), size);
	
	try {
		module->second.lazyLoad(temp, name);
	}
	catch(...) {
		_unlock();
		_modules.erase(module);
		throw;
	}
		
	_unlock();
}

void cuda::CudaRuntime::registerTexture(const void* texref,
	const std::string& moduleName,
	const std::string& textureName, bool normalize) {
	_lock();
	
	report("registerTexture('" << textureName << ", norm: " << normalize );

	_textures[(void*)texref] = RegisteredTexture(moduleName,
		textureName, normalize);
	
	_unlock();
}

void cuda::CudaRuntime::clearErrors() {
	_lock();
	HostThreadContext& thread = _getCurrentThread();
	thread.lastError = cudaSuccess;
	_unlock();
}

void cuda::CudaRuntime::reset() {
	_wait();

	_lock();

	report("Resetting cuda runtime.");
	HostThreadContext& thread = _getCurrentThread();
	thread.clear();
	_dimensions.clear();
	
	for(DeviceVector::iterator device = _devices.begin(); 
		device != _devices.end(); ++device)
	{
		report(" Clearing memory on device - " << (*device)->properties().name);
		(*device)->clearMemory();
	}
	
	for(ModuleMap::iterator module = _modules.begin(); module != _modules.end();
		module != _modules.end())
	{
		bool found = false;
		report(" Unloading module - " << module->first);
		for(FatBinaryMap::iterator bin = _fatBinaries.begin(); 
			bin != _fatBinaries.end(); ++bin)
		{
			if(bin->second.name() == module->first)
			{
				found = true;
				break;
			}
		}
		
		if(!found)
		{
			for(DeviceVector::iterator device = _devices.begin(); 
				device != _devices.end(); ++device)
			{
				(*device)->select();
				(*device)->unload(module->first);
				(*device)->unselect();
			}
			
			_modules.erase(module++);
		}
		else
		{
			++module;
		}
	}
	_unlock();
}

ocelot::PointerMap cuda::CudaRuntime::contextSwitch(unsigned int destinationId, 
	unsigned int sourceId) {
	report("Context switching from " << sourceId << " to " << destinationId);
	
	if(!_devicesLoaded) return ocelot::PointerMap();
	
	ocelot::PointerMap mappings;

	_wait();

	_acquire();
	
	if(sourceId >= _devices.size())
	{
		_release();
		Ocelot_Exception("Invalid source device - " << sourceId);
	}
	
	if(destinationId >= _devices.size())
	{
		_release();
		Ocelot_Exception("Invalid destination device - " << destinationId);
	}

	executive::Device& source = *_devices[sourceId];
	executive::Device& destination = *_devices[destinationId];
	
	_unbind();
	
	source.select();
	executive::Device::MemoryAllocationVector sourceAllocations = 
		source.getAllAllocations();
	source.unselect();
		
	for(executive::Device::MemoryAllocationVector::iterator 
		allocation = sourceAllocations.begin();
		allocation != sourceAllocations.end(); ++allocation)
	{
		size_t size = (*allocation)->size();
		void* pointer = (*allocation)->pointer();
		
		if(!(*allocation)->global())
		{
			char* temp = new char[size];
			source.select();
			(*allocation)->copy(temp, 0, size);
			source.free(pointer);
			source.unselect();

			destination.select();
			executive::Device::MemoryAllocation* dest = destination.allocate(
				size);
			dest->copy(0, temp, size);
			destination.unselect();
			
			report(" Mapping device allocation at " << pointer 
				<< " to " << dest->pointer());
			mappings.insert(std::make_pair(pointer,	dest->pointer()));
			delete[] temp;
		}
		else if((*allocation)->host())
		{
			destination.select();
			executive::Device::MemoryAllocation* dest = 
				destination.allocateHost(size, (*allocation)->flags());
			dest->copy(0, pointer, size);
			destination.unselect();

			mappings.insert(std::make_pair(pointer, dest->pointer()));
			
			source.select();
			source.free(pointer);
			source.unselect();
		}
	}

	for(ModuleMap::iterator module = _modules.begin(); 
		module != _modules.end(); ++module)
	{
		if( !module->second.loaded() ) continue;
		for(ir::Module::GlobalMap::const_iterator 
			global = module->second.globals().begin();
			global != module->second.globals().end(); ++global)
		{
			source.select();
			executive::Device::MemoryAllocation* sourceGlobal = 
				source.getGlobalAllocation(module->first, global->first);
			assert(sourceGlobal != 0);
			source.unselect();

			destination.select();
			executive::Device::MemoryAllocation* destinationGlobal = 
				destination.getGlobalAllocation(module->first, global->first);
			assert(destinationGlobal != 0);
			destination.unselect();
			
			char* temp = new char[sourceGlobal->size()];
			source.select();
			sourceGlobal->copy(temp, 0, sourceGlobal->size());
			source.unselect();

			destination.select();
			destinationGlobal->copy(0, temp, destinationGlobal->size());
			destination.unselect();
			delete[] temp;
		}
	}
		
	_unlock();
	
	return mappings;
}

void cuda::CudaRuntime::unregisterModule(const std::string& name) {
	
	_wait();

	_lock();

	ModuleMap::iterator module = _modules.find(name);
	if (module == _modules.end()) {
		_unlock();
		Ocelot_Exception("Module - " << name << " - is not registered.");
	}

	for (DeviceVector::iterator device = _devices.begin(); 
		device != _devices.end(); ++device) {
		(*device)->select();
		(*device)->unload(name);
		(*device)->unselect();
	}
	
	_modules.erase(module);
	
	_unlock();
}

void cuda::CudaRuntime::launch(const std::string& moduleName, 
	const std::string& kernelName) {
	_launchKernel(moduleName, kernelName);
}

void cuda::CudaRuntime::setOptimizationLevel(
	translator::Translator::OptimizationLevel l) {
	_wait();

	_lock();

	_optimization = l;
	for (DeviceVector::iterator device = _devices.begin(); 
		device != _devices.end(); ++device) {
		(*device)->select();
		(*device)->setOptimizationLevel(l);
		(*device)->unselect();
	}

	_unlock();
}

void cuda::CudaRuntime::registerExternalFunction(const std::string& name,
	void* function) {

	_wait();
	
	_lock();

	report("Adding external function '" << name << "'");
	_externals.add(name, function);

	_unlock();
}

void cuda::CudaRuntime::removeExternalFunction(const std::string& name) {
	_wait();

	_lock();

	report("Removing external function '" << name << "'");

	_externals.remove(name);

	_unlock();	
}

bool cuda::CudaRuntime::isExternalFunction(const std::string& name) {
	_lock();

	bool isExternal = _externals.find(name) != 0;

	_unlock();	
	
	return isExternal;
}

void cuda::CudaRuntime::getDeviceProperties(executive::DeviceProperties &properties, 
	int deviceIndex) {

	_lock();	
	bool notLoaded = !_devicesLoaded;
	_enumerateDevices();

	if (deviceIndex < 0) {
		HostThreadContext& thread = _getCurrentThread();
		deviceIndex = thread.selectedDevice;
	}
	
	assert(deviceIndex >=0 && deviceIndex < (int)_devices.size());
	properties = _devices[deviceIndex]->properties();
	
	// this is a horrible hack needed because cudaGetDeviceProperties can be 
	// called before setflags
	if (notLoaded) {
		_devicesLoaded = false;
		_workers.clear();

		for (DeviceVector::iterator device = _devices.begin(); 
			device != _devices.end(); ++device) {
			delete *device;
		}
		
		_devices.clear();
	}
	_unlock();	
}

////////////////////////////////////////////////////////////////////////////////

