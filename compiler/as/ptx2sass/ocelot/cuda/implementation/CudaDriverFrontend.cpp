/*!
	\file CudaDriverFrontend.cpp

	\author Andrew Kerr <arkerr@gatech.edu>
	\brief implements a CUDA Driver API front-end interface for GPU Ocelot
	\date Sept 16 2010
	\location somewhere over western Europe
*/

// C standard library includes
#include <assert.h>

// C++ standard library includes
#include <sstream>
#include <algorithm>

// Ocelot includes
#include <ocelot/cuda/interface/cuda_internal.h>
#include <ocelot/api/interface/OcelotConfiguration.h>
#include <ocelot/cuda/interface/CudaRuntimeContext.h>
#include <ocelot/cuda/interface/CudaDriverFrontend.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/executive/interface/RuntimeException.h>
#include <ocelot/executive/interface/Device.h>
#include <ocelot/executive/interface/ExecutableKernel.h>

// Hydrazine includes
#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/string.h>
#include <hydrazine/interface/debug.h>

// GL includes
//#include <GL/glew.h>

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

////////////////////////////////////////////////////////////////////////////////
//
// Error handling macros

#define Ocelot_Exception(x) { std::stringstream ss; ss << x; \
	throw hydrazine::Exception(ss.str()); }

////////////////////////////////////////////////////////////////////////////////

typedef api::OcelotConfiguration config;

cuda::CudaDriverFrontend *cuda::CudaDriverFrontend::_instance = 0;

cuda::CudaDriverInterface * cuda::CudaDriverInterface::get() {
	if (!cuda::CudaDriverFrontend::_instance) {
		cuda::CudaDriverFrontend::_instance = new CudaDriverFrontend;
	}
	return cuda::CudaDriverFrontend::_instance;
}

////////////////////////////////////////////////////////////////////////////////

class CudaDriverFrontendDestructor {
public:
	~CudaDriverFrontendDestructor() {
		if (cuda::CudaDriverFrontend::_instance) {
			report("!CudaDriverFrontendDestructor()");
			delete cuda::CudaDriverFrontend::_instance;
		}
	}
};

cuda::CudaDriverFrontend::CudaDriverFrontend():
	_devicesLoaded(false), _computeCapability(2)
{
	_flags = 0;
	_enumerateDevices();
}

cuda::CudaDriverFrontend::~CudaDriverFrontend() {
	// delete contexts
	for (ContextQueueThreadMap::iterator ctx_thread_it = _contexts.begin();
		ctx_thread_it != _contexts.end(); ++ctx_thread_it) {
		for (ContextQueue::iterator ctx_it = ctx_thread_it->second.begin(); 
			ctx_it != ctx_thread_it->second.end(); ++ctx_it) {
			delete *ctx_it;
		}
	}
}


//! \brief create devices if they do not exist
void cuda::CudaDriverFrontend::_enumerateDevices() {
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
	}
	if(config::get().executive.enableAMD) {
		executive::DeviceVector d =
			executive::Device::createDevices(ir::Instruction::CAL, _flags,
				_computeCapability);
		report(" - Added " << d.size() << " amd gpu devices." );
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
}

//! \brief gets active context
cuda::CudaDriverFrontend::Context * cuda::CudaDriverFrontend::_getContext() {
	return _getThreadContextQueue().back();
}

//! \brief locks context thread map
void cuda::CudaDriverFrontend::_lock() {
//	_mutex.lock();
}

//! \brief unlocks context thread map
void cuda::CudaDriverFrontend::_unlock() {
//	_mutex.unlock();
}

cuda::CudaDriverFrontend::ContextQueue & cuda::CudaDriverFrontend::_getThreadContextQueue() {
	return _contexts[_getThreadId()];
}

//! \brief gets the calling thread's ID
boost::thread::id cuda::CudaDriverFrontend::_getThreadId() {
	return boost::this_thread::get_id();
}

//! \brief locks and gets the thread's active context
cuda::CudaDriverFrontend::Context * cuda::CudaDriverFrontend::_bind() {

#if REPORT_BASE
	std::cout << "  _bind()" << std::endl;
#endif
	_lock();
	cuda::CudaDriverFrontend::Context *ctx = _getContext();
	if (!ctx->_getDevice().selected()) {
		ctx->_getDevice().select();
	}
	return ctx;
}

//! \brief unlocks thread's active context
void cuda::CudaDriverFrontend::_unbind() {
	_unlock();
	
#if REPORT_BASE
	std::cout << "  _unbind()" << std::endl;
#endif
}


/////////////////////////////////////////////////////////////////////////////////

cuda::CudaDriverFrontend::Context::Context():
	_selectedDevice(0), _nextSymbol(0), _flags(0),
	_optimization((translator::Translator::OptimizationLevel)
		config::get().executive.optimizationLevel), _referenceCount(1) {

}

cuda::CudaDriverFrontend::Context::~Context() { 

}

//! \brief performs a memcpy on selected device
void cuda::CudaDriverFrontend::Context::_memcpy(void *dst, const void *src, size_t count, 
	MemcpyKind kind) {

}

//! \brief report a memory error and throw an exception
void cuda::CudaDriverFrontend::Context::_memoryError(const void *address, size_t count, 
	const std::string &func) {

}

//! \brief gets the current device for the current thread
executive::Device& cuda::CudaDriverFrontend::Context::_getDevice() {
	return *_device;
}

//! \brief returns an Ocelot-formatted error message
std::string cuda::CudaDriverFrontend::Context::_formatError(const std::string & message) {
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

// Load module and register it with all devices
void cuda::CudaDriverFrontend::Context::_registerModule(ModuleMap::iterator module) {

}

// Load module and register it with all devices
void cuda::CudaDriverFrontend::Context::_registerModule(const std::string& name) {

}

// Load all modules and register them with all devices
void cuda::CudaDriverFrontend::Context::_registerAllModules() {

}

////////////////////////////////////////////////////////////////////////////////
			
/*********************************
** Initialization
*********************************/
CUresult cuda::CudaDriverFrontend::cuInit(unsigned int Flags) {
	//
	//
	// 
	return CUDA_SUCCESS;
}


/*********************************
** Driver Version Query
*********************************/
CUresult cuda::CudaDriverFrontend::cuDriverGetVersion(int *driverVersion) {
	*driverVersion = 0x3002;
	return CUDA_SUCCESS;
}


/************************************
**
**    Device management
**
***********************************/

CUresult cuda::CudaDriverFrontend::cuDeviceGet(CUdevice *device, int ordinal) {
	CUresult result = CUDA_ERROR_NOT_FOUND;
	_lock();
	if ((size_t)ordinal < _devices.size() && ordinal >= 0) {
		*device = ordinal;
		result = CUDA_SUCCESS;
	}
	_unlock();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuDeviceGetCount(int *count) {
	CUresult result = CUDA_ERROR_NOT_FOUND;
	_lock();
	result = CUDA_SUCCESS;
	*count = (int)_devices.size();
	_unlock();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuDeviceGetName(char *name, int len, CUdevice dev) {
	CUresult result = CUDA_ERROR_NOT_FOUND;
	_lock();
	int ordinal = (int)dev;
	executive::Device *device = _devices.at(ordinal);
	
	std::string devName = device->properties().name;
	std::memcpy(name, devName.c_str(), std::min(len, (int)devName.size()));
	result = CUDA_SUCCESS;
	
	_unlock();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuDeviceComputeCapability(int *major, int *minor, 
	CUdevice dev) {
	CUresult result = CUDA_ERROR_NOT_FOUND;
	_lock();

	int ordinal = (int)dev;
	executive::Device *device = _devices.at(ordinal);

	*major = device->properties().major;
	*minor = device->properties().minor;
	
	result = CUDA_SUCCESS;
	_unlock();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
	CUresult result = CUDA_ERROR_NOT_FOUND;
	_lock();

	int ordinal = (int)dev;
	executive::Device *device = _devices.at(ordinal);
	
	*bytes = device->properties().totalMemory;
	result = CUDA_SUCCESS;

	_unlock();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
	CUresult result = CUDA_ERROR_NOT_FOUND;
	_lock();

	int ordinal = (int)dev;
	executive::Device *device = _devices.at(ordinal);
	if (device) {
		executive::DeviceProperties properties = device->properties();
		prop->maxThreadsPerBlock = properties.maxThreadsPerBlock;
		for (int i = 0; i < 3; i++) {
			prop->maxThreadsDim[i] = properties.maxThreadsDim[i];
			prop->maxGridSize[i] = properties.maxGridSize[i];
		}
		prop->sharedMemPerBlock = properties.sharedMemPerBlock;
		prop->totalConstantMemory = properties.totalConstantMemory;
		prop->SIMDWidth = properties.SIMDWidth;
		prop->regsPerBlock = properties.regsPerBlock;
		prop->clockRate = properties.clockRate;
		prop->textureAlign = properties.textureAlign;
		result = CUDA_SUCCESS;
	}
	else {
		result = CUDA_ERROR_UNKNOWN;
	}

	_unlock();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, 
	CUdevice dev) {

	CUresult result = CUDA_SUCCESS;
	
	_lock();

	int ordinal = (int)dev;
	executive::Device *device = _devices.at(ordinal);
	assert(device);
	switch (attrib) {
		case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
			*pi = 512;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X :
			*pi = 1024;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
			*pi = 1024;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
			*pi = 1024;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
			*pi = (1 << 20);
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
			*pi = (1 << 20);
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
			*pi = (1 << 20);
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
			*pi = (1 << 16);
			break;
		case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
			*pi = (1 << 16);
			break;
		case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
			*pi = 32;
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
			*pi = (1 << 8);
			break;
		case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
			*pi = (1 << 10);
			break;
		case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
			*pi = (1000000);
			break;
		case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
			*pi = (256);
			break;
		case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP:
			*pi = 0;
			break;
		case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
			*pi = 1;
			break;
		case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
			*pi = 1000;
			break;
		case CU_DEVICE_ATTRIBUTE_INTEGRATED:
			*pi = 0;
			break;
		case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
			*pi = 1;
			break;
			
		case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
			*pi = CU_COMPUTEMODE_EXCLUSIVE;
			break;
			
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH:
			*pi = 4092;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH:
			*pi = 4092;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT:
			*pi = 4092;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH:
			*pi = 4092;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT:
			*pi = 4092;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH:
			*pi = 4092;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH:
			*pi = 4092;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT:
			*pi = 4092;
			break;
		case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES:
			*pi = 4092;
			break;
		case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT:
			*pi = 256;
			break;
		case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
			*pi = 1;
			break;
		case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
			*pi = 0;
			break;
		default:
			*pi = 0;
			result = CUDA_ERROR_NOT_FOUND;
			report("cuDeviceGetAttribute() - unsupported attribute requested: " << attrib);
			assert(0 && "cuDeviceGetAttribute() - unsupported attribute requested: ");
			break;
	}
	
	_unlock();
	return result;
}

/************************************
**
**    Context management
**
***********************************/

CUresult cuda::CudaDriverFrontend::cuCtxCreate(CUcontext *pctx, unsigned int flags, 
	CUdevice dev ) {

	_lock();

	// check to see if the device is in use - only support exclusive mode use of devices
	Context *newContext = new Context;
	newContext->_device = _devices.at((int)dev);
	*pctx = reinterpret_cast<CUcontext>(newContext);
	_getThreadContextQueue().push_back(newContext);
	_unlock();
	
	return CUDA_SUCCESS;
}

CUresult cuda::CudaDriverFrontend::cuCtxDestroy( CUcontext ctx ) {
	CUresult result = CUDA_ERROR_INVALID_VALUE;
	Context *context = reinterpret_cast<Context *>(ctx);
	_lock();
	if (_getThreadContextQueue().size() == 0 || _getThreadContextQueue().back() != context) {
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	else if (context->_referenceCount == 1) {
		_getThreadContextQueue().pop_back();
		delete context;
		result = CUDA_SUCCESS;
	}
	_unlock();

	return result;
}

CUresult cuda::CudaDriverFrontend::cuCtxAttach(CUcontext *pctx, unsigned int flags) {
	CUresult result = CUDA_ERROR_INVALID_VALUE;
	_lock();
	Context *context = 0;
	ContextQueue &queue = _getThreadContextQueue();
	if (queue.size()) {
		context = queue.back();
		++ context->_referenceCount;
		*pctx = reinterpret_cast<CUcontext>(context);
		result = CUDA_SUCCESS;
	}
	_unlock();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuCtxDetach(CUcontext ctx) {
	CUresult result = CUDA_ERROR_INVALID_CONTEXT;
	_lock();
	Context *context = reinterpret_cast<Context *>(ctx);
	ContextQueue &queue = _getThreadContextQueue();
	if (queue.size() == 0 || queue.back() != context) {
		result = CUDA_ERROR_INVALID_CONTEXT;
	}	
	else {
		if (!(--context->_referenceCount)) {
			// free context
		}
		result = CUDA_SUCCESS;
	}
	_unlock();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuCtxPushCurrent( CUcontext ctx ) {
	CUresult result = CUDA_SUCCESS;
	_lock();
	Context *context = reinterpret_cast<Context *>(ctx);
	ContextQueue &queue = _getThreadContextQueue();
	queue.push_back(context);
	report("    - pushed current, context stack has " << queue.size() << " elements");
	_unlock();
	
	return result;
}

CUresult cuda::CudaDriverFrontend::cuCtxPopCurrent( CUcontext *pctx ) {
	CUresult result = CUDA_ERROR_INVALID_CONTEXT;
	
	std::cout << "   - tid: " << boost::this_thread::get_id() << std::endl;
	
	_lock();
	ContextQueue &queue = _getThreadContextQueue();
	if (queue.size()) {
		Context *context = queue.back();		
		queue.pop_back();
		if (pctx) {
			*pctx = reinterpret_cast<CUcontext>(context);
		}
		result = CUDA_SUCCESS;
		report("    popped, context stack has " << queue.size() << " elements");
	}
	else {
		report("  cuCtxPopCurrent() - no contexts");
	}
	_unlock();
	report("  returning");
	return result;
}

CUresult cuda::CudaDriverFrontend::cuCtxGetDevice(CUdevice *device) {
	CUresult result = CUDA_ERROR_INVALID_CONTEXT;
	_lock();
	ContextQueue &queue = _getThreadContextQueue();
	if (queue.size()) {
		Context *context = queue.back();
		// return the thread's device
		*device = context->_selectedDevice;
		result = CUDA_SUCCESS;
	}
	_unlock();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuCtxSynchronize(void) {
	CUresult result = CUDA_ERROR_INVALID_CONTEXT;
	_lock();
	ContextQueue &queue = _getThreadContextQueue();
	if (queue.size()) {
		Context *context = queue.back();
		_unlock();
		// block on waiting operations
		context->_getDevice().synchronize();
		result = CUDA_SUCCESS;
	}
	else {
		_unlock();
	}
	return result;
}


/************************************
**
**    Module management
**

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		
		result = CUDA_SUCCESS;
	}
	_unbind();
	return result;

***********************************/

CUresult cuda::CudaDriverFrontend::cuModuleLoad(CUmodule *cuModule, const char *fname) {
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		std::ifstream file(fname);
		if (file.good()) {
			ModuleMap::iterator module = context->_modules.insert(
				std::make_pair(fname, ir::Module())).first;
			if (module->second.load(fname)) {
				*cuModule = reinterpret_cast<CUmodule>(& module->second);
				context->_getDevice().load(&module->second);
				context->_getDevice().setOptimizationLevel(context->_optimization);
				result = CUDA_SUCCESS;
			}
			else {
				context->_modules.erase(module);
				result = CUDA_ERROR_INVALID_VALUE;
			}
		}	
		else {
			report("cuModuleLoad() - failed to load module '" << fname << "' from file");
			result = CUDA_ERROR_FILE_NOT_FOUND;
		}
	}
	else {
		report("cuModuleLoad() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuModuleLoadData(CUmodule *module, 
	const void *image) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuModuleLoadDataEx(CUmodule *cuModule, 
	const void *image, unsigned int numOptions, 
	CUjit_option *options, void **optionValues) {
	
	report("   entered");
	
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		report("  obtained context");
		
		std::stringstream ss;
		ss << (const char *)image;
		
		std::stringstream modname;
		modname << "ocelotModule" << context->_modules.size() << ".ptx";
		
		{
			std::ofstream file(modname.str().c_str());
			file << (const char *)image;
		}
	
		ModuleMap::iterator module = context->_modules.insert(std::make_pair(modname.str(), ir::Module())).first;
		
		report("  created module, now loading..");

		if (module->second.load(ss, modname.str())) {
			*cuModule = reinterpret_cast<CUmodule>(& module->second);
			
			report("  loaded PTX module, now loading into emulator");
			
			context->_getDevice().load(&module->second);
			context->_getDevice().setOptimizationLevel(context->_optimization);
			result = CUDA_SUCCESS;
			
			report("    load Successful");
		}
		else {
			report("  load failed");
			context->_modules.erase(module);
			result = CUDA_ERROR_INVALID_VALUE;
		}
	}
	else {
		report("cuModuleLoadDataEx() - invalid context");
		result = CUDA_ERROR_INVALID_CONTEXT;
			report("    load failed");
	}
	report("  unbinding");
	_unbind();
	report("  returning");
	return result;
}

CUresult cuda::CudaDriverFrontend::cuModuleLoadFatBinary(CUmodule *module, 
	const void *fatCubin) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuModuleUnload(CUmodule hmod) {
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		context->_getDevice().unload(reinterpret_cast<const ir::Module *>(hmod)->path());
		result = CUDA_SUCCESS;
	}
	else {
		report("cuModuleLoad() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuModuleGetFunction(CUfunction *hfunc, 
	CUmodule hmod, const char *name) {
	
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		ir::Module *module = reinterpret_cast<ir::Module *>(hmod);
		if (module) {
			ir::PTXKernel *kernel = module->getKernel(name);
			if (kernel) {
				*hfunc = reinterpret_cast<CUfunction>(kernel);
				result = CUDA_SUCCESS;
			}
			else {
				result = CUDA_ERROR_NOT_FOUND;
				report("cuModuleGetFunction() - failed to get kernel " << name);
			}
		}
		else {
			result = CUDA_ERROR_INVALID_VALUE;
			report("cuModuleGetFunction() - invalid module");
		}
	}
	else {
		report("cuModuleGetFunction() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuModuleGetGlobal(CUdeviceptr *dptr, 
	size_t *bytes, CUmodule hmod, const char *name) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		ir::Module *module = reinterpret_cast<ir::Module *>(hmod);
		if (module) {
			//
			// get global
			//
			executive::Device::MemoryAllocation *allocation = 
				context->_getDevice().getGlobalAllocation(module->path(), std::string(name));
			if (allocation) {
				*dptr = hydrazine::bit_cast<CUdeviceptr, void *>(allocation->pointer());
				result = CUDA_SUCCESS;
				report("  obtained global " << name << " at " << allocation->pointer());
			}
			else {
				result = CUDA_ERROR_NOT_FOUND;
				report("cuModuleGetGlobal(" << name << ") - failed to get allocation");
			}
		}
		else {
			result = CUDA_ERROR_INVALID_VALUE;
			report("cuModuleGetGlobal() - invalid module");
		}
	}
	else {
		report("cuModuleGetGlobal() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, 
	const char *name) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Memory management
**
***********************************/

CUresult cuda::CudaDriverFrontend::cuMemGetInfo(size_t *free, 
	size_t *total) {
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		*total = context->_getDevice().properties().totalMemory;

		size_t consumed = 0;
		executive::Device::MemoryAllocationVector allocationVector = 
			context->_getDevice().getAllAllocations();
		for (executive::Device::MemoryAllocationVector::iterator alloc_it = allocationVector.begin();
			alloc_it != allocationVector.end(); ++alloc_it) {
			consumed += (*alloc_it)->size();
		}
		*free = *total - consumed;
		result = CUDA_SUCCESS;
	}
	else {
		report("cuMemGetInfo() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}


CUresult cuda::CudaDriverFrontend::cuMemAlloc( CUdeviceptr *dptr, 
	unsigned int bytesize) {
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		executive::Device &device = context->_getDevice();
		executive::Device::MemoryAllocation * allocation = device.allocate(bytesize);
		if (allocation) {
			*dptr = hydrazine::bit_cast<CUdeviceptr, void *>(allocation->pointer());
			result = CUDA_SUCCESS;
		}
		else {
			result = CUDA_ERROR_INVALID_VALUE;
		}
	}
	else {
		report("cuModuleGetFunction() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuMemAllocPitch( CUdeviceptr *dptr, 
			          size_t *pPitch,
			          unsigned int WidthInBytes, 
			          unsigned int Height, 
			          unsigned int ElementSizeBytes
			         ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemFree(CUdeviceptr dptr) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		executive::Device &device = context->_getDevice();
		
		executive::Device::MemoryAllocation *allocation = device.getMemoryAllocation((void *)dptr);
		if (allocation) {
			device.free((void *)dptr);
			result = CUDA_SUCCESS;
		}
		else {
			report("cuMemFree() - invalid value");
			result = CUDA_ERROR_INVALID_VALUE;
		}
	}
	else {
		report("cuModuleGetFunction() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;

}

CUresult cuda::CudaDriverFrontend::cuMemGetAddressRange( CUdeviceptr *pbase, 
	size_t *psize, CUdeviceptr dptr ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverFrontend::cuMemAllocHost(void **pp, unsigned int bytesize) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemFreeHost(void *p) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverFrontend::cuMemHostAlloc(void **pp, 
	unsigned long long bytesize, unsigned int Flags ) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		executive::Device &device = context->_getDevice();
		executive::Device::MemoryAllocation * allocation = device.allocateHost(bytesize, Flags);
		if (allocation) {
			*pp = allocation->mappedPointer();
			result = CUDA_SUCCESS;
		}
		else {
			result = CUDA_ERROR_INVALID_VALUE;
		}
	}
	else {
		report("cuMemHostAlloc() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}


CUresult cuda::CudaDriverFrontend::cuMemHostGetDevicePointer( CUdeviceptr *pdptr, 
	void *p, unsigned int Flags ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemHostGetFlags( unsigned int *pFlags, void *p ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Synchronous Memcpy
**
** Intra-device memcpy's done with these functions may execute 
**	in parallel with the CPU,
** but if host memory is involved, they wait until the copy is 
**	done before returning.
**
***********************************/

// 1D functions
// system <-> device memory
CUresult cuda::CudaDriverFrontend::cuMemcpyHtoD (CUdeviceptr dstDevice, 
	const void *srcHost, unsigned int ByteCount ) {
	
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		executive::Device &device = context->_getDevice();
		executive::Device::MemoryAllocation * allocation = device.getMemoryAllocation((void *)dstDevice);
		if (allocation) {
			allocation->copy(0, srcHost, (size_t)ByteCount);
			result = CUDA_SUCCESS;
		}
		else {
			result = CUDA_ERROR_INVALID_VALUE;
		}
	}
	else {
		report("cuModuleGetFunction() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, 
	unsigned int ByteCount ) {
	
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		executive::Device &device = context->_getDevice();
		executive::Device::MemoryAllocation * allocation = device.getMemoryAllocation((void *)srcDevice);
		if (allocation) {
			allocation->copy(dstHost, 0, (size_t)ByteCount);
			result = CUDA_SUCCESS;
		}
		else {
			result = CUDA_ERROR_INVALID_VALUE;
		}
	}
	else {
		report("cuModuleGetFunction() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}


// device <-> device memory
CUresult cuda::CudaDriverFrontend::cuMemcpyDtoD (CUdeviceptr dstDevice, 
	CUdeviceptr srcDevice, unsigned int ByteCount ) {
	
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		executive::Device &device = context->_getDevice();
		executive::Device::MemoryAllocation * destAllocation = 
			device.getMemoryAllocation((void *)dstDevice);
		executive::Device::MemoryAllocation * srcAllocation = 
			device.getMemoryAllocation((void *)srcDevice);
			
		if (destAllocation && srcAllocation) {
			long int destOffset = (char *)dstDevice - (char *)destAllocation->pointer();
			long int srcOffset = (char *)srcDevice - (char *)srcAllocation->pointer();
			
			srcAllocation->copy(destAllocation, destOffset, srcOffset, ByteCount);
			result = CUDA_SUCCESS;
		}
		else {
			result = CUDA_ERROR_INVALID_VALUE;
		}
	}
	else {
		report("cuModuleGetFunction() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuMemcpyHtoH (void *dstHost, const void *srcHost, 
	unsigned int ByteCount ) {
	std::memcpy(dstHost, srcHost, ByteCount);
	return CUDA_SUCCESS;
}

// device <-> array memory
CUresult cuda::CudaDriverFrontend::cuMemcpyDtoA ( CUarray dstArray, 
	unsigned int dstIndex, CUdeviceptr srcDevice, 
	unsigned int ByteCount ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemcpyAtoD ( CUdeviceptr dstDevice, 
	CUarray hSrc, unsigned int SrcIndex, unsigned int ByteCount ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


// system <-> array memory
CUresult cuda::CudaDriverFrontend::cuMemcpyHtoA( CUarray dstArray, 
	unsigned int dstIndex, const void *pSrc, 
	unsigned int ByteCount ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemcpyAtoH( void *dstHost, CUarray srcArray, 
	unsigned int srcIndex, unsigned int ByteCount ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


// array <-> array memory
CUresult cuda::CudaDriverFrontend::cuMemcpyAtoA( CUarray dstArray, 
	unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, 
	unsigned int ByteCount ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


// 2D memcpy

CUresult cuda::CudaDriverFrontend::cuMemcpy2D( const CUDA_MEMCPY2D *pCopy ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


// 3D memcpy

CUresult cuda::CudaDriverFrontend::cuMemcpy3D( const CUDA_MEMCPY3D *pCopy ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Asynchronous Memcpy
**
** Any host memory involved must be DMA'able (e.g., 
** allocated with cuMemAllocHost).
** memcpy's done with these functions execute in parallel with 
** the CPU and, if
** the hardware is available, may execute in parallel with the GPU.
** Asynchronous memcpy must be accompanied by appropriate stream 
** synchronization.
**
***********************************/

// 1D functions
// system <-> device memory
CUresult cuda::CudaDriverFrontend::cuMemcpyHtoDAsync (CUdeviceptr dstDevice, 
const void *srcHost, unsigned int ByteCount, CUstream hStream ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemcpyDtoHAsync (void *dstHost, 
CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

// system <-> array memory
CUresult cuda::CudaDriverFrontend::cuMemcpyHtoAAsync( CUarray dstArray, 
	unsigned int dstIndex, const void *pSrc, 
	unsigned int ByteCount, CUstream hStream ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, 
	unsigned int srcIndex, unsigned int ByteCount, 
	CUstream hStream ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


// 2D memcpy
CUresult cuda::CudaDriverFrontend::cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, 
	CUstream hStream ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


// 3D memcpy
CUresult cuda::CudaDriverFrontend::cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, 
	CUstream hStream ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Memset
**
***********************************/
CUresult cuda::CudaDriverFrontend::cuMemsetD8( CUdeviceptr dstDevice, 
	unsigned char uc, unsigned int N ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemsetD16( CUdeviceptr dstDevice, 
	unsigned short us, unsigned int N ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemsetD32( CUdeviceptr dstDevice, 
	unsigned int ui, unsigned int N ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverFrontend::cuMemsetD2D8( CUdeviceptr dstDevice,
	unsigned int dstPitch, unsigned char uc, unsigned int Width, 
	unsigned int Height ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemsetD2D16( CUdeviceptr dstDevice, 
	unsigned int dstPitch, unsigned short us, unsigned int Width, 
	unsigned int Height ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuMemsetD2D32( CUdeviceptr dstDevice, 
	unsigned int dstPitch, unsigned int ui, unsigned int Width, 
	unsigned int Height ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Function management
**
***********************************/

CUresult cuda::CudaDriverFrontend::cuFuncSetBlockShape(
	CUfunction hfunc, int x, int y, int z) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		ir::PTXKernel *ptxKernel = reinterpret_cast<ir::PTXKernel *>(hfunc);
		if (ptxKernel) {
			context->_launchConfiguration.blockDim.x = x;
			context->_launchConfiguration.blockDim.y = y;
			context->_launchConfiguration.blockDim.z = z;

			result = CUDA_SUCCESS;
			report("cuFuncSetBlockShape() - setting block dim to: " 
				<< context->_launchConfiguration.blockDim.x << ", "
				<< context->_launchConfiguration.blockDim.y << ", "
				<< context->_launchConfiguration.blockDim.z);
		}
		else {
			report("cuFuncSetBlockShape() - kernel not found");
			result = CUDA_ERROR_INVALID_CONTEXT;
		}
	}
	else {
		report("cuFuncSetBlockShape() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuFuncSetSharedSize(
	CUfunction hfunc, unsigned int bytes) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		ir::PTXKernel *ptxKernel = reinterpret_cast<ir::PTXKernel *>(hfunc);
		if (ptxKernel) {
			
			context->_launchConfiguration.sharedMemory = bytes;

			result = CUDA_SUCCESS;
		}
		else {
			report("cuParamSetSize() - kernel not found");
			result = CUDA_ERROR_INVALID_CONTEXT;
		}
	}
	else {
		report("cuParamSetSize() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuFuncGetAttribute (int *pi, CUfunction_attribute attrib,
	CUfunction hfunc) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		ir::PTXKernel *ptxKernel = reinterpret_cast<ir::PTXKernel *>(hfunc);
		if (ptxKernel) {
			result = CUDA_SUCCESS;
			// thsi function isn't needed
		}
		else {
			report("cuFuncGetAttribute() - kernel not found");
			result = CUDA_ERROR_INVALID_CONTEXT;
		}
	}
	else {
		report("cuFuncGetAttribute() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

static executive::ExecutableKernel::CacheConfiguration _translateCacheConfiguration(CUfunc_cache config) {
	switch (config) {
		case CU_FUNC_CACHE_PREFER_SHARED:
			return executive::ExecutableKernel::CachePreferShared;
		case CU_FUNC_CACHE_PREFER_L1:
			return executive::ExecutableKernel::CachePreferL1;
		default:
			break;
	}
	return executive::ExecutableKernel::CacheConfigurationDefault;
}

CUresult cuda::CudaDriverFrontend::cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		ir::PTXKernel *ptxKernel = reinterpret_cast<ir::PTXKernel *>(hfunc);
		executive::ExecutableKernel *executableKernel = context->_getDevice().getKernel(
			ptxKernel->module->path(), ptxKernel->name);
		executableKernel->setCacheConfiguration(_translateCacheConfiguration(config));
	}
	else {
		report("cuFuncSetCacheConfig() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

/************************************
**
**    Array management 
**
***********************************/

CUresult cuda::CudaDriverFrontend::cuArrayCreate( CUarray *pHandle, 
	const CUDA_ARRAY_DESCRIPTOR *pAllocateArray ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuArrayGetDescriptor( 
	CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuArrayDestroy( CUarray hArray ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverFrontend::cuArray3DCreate( CUarray *pHandle, 
	const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuArray3DGetDescriptor( 
	CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}



/************************************
**
**    Texture reference management
**
***********************************/
CUresult cuda::CudaDriverFrontend::cuTexRefCreate( CUtexref *pTexRef ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefDestroy( CUtexref hTexRef ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverFrontend::cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, 
	unsigned int Flags ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefSetAddress( size_t *ByteOffset, 
	CUtexref hTexRef, CUdeviceptr dptr, unsigned int bytes ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefSetAddress2D( CUtexref hTexRef, 
	const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, 
	unsigned int Pitch) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefSetFormat( CUtexref hTexRef, 
	CUarray_format fmt, int NumPackedComponents ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefSetAddressMode( CUtexref hTexRef, int dim, 
	CUaddress_mode am ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefSetFilterMode( CUtexref hTexRef, 
	CUfilter_mode fm ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefSetFlags( CUtexref hTexRef, 
	unsigned int Flags ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


CUresult cuda::CudaDriverFrontend::cuTexRefGetAddress( CUdeviceptr *pdptr, 
	CUtexref hTexRef ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefGetArray( CUarray *phArray, 
	CUtexref hTexRef ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefGetAddressMode( CUaddress_mode *pam, 
	CUtexref hTexRef, int dim ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefGetFilterMode( CUfilter_mode *pfm, 
	CUtexref hTexRef ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefGetFormat( CUarray_format *pFormat, 
	int *pNumChannels, CUtexref hTexRef ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuTexRefGetFlags( unsigned int *pFlags, 
	CUtexref hTexRef ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Parameter management
**
***********************************/

CUresult cuda::CudaDriverFrontend::cuParamSetSize (CUfunction hfunc, 
	unsigned int numbytes) {
	
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		ir::PTXKernel *ptxKernel = reinterpret_cast<ir::PTXKernel *>(hfunc);
		if (ptxKernel) {
			result = CUDA_SUCCESS;	// this function isn't needed
		}
		else {
			report("cuParamSetSize() - kernel not found");
			result = CUDA_ERROR_INVALID_CONTEXT;
		}
	}
	else {
		report("cuParamSetSize() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuParamSeti    (CUfunction hfunc, int offset, 
	unsigned int value) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuParamSetf    (CUfunction hfunc, int offset, 
	float value) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

//
//
//
CUresult cuda::CudaDriverFrontend::cuParamSetv(CUfunction hfunc, int offset, 
	void * ptr, unsigned int size) {
	
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		ir::PTXKernel *ptxKernel = reinterpret_cast<ir::PTXKernel *>(hfunc);
		if (ptxKernel) {
			// set an argument
			std::memcpy(context->_hostThreadContext.parameterBlock + offset, ptr, size);
	
			context->_hostThreadContext.parameterIndices.push_back(offset);
			context->_hostThreadContext.parameterSizes.push_back(size);
			
			result = CUDA_SUCCESS;
		}
		else {
			report("cuParamSetSize() - kernel not found");
			result = CUDA_ERROR_INVALID_CONTEXT;
		}
	}
	else {
		report("cuParamSetSize() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuParamSetTexRef(CUfunction hfunc, int texunit, 
	CUtexref hTexRef) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    Launch functions
**
***********************************/

static ir::Dim3 convert(const dim3& d) {
	return std::move(ir::Dim3(d.x, d.y, d.z));
}

CUresult cuda::CudaDriverFrontend::cuLaunch(CUfunction hfunc) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		ir::PTXKernel *ptxKernel = reinterpret_cast<ir::PTXKernel *>(hfunc);
		if (ptxKernel) {
	
			context->_hostThreadContext.mapParameters(ptxKernel);
			
			report("kernel launch (" << ptxKernel->name 
				<< ") on thread " << boost::this_thread::get_id());
	
			try {
				trace::TraceGeneratorVector traceGens;

				traceGens = context->_hostThreadContext.persistentTraceGenerators;
				traceGens.insert(traceGens.end(),
					context->_hostThreadContext.nextTraceGenerators.begin(), 
					context->_hostThreadContext.nextTraceGenerators.end());

				context->_getDevice().launch(ptxKernel->module->path(), 
					ptxKernel->name, 
					convert(context->_launchConfiguration.gridDim), 
					convert(context->_launchConfiguration.blockDim), 
					context->_launchConfiguration.sharedMemory, 
					context->_hostThreadContext.parameterBlock, 
					context->_hostThreadContext.parameterBlockSize, traceGens);
					
				report(" launch completed successfully");	
			}
			catch( const executive::RuntimeException& e ) {
				std::cerr << "==Ocelot== PTX Emulator failed to run kernel \"" 
					<< ptxKernel->name 
					<< "\" with exception: \n";
				std::cerr << e.toString()  
					<< "\n" << std::flush;
				_unbind();
				throw;
			}
			catch( const std::exception& e ) {
				std::cerr << "==Ocelot== " << context->_getDevice().properties().name
					<< " failed to run kernel \""
					<< ptxKernel->name
					<< "\" with exception: \n";
				std::cerr << e.what()
					<< "\n" << std::flush;
				_unbind();
				throw;
			}
			catch(...) {
				_unbind();
				throw;
			}

			result = CUDA_SUCCESS;
		}
		else {
			report("cuLaunch() - kernel not found");
			result = CUDA_ERROR_INVALID_CONTEXT;
		}
	}
	else {
		report("cuLaunch() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;

}

CUresult cuda::CudaDriverFrontend::cuLaunchGrid (CUfunction hfunc, int grid_width, 
	int grid_height) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		ir::PTXKernel *ptxKernel = reinterpret_cast<ir::PTXKernel *>(hfunc);
		if (ptxKernel) {
	
			context->_hostThreadContext.mapParameters(ptxKernel);
			context->_launchConfiguration.gridDim.x = grid_width;
			context->_launchConfiguration.gridDim.y = grid_height;
			context->_launchConfiguration.gridDim.z = 1;
			
			report("kernel launch (" << ptxKernel->name 
				<< ") on thread " << boost::this_thread::get_id());
	
			try {
				trace::TraceGeneratorVector traceGens;

				traceGens = context->_hostThreadContext.persistentTraceGenerators;
				traceGens.insert(traceGens.end(),
					context->_hostThreadContext.nextTraceGenerators.begin(), 
					context->_hostThreadContext.nextTraceGenerators.end());

				context->_getDevice().launch(ptxKernel->module->path(), 
					ptxKernel->name, 
					convert(context->_launchConfiguration.gridDim), 
					convert(context->_launchConfiguration.blockDim), 
					context->_launchConfiguration.sharedMemory, 
					context->_hostThreadContext.parameterBlock, 
					context->_hostThreadContext.parameterBlockSize, traceGens);
					
				report(" launch completed successfully");	
			}
			catch( const executive::RuntimeException& e ) {
				std::cerr << "==Ocelot== PTX Emulator failed to run kernel \"" 
					<< ptxKernel->name 
					<< "\" with exception: \n";
				std::cerr << e.toString()  
					<< "\n" << std::flush;
				_unbind();
				throw;
			}
			catch( const std::exception& e ) {
				std::cerr << "==Ocelot== " << context->_getDevice().properties().name
					<< " failed to run kernel \""
					<< ptxKernel->name
					<< "\" with exception: \n";
				std::cerr << e.what()
					<< "\n" << std::flush;
				_unbind();
				throw;
			}
			catch(...) {
				_unbind();
				throw;
			}

			result = CUDA_SUCCESS;
		}
		else {
			report("cuLaunch() - kernel not found");
			result = CUDA_ERROR_INVALID_CONTEXT;
		}
	}
	else {
		report("cuLaunch() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;

}

CUresult cuda::CudaDriverFrontend::cuLaunchGridAsync( CUfunction f, int grid_width, 
	int grid_height, CUstream hStream ) {
	return cuLaunchGrid(f, grid_width, grid_height);
}


/************************************
**
**    Events
**
***********************************/
CUresult cuda::CudaDriverFrontend::cuEventCreate( CUevent *phEvent, 
	unsigned int Flags ) {
	
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		unsigned int eventHandle = context->_getDevice().createEvent(Flags);
		*phEvent = hydrazine::bit_cast<CUevent>(eventHandle);
		context->_events.insert(*phEvent);
		result = CUDA_SUCCESS;
	}
	else {
		report("cuEventCreate() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuEventRecord( CUevent hEvent, CUstream hStream ) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		unsigned int eventHandle = hydrazine::bit_cast<unsigned int, CUevent>(hEvent);
		unsigned int streamHandle = hydrazine::bit_cast<unsigned int, CUstream>(hStream);
		if (context->_events.find(hEvent) != context->_events.end() &&
			(!streamHandle || context->_streams.find(hStream) != context->_streams.end())) {
			context->_getDevice().recordEvent(eventHandle, streamHandle);
			result = CUDA_SUCCESS;
		}
		else {
			report("cuEventRecord() - event not found");
			result = CUDA_ERROR_NOT_FOUND;
		}
	}
	else {
		report("cuParamSetSize() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuEventQuery( CUevent hEvent ) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		unsigned int eventHandle = hydrazine::bit_cast<unsigned int, CUevent>(hEvent);
		if (context->_events.find(hEvent) != context->_events.end()) {
			bool eventResult = context->_getDevice().queryEvent(eventHandle);
			if (eventResult) {
				result = CUDA_SUCCESS;
			}
			else {
				result = CUDA_ERROR_NOT_READY;
			}
		}
		else {
			report("cuEventRecord() - event not found");
			result = CUDA_ERROR_NOT_FOUND;
		}
	}
	else {
		report("cuParamSetSize() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuEventSynchronize( CUevent hEvent ) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		unsigned int eventHandle = hydrazine::bit_cast<unsigned int, CUevent>(hEvent);
		if (context->_events.find(hEvent) != context->_events.end()) {
			context->_getDevice().synchronizeEvent(eventHandle);
			result = CUDA_SUCCESS;
		}
		else {
			report("cuEventSynchronize() - event not found");
			result = CUDA_ERROR_NOT_FOUND;
		}
	}
	else {
		report("cuEventSynchronize() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuEventDestroy( CUevent hEvent ) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		if (context->_events.find(hEvent) != context->_events.end()) {
			unsigned int eventHandle = hydrazine::bit_cast<unsigned int, CUevent>(hEvent);
			context->_getDevice().destroyEvent(eventHandle);
			context->_events.erase(context->_events.find(hEvent));
			result = CUDA_SUCCESS;
		}
		else {
			report("cuEventDestroy() - event not found");
			result = CUDA_ERROR_NOT_FOUND;
		}
	}
	else {
		report("cuEventDestroy() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuEventElapsedTime( float *pMilliseconds, 
	CUevent hStart, CUevent hEnd ) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		if (context->_events.find(hStart) != context->_events.end() &&
			context->_events.find(hEnd) != context->_events.end()) {
			
			unsigned int startHandle = hydrazine::bit_cast<unsigned int, CUevent>(hStart);
			unsigned int endHandle = hydrazine::bit_cast<unsigned int, CUevent>(hEnd);
			*pMilliseconds = context->_getDevice().getEventTime(startHandle, endHandle);
			result = CUDA_SUCCESS;
		}
		else {
			report("cuEventElapsedTime() - event not found");
			result = CUDA_ERROR_NOT_FOUND;
		}
	}
	else {
		report("cuEventElapsedTime() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}


/************************************
**
**    Streams
**
***********************************/
CUresult cuda::CudaDriverFrontend::cuStreamCreate( CUstream *phStream, 
	unsigned int Flags ) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		unsigned int streamHandle = context->_getDevice().createStream();
		*phStream = hydrazine::bit_cast<CUstream>(streamHandle);
		context->_streams.insert(*phStream);
		result = CUDA_SUCCESS;
	}
	else {
		report("cuEventElapsedTime() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuStreamQuery( CUstream hStream ) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		if (context->_streams.find(hStream) != context->_streams.end()) {
			unsigned int streamHandle = hydrazine::bit_cast<unsigned int, CUstream>(hStream);
			bool streamResult = context->_getDevice().queryStream(streamHandle);
			if (streamResult) {
				result = CUDA_SUCCESS;
			}
			else {
				result = CUDA_ERROR_NOT_READY;
			}
		}
		else {
			report("cuStreamQuery() - stream not found");
			result = CUDA_ERROR_NOT_FOUND;
		}
	}
	else {
		report("cuStreamQuery() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuStreamSynchronize( CUstream hStream ) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		if (context->_streams.find(hStream) != context->_streams.end()) {
			unsigned int streamHandle = hydrazine::bit_cast<unsigned int, CUstream>(hStream);
			context->_getDevice().synchronizeStream(streamHandle);
			result = CUDA_SUCCESS;
		}
		else {
			report("cuStreamSynchronize() - stream not found");
			result = CUDA_ERROR_NOT_FOUND;
		}
	}
	else {
		report("cuStreamSynchronize() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}

CUresult cuda::CudaDriverFrontend::cuStreamDestroy( CUstream hStream ) {

	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		if (context->_streams.find(hStream) != context->_streams.end()) {
			unsigned int streamHandle = hydrazine::bit_cast<unsigned int, CUstream>(hStream);
			context->_getDevice().destroyStream(streamHandle);
			context->_streams.erase(context->_streams.find(hStream));
			result = CUDA_SUCCESS;
		}
		else {
			report("cuStreamDestroy() - stream not found");
			result = CUDA_ERROR_NOT_FOUND;
		}
	}
	else {
		report("cuStreamDestroy() - context not valid");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	return result;
}


/************************************
**
**    Graphics
**
***********************************/
CUresult cuda::CudaDriverFrontend::cuGraphicsUnregisterResource(
	CUgraphicsResource resource) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuGraphicsSubResourceGetMappedArray(
	CUarray *pArray, CUgraphicsResource resource, 
	unsigned int arrayIndex, unsigned int mipLevel ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuGraphicsResourceGetMappedPointer(
	CUdeviceptr *pDevPtr, size_t *pSize, 
	CUgraphicsResource resource ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuGraphicsResourceSetMapFlags(
	CUgraphicsResource resource, unsigned int flags ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}
 
CUresult cuda::CudaDriverFrontend::cuGraphicsMapResources(unsigned int count, 
	CUgraphicsResource *resources, CUstream hStream ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuGraphicsUnmapResources(unsigned int count, 
	CUgraphicsResource *resources, CUstream hStream ) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}


/************************************
**
**    OpenGL
**
***********************************/
CUresult cuda::CudaDriverFrontend::cuGLInit() {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

CUresult cuda::CudaDriverFrontend::cuGLCtxCreate(CUcontext *pCtx, 
	unsigned int Flags, CUdevice device) {
	return cuCtxCreate(pCtx, Flags, device);
}

CUresult cuda::CudaDriverFrontend::cuGraphicsGLRegisterBuffer( 
	CUgraphicsResource *pCudaResource, unsigned int bufObj, 
	unsigned int Flags ) {
	
	CUresult result = CUDA_ERROR_NOT_FOUND;
	Context *context = _bind();
	if (context) {
		if (context->_buffers.count(bufObj) == 0) {
			void* graphic = context->_getDevice().glRegisterBuffer(bufObj, 0);
			context->_buffers.insert(std::make_pair(bufObj, graphic));
			result = CUDA_SUCCESS;
		}
	}
	else {
		report("  cuGraphicsGLRegisterBuffer() - invalid context");
		result = CUDA_ERROR_INVALID_CONTEXT;
	}
	_unbind();
	
	return result;
}

CUresult cuda::CudaDriverFrontend::cuGraphicsGLRegisterImage( 
	CUgraphicsResource *pCudaResource, unsigned int image, 
	int target, unsigned int Flags) {
	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

////////////////////////////////////////////////////////////////////////////////

CUresult cuda::CudaDriverFrontend::cuGetExportTable(
	const void **ppExportTable, 
	const CUuuid *pExportTableId) {

	assert(0 && "unimplemented");
	return CUDA_ERROR_NOT_FOUND;
}

std::string cuda::CudaDriverFrontend::toString(CUresult result) {
	assert(0 && "unimplemented");
	return "CUresult";
}

////////////////////////////////////////////////////////////////////////////////

