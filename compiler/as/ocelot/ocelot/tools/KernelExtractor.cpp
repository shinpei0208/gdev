/*!
	\file KernelExtractor.cpp

	\author Andrew Kerr <arkerr@gatech.edu>
	\brief implements CUDA Driver API functions which call through to the implementation of the
		appropriate CUDA Driver API front-end implementation
	\date 31 Jan 2011
	\location somewhere over Atlanta

	#
	# to swap this in
	#	
	rm /usr/lib64/libcuda.so
	ln -s /usr/local/lib/libkernelExtractor.so /usr/lib64/libcuda.so
	
	#
	# to swap this out
	#
	rm /usr/lib64/libcuda.so
	ln -s /usr/lib64/libcuda.so.1 /usr/lib64/libcuda.so

*/

// C++ includes
#include <string>
#include <sstream>
#include <cstring>
#include <iomanip>

// Ocelot includes
#include <ocelot/cuda/interface/cuda_internal.h>
#include <ocelot/cuda/interface/FatBinaryContext.h>
#include <ocelot/util/interface/KernelExtractor.h>

// Hydrazine includes
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Casts.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////

// whether CUDA runtime catches exceptions thrown by Ocelot
#define CATCH_RUNTIME_EXCEPTIONS 0

// whether verbose error messages are printed
#define CUDA_VERBOSE 1

// whether debugging messages are printed
#define REPORT_BASE 1

//////////////////////////////////////////////////////////////////////////////////////////////////
//
// Error handling macros

#define Ocelot_Exception(x) { std::stringstream ss; ss << x; throw hydrazine::Exception(ss.str()) ); }

///////////////////////////////////////////////////////////////////////////////////////////////////

#if REPORT_BASE
#define trace() { std::cout << __FILE__ << ":" << __LINE__ << " - " << __func__ << "() " << std::endl; }
#else
#define trace()
#endif

#if REPORT_BASE
#define RETURN(x) CUresult result = x; \
	if (result != CUDA_SUCCESS) { std::cout << __FILE__ << ":" << __LINE__ << "  error: " << (int)result << std::endl; } \
	return result;
#else
#define RETURN(x) return x 
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

util::KernelExtractorDriver util::KernelExtractorDriver::instance;

cuda::CudaDriverInterface * cuda::CudaDriverInterface::get() {
	return &util::KernelExtractorDriver::instance;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//
//static CUdeviceptr toDevicePtr(void *ptr) {
//	return hydrazine::bit_cast<CUdeviceptr>(ptr);
//}
//
static CUdeviceptr toDevicePtr(const void *ptr) {
	return hydrazine::bit_cast<CUdeviceptr>(ptr);
}

static void * fromDevicePtr(CUdeviceptr ptr) {
//	return reinterpret_cast<void *>(hydrazine::bit_cast<size_t>(ptr) & ((1ULL << 8*sizeof(unsigned int))-1));
	return hydrazine::bit_cast<void *>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

util::KernelExtractorDriver::KernelExtractorDriver() {
	
	cudaDriver._libname = "libcuda.so.1";
	cudaDriver.load();
	
	enabled = true;
}

util::KernelExtractorDriver::~KernelExtractorDriver() {
	report("~KernelExtractorDriver()");
}

////////////////////////////////////////////////////////////////////////////////////////////////////


//! \brief binds module handle to PTX image
void util::KernelExtractorDriver::loadModule(
	CUresult result, CUmodule module, 
	const char *ptxImage, 
	const char *name) {
	
	std::stringstream ss;
	ss << "module" << moduleNameMap.size();
	std::string modName = ss.str();
	
	if (name) {
		modName = name;	
	}
	
	if (ptxImage) {
		std::stringstream fss;
		fss << modName << ".ptx";
		state.modules[modName]->name = modName;
		state.modules[modName]->ptx = fss.str();
		std::ofstream file(fss.str());
		file << ptxImage;
	}
	else if (name) {
		modName = name;
		state.modules[modName]->name = modName;
		state.modules[modName]->ptx = name;
	}
	moduleNameMap[module] = modName;
}

//! \brief binds a function handle to a module and kernel name
void util::KernelExtractorDriver::bindKernel(
	CUresult result, 
	CUmodule module,
	CUfunction function, 
	const char *name) {

	ModuleNameMap::const_iterator mod_it = moduleNameMap.find(module);
	if (mod_it != moduleNameMap.end()) {
		std::pair<std::string,std::string> fname(mod_it->second, name);
		functionNameMap[function] = fname;
	}
}

//! \brief binds a texture handle to a module and texture name
void util::KernelExtractorDriver::bindTexture(
	CUresult result, 
	CUmodule module, 
	CUtexref texture, 
	const char *name) {
	
	ModuleNameMap::const_iterator mod_it = moduleNameMap.find(module);
	if (mod_it != moduleNameMap.end()) {
		std::pair<std::string,std::string> fname(mod_it->second, name);
		textureNameMap[texture] = fname;
	}
}


//! \brief binds a global variable to a pointer
void util::KernelExtractorDriver::bindGlobal(
	CUresult result, 
	CUmodule module, 
	void *ptr, 
	const char *name) {

	
}

/*!

*/
void util::KernelExtractorDriver::kernelLaunch(CUfunction f, int gridX, int gridY) {
	state.launch.gridDim = ir::Dim3(gridX, gridY, 1);
	FunctionNameMap::const_iterator f_it = functionNameMap.find(f);
	
	cuCtxSynchronize();	// wait for previous launches to conclude [somehow]	
	synchronizeFromDevice();
	
	if (f_it != functionNameMap.end()) {
		state.launch.moduleName = f_it->second.first;
		state.launch.kernelName = f_it->second.second;
	}
	else {
		state.launch.moduleName = "unknown_module";
		state.launch.kernelName = "unknown_kernel";
	}
	
	// serialize 'before' state
	std::string launchName = state.launch.moduleName + "-" + state.launch.kernelName;
	std::ofstream file(state.application.name + "-" + launchName + ".json");
	
	std::string app = state.application.name;
	state.application.name += "-before-" + launchName;
	state.serialize(file);
	state.application.name = app;
}


void util::KernelExtractorDriver::kernelReturn(CUresult result) {
	cuCtxSynchronize();
	
	synchronizeFromDevice();
	
	
	std::string launchName = state.launch.moduleName + "-" + state.launch.kernelName;
	std::ofstream file(state.application.name + "-" + launchName + ".json", std::ios_base::app);
	
	std::string app = state.application.name;
	state.application.name += "-after-" + state.launch.moduleName + "-" + state.launch.kernelName;
	state.serialize(file);
	state.application.name = app;
}


//! \brief copies data from device to host-side allocations
void util::KernelExtractorDriver::synchronizeFromDevice() {
	ExtractedDeviceState::GlobalAllocationMap::const_iterator alloc_it = state.globalAllocations.begin();
	for (; alloc_it != state.globalAllocations.end(); ++alloc_it) {
		ExtractedDeviceState::MemoryAllocation *allocation = alloc_it->second;
		CUresult result = cudaDriver.cuMemcpyDtoH(&allocation->data[0], 
			toDevicePtr(allocation->devicePointer), allocation->data.size());
		if (result != CUDA_SUCCESS) {
			// failed
			report("KernelExtractorDriver::synchronizeFromDevice() - failed to copy from device " 
				<< cuda::CudaDriver::toString(result));
			report("  source: " << (void *)allocation->devicePointer);
			break;
		}
	}
	
	ExtractedDeviceState::GlobalVariableMap::iterator global_it = state.globalVariables.begin();
	for (; global_it != state.globalVariables.end(); ++global_it) {
		ExtractedDeviceState::GlobalAllocation *allocation = global_it->second;
		
		CUmodule hModule = inverseModuleLookup(allocation->module);
		CUdeviceptr devicePtr;
		size_t size = 0;
		CUresult result = cudaDriver.cuModuleGetGlobal(&devicePtr, &size, hModule, allocation->name.c_str());
		if (result == CUDA_SUCCESS) {
			assert(allocation->data.size() >= size);
			CUresult result = cudaDriver.cuMemcpyDtoH(&allocation->data[0], devicePtr, 
				allocation->data.size());
			if (result != CUDA_SUCCESS) {
				// failed
				report("KernelExtractorDriver::synchronizeFromDevice() - failed to copy from device " 
					<< cuda::CudaDriver::toString(result));
				report("  source: " << (void *)devicePtr);
				break;
			}
		}
	}
}

//! \brief copies data to host-side allocations to device
void util::KernelExtractorDriver::synchronizeToDevice() {
	assert(0 && "unimplemented");
}

//! \brief allocates device memory
void util::KernelExtractorDriver::allocate(CUresult result, void *dptr, size_t bytes) {
	ExtractedDeviceState::MemoryAllocation *allocation = 
		new ExtractedDeviceState::MemoryAllocation(dptr, bytes);
	state.globalAllocations[dptr] = allocation;
	report("New allocation: " << dptr << " (" << bytes << ")");
}

//! \brief deletes an allocation
void util::KernelExtractorDriver::free(void *dptr) {
	ExtractedDeviceState::GlobalAllocationMap::iterator it = state.globalAllocations.find(dptr);
	if (it != state.globalAllocations.end()) {
		ExtractedDeviceState::MemoryAllocation *allocation = it->second;
		state.globalAllocations.erase(it);
		delete allocation;
	}
}

CUmodule util::KernelExtractorDriver::inverseModuleLookup(const std::string &name) {
	CUmodule hModule = 0;
	for (ModuleNameMap::const_iterator mod_it = moduleNameMap.begin(); mod_it != moduleNameMap.end();
		++mod_it) {
		if (mod_it->second == name) {
			return mod_it->first;
		}
	}
	return hModule;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*********************************
** Initialization
*********************************/
CUresult util::KernelExtractorDriver::cuInit(unsigned int Flags) {
	trace();
	RETURN( cudaDriver.cuInit(Flags) );
}


/*********************************
** Driver Version Query
*********************************/
CUresult util::KernelExtractorDriver::cuDriverGetVersion(int *driverVersion) {
	trace();
	CUresult res = (*cudaDriver.cuDriverGetVersion)(driverVersion);
	report(" cuDriverGetVersion = " << *driverVersion);
	RETURN( res );
}

CUresult util::KernelExtractorDriver::cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
	trace();
	CUresult res = cudaDriver.cuGetExportTable(ppExportTable, pExportTableId);	
	RETURN( res );
}

/************************************
**
**    Device management
**
***********************************/

CUresult util::KernelExtractorDriver::cuDeviceGet(CUdevice *device, int ordinal) {
	trace();	
	RETURN( cudaDriver.cuDeviceGet(device, ordinal) );
}

CUresult util::KernelExtractorDriver::cuDeviceGetCount(int *count) {
	trace();	
	RETURN( cudaDriver.cuDeviceGetCount(count) );
}

CUresult util::KernelExtractorDriver::cuDeviceGetName(char *name, int len, CUdevice dev) {
	trace();	
	RETURN( cudaDriver.cuDeviceGetName(name, len, dev) );
}

CUresult util::KernelExtractorDriver::cuDeviceComputeCapability(int *major, int *minor, 
	CUdevice dev) {
	trace();	
	RETURN( cudaDriver.cuDeviceComputeCapability(major, minor, dev) );
}

CUresult util::KernelExtractorDriver::cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
	trace();	
	RETURN( cudaDriver.cuDeviceTotalMem(bytes, dev) );
}

CUresult util::KernelExtractorDriver::cuDeviceGetProperties(CUdevprop *prop, 
	CUdevice dev) {
	trace();	
	RETURN( cudaDriver.cuDeviceGetProperties(prop, dev) );
}

CUresult util::KernelExtractorDriver::cuDeviceGetAttribute(int *pi, 
	CUdevice_attribute attrib, CUdevice dev) {
	trace();	
	RETURN( cudaDriver.cuDeviceGetAttribute(pi, attrib, dev) );
}


/************************************
**
**    Context management
**
***********************************/

CUresult util::KernelExtractorDriver::cuCtxCreate(CUcontext *pctx, unsigned int flags, 
	CUdevice dev ) {
	trace();	
	RETURN( cudaDriver.cuCtxCreate(pctx, flags, dev) );
}

CUresult util::KernelExtractorDriver::cuCtxDestroy( CUcontext ctx ) {
	trace();	
	RETURN( cudaDriver.cuCtxDestroy(ctx) );
}

CUresult util::KernelExtractorDriver::cuCtxAttach(CUcontext *pctx, unsigned int flags) {
	trace();	
	RETURN( cudaDriver.cuCtxAttach(pctx, flags) );
}

CUresult util::KernelExtractorDriver::cuCtxDetach(CUcontext ctx) {
	trace();	
	RETURN( cudaDriver.cuCtxDetach(ctx) );
}

CUresult util::KernelExtractorDriver::cuCtxPushCurrent( CUcontext ctx ) {
	trace();	
	RETURN( cudaDriver.cuCtxPushCurrent(ctx) );
}

CUresult util::KernelExtractorDriver::cuCtxPopCurrent( CUcontext *pctx ) {
	trace();	
	RETURN( cudaDriver.cuCtxPopCurrent(pctx) );
}

CUresult util::KernelExtractorDriver::cuCtxGetDevice(CUdevice *device) {
	trace();	
	RETURN( cudaDriver.cuCtxGetDevice(device) );
}

CUresult util::KernelExtractorDriver::cuCtxSynchronize(void) {
	trace();	
	RETURN( cudaDriver.cuCtxSynchronize() );
}


/************************************
**
**    Module management
**
***********************************/

CUresult util::KernelExtractorDriver::cuModuleLoad(CUmodule *module, const char *fname) {
	trace();	
	CUresult res = cudaDriver.cuModuleLoad(module, fname);
	if (enabled) {
		loadModule(res, *module, 0, fname);
	}
	RETURN( res );
}

CUresult util::KernelExtractorDriver::cuModuleLoadData(CUmodule *module, 
	const void *image) {
	trace();	
	CUresult res = cudaDriver.cuModuleLoadData(module, image);
	if (enabled) {
		loadModule(res, *module, (const char *)image);
	}
	RETURN( res );
}

CUresult util::KernelExtractorDriver::cuModuleLoadDataEx(CUmodule *module, 
	const void *image, unsigned int numOptions, 
	CUjit_option *options, void **optionValues) {
	trace();
	CUresult res = cudaDriver.cuModuleLoadDataEx(module, image, numOptions, options, optionValues);
	if (enabled) {
		loadModule(res, *module, (const char *)image);
	}
	RETURN( res );
}

CUresult util::KernelExtractorDriver::cuModuleLoadFatBinary(CUmodule *module, 
	const void *fatCubin) {
	trace();
	cuda::FatBinaryContext fatBin(fatCubin);
	CUresult res = cudaDriver.cuModuleLoadFatBinary(module, fatCubin);
	if (enabled) {
		loadModule(res, *module, fatBin.ptx(), fatBin.name());	
	}
	RETURN( res );
}

CUresult util::KernelExtractorDriver::cuModuleUnload(CUmodule hmod) {
	trace();	
	RETURN( cudaDriver.cuModuleUnload(hmod) );
}

CUresult util::KernelExtractorDriver::cuModuleGetFunction(CUfunction *hfunc, 
	CUmodule hmod, const char *name) {
	trace();	
	CUresult res = cudaDriver.cuModuleGetFunction(hfunc, hmod, name);
	if (enabled) {
		bindKernel(res, hmod, *hfunc, name);
	}	
	RETURN( res );
}

CUresult util::KernelExtractorDriver::cuModuleGetGlobal(CUdeviceptr *dptr, 
	size_t *bytes, CUmodule hmod, const char *name) {
	trace();	
	RETURN( cudaDriver.cuModuleGetGlobal(dptr, bytes, hmod, name) );
}

CUresult util::KernelExtractorDriver::cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, 
	const char *name) {
	trace();
	CUresult res = cudaDriver.cuModuleGetTexRef(pTexRef, hmod, name);
	if (enabled) {
		bindTexture(res, hmod, *pTexRef, name);
	}
	RETURN( res );
}


/************************************
**
**    Memory management
**
***********************************/

CUresult util::KernelExtractorDriver::cuMemGetInfo(size_t *free, 
	size_t *total) {
	trace();	
	RETURN( cudaDriver.cuMemGetInfo(free, total) );
}


CUresult util::KernelExtractorDriver::cuMemAlloc( CUdeviceptr *dptr, 
	unsigned int bytesize) {
	trace();	
	
	CUresult res = cudaDriver.cuMemAlloc(dptr, bytesize);
	
	if (enabled) {
		allocate(res, fromDevicePtr(*dptr), bytesize);
	}
	
	RETURN( res );
}

CUresult util::KernelExtractorDriver::cuMemAllocPitch( CUdeviceptr *dptr, 
			          size_t *pPitch,
			          unsigned int WidthInBytes, 
			          unsigned int Height, 
			          unsigned int ElementSizeBytes
			         ) {
	trace();	
	CUresult res = cudaDriver.cuMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
	if (enabled) {
		allocate(res, fromDevicePtr(*dptr), *pPitch * Height);
	}
	RETURN( res );
}

CUresult util::KernelExtractorDriver::cuMemFree(CUdeviceptr dptr) {
	trace();
	CUresult res = cudaDriver.cuMemFree(dptr);
	if (enabled) {
		free(fromDevicePtr(dptr));
	}
	RETURN( res );
}

CUresult util::KernelExtractorDriver::cuMemGetAddressRange( CUdeviceptr *pbase, 
	size_t *psize, CUdeviceptr dptr ) {
	trace();	
	RETURN( cudaDriver.cuMemGetAddressRange(pbase, psize, dptr) );
}


CUresult util::KernelExtractorDriver::cuMemAllocHost(void **pp, unsigned int bytesize) {
	trace();	
	RETURN( cudaDriver.cuMemAllocHost(pp, bytesize) );
}

CUresult util::KernelExtractorDriver::cuMemFreeHost(void *p) {
	trace();	
	RETURN( cudaDriver.cuMemFreeHost(p) );
}


CUresult util::KernelExtractorDriver::cuMemHostAlloc(void **pp, 
	unsigned long long bytesize, unsigned int Flags ) {
	trace();	
	RETURN( cudaDriver.cuMemHostAlloc(pp, bytesize, Flags) );
}


CUresult util::KernelExtractorDriver::cuMemHostGetDevicePointer( CUdeviceptr *pdptr, 
	void *p, unsigned int Flags ) {
	trace();	
	RETURN( cudaDriver.cuMemHostGetDevicePointer(pdptr, p, Flags) );
}

CUresult util::KernelExtractorDriver::cuMemHostGetFlags( unsigned int *pFlags, void *p ) {
	trace();	
	RETURN( cudaDriver.cuMemHostGetFlags(pFlags, p) );
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
CUresult util::KernelExtractorDriver::cuMemcpyHtoD (CUdeviceptr dstDevice, 
	const void *srcHost, unsigned int ByteCount ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyHtoD(dstDevice, srcHost, ByteCount) );
}

CUresult util::KernelExtractorDriver::cuMemcpyDtoH (void *dstHost, CUdeviceptr srcDevice, 
	unsigned int ByteCount ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyDtoH(dstHost, srcDevice, ByteCount) );
}


// device <-> device memory
CUresult util::KernelExtractorDriver::cuMemcpyDtoD (CUdeviceptr dstDevice, 
	CUdeviceptr srcDevice, unsigned int ByteCount ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyDtoD(dstDevice, srcDevice, ByteCount) );
}


// device <-> array memory
CUresult util::KernelExtractorDriver::cuMemcpyDtoA ( CUarray dstArray, 
	unsigned int dstIndex, CUdeviceptr srcDevice, 
	unsigned int ByteCount ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyDtoA(dstArray, dstIndex, srcDevice, ByteCount) );
}

CUresult util::KernelExtractorDriver::cuMemcpyAtoD ( CUdeviceptr dstDevice, 
	CUarray hSrc, unsigned int SrcIndex, unsigned int ByteCount ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyAtoD(dstDevice, hSrc, SrcIndex, ByteCount) );
}


// system <-> array memory
CUresult util::KernelExtractorDriver::cuMemcpyHtoA( CUarray dstArray, 
	unsigned int dstIndex, const void *pSrc, 
	unsigned int ByteCount ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyHtoA(dstArray, dstIndex, pSrc, ByteCount) );
}

CUresult util::KernelExtractorDriver::cuMemcpyAtoH( void *dstHost, CUarray srcArray, 
	unsigned int srcIndex, unsigned int ByteCount ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyAtoH(dstHost, srcArray, srcIndex, ByteCount) );
}


// array <-> array memory
CUresult util::KernelExtractorDriver::cuMemcpyAtoA( CUarray dstArray, 
	unsigned int dstIndex, CUarray srcArray, unsigned int srcIndex, 
	unsigned int ByteCount ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyAtoA(dstArray, dstIndex, srcArray, srcIndex, ByteCount) );
}


// 2D memcpy

CUresult util::KernelExtractorDriver::cuMemcpy2D( const CUDA_MEMCPY2D *pCopy ) {
	trace();	
	RETURN( cudaDriver.cuMemcpy2D(pCopy) );
}

CUresult util::KernelExtractorDriver::cuMemcpy2DUnaligned( const CUDA_MEMCPY2D *pCopy ) {
	trace();	
	RETURN( cudaDriver.cuMemcpy2DUnaligned(pCopy) );
}


// 3D memcpy

CUresult util::KernelExtractorDriver::cuMemcpy3D( const CUDA_MEMCPY3D *pCopy ) {
	trace();	
	RETURN( cudaDriver.cuMemcpy3D(pCopy) );
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
CUresult util::KernelExtractorDriver::cuMemcpyHtoDAsync (CUdeviceptr dstDevice, 
	const void *srcHost, unsigned int ByteCount, CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyHtoDAsync(dstDevice, srcHost, ByteCount, hStream) );
}

CUresult util::KernelExtractorDriver::cuMemcpyDtoHAsync (void *dstHost, 
	CUdeviceptr srcDevice, unsigned int ByteCount, CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyDtoHAsync(dstHost, srcDevice, ByteCount, hStream) );
}


// system <-> array memory
CUresult util::KernelExtractorDriver::cuMemcpyHtoAAsync( CUarray dstArray, 
	unsigned int dstIndex, const void *pSrc, 
	unsigned int ByteCount, CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyHtoAAsync(dstArray, dstIndex, pSrc, ByteCount, hStream) );
}

CUresult util::KernelExtractorDriver::cuMemcpyAtoHAsync( void *dstHost, CUarray srcArray, 
	unsigned int srcIndex, unsigned int ByteCount, 
	CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuMemcpyAtoHAsync(dstHost, srcArray, srcIndex, ByteCount, hStream) );
}

// 2D memcpy
CUresult util::KernelExtractorDriver::cuMemcpy2DAsync( const CUDA_MEMCPY2D *pCopy, 
	CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuMemcpy2DAsync(pCopy, hStream) );
}


// 3D memcpy
CUresult util::KernelExtractorDriver::cuMemcpy3DAsync( const CUDA_MEMCPY3D *pCopy, 
	CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuMemcpy3DAsync(pCopy, hStream) );
}


/************************************
**
**    Memset
**
***********************************/
CUresult util::KernelExtractorDriver::cuMemsetD8( CUdeviceptr dstDevice, 
	unsigned char uc, unsigned int N ) {
	trace();	
	RETURN( cudaDriver.cuMemsetD8(dstDevice, uc, N) );
}

CUresult util::KernelExtractorDriver::cuMemsetD16( CUdeviceptr dstDevice, 
	unsigned short us, unsigned int N ) {
	trace();	
	RETURN( cudaDriver.cuMemsetD16(dstDevice, us, N) );
}

CUresult util::KernelExtractorDriver::cuMemsetD32( CUdeviceptr dstDevice, 
	unsigned int ui, unsigned int N ) {
	trace();	
	RETURN( cudaDriver.cuMemsetD32(dstDevice, ui, N) );
}


CUresult util::KernelExtractorDriver::cuMemsetD2D8( CUdeviceptr dstDevice,
	unsigned int dstPitch, unsigned char uc, unsigned int Width, 
	unsigned int Height ) {
	trace();	
	RETURN( cudaDriver.cuMemsetD2D8(dstDevice, dstPitch, uc, Width, Height) );
}

CUresult util::KernelExtractorDriver::cuMemsetD2D16( CUdeviceptr dstDevice, 
	unsigned int dstPitch, unsigned short us, unsigned int Width, 
	unsigned int Height ) {
	trace();	
	RETURN( cudaDriver.cuMemsetD2D16(dstDevice, dstPitch, us, Width, Height) );
}

CUresult util::KernelExtractorDriver::cuMemsetD2D32( CUdeviceptr dstDevice, 
	unsigned int dstPitch, unsigned int ui, unsigned int Width, 
	unsigned int Height ) {
	trace();	
	RETURN( cudaDriver.cuMemsetD2D32(dstDevice, dstPitch, ui, Width, Height) );
}


/************************************
**
**    Function management
**
***********************************/


CUresult util::KernelExtractorDriver::cuFuncSetBlockShape (CUfunction hfunc, int x, 
	int y, int z) {
	trace();
	if (enabled) {
		state.launch.blockDim = ir::Dim3(x, y, z);
	}
	RETURN( cudaDriver.cuFuncSetBlockShape(hfunc, x, y, z) );
}

CUresult util::KernelExtractorDriver::cuFuncSetSharedSize (CUfunction hfunc, 
	unsigned int bytes) {
	trace();	
	if (enabled) {
		state.launch.sharedMemorySize = bytes;
	}
	RETURN( cudaDriver.cuFuncSetSharedSize(hfunc, bytes) );
}

CUresult util::KernelExtractorDriver::cuFuncGetAttribute (int *pi, 
	CUfunction_attribute attrib, CUfunction hfunc) {
	trace();	
	RETURN( cudaDriver.cuFuncGetAttribute(pi, attrib, hfunc) );
}


/************************************
**
**    Array management 
**
***********************************/

CUresult util::KernelExtractorDriver::cuArrayCreate( CUarray *pHandle, 
	const CUDA_ARRAY_DESCRIPTOR *pAllocateArray ) {
	trace();	
	RETURN( cudaDriver.cuArrayCreate(pHandle, pAllocateArray) );
}

CUresult util::KernelExtractorDriver::cuArrayGetDescriptor( 
	CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray ) {
	trace();	
	RETURN( cudaDriver.cuArrayGetDescriptor(pArrayDescriptor, hArray) );
}

CUresult util::KernelExtractorDriver::cuArrayDestroy( CUarray hArray ) {
	trace();	
	RETURN( cudaDriver.cuArrayDestroy(hArray) );
}


CUresult util::KernelExtractorDriver::cuArray3DCreate( CUarray *pHandle, 
	const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray ) {
	trace();	
	RETURN( cudaDriver.cuArray3DCreate(pHandle, pAllocateArray) );
}

CUresult util::KernelExtractorDriver::cuArray3DGetDescriptor( 
	CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray ) {
	trace();	
	RETURN( cudaDriver.cuArray3DGetDescriptor(pArrayDescriptor, hArray) );
}



/************************************
**
**    Texture reference management
**
***********************************/
CUresult util::KernelExtractorDriver::cuTexRefCreate( CUtexref *pTexRef ) {
	trace();	
	RETURN( cudaDriver.cuTexRefCreate(pTexRef) );
}

CUresult util::KernelExtractorDriver::cuTexRefDestroy( CUtexref hTexRef ) {
	trace();	
	RETURN( cudaDriver.cuTexRefDestroy(hTexRef) );
}


CUresult util::KernelExtractorDriver::cuTexRefSetArray( CUtexref hTexRef, CUarray hArray, 
	unsigned int Flags ) {
	trace();	
	RETURN( cudaDriver.cuTexRefSetArray(hTexRef, hArray, Flags) );
}

CUresult util::KernelExtractorDriver::cuTexRefSetAddress( size_t *ByteOffset, 
	CUtexref hTexRef, CUdeviceptr dptr, size_t bytes ) {
	trace();	
	RETURN( cudaDriver.cuTexRefSetAddress(ByteOffset, hTexRef, dptr, bytes) );
}

CUresult util::KernelExtractorDriver::cuTexRefSetAddress2D( CUtexref hTexRef, 
	const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, 
	unsigned int Pitch) {
	trace();	
	RETURN( cudaDriver.cuTexRefSetAddress2D(hTexRef, desc, dptr, Pitch) );
}

CUresult util::KernelExtractorDriver::cuTexRefSetFormat( CUtexref hTexRef, 
	CUarray_format fmt, int NumPackedComponents ) {
	trace();	
	RETURN( cudaDriver.cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents) );
}

CUresult util::KernelExtractorDriver::cuTexRefSetAddressMode( CUtexref hTexRef, int dim, 
	CUaddress_mode am ) {
	trace();	
	RETURN( cudaDriver.cuTexRefSetAddressMode(hTexRef, dim, am) );
}

CUresult util::KernelExtractorDriver::cuTexRefSetFilterMode( CUtexref hTexRef, 
	CUfilter_mode fm ) {
	trace();	
	RETURN( cudaDriver.cuTexRefSetFilterMode(hTexRef, fm) );
}

CUresult util::KernelExtractorDriver::cuTexRefSetFlags( CUtexref hTexRef, 
	unsigned int Flags ) {
	trace();	
	RETURN( cudaDriver.cuTexRefSetFlags(hTexRef, Flags) );
}


CUresult util::KernelExtractorDriver::cuTexRefGetAddress( CUdeviceptr *pdptr, 
	CUtexref hTexRef ) {
	trace();	
	RETURN( cudaDriver.cuTexRefGetAddress(pdptr, hTexRef) );
}

CUresult util::KernelExtractorDriver::cuTexRefGetArray( CUarray *phArray, 
	CUtexref hTexRef ) {
	trace();	
	RETURN( cudaDriver.cuTexRefGetArray(phArray, hTexRef) );
}

CUresult util::KernelExtractorDriver::cuTexRefGetAddressMode( CUaddress_mode *pam, 
	CUtexref hTexRef, int dim ) {
	trace();	
	RETURN( cudaDriver.cuTexRefGetAddressMode(pam, hTexRef, dim) );
}

CUresult util::KernelExtractorDriver::cuTexRefGetFilterMode( CUfilter_mode *pfm, 
	CUtexref hTexRef ) {
	trace();	
	RETURN( cudaDriver.cuTexRefGetFilterMode(pfm, hTexRef) );
}

CUresult util::KernelExtractorDriver::cuTexRefGetFormat( CUarray_format *pFormat, 
	int *pNumChannels, CUtexref hTexRef ) {
	trace();	
	RETURN( cudaDriver.cuTexRefGetFormat(pFormat, pNumChannels, hTexRef) );
}

CUresult util::KernelExtractorDriver::cuTexRefGetFlags( unsigned int *pFlags, 
	CUtexref hTexRef ) {
	trace();	
	RETURN( cudaDriver.cuTexRefGetFlags(pFlags, hTexRef) );
}


/************************************
**
**    Parameter management
**
***********************************/

CUresult util::KernelExtractorDriver::cuParamSetSize (CUfunction hfunc, 
	unsigned int numbytes) {
	trace();
	if (enabled) {
		state.launch.parameterMemory.resize(numbytes, 0);
		report("  setting size " << numbytes);
	}
	RETURN( cudaDriver.cuParamSetSize(hfunc, numbytes) );
}

CUresult util::KernelExtractorDriver::cuParamSeti    (CUfunction hfunc, int offset, 
	unsigned int value) {
	trace();
	if (enabled) {
		std::memcpy(&state.launch.parameterMemory[offset], &value, sizeof(value));
		report("  offset " << offset << ", value " << value);
	}
	RETURN( cudaDriver.cuParamSeti(hfunc, offset, value) );
}

CUresult util::KernelExtractorDriver::cuParamSetf    (CUfunction hfunc, int offset, 
	float value) {
	trace();	
	if (enabled) {
		std::memcpy(&state.launch.parameterMemory[offset], &value, sizeof(value));
		report("  offset " << offset << ", value " << value);
	}
	RETURN( cudaDriver.cuParamSetf(hfunc, offset, value) );
}

CUresult util::KernelExtractorDriver::cuParamSetv    (CUfunction hfunc, int offset, 
	void * ptr, unsigned int numbytes) {
	trace();	
	if (enabled) {
		std::memcpy(&state.launch.parameterMemory[offset], ptr, numbytes);
		
		report("  offset: " << offset << ", size: " << numbytes << " bytes");
		unsigned int *intPtr = (unsigned int *)ptr;
		for (unsigned int i = 0; i < numbytes/4; i++) {
			report("  0x" << std::setw(8) << std::setfill('0') << std::setbase(16) << intPtr[i] << std::setbase(10));
		}
	}
	RETURN( cudaDriver.cuParamSetv(hfunc, offset, ptr, numbytes) );
}

CUresult util::KernelExtractorDriver::cuParamSetTexRef(CUfunction hfunc, int texunit, 
	CUtexref hTexRef) {
	trace();
	assert(0 && "unimplemented");
	RETURN( cudaDriver.cuParamSetTexRef(hfunc, texunit, hTexRef) );
}


/************************************
**
**    Launch functions
**

Serialize before state of each kernel,
time kernel execution,
serialize after state

***********************************/

CUresult util::KernelExtractorDriver::cuLaunch ( CUfunction f ) {
	trace();	
	if (enabled) {
		kernelLaunch(f);
	}
	
	CUresult res = cudaDriver.cuLaunch(f);
	if (enabled) {
		kernelReturn(res);
	}
	
	RETURN( res );
}

CUresult util::KernelExtractorDriver::cuLaunchGrid (CUfunction f, int grid_width, 
	int grid_height) {
	trace();
	if (enabled) {
		kernelLaunch(f, grid_width, grid_height);
	}
	CUresult res = cudaDriver.cuLaunchGrid(f, grid_width, grid_height);
	if (enabled) {
		kernelReturn(res);
	}
	RETURN( res );
}

CUresult util::KernelExtractorDriver::cuLaunchGridAsync( CUfunction f, int grid_width, 
	int grid_height, CUstream hStream ) {
	trace();	
	
	if (enabled) {
		kernelLaunch(f, grid_width, grid_height);
	}
	CUresult res = cudaDriver.cuLaunchGridAsync(f, grid_width, grid_height, hStream);
	if (enabled) {
		kernelReturn(res);
	}
	RETURN(res);
}


/************************************
**
**    Events
**
***********************************/
CUresult util::KernelExtractorDriver::cuEventCreate( CUevent *phEvent, 
	unsigned int Flags ) {
	trace();	
	RETURN( cudaDriver.cuEventCreate(phEvent, Flags) );
}

CUresult util::KernelExtractorDriver::cuEventRecord( CUevent hEvent, CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuEventRecord(hEvent, hStream) );
}

CUresult util::KernelExtractorDriver::cuEventQuery( CUevent hEvent ) {
	trace();	
	RETURN( cudaDriver.cuEventQuery(hEvent) );
}

CUresult util::KernelExtractorDriver::cuEventSynchronize( CUevent hEvent ) {
	trace();	
	RETURN( cudaDriver.cuEventSynchronize(hEvent) );
}

CUresult util::KernelExtractorDriver::cuEventDestroy( CUevent hEvent ) {
	trace();	
	RETURN( cudaDriver.cuEventDestroy(hEvent) );
}

CUresult util::KernelExtractorDriver::cuEventElapsedTime( float *pMilliseconds, 
	CUevent hStart, CUevent hEnd ) {
	trace();	
	RETURN( cudaDriver.cuEventElapsedTime(pMilliseconds, hStart, hEnd) );
}


/************************************
**
**    Streams
**
***********************************/
CUresult util::KernelExtractorDriver::cuStreamCreate( CUstream *phStream, 
	unsigned int Flags ) {
	trace();	
	RETURN( cudaDriver.cuStreamCreate(phStream, Flags) );
}

CUresult util::KernelExtractorDriver::cuStreamQuery( CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuStreamQuery(hStream) );
}

CUresult util::KernelExtractorDriver::cuStreamSynchronize( CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuStreamSynchronize(hStream) );
}

CUresult util::KernelExtractorDriver::cuStreamDestroy( CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuStreamDestroy(hStream) );
}


/************************************
**
**    Graphics
**
***********************************/
CUresult util::KernelExtractorDriver::cuGraphicsUnregisterResource(
	CUgraphicsResource resource) {
	trace();	
	RETURN( cudaDriver.cuGraphicsUnregisterResource(resource) );
}

CUresult util::KernelExtractorDriver::cuGraphicsSubResourceGetMappedArray(
	CUarray *pArray, CUgraphicsResource resource, 
	unsigned int arrayIndex, unsigned int mipLevel ) {
	trace();	
	RETURN( cudaDriver.cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel) );
}

CUresult util::KernelExtractorDriver::cuGraphicsResourceGetMappedPointer(
	CUdeviceptr *pDevPtr, size_t *pSize, 
	CUgraphicsResource resource ) {
	trace();	
	RETURN( cudaDriver.cuGraphicsResourceGetMappedPointer(pDevPtr, pSize, resource) );
}

CUresult util::KernelExtractorDriver::cuGraphicsResourceSetMapFlags(
	CUgraphicsResource resource, unsigned int flags ) {
	trace();	
	RETURN( cudaDriver.cuGraphicsResourceSetMapFlags(resource, flags) );
}
 
CUresult util::KernelExtractorDriver::cuGraphicsMapResources(unsigned int count, 
	CUgraphicsResource *resources, CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuGraphicsMapResources(count, resources, hStream) );
}

CUresult util::KernelExtractorDriver::cuGraphicsUnmapResources(unsigned int count, 
	CUgraphicsResource *resources, CUstream hStream ) {
	trace();	
	RETURN( cudaDriver.cuGraphicsUnmapResources(count, resources, hStream) );
}


/************************************
**
**    OpenGL
**
***********************************/
CUresult util::KernelExtractorDriver::cuGLInit() {
	trace();	
	RETURN( cudaDriver.cuGLInit() );
}

CUresult util::KernelExtractorDriver::cuGLCtxCreate(CUcontext *pCtx, 
	unsigned int Flags, CUdevice device) {
	trace();	
	RETURN( cudaDriver.cuGLCtxCreate(pCtx, Flags, device) );
}

CUresult util::KernelExtractorDriver::cuGraphicsGLRegisterBuffer( 
	CUgraphicsResource *pCudaResource, unsigned int buffer, 
	unsigned int Flags ) {
	trace();	
	RETURN( cudaDriver.cuGraphicsGLRegisterBuffer(pCudaResource, buffer, Flags) );
}

CUresult util::KernelExtractorDriver::cuGraphicsGLRegisterImage( 
	CUgraphicsResource *pCudaResource, unsigned int image, 
	int target, unsigned int Flags) {
	trace();	
	RETURN( cudaDriver.cuGraphicsGLRegisterImage(pCudaResource, image, target, Flags) );
}

CUresult util::KernelExtractorDriver::cuGLRegisterBufferObject(GLuint bufferobj) {
	trace();
	RETURN (cudaDriver.cuGLRegisterBufferObject(bufferobj));
}

CUresult util::KernelExtractorDriver::cuGLSetBufferObjectMapFlags(GLuint buffer, unsigned int flags) {
	trace();
	RETURN ( cudaDriver.cuGLSetBufferObjectMapFlags(buffer, flags) );
}


