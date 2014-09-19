/*! \file CalDriver.cpp
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 8, 2010
 *  \brief The source file for the CalDriver class.
 */

// Ocelot includes
#include <ocelot/cal/interface/CalDriver.h>

// hydrazine includes
#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>

// Linux system headers
#if __GNUC__
	#include <dlfcn.h>
#else 
	// TODO Add dynamic loading support on windows
	#define dlopen(a,b) 0
	#define dlclose(a) -1
	#define dlerror() "Unknown error"
	#define dlsym(a,b) 0
#endif

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

#define Throw(x) {std::stringstream s; s << x; \
	throw hydrazine::Exception(s.str()); }

namespace cal
{
	// start with an uninitialized singleton
	const CalDriver *CalDriver::_instance = 0;

	CalDriver::CalDriver() : _driver(0), _compiler(0)
	{
		report("Loading libaticalrt.so");
		#if __GNUC__
		_driver = dlopen("libaticalrt.so", RTLD_LAZY);
		if (_driver == 0) {
			Throw("Failed to load cal driver: " << dlerror());
		}

		report("Loading libaticalcl.so");
		_compiler = dlopen("libaticalcl.so", RTLD_LAZY);
		if (_compiler == 0) {
			dlclose(_driver);
			_driver = 0;
			Throw("Failed to load cal compiler driver: " << dlerror());
		}

		hydrazine::bit_cast(_calInit,              dlsym(_driver, "calInit"));
		hydrazine::bit_cast(_calShutdown,          dlsym(_driver, "calShutdown"));
		hydrazine::bit_cast(_calDeviceGetCount,    dlsym(_driver, "calDeviceGetCount"));
		hydrazine::bit_cast(_calDeviceGetAttribs,  dlsym(_driver, "calDeviceGetAttribs"));
		hydrazine::bit_cast(_calDeviceGetStatus,   dlsym(_driver, "calDeviceGetStatus"));
		hydrazine::bit_cast(_calDeviceOpen,        dlsym(_driver, "calDeviceOpen"));
		hydrazine::bit_cast(_calDeviceClose,       dlsym(_driver, "calDeviceClose"));
		hydrazine::bit_cast(_calDeviceGetInfo,     dlsym(_driver, "calDeviceGetInfo"));
		hydrazine::bit_cast(_calCtxCreate,         dlsym(_driver, "calCtxCreate"));
		hydrazine::bit_cast(_calCtxDestroy,        dlsym(_driver, "calCtxDestroy"));
		hydrazine::bit_cast(_calCtxGetMem,         dlsym(_driver, "calCtxGetMem"));
		hydrazine::bit_cast(_calCtxReleaseMem,     dlsym(_driver, "calCtxReleaseMem"));
		hydrazine::bit_cast(_calCtxSetMem,         dlsym(_driver, "calCtxSetMem"));
		hydrazine::bit_cast(_calModuleLoad,        dlsym(_driver, "calModuleLoad"));
		hydrazine::bit_cast(_calModuleUnload,      dlsym(_driver, "calModuleUnload"));
		hydrazine::bit_cast(_calModuleGetEntry,    dlsym(_driver, "calModuleGetEntry"));
		hydrazine::bit_cast(_calModuleGetName,     dlsym(_driver, "calModuleGetName"));
		hydrazine::bit_cast(_calResAllocLocal1D,   dlsym(_driver, "calResAllocLocal1D"));
		hydrazine::bit_cast(_calResAllocRemote1D,  dlsym(_driver, "calResAllocRemote1D"));
		hydrazine::bit_cast(_calResFree,           dlsym(_driver, "calResFree"));
		hydrazine::bit_cast(_calResMap,            dlsym(_driver, "calResMap"));
		hydrazine::bit_cast(_calResUnmap,          dlsym(_driver, "calResUnmap"));
		hydrazine::bit_cast(_calCtxRunProgramGrid, dlsym(_driver, "calCtxRunProgramGrid"));
		hydrazine::bit_cast(_calCtxIsEventDone,    dlsym(_driver, "calCtxIsEventDone"));
		hydrazine::bit_cast(_calGetErrorString,    dlsym(_driver, "calGetErrorString"));

		hydrazine::bit_cast(_calclCompile,    dlsym(_compiler, "calclCompile"));
		hydrazine::bit_cast(_calclLink,       dlsym(_compiler, "calclLink"));
		hydrazine::bit_cast(_calclFreeObject, dlsym(_compiler, "calclFreeObject"));
		hydrazine::bit_cast(_calclFreeImage,  dlsym(_compiler, "calclFreeImage"));

		calInit();
		#else
		assertM(false, "CAL Driver support not compiled into Ocelot.");
		#endif
	}

	CalDriver::~CalDriver()
	{
		calShutdown();

		#if __GNUC__
		if (_driver) {
			dlclose(_driver);
		}

		if (_compiler) {
			dlclose(_compiler);
		}
		#else
		assertM(false, "CAL Driver support not compiled into Ocelot.");
		#endif

        // don't delete _instance as this would call the destructor again
        // let the memory be reclaimed when the program terminates
	}

	const CalDriver *CalDriver::Instance()
	{
		if (_instance == 0) {
			_instance = new CalDriver;
		}

		return _instance;
	}

	
	void CalDriver::calInit() const
	{
		_checkError((*_calInit)());

		report("calInit()");
	}

	void CalDriver::calShutdown() const
	{
		_checkError((*_calShutdown)());

		report("calShutdown()");
	}

	void CalDriver::calDeviceGetCount(CALuint *count) const
	{
		if(_driver == 0)
		{
			*count = 0;
		}
		else
		{	
		_checkError((*_calDeviceGetCount)(count));

		report("calDeviceGetCount("
				<< "*count = " << std::dec << *count
				<< ")");
		}
	}

	void CalDriver::calDeviceGetAttribs(CALdeviceattribs *attribs,
			CALuint ordinal) const
	{
		_checkError((*_calDeviceGetAttribs)(attribs, ordinal));

		report("calDeviceGetAttribs()");
	}

	void CalDriver::calDeviceGetStatus(CALdevicestatus *status, 
			CALdevice dev) const
	{
		_checkError((*_calDeviceGetStatus)(status, dev));

		report("calDeviceGetStatus()");
	}	

	void CalDriver::calDeviceOpen(CALdevice *dev, CALuint ordinal) const
	{
		_checkError((*_calDeviceOpen)(dev, ordinal));

		report("calDeviceOpen("
				<< "*dev = " << std::hex << std::showbase << *dev
				<< ", ordinal = " << std::dec << ordinal
				<< ")");
	}

	void CalDriver::calDeviceClose(CALdevice dev) const
	{
		_checkError((*_calDeviceClose)(dev));

		report("calDeviceClose("
				<< "dev = " << std::hex << std::showbase << dev
				<< ")");
	}

	void CalDriver::calDeviceGetInfo(CALdeviceinfo *info, CALuint ordinal) const
	{
		_checkError((*_calDeviceGetInfo)(info, ordinal));
		
		report("calDeviceGetInfo("
				<< "info = " << std::hex << std::showbase << info
				<< ", ordinal = " << std::dec << ordinal
				<< ")");
	}

	void CalDriver::calCtxCreate(CALcontext *ctx, CALdevice dev) const
	{
		_checkError((*_calCtxCreate)(ctx, dev));
		
		report("calCtxCreate("
				<< "*ctx = " << std::hex << std::showbase << *ctx
				<< ", dev = " << std::hex << std::showbase << dev
				<< ")");
	}

	void CalDriver::calCtxDestroy(CALcontext ctx) const
	{
		_checkError((*_calCtxDestroy)(ctx));
		
		report("calCtxDestroy("
				<< "ctx = " << std::hex << std::showbase << ctx
				<< ")");
	}

	void CalDriver::calCtxGetMem(CALmem *mem, CALcontext ctx, 
			CALresource res) const
	{
		_checkError((*_calCtxGetMem)(mem, ctx, res));

		report("calCtxGetMem("
				<< "*mem = " << std::hex << std::showbase << *mem
				<< ", ctx = " << std::hex << std::showbase << ctx
				<< ", res = " << std::hex << std::showbase << res
				<< ")");
	}

	void CalDriver::calCtxReleaseMem(CALcontext ctx, CALmem mem) const
	{
		_checkError((*_calCtxReleaseMem)(ctx, mem));

		report("calCtxReleaseMem("
				<< "ctx = " << std::hex << std::showbase << ctx
				<< ", mem = " << std::hex << std::showbase << mem
				<< ")");
	}

	void CalDriver::calCtxSetMem(CALcontext ctx, CALname name, CALmem mem) const
	{
		_checkError((*_calCtxSetMem)(ctx, name, mem));

		report("calCtxSetMem("
				<< "ctx = " << std::hex << std::showbase << ctx
				<< ", name = " << std::hex << std::showbase << name
				<< ", mem = " << std::hex << std::showbase << mem
				<< ")");
	}

	void CalDriver::calModuleLoad(CALmodule* module, CALcontext ctx, 
			CALimage image) const
	{
		_checkError((*_calModuleLoad)(module, ctx, image));

		report("calModuleLoad("
				<< "*module = " << std::hex << std::showbase << *module
				<< ", ctx = " << std::hex << std::showbase << ctx
				<< ", image = " << std::hex << std::showbase << image
				<< ")");
	}

	void CalDriver::calModuleUnload(CALcontext ctx, CALmodule module) const
	{
		_checkError((*_calModuleUnload)(ctx, module));

		report("calModuleUnLoad("
				<< "ctx = " << std::hex << std::showbase << ctx
				<< ", module = " << std::hex << std::showbase << module
				<< ")");
	}

	void CalDriver::calModuleGetEntry(CALfunc* func, CALcontext ctx, 
			CALmodule module, const CALchar* procName) const
	{
		_checkError((*_calModuleGetEntry)(func, ctx, module, procName));

		report("calModuleGetEntry("
				<< "*func = " << std::hex << std::showbase << *func
				<< ", ctx = " << std::hex << std::showbase << ctx
				<< ", module = " << std::hex << std::showbase << module
				<< ", procName = " << procName
				<< ")");
	}

	void CalDriver::calModuleGetName(CALname* name, CALcontext ctx, 
			CALmodule module, const CALchar* varName) const
	{
		_checkError((*_calModuleGetName)(name, ctx, module, varName));

		report("calModuleGetName("
				<< "*name = " << std::hex << std::showbase << *name
				<< ", ctx = " << std::hex << std::showbase << ctx
				<< ", module = " << std::hex << std::showbase << module
				<< ", varName = " << varName
				<< ")");
	}

	void CalDriver::calResAllocLocal1D(CALresource* res, CALdevice dev, 
			CALuint width, CALformat format, CALuint flags) const
	{
		_checkError((*_calResAllocLocal1D)(res, dev, width, format, flags));

		report("calResAllocLocal1D("
				<< "*res = " << std::hex << std::showbase << *res
				<< ", dev = " << dev
				<< ", width = " << std::dec << width
				<< ", format = " << std::dec << format
				<< ", flags = " << std::dec << flags 
				<< ")");
	}

	void CalDriver::calResAllocRemote1D(CALresource* res, CALdevice* dev, 
			CALuint deviceCount, CALuint width, CALformat format, 
			CALuint flags) const
	{
		_checkError((*_calResAllocRemote1D)(res, dev, deviceCount, width, 
					format, flags));

		report("calResAllocRemote1D("
				<< "*res = " << std::hex << std::showbase << *res
				<< ", *dev = " << *dev
				<< ", deviceCount = " << std::dec << deviceCount
				<< ", width = " << std::dec << width
				<< ", format = " << std::dec << format
				<< ", flags = " << std::dec << flags 
				<< ")");
	}

	void CalDriver::calResFree(CALresource res) const
	{
		_checkError((*_calResFree)(res));

		report("calResFree("
				<< "res = " << std::hex << std::showbase << res
				<< ")");
	}

	void CalDriver::calResMap(CALvoid** pPtr, CALuint* pitch, CALresource res, 
			CALuint flags) const
	{
		_checkError((*_calResMap)(pPtr, pitch, res, flags));

		report("calResMap("
				<< "*pPtr = " << std::hex << std::showbase << *pPtr
				<< ", *pitch = " << std::hex << std::showbase << *pitch
				<< ", res = " << std::hex << std::showbase << res
				<< ", flags = " << std::dec << flags
				<< ")");
	}

	void CalDriver::calResUnmap(CALresource res) const
	{
		_checkError((*_calResUnmap)(res));

		report("calResUnmap("
				<< "res = " << std::hex << std::showbase << res
				<< ")");
	}

	void CalDriver::calCtxRunProgramGrid(CALevent* event, CALcontext ctx, 
			CALprogramGrid* pProgramGrid) const
	{
		_checkError((*_calCtxRunProgramGrid)(event, ctx, pProgramGrid));

		report("calCtxRunProgramGrid("
				<< "event = " << std::hex << std::showbase << event
				<< ", ctx = " << std::hex << std::showbase << ctx
				<< ", pProgramGrid = " << std::hex << std::showbase << pProgramGrid
				<< ")");
	}

	bool CalDriver::calCtxIsEventDone(CALcontext ctx, CALevent event) const
	{
		CALresult result;
		result = (*_calCtxIsEventDone)(ctx, event);

		report("calCtxIsEventDone("
				<< "ctx = " << std::hex << std::showbase << ctx
				<< ", event = " << std::hex << std::showbase << event
				<< ")");

		switch(result)
		{
			case CAL_RESULT_OK:      return true; break;
			case CAL_RESULT_PENDING: return false; break;
			default: Throw(calGetErrorString()); break;
		}

		return true;
	}

	void CalDriver::calclCompile(CALobject* obj, CALlanguage language, 
			const CALchar* source, CALtarget target) const
	{
		_checkError((*_calclCompile)(obj, language, source, target));

		report("calclCompile("
				<< "*obj = " << std::hex << std::showbase << *obj
				<< ", language = " << std::dec << language
				<< ", source = \"" << std::string(source).substr(0, 10) << "...\""
				<< ", target = " << std::dec << target
				<< ")");
	}

	void CalDriver::calclLink(CALimage* image, CALobject* obj, 
			CALuint objCount) const
	{
		_checkError((*_calclLink)(image, obj, objCount));

		report("calclLink("
				<< "*image = " << std::hex << std::showbase << *image
				<< ", *obj = " << std::hex << std::showbase << *obj
				<< ", objCount = " << std::dec << objCount
				<< ")");
	}

	void CalDriver::calclFreeObject(CALobject obj) const
	{
		_checkError((*_calclFreeObject)(obj));

		report("calclFreeObject("
				<< "obj = " << std::hex << std::showbase << obj
				<< ")");
	}

	void CalDriver::calclFreeImage(CALimage image) const
	{
		_checkError((*_calclFreeImage)(image));

		report("calclFreeImage("
				<< "image = " << std::hex << std::showbase << image
				<< ")");
	}

	const CALchar *CalDriver::calGetErrorString() const
	{
		const CALchar *result;
		result = (*_calGetErrorString)();

		report("calGetErrorString()");

		return result;
	}

	inline void CalDriver::_checkError(CALresult r) const
	{
		if (r != CAL_RESULT_OK) Throw(calGetErrorString());
	}
}
