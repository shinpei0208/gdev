/*! \file CalDriver.h
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 8, 2010
 *  \brief The header file for the CalDriver class
 */

#ifndef CAL_DRIVER_H_INCLUDED
#define CAL_DRIVER_H_INCLUDED

// CAL includes
#include <ocelot/cal/include/cal.h>
#include <ocelot/cal/include/calcl.h>
			
/*! \brief CAL device pointer */
typedef unsigned int CALdeviceptr;

namespace cal
{
	/*! \brief Provides access to the CAL runtime/driver
	 *
	 * Implemented as a Singleton. It is non-thread-safe for now (is CAL
	 * thread-safe?)
	 */
	class CalDriver
	{
		public:
			/*! \brief Singleton instance getter */
			static const CalDriver *Instance();

			/*****************************//**
			 * \name Initialization
			 ********************************/
			//@{
			void calInit() const;
			void calShutdown() const;
			//@}

			/*****************************//**
			 * \name Device Management
			 ********************************/
			//@{
			void calDeviceGetCount(CALuint *count) const;
			void calDeviceGetAttribs(CALdeviceattribs *attribs, 
					CALuint ordinal) const;
			void calDeviceGetStatus(CALdevicestatus *status, 
					CALdevice dev) const;
			void calDeviceOpen(CALdevice *dev, CALuint ordinal) const;
			void calDeviceClose(CALdevice dev) const;
			void calDeviceGetInfo(CALdeviceinfo *info, CALuint ordinal) const;
			//@}

			/*****************************//**
			 * \name Context Management
			 ********************************/
			//@{
			void calCtxCreate(CALcontext *ctx, CALdevice dev) const;
			void calCtxDestroy(CALcontext ctx) const;
			void calCtxGetMem(CALmem *mem, CALcontext ctx, 
					CALresource res) const;
			void calCtxReleaseMem(CALcontext ctx, CALmem mem) const;
			void calCtxSetMem(CALcontext ctx, CALname name, CALmem mem) const;
			//@}

			/*****************************//**
			 * \name Module Management
			 ********************************/
			//@{
			void calModuleLoad(CALmodule* module, CALcontext ctx, 
					CALimage image) const;
			void calModuleUnload(CALcontext ctx, CALmodule module) const;
			void calModuleGetEntry(CALfunc* func, CALcontext ctx, 
					CALmodule module, const CALchar* procName) const;
			void calModuleGetName(CALname* name, CALcontext ctx, 
					CALmodule module, const CALchar* varName) const;
			//@}

			/*****************************//**
			 * \name Memory Management
			 ********************************/
			//@{
			void calResAllocLocal1D(CALresource* res, CALdevice dev, 
					CALuint width, CALformat format, CALuint flags) const;
			void calResAllocRemote1D(CALresource* res, CALdevice* dev, 
					CALuint deviceCount, CALuint width, CALformat format, 
					CALuint flags) const;
			void calResFree(CALresource res) const;
			void calResMap(CALvoid** pPtr, CALuint* pitch, CALresource res, 
					CALuint flags) const;
			void calResUnmap(CALresource res) const;
			//@}

			/*****************************//**
			 * \name Execution Management
			 ********************************/
			//@{
			void calCtxRunProgramGrid(CALevent* event, CALcontext ctx, 
					CALprogramGrid* pProgramGrid) const;
			bool calCtxIsEventDone(CALcontext ctx, CALevent event) const;
			//@}

			/*****************************//**
			 * \name Compiler Interface
			 ********************************/
			//@{
			void calclCompile(CALobject* obj, CALlanguage language, 
					const CALchar* source, CALtarget target) const;
			void calclLink(CALimage* image, CALobject* obj, 
					CALuint objCount) const;		
			void calclFreeObject(CALobject obj) const;
			void calclFreeImage(CALimage image) const;
			//@}

			/*****************************//**
			 * \name Error Reporting
			 ********************************/
			//@{
			const CALchar *calGetErrorString() const;
			//@}
		
		private:
			/*! \brief Singleton instance */
			static const CalDriver *_instance;
			/*! \brief Runtime/Driver library (libaticalrt.so) handle */
            void *_driver;
			/*! \brief Compiler library (libaticalcl.so) handle */
            void *_compiler;

			/*****************************//**
			 * \name CAL function pointers
			 ********************************/
			//@{
			CALresult (*_calInit)();
			CALresult (*_calShutdown)();
			CALresult (*_calDeviceGetCount)(CALuint *count);
			CALresult (*_calDeviceGetAttribs)(CALdeviceattribs *attribs, CALuint ordinal);
			CALresult (*_calDeviceGetStatus)(CALdevicestatus *status, CALdevice dev);
			CALresult (*_calDeviceOpen)(CALdevice *dev, CALuint ordinal);
			CALresult (*_calDeviceClose)(CALdevice dev);
			CALresult (*_calDeviceGetInfo)(CALdeviceinfo *info, CALuint ordinal);
			CALresult (*_calCtxCreate)(CALcontext *ctx, CALdevice dev);
			CALresult (*_calCtxDestroy)(CALcontext ctx);
			CALresult (*_calCtxGetMem)(CALmem *mem, CALcontext ctx, CALresource res);
			CALresult (*_calCtxReleaseMem)(CALcontext ctx, CALmem mem);
			CALresult (*_calCtxSetMem)(CALcontext ctx, CALname name, CALmem mem);
			CALresult (*_calModuleLoad)(CALmodule* module, CALcontext ctx, CALimage image);
			CALresult (*_calModuleUnload)(CALcontext ctx, CALmodule module);
			CALresult (*_calModuleGetEntry)(CALfunc* func, CALcontext ctx, CALmodule module, const CALchar* procName);
			CALresult (*_calModuleGetName)(CALname* name, CALcontext ctx, CALmodule module, const CALchar* varName);
			CALresult (*_calResAllocLocal1D)(CALresource* res, CALdevice dev, CALuint width, CALformat format, CALuint flags);
			CALresult (*_calResAllocRemote1D)(CALresource* res, CALdevice* dev, CALuint deviceCount, CALuint width, CALformat format, CALuint flags);
			CALresult (*_calResFree)(CALresource res);
			CALresult (*_calResMap)(CALvoid** pPtr, CALuint* pitch, CALresource res, CALuint flags);
			CALresult (*_calResUnmap)(CALresource res);
			CALresult (*_calCtxRunProgramGrid)(CALevent* event, CALcontext ctx, CALprogramGrid* pProgramGrid);
			CALresult (*_calCtxIsEventDone)(CALcontext ctx, CALevent event);
			CALresult (*_calclCompile)(CALobject* obj, CALlanguage language, const CALchar* source, CALtarget target);
			CALresult (*_calclLink)(CALimage* image, CALobject* obj, CALuint objCount);
			CALresult (*_calclFreeObject)(CALobject obj);
			CALresult (*_calclFreeImage)(CALimage image);
			const CALchar *(*_calGetErrorString)();
			//@}

			/*! \brief Check result and throw exception if error */
			void _checkError(CALresult r) const;

		private:
			/*! \brief Constructor */
			CalDriver();
			/*! \brief Destructor */
			~CalDriver();
	};
}

#endif
