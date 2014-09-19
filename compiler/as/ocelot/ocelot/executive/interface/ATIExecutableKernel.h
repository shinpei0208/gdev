/*! \file ATIExecutableKernel.h
 *  \author Rodrigo Dominguez <rdomingu@ece.neu.edu>
 *  \date April 10, 2010
 *  \brief The header file for the ATI Executable Kernel class.
 */

#ifndef ATIEXECUTABLEKERNEL_H_INCLUDED
#define ATIEXECUTABLEKERNEL_H_INCLUDED

// Ocelot includes
#include <ocelot/executive/interface/ExecutableKernel.h>
#include <ocelot/executive/interface/ATIGPUDevice.h>

#include <boost/cstdint.hpp>

namespace executive 
{
	class ATIExecutableKernel : public executive::ExecutableKernel {
		public:
			/*! \brief Constructor */
			ATIExecutableKernel(ir::IRKernel &k, CALcontext *context, 
					CALevent *event, CALresource *uav0, CALresource *cb0, 
					CALresource *cb1, Device* d);

			/*!	\brief Launch a kernel on a 2D grid */
			void launchGrid(int width, int height, int depth);

			/*!	\brief Sets the shape of a kernel */
			void setKernelShape(int x, int y, int z);

			/*! \brief Changes the amount of external shared memory */
			void setExternSharedMemorySize(unsigned int bytes);

			/*! \brief Changes the amount of vote memory */
			void setVoteMemorySize(unsigned int bytes);

			/*! \brief Sets the max number of pthreads this kernel can use */
			void setWorkerThreads(unsigned int workerThreadLimit);

			/*! \brief Indicate that the kernels arguments have been updated */
			void updateArgumentMemory();

			/*! \brief Indicate that other memory has been updated */
			void updateMemory();

			unsigned int voteMemorySize() const;

			/*! \brief Get a vector of all textures references by the kernel */
			TextureVector textureReferences() const;

			/*!	\brief adds a trace generator to the EmulatedKernel */
			void addTraceGenerator(trace::TraceGenerator* generator);

			/*!	\brief removes a trace generator from an EmulatedKernel */
			void removeTraceGenerator(trace::TraceGenerator* generator);

			/*! sets an external function table for the emulated kernel */
			void setExternalFunctionSet(const ir::ExternalFunctionSet& s);

			/*! clear the external function table for the emulated kernel */
			void clearExternalFunctionSet();

			/*! \brief Copies data from global objects into const and global
			 * memory */
			void updateGlobals();

			/*! A map of register name to register number */
			ir::PTXKernel::RegisterMap registerMap;

		private:
			/*!	\brief On construction, allocates registers by computing live 
			 * ranges */
			void registerAllocation();

			/*! \brief Initialize shared memory */
			void initializeSharedMemory();

			/*! \brief Maps identifiers to const memory allocations */
			void initializeGlobalMemory();

			/*! \brief Determine the padding required to satisfy alignment */
			static unsigned int _pad(size_t& size, unsigned int alignment);

			/*! \brief Create the IL module from the code */
			void _translateKernel();

			/*! \brief The amount of allocated shared memory for voting */
			unsigned int _voteMemorySize;

			/*! \brief Type of the constant buffer (cb) */
			typedef struct { unsigned int x, y, z, w; } cb_t;

			/*! \brief CAL Context */
			CALcontext *_context;
			/*! \brief CAL Event */
			CALevent *_event;
			/*! \brief CAL Device Info */
			CALdeviceinfo _info;
			/*! \brief CAL Module */
			CALmodule _module;
			/*! \brief CAL Object */
			CALobject _object;
			/*! \brief CAL Image */
			CALimage _image;

			/********************************************************//**
			 * \name uav0 Memory Manager
			 ***********************************************************/
			//@{
			/*! \brief CAL uav0 resource */
			CALresource *_uav0Resource;
			/*! \brief CAL uav0 memory handle */
			CALmem _uav0Mem;
			/*! \brief CAL uav0 module name */
			CALname _uav0Name;
			/*! \brief CAL uav8 module name (for less-than-32bits memory ops) */
			CALname _uav8Name;
			//@}

			/********************************************************//**
			 * \name cb0 Memory Manager (ABI data)
			 ***********************************************************/
			//@{
			/*! \brief CAL cb0 resource */
			CALresource *_cb0Resource;
			/*! \brief CAL cb0 memory handle */
			CALmem _cb0Mem;
			/*! \brief CAL cb0 module name */
			CALname _cb0Name;
			//@}
			
			/********************************************************//**
			 * \name cb1 Memory Manager (param)
			 ***********************************************************/
			//@{
			/*! \brief CAL cb1 resource */
			CALresource *_cb1Resource;
			/*! \brief CAL cb1 memory handle */
			CALmem _cb1Mem;
			/*! \brief CAL cb1 module name */
			CALname _cb1Name;
			//@}
			
			/*! \brief Returns a pointer to an instance to the 
				CalDriver singleton */
			static const cal::CalDriver *CalDriver();
	};
}

#endif

