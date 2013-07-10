/*! \file DeviceProperties.h
	\author Gregory Diamos
	\date April 1, 2010
	\brief The header file for the Device class.
*/

#ifndef OECLOT_EXECUTIVE_DEVICEPROPERTIES_H_INCLUDED
#define OECLOT_EXECUTIVE_DEVICEPROPERTIES_H_INCLUDED

// Ocelot includes
#include <ocelot/ir/interface/Instruction.h>

namespace executive {

	class DeviceProperties
	{
		public:
			/*! "native" ISA of the device */
			ir::Instruction::Architecture ISA;
			/*! identifies the device's address space */
			int addressSpace;
			/*! human-readable device name */
			char name[256];

			/*! number of bytes of global memory available to
				the device */
			size_t totalMemory;
			/*! gets the number of multiprocessors/cores on
				the device */
			unsigned int multiprocessorCount;
			/*! true if the device can simultaneously execute a kernel
				while performing data transfer */
			int memcpyOverlap;
			/*! maximum number of threads per block */
			int maxThreadsPerBlock;
			/*! maximum size of each dimension of a block */
			int maxThreadsDim[3];
			/*! maximum size of each dimension of a grid */
			int maxGridSize[3];
			/*! total amount of shared memory available per block
				in bytes */
			int sharedMemPerBlock;
			/*! total amount of constant memory on the device */
			int totalConstantMemory;
			/*! warp size */
			int SIMDWidth;
			/*! maximum pitch allowed by memory copy functions */
			int memPitch;
			/*! total registers allowed per block */
			int regsPerBlock;
			/*! clock frequency in kHz */
			int clockRate;
			/*! alignment requirement for textures */
			int textureAlign;
			/*! Is the device integrated or discrete */
			int integrated;
			/*! Concurrent kernel execution */
			int concurrentKernels;
			/*! major shader module revision */
			int major;
			/*! minor shader model revision */
			int minor;
			/*! stack size */
			size_t stackSize;
			/*! printfFIFOSize */
			size_t printfFIFOSize;
			/**< This device shares a unified address with the host */
			bool unifiedAddressing;          
			/**< Peak memory clock frequency in kilohertz */
			int memoryClockRate;
			/**< Global memory bus width in bits */
			int memoryBusWidth;
			/**< Size of L2 cache in bytes */
			int l2CacheSize;
			/**< Maximum resident threads per multiprocessor */
			int maxThreadsPerMultiProcessor;
	};
}

#endif

