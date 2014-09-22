/*! \file EmulatorCallStack.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Friday July 9, 2010
	\brief The header file for the EmulatorCallStack class.
*/

#ifndef EMULATOR_CALL_STACK_H_INCLUDED
#define EMULATOR_CALL_STACK_H_INCLUDED

// Ocelot Includes
#include <ocelot/executive/interface/FrameInfo.h>

// Standard Library Includes
#include <vector>

namespace executive
{

/*! \brief A class implementing a multi-threaded variably sized call stack
	for the emulator device.
*/
class EmulatorCallStack
{
	private:
		typedef long long unsigned int RegisterType;
		typedef std::vector<unsigned int> SizeStack;
		typedef std::vector<char> DataVector;

	private:
		/*! \brief The current stack pointer */
		unsigned int _stackPointer;
		/*! \brief The number of threads used by this stack */
		unsigned int _threadCount;
		/*! \brief The offset for local memory */
		unsigned int _localMemoryBase;
		/*! \brief The offset for the register file */
		unsigned int _registerFileBase;
		/*! \brief A history of the sizes of stack frames */
		SizeStack _stackFrameSizes;
		/*! \brief A history of the sizes of local memory */
		SizeStack _localMemorySizes;
		/*! \brief A history of the sizes of register files */
		SizeStack _registerFileSizes;
		/*! \brief A history of the sizes of the shared memory block */
		SizeStack _sharedMemorySizes;
		/*! \brief The actual data for the stack */
		DataVector _stack;
		/*! \brief The actual data for shared memory */
		DataVector _sharedMemory;
		/*! \brief The actual data for global local memory */
		DataVector _globalLocalMemory;
		/*! \brief The amount of per-thread global local memory */
		unsigned int _globalLocalSize;
		/*! \brief The offset of the previously svaed frame */
		unsigned int _savedOffset;
		/*! \brief The offset of the previously svaed frame */
		unsigned int _savedFrameSize;

	public:
		/*! \brief Create a new call stack for a set number of threads */
		EmulatorCallStack(unsigned int threads = 0,
			unsigned int initialArgumentSize = 0, 
			unsigned int initialFrameSize = 0, unsigned int registers = 0, 
			unsigned int localSize = 0, unsigned int globalLocalSize = 0,
			unsigned int sharedSize = 0);
	
		/*! \brief Get a pointer to the base of the stack */
		void* stackBase();
		/*! \brief Get the total size of the stack */
		unsigned int totalStackSize() const;
		
		/*! \brief Get a pointer to the base of the current stack frame */
		void* stackFramePointer(unsigned int thread);
		/*! \brief Get a pointer to the stack frame at the saved offset */
		void* previousStackFramePointer(unsigned int thread);
		/*! \brief Get a pointer to the stack frame at the saved offset */
		void* savedStackFramePointer(unsigned int thread);
		/*! \brief Get a pointer to the previous stack frame */
		void* callerFramePointer(unsigned int thread);

		/*! \brief Get the offset of the current stack frame, save it */
		unsigned int offset() const;
		/*! \brief Save the offset of the current stack frame */
		void saveFrame();

		/*! \brief Get a pointer to the register file for a given thread */
		RegisterType* registerFilePointer(unsigned int thread);
		/*! \brief Get a pointer to the register file for a given thread */
		const RegisterType* registerFilePointer(unsigned int thread) const;
		/*! \brief Get a pointer to local memory for a given thread */
		void* localMemoryPointer(unsigned int thread);
		/*! \brief Get a pointer to shared memory */
		void* sharedMemoryPointer();
		/*! \brief Get a pointer to global local memory */
		void* globalLocalMemoryPointer(unsigned int thread);
		
		/*! \brief Get the current register file size */
		unsigned int registerCount() const;
		/*! \brief Get the current size of local memory */
		unsigned int localMemorySize() const;
		/*! \brief Get the shared memory size */
		unsigned int sharedMemorySize() const;
		/*! \brief Get the global local memory size */
		unsigned int globalLocalMemorySize() const;
		
		/*! \brief Get the size of the current frame */
		unsigned int stackFrameSize() const;
		/*! \brief Get the size of the previous frame */
		unsigned int previousFrameSize() const;
		/*! \brief Get the return program counter */
		unsigned int returnPC() const;
		/*! \brief Get the offset of the caller frame */
		unsigned int callerOffset() const;
		/*! \brief Get the offset of the caller frame */
		unsigned int callerFrameSize() const;
		/*! \brief Is the only frame on the stack the entry point? */
		bool isTheCurrentFrameMain() const;

		/*! \brief Push a new frame onto the stack */
		void pushFrame(unsigned int stackSize, unsigned int registers, 
			unsigned int localSize, unsigned int sharedSize, 
			unsigned int callPC, unsigned int callerStackFrame,
			unsigned int callerStackSize);
		/*! \brief Pop the current frame */
		void popFrame();

	public:
		/*! \brief Get the number of allocated frames */
		unsigned int getFrameCount() const;
		/*! \brief Get info for a specific frame */
		FrameInfo getFrameInfo(unsigned int frame) const;

	private:
		/*! \brief Get a pointer to the base of the stack. */
		void* _stackBase(unsigned int byteOffset = 0) const;
		/*! \brieg Resize the stack */
		void _resizeStack(unsigned int size);
		/*! \brief Get the aligned stack size */
		unsigned int _stackSize() const;
		/*! \brief Resize shared memory */
		void _resizeSharedMemory(unsigned int size);
		/*! \brief Get the aligned shared memory size */
		unsigned int _sharedMemorySize() const;
};

}

#endif

