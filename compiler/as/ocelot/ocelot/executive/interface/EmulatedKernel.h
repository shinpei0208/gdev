/*! \file EmulatedKernel.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\date Jan 19, 2009
	\brief implements a kernel emulated on the host CPU
*/

#ifndef EXECUTIVE_EMULATEDKERNEL_H_INCLUDED
#define EXECUTIVE_EMULATEDKERNEL_H_INCLUDED

// Ocelot Includes
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/Texture.h>

#include <ocelot/executive/interface/ExecutableKernel.h>
#include <ocelot/executive/interface/CTAContext.h>
#include <ocelot/executive/interface/FrameInfo.h>
#include <ocelot/executive/interface/CooperativeThreadArray.h>

// Forward Declarations
namespace trace     { class TraceGenerator;          }
namespace executive { class EmulatedKernelScheduler; }

namespace executive {
		
	class EmulatedKernel: public ExecutableKernel {
	public:
		typedef std::vector<ir::PTXInstruction> PTXInstructionVector;
		typedef std::map<int, std::string> ProgramCounterBlockMap;
		typedef std::unordered_map<std::string, int> FunctionNameMap;
		typedef std::map< std::string, std::pair<int, int> > BlockRangeMap;
		typedef std::unordered_map<int, const EmulatedKernel*> PCToKernelMap;
		typedef CooperativeThreadArray::RegisterFile RegisterFile;

	private:
		static void _computeOffset(const ir::PTXStatement& it, 
			unsigned int& offset, unsigned int& totalOffset);

	public:
		EmulatedKernel(ir::IRKernel* kernel, Device* d = 0, 
			bool initialize = true);
		EmulatedKernel(Device *c);
		EmulatedKernel();
		virtual ~EmulatedKernel();
	
		/*!	\brief Determines whether kernel is executable */
		bool executable() const;
		
		/*!	Launch a kernel on a 2D grid */
		void launchGrid(int width, int height, int depth);
	
		/*!	Sets the shape of a kernel */
		void setKernelShape(int x, int y, int z);

		ir::Dim3 getKernelShape() const;
		
		/*! \brief Changes the amount of external shared memory */
		void setExternSharedMemorySize(unsigned int bytes);
		
		/*!	Sets device used to execute the kernel */
		void setWorkerThreads(unsigned int limit);

		/*! \brief Indicate that the kernels parameters have been updated */
		void updateArgumentMemory();
		
		/*! \brief Indicate that other memory has been updated */
		void updateMemory();

		/*! \brief Get a vector of all textures references by the kernel */
		TextureVector textureReferences() const;

	public:

		/*! sets an external function table for the emulated kernel */
		void setExternalFunctionSet(const ir::ExternalFunctionSet& s);
		
		/*! clear the external function table for the emulated kernel */
		void clearExternalFunctionSet();

		/*! \brief Initialize the kernel */
		void initialize();

		/*!	Maps identifiers to global memory allocations. */
		void initializeGlobalMemory();
	
		/*! \brief Sets the CTA */
		void setCTA(CooperativeThreadArray* cta);
	
	public:
		/*! Lazily sets the target of a call instruction to the entry point
			of the specified function.  This function will be inserted into
			the instruction sequence if it does not already exist */
		void lazyLink(int callPC, const std::string& functionName);

		/*! Code that was linked in still uses absolute branch targets, they
			need to be updated now that the code is located in a different
			place.
		*/
		void fixBranchTargets(size_t newPC);

		/*! Looks up a named function in the module and inserts it into the
			instruction stream if it does not exist.  Returns the PC of the
			function in the instruction stream. */
		size_t link(const std::string& functionName);

		/*! Finds the kernel beginning at the specified PC */
		const EmulatedKernel* getKernel(int PC) const;

		/*! Finds the kernel containing the specified PC */
		const EmulatedKernel* getKernelContainingThisPC(int PC) const;
		
		/*! If the kernel is executing, jump to the specified PC */
		void jumpToPC(int PC);

	public:
		/* Get a snapshot of the current register file */
		RegisterFile getCurrentRegisterFile() const;

		/* Get a pointer to the base of the current shared memory block */
		const char* getSharedMemory() const;

		/* Get a pointer to the base of the current local memory block
			for the specified thread */
		const char* getLocalMemory(unsigned int threadId) const;

		/* Get a pointer to the base of the current global local memory block

			for the specified thread */
		const char* getGlobalLocalMemory(unsigned int threadId) const;

		/* Get the argument memory size of the current frame */
		unsigned int getCurrentFrameArgumentMemorySize() const;

		/* Get the local memory size of the current frame */
		unsigned int getCurrentFrameLocalMemorySize() const;

		/* Get the parameter memory size of the current frame */
		unsigned int getCurrentFrameParameterMemorySize() const;

		/* Get a pointer to the base of stack memory for the specified thread */
		const char* getStackBase(unsigned int threadId) const;

		/* Get the total stack size for the specified thread */
		unsigned int getTotalStackSize(unsigned int threadId) const;

		/* Get the number of stack frames */
		unsigned int getStackFrameCount() const;

		/* Get info for a specific stack frame */
		FrameInfo getStackFrameInfo(unsigned int frame) const;
		
	protected:
		/*! Cleans up the EmulatedKernel instance*/
		void freeAll();

		/*!	On construction, allocates registers by computing live ranges */
		void registerAllocation();
		
		/*!	Produces a packed vector of instructions, updates each operand, 
			and changes labels to indices.
		*/
		void constructInstructionSequence();

		/*!	After emitting the instruction sequence, visit each memory move 
			operation and replace references to parameters with offsets into 
			parameter memory.
		*/
		void updateParamReferences();

		/*!	Allocate parameter memory*/	
		void initializeArgumentMemory();

		/*!	Allocates arrays in shared memory and maps identifiers to 
			allocations. */
		void initializeSharedMemory();

		/*!	Allocates arrays in local memory and maps identifiers to 
			allocations. */
		void initializeLocalMemory();

		/*!	Allocates arrays in globally scoped memory and maps identifiers to 
			allocations. */
		void initializeGlobalLocalMemory();

		/*!	Maps identifiers to const memory allocations. */
		void initializeConstMemory();

		/*!	Maps identifiers to global shared memory allocations. */
		void initializeGlobalSharedMemory();
		
		/*! Determines stack memory size and maps identifiers to allocations */
		void initializeStackMemory();
		
		/*!	Scans the kernel and builds the set of textures using references 
				in tex instructions */
		void initializeTextureMemory();

		/*! Setup symbols that are referenced in global variables.		
		*/
		void initializeSymbolReferences();

		/*! Sets the target of call instructions to invalid pcs so that they
			can be lazily compiled and allocated */
		void invalidateCallTargets();

	public:
		/*! A map of register name to register number */
		ir::PTXKernel::RegisterMap registerMap;

		/*!	Pointer to block of memory used to store argument data */
		char* ArgumentMemory;

		/*!	Pointer to byte-addressable const memory */
		char* ConstMemory;
		
		/*!	Packed and allocated vector of instructions */
		PTXInstructionVector instructions;

		/*! Maps program counters of header instructions to basic block label */
		ProgramCounterBlockMap branchTargetsToBlock;
		
		/*! maps the program counter of the terminating
			instructions to owning basic block */
		ProgramCounterBlockMap basicBlockMap;
		
		/*! maps a PC to the basic block it starts */
		ProgramCounterBlockMap basicBlockPC;
		
		/*! maps a block label to the PCs of the first
			and last instructions in the block */
		BlockRangeMap blockPCRange;

		/*!	Packed vector of mapped textures */
		TextureVector textures;

		/*! A handle to the current scheduler, or 0 if none is executing */
		EmulatedKernelScheduler* scheduler;

	private:
		/*! Maps program counter to the kernel that begins there */
		PCToKernelMap kernelEntryPoints;

		/*! A map of function names to the PC of their entry point */
		FunctionNameMap functionEntryPoints;

		/*! A handle to the current CTA, or 0 if none is executing */
		CooperativeThreadArray* CTA;

		/*! Is the kernel initizlied ? */
		bool _initialized;

	public:
		/*! \brief Check to see if a memory access is valid */
		bool checkMemoryAccess(const void* base, size_t size) const;
	
	public:
		/*! Copies data from global objects into const and global memory */
		void updateGlobals();

	public:
		/*!	Print out every instruction	*/
		std::string toString() const;
		
		/*! \brief Get the file name that the kernel resides in */
		std::string fileName() const;
		
		/*! \brief Get the nearest location to an instruction at a given PC */
		std::string location(unsigned int PC) const;
		
		/*!	\brief gets the basic block label owning the instruction 
			specified by the PC */
		std::string getInstructionBlock(int PC) const;
		
		/*! \brief accessor for obtaining PCs of first and
			last instructions in a block */
		std::pair<int,int> getBlockRange(const std::string &label) const;


	public:
		unsigned int _getSharedMemorySizeOfReachableKernels() const;
	};

}

#endif

