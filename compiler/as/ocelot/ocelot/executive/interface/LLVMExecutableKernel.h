/*! \file LLVMExecutableKernel.h
	\date Friday September 4, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMRuntime class
*/

#ifndef LLVM_EXECUTABLE_KERNEL_H_INCLUDED
#define LLVM_EXECUTABLE_KERNEL_H_INCLUDED

// Ocelot Includes
#include <ocelot/translator/interface/Translator.h>
#include <ocelot/executive/interface/ExecutableKernel.h>

// Standard Library Includes
#include <unordered_map>

// Forward Declarations
namespace ir
{
	class PTXKernel;
}

namespace executive
{

/*! \brief Executes an LLVMKernel using the LLVM JIT */
class LLVMExecutableKernel : public executive::ExecutableKernel
{
public:
	/*! \brief Types of call instructions */
	enum CallType
	{
		TailCall        = 0x1,
		ReturnCall      = 0x2,
		NormalCall      = 0x3,
		ExitCall        = 0x4,
		BarrierCall     = 0x5,
		InvalidCallType = 0x0
	};

	typedef translator::Translator::OptimizationLevel OptimizationLevel;

public:
	/*! \brief Creates a new instance of the runtime bound to a kernel*/
	LLVMExecutableKernel(const ir::IRKernel& kernel, Device* d = 0,
		OptimizationLevel l = translator::Translator::NoOptimization);
	/*! \brief Clean up the runtime */
	~LLVMExecutableKernel();

public:
	/*! \brief Launch a kernel on a 2D grid */
	void launchGrid(int width, int height, int depth);
	/*! \brief Sets the shape of a cta in the kernel */
	void setKernelShape(int x, int y, int z);
	/*! \brief Declare an amount of external shared memory */
	void setExternSharedMemorySize(unsigned int bytes);
	/*! \brief Describes the device used to execute the kernel */
	void setWorkerThreads(unsigned int threadLimit);
	/*! \brief Reload argument memory */
	void updateArgumentMemory();
	/*! \brief Indicate that other memory has been updated */
	void updateMemory();
	/*! \brief Get a vector of all textures references by the kernel */
	TextureVector textureReferences() const;

public:
	/*! \brief Get the block of argument memory associated with the kernel */
	char* argumentMemory() const;
	/*! \brief Get the block of constant memory associated with the kernel */
	char* constantMemory() const;
	/*! \brief Get the optimization level of the kernel */
	OptimizationLevel optimization() const;

public:
	/*!	adds a trace generator to the EmulatedKernel */
	void addTraceGenerator(trace::TraceGenerator *generator);
	/*!	removes a trace generator from an EmulatedKernel */
	void removeTraceGenerator(trace::TraceGenerator *generator);
	/*! sets an external function table for the emulated kernel */
	void setExternalFunctionSet(const ir::ExternalFunctionSet& s);
	/*! clear the external function table for the emulated kernel */
	void clearExternalFunctionSet();

private:
	typedef std::unordered_map<std::string, size_t> AllocationMap;
                        
private:
	/*! \brief Allocate memory for the kernel */
	void _allocateMemory();
	/*! \brief Allocate argument memory for the kernel */
	void _allocateArgumentMemory();
	/*! \brief Allocate constant memory for the kernel */
	void _allocateConstantMemory();

private:
	const ir::PTXKernel* _kernel;
	const Device*        _device;
	OptimizationLevel    _optimizationLevel;
	char*                _argumentMemory;
	char*                _constantMemory;
};

}

#endif

