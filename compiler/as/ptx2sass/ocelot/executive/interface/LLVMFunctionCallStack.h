/*! \file LLVMFunctionCallStack.h
	\date Monday September 27, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMFunctionCallStack class.
*/

#ifndef LLVM_FUNCTION_CALL_STACK_H_INCLUDED
#define LLVM_FUNCTION_CALL_STACK_H_INCLUDED

// Standard Library Includes
#include <vector>

namespace executive
{

/*! \brief A class for managing a call stack for a single PTX thread */
class LLVMFunctionCallStack
{
public:
	void call(unsigned int localSize, unsigned int parameterSize, 
		unsigned int functionId, unsigned int resumePoint = -1);
	unsigned int returned();

	void setKernelArgumentMemory(char* memory, unsigned int argumentSize);
	void resizeCurrentLocalMemory(unsigned int size);

	char* localMemory();
	char* parameterMemory();
	char* argumentMemory();

	unsigned int localSize() const;
	unsigned int parameterSize() const;
	unsigned int argumentSize() const;

	unsigned int functionId() const;

private:
	class ParameterAndLocalSize
	{
	public:
		ParameterAndLocalSize(unsigned int localSize,
			unsigned int parameterSize, unsigned int functionId,
			unsigned int resumePoint);
	
	public:
		unsigned int localSize;
		unsigned int parameterSize;
		unsigned int functionId;
		unsigned int resumePoint;
	};

	typedef std::vector<char> DataVector;
	typedef std::vector<ParameterAndLocalSize> SizeVector;
	
private:
	DataVector   _stack;
	SizeVector   _sizes;
	char*        _argumentMemory;
	unsigned int _argumentSize;

};

}

#endif

