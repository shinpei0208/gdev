/*! \file LLVMFunctionCallStack.cpp
	\date Monday September 27, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the LLVMFunctionCallStack class.
*/

#ifndef LLVM_FUNCTION_CALL_STACK_CPP_INCLUDED
#define LLVM_FUNCTION_CALL_STACK_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/executive/interface/LLVMFunctionCallStack.h>

// Standard Library Includes
#include <cassert>

namespace executive
{

void LLVMFunctionCallStack::call(unsigned int l, unsigned int a, unsigned int i,
	unsigned int r)
{
	_stack.resize(_stack.size() + l + a);
	_sizes.push_back(ParameterAndLocalSize(l, a, i, r));
}

unsigned int LLVMFunctionCallStack::returned()
{
	if (_sizes.size()) {
		const ParameterAndLocalSize& sizes = _sizes.back();
		_stack.resize(_stack.size() - sizes.localSize - sizes.parameterSize);
		unsigned int resumePoint = sizes.resumePoint;
		_sizes.pop_back();
		return resumePoint;
	}
	return 0xffffffff;
}

void LLVMFunctionCallStack::setKernelArgumentMemory(
	char* memory, unsigned int size)
{
	assert(_stack.size() == 0);
	_argumentMemory = memory;
	_argumentSize   = size;
}

void LLVMFunctionCallStack::resizeCurrentLocalMemory(unsigned int size)
{
	assert(_stack.size() != 0);
	
	_stack.resize( _stack.size() - _sizes.back().localSize + size );
	_sizes.back().localSize = size;
}

char* LLVMFunctionCallStack::localMemory()
{
	const ParameterAndLocalSize& sizes = _sizes.back();
	return &_stack[_stack.size() - sizes.localSize];
}

char* LLVMFunctionCallStack::parameterMemory()
{
	const ParameterAndLocalSize& sizes = _sizes.back();
	return &_stack[_stack.size() - sizes.localSize - sizes.parameterSize];
}

char* LLVMFunctionCallStack::argumentMemory()
{
	if(_sizes.size() < 2) return _argumentMemory;
	
	const ParameterAndLocalSize& sizes = _sizes.back();
	unsigned int previousParameterSize = _sizes[_sizes.size()-2].parameterSize;
	unsigned int previousLocalSize     = _sizes[_sizes.size()-2].localSize;
	return &_stack[_stack.size() - sizes.localSize - sizes.parameterSize
		- previousParameterSize - previousLocalSize];
}

unsigned int LLVMFunctionCallStack::localSize() const
{
	const ParameterAndLocalSize& sizes = _sizes.back();
	return sizes.localSize;
}

unsigned int LLVMFunctionCallStack::parameterSize() const
{
	const ParameterAndLocalSize& sizes = _sizes.back();
	return sizes.parameterSize;
}

unsigned int LLVMFunctionCallStack::argumentSize() const
{
	if(_sizes.size() < 2) return _argumentSize;
	
	return _sizes[_sizes.size()-2].parameterSize;
}

unsigned int LLVMFunctionCallStack::functionId() const
{
	const ParameterAndLocalSize& sizes = _sizes.back();
	return sizes.functionId;
}

LLVMFunctionCallStack::ParameterAndLocalSize::ParameterAndLocalSize(
	unsigned int l, unsigned int a, unsigned int id, unsigned int r)
	: localSize(l), parameterSize(a), functionId(id), resumePoint(r)
{
	
}

}

#endif

