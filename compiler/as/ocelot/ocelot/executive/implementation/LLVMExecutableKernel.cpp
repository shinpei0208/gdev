/*! \file LLVMExecutableKernel.cpp
	\date Friday September 4, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the LLVMExecutableKernel class
*/

#ifndef LLVM_EXECUTABLE_KERNEL_CPP_INCLUDED
#define LLVM_EXECUTABLE_KERNEL_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/executive/interface/LLVMExecutableKernel.h>
#include <ocelot/executive/interface/LLVMExecutionManager.h>
#include <ocelot/executive/interface/LLVMModuleManager.h>
#include <ocelot/executive/interface/Device.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cstring>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace executive
{

static unsigned int pad(unsigned int& size, unsigned int alignment)
{
	unsigned int padding = alignment - (size % alignment);
	padding = (alignment == padding) ? 0 : padding;
	size += padding;
	return padding;
}

LLVMExecutableKernel::LLVMExecutableKernel(const ir::IRKernel& k, 
	executive::Device* d, translator::Translator::OptimizationLevel l) : 
	ExecutableKernel(d), _optimizationLevel(l), _argumentMemory(0),
	_constantMemory(0)
{
	assert(!function());
	assertM(k.ISA == ir::Instruction::PTX, 
		"LLVMExecutable kernel must be constructed from a PTXKernel");
	ISA = ir::Instruction::LLVM;
	
	name = k.name;
	arguments = k.arguments;
	module = k.module;
	
	mapArgumentOffsets();
}

LLVMExecutableKernel::~LLVMExecutableKernel()
{	
	delete[] _argumentMemory;
	delete[] _constantMemory;
}

void LLVMExecutableKernel::launchGrid(int x, int y, int z)
{	
	report( "Launching kernel \"" << name << "\" on grid ( x = " 
		<< x << ", y = " << y << " )"  );
	
	_gridDim.x = x;
	_gridDim.y = y;
	_gridDim.z = z;
	
	initializeTraceGenerators();
	
	LLVMExecutionManager::launch(*this);
	
	finalizeTraceGenerators();
}

void LLVMExecutableKernel::setKernelShape( int x, int y, int z )
{
	report( "Setting CTA shape to ( x = " << x << ", y = " 
		<< y << ", z = " << z << " ) for kernel \"" << name << "\""  );

	_blockDim.x = x;
	_blockDim.y = y;
	_blockDim.z = z;
}

void LLVMExecutableKernel::setExternSharedMemorySize(unsigned int bytes)
{
	_externSharedMemorySize = bytes;
}

void LLVMExecutableKernel::setWorkerThreads(unsigned int threadLimit)
{
	LLVMExecutionManager::setWorkerThreadCount(
		std::min(device->properties().multiprocessorCount, threadLimit));
}

void LLVMExecutableKernel::updateArgumentMemory()
{
	_allocateMemory();
	unsigned int size = 0;
	for(ParameterVector::iterator argument = arguments.begin();
		argument != arguments.end(); ++argument) 
	{
		pad(size, argument->getAlignment());
		for(ir::Parameter::ValueVector::iterator 
			value = argument->arrayValues.begin(); 
			value != argument->arrayValues.end(); ++value) 
		{
			assertM(size < argumentMemorySize(), "Size " << size 
				<< " not less than allocated parameter size " 
				<< argumentMemorySize());
			std::memcpy(_argumentMemory + size, &value->val_b16, 
				argument->getElementSize());
			size += argument->getElementSize();
		}
	}
}

void LLVMExecutableKernel::updateMemory()
{
	report( "Updating Memory" );

	report( " Updating Constant Memory" );
	unsigned int bytes = 0;

	for(ir::Module::GlobalMap::const_iterator 
		constant = module->globals().begin(); 
		constant != module->globals().end(); ++constant) 
	{
		if(constant->second.statement.directive == ir::PTXStatement::Const) 
		{
			report( "   Updating global constant variable " 
				<< constant->second.statement.name << " of size " 
				<< constant->second.statement.bytes() );
			pad(bytes, constant->second.statement.alignment);

			assert(device != 0);
			Device::MemoryAllocation* global = device->getGlobalAllocation(
				module->path(), constant->second.statement.name);

			assert(global != 0);
			assert(global->size() + bytes <= _constMemorySize);

			memcpy(_constantMemory + bytes, global->pointer(), global->size());

			bytes += global->size();
		}
	}
}

ExecutableKernel::TextureVector LLVMExecutableKernel::textureReferences() const
{
	return TextureVector();
}

char* LLVMExecutableKernel::argumentMemory() const
{
	return _argumentMemory;
}

char* LLVMExecutableKernel::constantMemory() const
{
	return _constantMemory;
}

LLVMExecutableKernel::OptimizationLevel
	LLVMExecutableKernel::optimization() const
{
	return _optimizationLevel;
}

void LLVMExecutableKernel::addTraceGenerator(
	trace::TraceGenerator *generator)
{
	assertM(false, "No trace generation support in LLVM kernel.");
}

void LLVMExecutableKernel::removeTraceGenerator(
	trace::TraceGenerator *generator)
{
	assertM(false, "No trace generation support in LLVM kernel.");	
}

void LLVMExecutableKernel::setExternalFunctionSet(
	const ir::ExternalFunctionSet& s)
{
	LLVMModuleManager::setExternalFunctionSet(s);
}

void LLVMExecutableKernel::clearExternalFunctionSet()
{
	LLVMModuleManager::clearExternalFunctionSet();
}

void LLVMExecutableKernel::_allocateMemory()
{
	_allocateArgumentMemory();
	_allocateConstantMemory();
}

void LLVMExecutableKernel::_allocateArgumentMemory()
{
	if(_argumentMemory != 0) return;
	report( "  Allocating argument memory." );

	_argumentMemorySize = 0;

	for(ParameterVector::iterator argument = arguments.begin(); 
		argument != arguments.end(); ++argument)
	{
		pad(_argumentMemorySize, argument->getAlignment());

		report("   Allocated argument " << argument->name << " from "
			<< _argumentMemorySize << " to " 
			<< (_argumentMemorySize + argument->getSize()));

		argument->offset = _argumentMemorySize;
		_argumentMemorySize += argument->getSize();
	}

	report("  Allocated " << _argumentMemorySize << " for argument memory.");

	_argumentMemory = new char[_argumentMemorySize];
}

void LLVMExecutableKernel::_allocateConstantMemory()
{
	if(_constantMemory != 0) return;
	
	report( " Allocating Constant Memory" );
	_constMemorySize = 0;

	for(ir::Module::GlobalMap::const_iterator 
		global = module->globals().begin(); 
		global != module->globals().end(); ++global) 
	{
		if(global->second.statement.directive == ir::PTXStatement::Const) 
		{
			report( "   Found global constant variable " 
				<< global->second.statement.name << " of size " 
				<< global->second.statement.bytes() );
			pad(_constMemorySize, global->second.statement.alignment);
			_constMemorySize += global->second.statement.bytes();
		}
	}

	report("   Total constant memory size is " << _constMemorySize << ".");

	_constantMemory = new char[_constMemorySize];
}

}

#endif

