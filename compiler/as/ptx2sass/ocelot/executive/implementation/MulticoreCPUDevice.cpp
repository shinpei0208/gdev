/* 	\file MulticoreCPUDevice.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday April 20, 2010
	\brief The source file for the MulticoreCPUDevice class.
*/

#ifndef MULTICORE_CPU_DEVICE_CPP_INCLUDED
#define MULTICORE_CPU_DEVICE_CPP_INCLUDED

// ocelot includes
#include <ocelot/executive/interface/MulticoreCPUDevice.h>
#include <ocelot/executive/interface/LLVMExecutableKernel.h>
#include <ocelot/executive/interface/LLVMModuleManager.h>
#include <ocelot/executive/interface/LLVMExecutionManager.h>

// hydrazine includes
#include <hydrazine/interface/SystemCompatibility.h>
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/Casts.h>

// Standard Library Includes
#include <cstring>

// Macros
#define Throw(x) {std::stringstream s; s << x; \
	throw hydrazine::Exception(s.str());}

namespace executive
{
	MulticoreCPUDevice::Module::Module(const ir::Module* m, Device* d) 
		: EmulatorDevice::Module(m, d)
	{
	
	}

	MulticoreCPUDevice::Module::~Module()
	{
		assert(ir->loaded());
		
		if(!LLVMModuleManager::isModuleLoaded(ir->path())) return;

		LLVMModuleManager::unloadModule(ir->path());
		LLVMExecutionManager::flushTranslatedKernels();
	}

	ExecutableKernel* MulticoreCPUDevice::Module::getKernel(
		const std::string& name)
	{
		KernelMap::iterator kernel = kernels.find(name);
		if(kernel != kernels.end())
		{
			return kernel->second;
		}
		
		ir::Module::KernelMap::const_iterator ptxKernel = 
			ir->kernels().find(name);
			
		MulticoreCPUDevice* cpu = static_cast<MulticoreCPUDevice*>(device);
		
		if(ptxKernel != ir->kernels().end())
		{
			kernel = kernels.insert(std::make_pair(name, 
				new LLVMExecutableKernel(*ptxKernel->second, device, 
				cpu->_optimizationLevel))).first;
			return kernel->second;
		}
		
		return 0;
	}

	MulticoreCPUDevice::MulticoreCPUDevice(unsigned int flags) 
		: EmulatorDevice(flags), _workerThreads(-1),
		_optimizationLevel(translator::Translator::NoOptimization)
	{
		_properties.ISA = ir::Instruction::LLVM;
		std::strcpy(_properties.name,
			"Ocelot Multicore CPU Backend (LLVM-JIT)");
		_properties.multiprocessorCount = hydrazine::getHardwareThreadCount();
		_properties.clockRate = 2000;
	}
	
	void MulticoreCPUDevice::load(const ir::Module* module)
	{
		if(_modules.count(module->path()) != 0)
		{
			Throw("Duplicate module - " << module->path());
		}
		_modules.insert(std::make_pair(module->path(), 
			new Module(module, this)));	
	}

	ExecutableKernel* MulticoreCPUDevice::getKernel(
		const std::string& moduleName, const std::string& kernelName)
	{
		ModuleMap::iterator module = _modules.find(moduleName);
		
		if(module == _modules.end()) return 0;
		
		return module->second->getKernel(kernelName);
	}
	
	void MulticoreCPUDevice::launch(const std::string& moduleName, 
		const std::string& kernelName, const ir::Dim3& grid, 
		const ir::Dim3& block, size_t sharedMemory, 
		const void* argumentBlock, size_t argumentBlockSize, 
		const trace::TraceGeneratorVector& traceGenerators,
		const ir::ExternalFunctionSet* externals)
	{
		ModuleMap::iterator module = _modules.find(moduleName);
		
		if(module == _modules.end())
		{
			Throw("Unknown module - " << moduleName);
		}
		
		ExecutableKernel* kernel = module->second->getKernel(kernelName);
		
		if(kernel == 0)
		{
			Throw("Unknown kernel - " << kernelName 
				<< " in module " << moduleName);
		}
		
		if(kernel->sharedMemorySize() + sharedMemory > 
			(size_t)properties().sharedMemPerBlock)
		{
			Throw("Out of shared memory for kernel \""
				<< kernel->name << "\" : \n\tpreallocated "
				<< kernel->sharedMemorySize() << " + requested " 
				<< sharedMemory << " is greater than available " 
				<< properties().sharedMemPerBlock << " for device " 
				<< properties().name);
		}
		
		if(kernel->constMemorySize() > (size_t)properties().totalConstantMemory)
		{
			Throw("Out of shared memory for kernel \""
				<< kernel->name << "\" : \n\tpreallocated "
				<< kernel->constMemorySize() << " is greater than available " 
				<< properties().totalConstantMemory << " for device " 
				<< properties().name);
		}
		
		kernel->device = this;
		kernel->setKernelShape(block.x, block.y, block.z);
		kernel->setArgumentBlock((const unsigned char*)argumentBlock, 
			argumentBlockSize);
		kernel->updateArgumentMemory();
		kernel->updateMemory();
		kernel->setExternSharedMemorySize(sharedMemory);
		kernel->setWorkerThreads(_workerThreads);
		kernel->setExternalFunctionSet(*externals);
		kernel->setTraceGenerators(traceGenerators);
		
		kernel->launchGrid(grid.x, grid.y, grid.z);
		
		kernel->clearExternalFunctionSet();
	}

	void MulticoreCPUDevice::limitWorkerThreads(unsigned int threads)
	{
		_workerThreads = threads;
	}

	void MulticoreCPUDevice::setOptimizationLevel(
		translator::Translator::OptimizationLevel level)
	{
		_optimizationLevel = level;
	}
}

#endif

