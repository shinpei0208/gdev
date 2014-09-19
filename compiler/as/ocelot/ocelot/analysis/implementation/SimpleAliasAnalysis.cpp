/*! \file   SimpleAliasAnalysis.cpp
	\date   Thursday November 8, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the SimpleAliasAnalysis class.
*/

// Ocelot Includes
#include <ocelot/analysis/interface/SimpleAliasAnalysis.h>

#include <ocelot/ir/interface/IRKernel.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>

namespace analysis
{

SimpleAliasAnalysis::SimpleAliasAnalysis()
: KernelAnalysis("SimpleAliasAnalysis"),
	_aStoreCanReachThisFunction(true), _kernel(0)
{

}

void SimpleAliasAnalysis::analyze(ir::IRKernel& kernel)
{
	// Functions can be called
	_aStoreCanReachThisFunction = !kernel.function();
	_kernel = &kernel;
	
	if(_aStoreCanReachThisFunction) return;
	
	for(auto block = kernel.cfg()->begin();
		block != kernel.cfg()->end(); ++block)
	{
		for(auto instruction = block->instructions.begin();
			instruction != block->instructions.end(); ++instruction)
		{
			auto ptx = static_cast<ir::PTXInstruction*>(*instruction);
			
			if(ptx->isStore())
			{
				_aStoreCanReachThisFunction = true;
				return;
			}
		}
	}
}

static bool referencesKernelArgument(ir::IRKernel* kernel,
	const ir::PTXInstruction* load)
{
	if(load->a.addressMode == ir::PTXOperand::Address)
	{
		for(auto argument = kernel->arguments.begin();
			argument != kernel->arguments.end(); ++argument)
		{
			if(load->a.identifier == argument->name) return true;
		}
	}
	
	return false;
}

bool SimpleAliasAnalysis::cannotAliasAnyStore(const ir::Instruction* load)
{
	if(!_aStoreCanReachThisFunction) return true;

	auto ptx = static_cast<const ir::PTXInstruction*>(load);

	if(ptx->addressSpace == ir::PTXInstruction::Param)
	{
		if(referencesKernelArgument(_kernel, ptx)) return true;
	}
	
	if(ptx->addressSpace == ir::PTXInstruction::Const) return true;

	return false;
}

bool SimpleAliasAnalysis::canAlias(const ir::Instruction* s,
	const ir::Instruction* l)
{
	auto load  = static_cast<const ir::PTXInstruction*>(l);
	auto store = static_cast<const ir::PTXInstruction*>(s);

	if(cannotAliasAnyStore(load)) return false;
	
	// check for address space differences
	if(load->addressSpace != ir::PTXInstruction::Generic &&
		store->addressSpace != ir::PTXInstruction::Generic)
	{
		if(load->addressSpace != store->addressSpace) return false;
	}
	
	// check for constant addresses
	if(load->a.addressMode == ir::PTXOperand::Immediate)
	{
		if(store->d.addressMode == ir::PTXOperand::Immediate)
		{
			uint64_t storeAddress = store->d.imm_uint + store->d.offset;
			uint64_t loadAddress  =  load->a.imm_uint +  load->a.offset;

			if(storeAddress + store->a.bytes() < loadAddress)
			{
				return false;
			}
			if(loadAddress + load->d.bytes() < storeAddress)
			{
				return false;
			}
		}
	}
	
	return true;
}

}

