/*! \file   DefaultLayoutPass.cpp
	\author Gregory Diamso <gregory.diamos@gatech.edu>
	\date   Monday May 16, 2011
	\brief  The source file for the DefaultLayoutPass class.
*/

#ifndef DEFAULT_LAYOUT_PASS_CPP_INCLUDED
#define DEFAULT_LAYOUT_PASS_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/transforms/interface/DefaultLayoutPass.h>

#include <ocelot/ir/interface/ControlFlowGraph.h>
#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <unordered_map>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace transforms
{

DefaultLayoutPass::DefaultLayoutPass()
: ImmutableKernelPass({}, "DefaultLayoutPass")
{

}

void DefaultLayoutPass::initialize(const ir::Module& m)
{
	instructions.clear();
}

void DefaultLayoutPass::runOnKernel(const ir::IRKernel& k)
{
	report("Running default layout pass");
	typedef std::unordered_map<ir::BasicBlock::Id, unsigned int> IdToPCMap;

	IdToPCMap pcs;

	// TODO get this from an analysis rather than directly from the CFG
	ir::ControlFlowGraph::BlockPointerVector
		blocks = const_cast<ir::IRKernel&>(k).cfg()->executable_sequence();
	
	report(" Packing instructions into a vector");
	for(ir::ControlFlowGraph::BlockPointerVector::const_iterator
		block = blocks.begin(); block != blocks.end(); ++block)
	{
		for(ir::ControlFlowGraph::InstructionList::iterator 
			instruction = (*block)->instructions.begin();
			instruction != (*block)->instructions.end(); ++instruction)
		{
			const ir::PTXInstruction& ptx = static_cast<
				const ir::PTXInstruction&>(**instruction);
				
			report("  [" << instructions.size() << "] '" << ptx.toString());
				
			if(instruction == (*block)->instructions.begin())
			{
				pcs.insert(std::make_pair((*block)->id, instructions.size()));
			}
			
			instructions.push_back(ptx);
			instructions.back().pc = instructions.size() - 1;
			
		}
	}
	
	report(" Updating branch targets");
	unsigned int pc = 0;
	for(ir::ControlFlowGraph::BlockPointerVector::const_iterator
		block = blocks.begin(); block != blocks.end(); ++block)
	{
		for(ir::ControlFlowGraph::InstructionList::iterator 
			instruction = (*block)->instructions.begin();
			instruction != (*block)->instructions.end(); ++instruction, ++pc)
		{
			const ir::PTXInstruction& ptx = static_cast<
				const ir::PTXInstruction&>(**instruction);
			if(ptx.opcode == ir::PTXInstruction::Bra)
			{
				IdToPCMap::const_iterator target = pcs.find(
					(*block)->get_branch_edge()->tail->id);
				assert(target != pcs.end());
				
				report("  setting branch target of '" << ptx.toString()
					<< "' to " << target->second);
				
				instructions[pc].branchTargetInstruction = target->second;
			}
		}
	}
}

void DefaultLayoutPass::finalize()
{
	
}

}

#endif

