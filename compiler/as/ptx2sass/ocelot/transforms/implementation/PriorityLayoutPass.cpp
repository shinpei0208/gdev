/*! \file   PriorityLayoutPass.cpp
	\author Gregory Diamso <gregory.diamos@gatech.edu>
	\date   Monday May 16, 2011
	\brief  The source file for the PriorityLayoutPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/PriorityLayoutPass.h>

#include <ocelot/analysis/interface/ThreadFrontierAnalysis.h>

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

PriorityLayoutPass::PriorityLayoutPass()
: ImmutableKernelPass({"ThreadFrontierAnalysis"}, "PriorityLayoutPass")
{

}

void PriorityLayoutPass::initialize(const ir::Module& m)
{
	instructions.clear();
}

void PriorityLayoutPass::runOnKernel(const ir::IRKernel& k)
{
	report("Running priority layout pass");
	
	typedef analysis::ThreadFrontierAnalysis::Priority Priority;
	typedef std::unordered_map<std::string, unsigned int> LabelToPCMap;
	typedef std::map<Priority, ir::ControlFlowGraph::const_iterator,
		std::greater<Priority>> PriorityToBlockMap;

	LabelToPCMap pcs;

	PriorityToBlockMap blocks;
	
	auto tfAnalysis = static_cast<analysis::ThreadFrontierAnalysis*>(
		getAnalysis("ThreadFrontierAnalysis"));

	// Sort blocks by priority
	for(auto block = k.cfg()->begin(); block != k.cfg()->end(); ++block)
	{
		blocks.insert(std::make_pair(tfAnalysis->getPriority(block), block));
	}
		
	report(" Packing instructions into a vector");
	for(auto block = blocks.begin(); block != blocks.end(); ++block)
	{
		for(auto instruction = block->second->instructions.begin();
			instruction != block->second->instructions.end(); ++instruction)
		{
			const ir::PTXInstruction& ptx = static_cast<
				const ir::PTXInstruction&>(**instruction);
				
			report("  [" << instructions.size() << "] '" << ptx.toString());
				
			if(instruction == block->second->instructions.begin())
			{
				pcs.insert(std::make_pair(block->second->label(),
					instructions.size()));
			}
			
			instructions.push_back(ptx);
			instructions.back().pc = instructions.size() - 1;
		}
	}
	
	report(" Updating branch targets");
	unsigned int pc = 0;
	for(auto block = blocks.begin(); block != blocks.end(); ++block)
	{
		for(auto instruction = block->second->instructions.begin();
			instruction != block->second->instructions.end();
			++instruction, ++pc)
		{
			const ir::PTXInstruction& ptx = static_cast<
				const ir::PTXInstruction&>(**instruction);
			if(ptx.opcode == ir::PTXInstruction::Bra)
			{
				LabelToPCMap::const_iterator target =
					pcs.find(ptx.d.identifier);
				assert(target != pcs.end());
				
				report("  setting branch target of '" << ptx.toString()
					<< "' to " << target->second);
				
				instructions[pc].branchTargetInstruction = target->second;
			}
		}
	}
}

void PriorityLayoutPass::finalize()
{
	
}

}


