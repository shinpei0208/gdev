/*! \file   ThreadFrontierReconvergencePass.cpp
	\author Gregory Diamso <gregory.diamos@gatech.edu>
	\date   Monday May 16, 2011
	\brief  The source file for the ThreadFrontierReconvergencePass class.
*/

#ifndef THREAD_FRONTIER_RECONVERGENCE_PASS_CPP_INCLUDED
#define THREAD_FRONTIER_RECONVERGENCE_PASS_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/transforms/interface/ThreadFrontierReconvergencePass.h>

#include <ocelot/analysis/interface/ThreadFrontierAnalysis.h>

#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <map>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace transforms
{

ThreadFrontierReconvergencePass::ThreadFrontierReconvergencePass(bool g)
: ImmutableKernelPass({"ThreadFrontierAnalysis"},
	"ThreadFrontierReconvergencePass"), _gen6(g)
{

}

void ThreadFrontierReconvergencePass::initialize(const ir::Module& m)
{
	instructions.clear();
}

void ThreadFrontierReconvergencePass::runOnKernel(const ir::IRKernel& k)
{
	report("Running thread frontier reconvergence pass");
	typedef analysis::ThreadFrontierAnalysis::Priority Priority;
	typedef std::multimap<Priority, ir::ControlFlowGraph::const_iterator,
		std::greater<Priority>> ReversePriorityMap;
	typedef analysis::ThreadFrontierAnalysis TFAnalysis;
	typedef ir::ControlFlowGraph::const_pointer_iterator const_pointer_iterator;

	Analysis* analysis = getAnalysis("ThreadFrontierAnalysis");
	assert(analysis != 0);
	
	TFAnalysis* tfAnalysis = static_cast<TFAnalysis*>(analysis);

	ReversePriorityMap priorityToBlocks;
	
	// sort by priority (high to low)
	for(ir::ControlFlowGraph::const_iterator block = k.cfg()->begin();
		block != k.cfg()->end(); ++block)
	{
		priorityToBlocks.insert(std::make_pair(tfAnalysis->getPriority(block),
			block));
	}
	
	typedef std::unordered_map<ir::BasicBlock::Id, unsigned int> IdToPCMap;
	typedef std::unordered_map<unsigned int,
		ir::ControlFlowGraph::const_iterator> PCToBlockMap;

	IdToPCMap     pcs;
	PCToBlockMap  branchPCs;
	PCToBlockMap  fallthroughPCs;
	
	// lay the code out in priority order
	report(" Packing instructions into a vector");
	for(ReversePriorityMap::const_iterator
		priorityAndBlock = priorityToBlocks.begin();
		priorityAndBlock != priorityToBlocks.end(); ++priorityAndBlock)
	{
		ir::ControlFlowGraph::const_iterator block = priorityAndBlock->second;

		report("  Basic Block " << block->label() << " ("
			<< block->id << ")");
			
		pcs.insert(std::make_pair(block->id, instructions.size()));
		
		for(ir::ControlFlowGraph::InstructionList::const_iterator 
			instruction = block->instructions.begin();
			instruction != block->instructions.end(); ++instruction)
		{
			const ir::PTXInstruction& ptx = static_cast<
				const ir::PTXInstruction&>(**instruction);
				
			report("   [" << instructions.size() << "] '" << ptx.toString());
			
			instructions.push_back(ptx);
			instructions.back().pc = instructions.size() - 1;
			
			if(ptx.opcode == ir::PTXInstruction::Bra)
			{
				branchPCs.insert(std::make_pair(instructions.back().pc, block));
			}
		}
		
		if(!_gen6)
		{
			// Add a branch for the fallthrough if it is in the TF
			if(block->has_fallthrough_edge())
			{
				ir::ControlFlowGraph::const_iterator target =
					block->get_fallthrough_edge()->tail;
				
				ReversePriorityMap::const_iterator next = priorityAndBlock;
				++next;
				
				bool needsCheck = target != next->second;
				
				TFAnalysis::BlockVector frontier =
						tfAnalysis->getThreadFrontier(block);
		
				for(TFAnalysis::BlockVector::const_iterator
					stalledBlock = frontier.begin();
					stalledBlock != frontier.end(); ++stalledBlock)
				{
					if((*stalledBlock)->id == target->id)
					{
						needsCheck = true;
						break;
					}
				}
				
				if(needsCheck)
				{
					fallthroughPCs.insert(std::make_pair(
						instructions.size(), block));
					
					instructions.push_back(ir::PTXInstruction(
						ir::PTXInstruction::Bra,
						ir::PTXOperand(target->label())));
		
					instructions.back().needsReconvergenceCheck = true;
					instructions.back().branchTargetInstruction = -1;
					report("   [" << (instructions.size() - 1) << "] '"
						<< instructions.back().toString());
					report("    - artificial branch for check on"
						" fallthrough into TF.");
				}
			}
		}
	}
	
	report(" Updating branch targets");
	for(PCToBlockMap::const_iterator pcAndBlock = branchPCs.begin();
		pcAndBlock != branchPCs.end(); ++pcAndBlock)
	{
		ir::ControlFlowGraph::const_iterator block = pcAndBlock->second;
		unsigned int pc                            = pcAndBlock->first;
		
		const ir::PTXInstruction& ptx = static_cast<
			const ir::PTXInstruction&>(instructions[pc]);
			
		ir::ControlFlowGraph::const_iterator target =
			block->get_branch_edge()->tail;
					
		IdToPCMap::const_iterator targetPC = pcs.find(target->id);
		assert(targetPC != pcs.end());
		
		report("  setting branch target of '" << ptx.toString()
			<< "' to " << targetPC->second);
		
		instructions[pc].branchTargetInstruction = targetPC->second;
		
		TFAnalysis::BlockVector frontier =
			tfAnalysis->getThreadFrontier(block);
		
		if(_gen6)
		{
			ir::ControlFlowGraph::const_iterator firstBlock =
				k.cfg()->end();
		
			TFAnalysis::Priority highest = 0;
		
			frontier.push_back(block->get_branch_edge()->tail);
			
			if(block->has_fallthrough_edge())
			{
				frontier.push_back(
					block->get_fallthrough_edge()->tail);
			}
		
			// gen6 jumps to the block with the highest priority
			for(TFAnalysis::BlockVector::const_iterator
				stalledBlock = frontier.begin();
				stalledBlock != frontier.end(); ++stalledBlock)
			{
				TFAnalysis::Priority priority =
					tfAnalysis->getPriority(*stalledBlock);
				if(priority >= highest)
				{
					highest    = priority;
					firstBlock = *stalledBlock;
				}
			}
		
			// the reconverge point is the first block in the frontier
			assert(firstBlock != k.cfg()->end());
	
			IdToPCMap::const_iterator reconverge =
				pcs.find(firstBlock->id);
			assert(reconverge != pcs.end());
			instructions[pc].reconvergeInstruction =
				reconverge->second;
			report("   re-converge point "  << reconverge->second
				<< ", " << firstBlock->label());
		}
		else
		{
			// Does this branch need to check for re-convergence?
			// Or: are any of the target's predecessors
			//       in the thread frontier?
			bool needsCheck = false;
			
			for(TFAnalysis::BlockVector::const_iterator
				stalledBlock = frontier.begin();
				stalledBlock != frontier.end(); ++stalledBlock)
			{
				if((*stalledBlock)->id == target->id)
				{
					needsCheck = true;
					report("   needs re-convergence check.");
					break;
				}
			}
			
			instructions[pc].needsReconvergenceCheck = needsCheck;
		}
	}
	
	report(" Updating fallthrough targets");
	for(PCToBlockMap::const_iterator pcAndBlock = fallthroughPCs.begin();
		pcAndBlock != fallthroughPCs.end(); ++pcAndBlock)
	{
		ir::ControlFlowGraph::const_iterator block = pcAndBlock->second;
		unsigned int pc                            = pcAndBlock->first;
		
		const ir::PTXInstruction& ptx = static_cast<
			const ir::PTXInstruction&>(instructions[pc]);
			
		ir::ControlFlowGraph::const_iterator target =
			block->get_fallthrough_edge()->tail;
					
		IdToPCMap::const_iterator targetPC = pcs.find(target->id);
		assert(targetPC != pcs.end());
		
		report("  setting branch target of '" << ptx.toString()
			<< "' to " << targetPC->second);
		
		instructions[pc].branchTargetInstruction = targetPC->second;
	}
	
}

void ThreadFrontierReconvergencePass::finalize()
{

}

}

#endif

