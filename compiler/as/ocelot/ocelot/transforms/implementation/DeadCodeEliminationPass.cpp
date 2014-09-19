/*! \file   DeadCodeEliminationPass.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday July 21, 2011
	\brief  The source file for the DeadCodeEliminationPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/DeadCodeEliminationPass.h>

#include <ocelot/analysis/interface/DataflowGraph.h>

#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0
#define REPORT_PTX  0

namespace transforms
{

DeadCodeEliminationPass::DeadCodeEliminationPass()
: KernelPass({"DataflowGraphAnalysis"}, "DeadCodeEliminationPass")
{

}

void DeadCodeEliminationPass::initialize(const ir::Module& m)
{
	
}

typedef analysis::DataflowGraph::iterator              iterator;
typedef analysis::DataflowGraph::Instruction           Instruction;
typedef analysis::DataflowGraph::RegisterPointerVector RegisterPointerVector;
typedef analysis::DataflowGraph::RegisterVector        RegisterVector;
typedef std::unordered_set<iterator>                   BlockSet;
typedef analysis::DataflowGraph::BlockPointerSet       BlockPointerSet;
typedef analysis::DataflowGraph::PhiInstruction        PhiInstruction;
typedef analysis::DataflowGraph::Register              Register;
typedef analysis::DataflowGraph::RegisterSet           RegisterSet;
typedef analysis::DataflowGraph::InstructionVector     InstructionVector;
typedef analysis::DataflowGraph::PhiInstructionVector  PhiInstructionVector;

static bool canRemoveAliveOut(analysis::DataflowGraph& dfg,
	iterator block, const Register& aliveOut)
{
	for(BlockPointerSet::iterator successor = block->targets().begin();
		successor != block->targets().end(); ++successor)
	{
		for(RegisterSet::iterator aliveIn = (*successor)->aliveIn().begin();
			aliveIn != (*successor)->aliveIn().end(); ++aliveIn)
		{
			// A target successor uses the value
			if(aliveIn->id == aliveOut.id) return false;
		}
	}
	
	if(block->fallthrough() != dfg.end())
	{
		iterator successor = block->fallthrough();
		for(RegisterSet::iterator aliveIn = successor->aliveIn().begin();
			aliveIn != successor->aliveIn().end(); ++aliveIn)
		{
			// A fallthrough successor uses the value
			if(aliveIn->id == aliveOut.id) return false;
		}
	}
	
	// No successors use the value
	return true;
}

static bool canRemoveInstruction(iterator block,
	InstructionVector::iterator instruction)
{
	ir::PTXInstruction& ptx = *static_cast<ir::PTXInstruction*>(
		instruction->i);

	if(ptx.hasSideEffects()) return false;
	
	for(RegisterPointerVector::iterator reg = instruction->d.begin();
		reg != instruction->d.end(); ++reg)
	{
		// the reg is alive outside the block
		if(block->aliveOut().count(*reg) != 0) return false;
		InstructionVector::iterator next = instruction;
		for(++next; next != block->instructions().end(); ++next)
		{
			for(RegisterPointerVector::iterator source = next->s.begin();
				source != next->s.end(); ++source)
			{
				// found a user in the block
				if(*source->pointer == *reg->pointer) return false;
			}
		}
	}
	
	// There are no users and the instruction has no side effects
	return true;
}

static bool canRemovePhi(iterator block, const PhiInstruction& phi)
{
	// The value produced by the phi is alive out of the block
	if(block->aliveOut().count(phi.d.id) != 0) return false;

	for(InstructionVector::iterator next = block->instructions().begin();
		next != block->instructions().end(); ++next)
	{
		for(RegisterPointerVector::iterator source = next->s.begin();
			source != next->s.end(); ++source)
		{
			// found a user in the block
			if(*source->pointer == phi.d.id) return false;
		}
	}
	
	// There are no users in the block and it is dead outside the block
	return true;
}

static bool canRemoveAliveIn(iterator block, const Register& aliveIn)
{
	for(PhiInstructionVector::iterator phi = block->phis().begin();
		phi != block->phis().end(); ++phi)
	{
		for(RegisterVector::iterator source = phi->s.begin();
			source != phi->s.end(); ++source)
		{
			// found a phi user
			if(source->id == aliveIn.id) return false;
		}
	}
	
	// The are no users in the block
	return true;
}

static void eliminateDeadInstructions(analysis::DataflowGraph& dfg,
	BlockSet& blocks, iterator block)
{
	typedef analysis::DataflowGraph::Block                Block;
	typedef analysis::DataflowGraph::RegisterSet   RegisterSet;
	typedef std::vector<unsigned int>                     KillList;
	typedef std::vector<PhiInstructionVector::iterator>   PhiKillList;
	typedef std::vector<RegisterSet::iterator>            AliveKillList;
	
	report(" Eliminating dead instructions from BB_" << block->id());
	
	report("  Removing dead alive out values");
	AliveKillList aliveOutKillList;
	for(RegisterSet::iterator aliveOut = block->aliveOut().begin();
		aliveOut != block->aliveOut().end(); ++aliveOut)
	{
		if(canRemoveAliveOut(dfg, block, *aliveOut))
		{
			report("   removed " << aliveOut->id);
			aliveOutKillList.push_back(aliveOut);
		}
	}
	
	for(AliveKillList::iterator killed = aliveOutKillList.begin();
		killed != aliveOutKillList.end(); ++killed)
	{
		block->aliveOut().erase(*killed);
	}
	
	KillList killList;
	
	report("  Removing dead instructions");
	unsigned int index = 0;
	for(InstructionVector::iterator instruction = block->instructions().begin();
		instruction != block->instructions().end(); ++instruction)
	{
		if(canRemoveInstruction(block, instruction))
		{
			report("   removed '" << instruction->i->toString() << "'");
			killList.push_back(index);
			
			// schedule the block for more work
			report("    scheduled this block again");
			blocks.insert(block);
		}
		else
		{
			++index;
		}
	}
	
	for(KillList::iterator killed = killList.begin();
		killed != killList.end(); ++killed)
	{
		dfg.erase(block, *killed);
	}
	
	PhiKillList phiKillList;
	
	report("  Removing dead phi instructions");
	for(PhiInstructionVector::iterator phi = block->phis().begin();
		phi != block->phis().end(); ++phi)
	{
		if(canRemovePhi(block, *phi))
		{
			report("   removed " << phi->d.id);
			phiKillList.push_back(phi);
		}
	}
	
	report("  Removing dead alive in values");
	AliveKillList aliveInKillList;
	for(RegisterSet::iterator aliveIn = block->aliveIn().begin();
		aliveIn != block->aliveIn().end(); ++aliveIn)
	{
		if(canRemoveAliveIn(block, *aliveIn))
		{
			report("   removed " << aliveIn->id);
			aliveInKillList.push_back(aliveIn);
			
			// schedule the predecessors for more work
			for(BlockPointerSet::iterator predecessor =
				block->predecessors().begin();
				predecessor != block->predecessors().end(); ++predecessor)
			{
				report("    scheduled predecessor BB_" << (*predecessor)->id());
				blocks.insert(*predecessor);
			}
		}
	}
	
	for(AliveKillList::iterator killed = aliveInKillList.begin();
		killed != aliveInKillList.end(); ++killed)
	{
		block->aliveIn().erase(*killed);
	}
}

void DeadCodeEliminationPass::runOnKernel(ir::IRKernel& k)
{
	report("Running dead code elimination on kernel " << k.name);
	reportE(REPORT_PTX, k);
	
	Analysis* dfgAnalysis = getAnalysis("DataflowGraphAnalysis");
	assert(dfgAnalysis != 0);

	analysis::DataflowGraph& dfg =
		*static_cast<analysis::DataflowGraph*>(dfgAnalysis);
	
	dfg.convertToSSAType(analysis::DataflowGraph::Minimal);
	
	assert(dfg.ssa() != analysis::DataflowGraph::SsaType::None);
	
	BlockSet blocks;
	
	report(" Starting by scanning all basic blocks");
	
	for(iterator block = dfg.begin(); block != dfg.end(); ++block)
	{
		report("  Queueing up BB_" << block->id());
		blocks.insert(block);
	}
	
	while(!blocks.empty())
	{
		iterator block = *blocks.begin();
		blocks.erase(blocks.begin());
	
		eliminateDeadInstructions(dfg, blocks, block);
	}
	
	report("Finished running dead code elimination on kernel " << k.name);
	reportE(REPORT_PTX, k);
}

void DeadCodeEliminationPass::finalize()
{

}

}

