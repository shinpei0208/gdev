/*! \file   MIMDThreadSchedulingPass.cpp
	\date   Friday February 18, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the MIMDThreadSchedulingPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/MIMDThreadSchedulingPass.h>

#include <ocelot/ir/interface/IRKernel.h>

#include <ocelot/analysis/interface/SafeRegionAnalysis.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <unordered_set>
#include <cassert>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace transforms
{

MIMDThreadSchedulingPass::MIMDThreadSchedulingPass()
: KernelPass({"SafeRegionAnalysis"}, "MIMDThreadSchedulingPass")
{

}

typedef ir::PTXInstruction PTXInstruction;
typedef MIMDThreadSchedulingPass::Statistics Statistics;
typedef analysis::SafeRegionAnalysis::SafeRegion SafeRegion;

static void insertYieldsBeforeBlocksWithSideEffects(const SafeRegion*,
	Statistics&);

void MIMDThreadSchedulingPass::runOnKernel(ir::IRKernel& k)
{
	statistics.reset();
	
	report("Discovering branches that require scheduler yields.");
	
	// Get analyses
	auto safeRegionAnalysis = static_cast<analysis::SafeRegionAnalysis*>(
		getAnalysis("SafeRegionAnalysis"));

	insertYieldsBeforeBlocksWithSideEffects(safeRegionAnalysis->getRoot(),
		statistics);

	report(" Total instructions:       " << statistics.totalInstructions);
	report(" Total safe instructions:  " << statistics.totalSafeInstructions);
	report(" Percent safe instructions "
		<< ((100.0 * statistics.totalSafeInstructions) /
			(statistics.totalInstructions)) << "%");
}

Statistics::Statistics()
: totalInstructions(0), totalSafeInstructions(0)
{

}

void Statistics::reset()
{
	totalInstructions     = 0;
	totalSafeInstructions = 0;
}

typedef ir::ControlFlowGraph::iterator block_iterator;

static void insertYield(block_iterator block);

static void insertYieldsBeforeBlocksWithSideEffects(const SafeRegion* region,
	Statistics& statistics)
{
	if(region->isLeaf())
	{
		unsigned int instructions = region->block->instructions.size();
	
		statistics.totalInstructions += instructions;
		
		if(region->doesNotDependOnSideEffects)
		{
			statistics.totalSafeInstructions += instructions;
		}
		else
		{
			insertYield(region->block);
		}
	}
	else
	{
		for(auto& child : region->children)
		{
			insertYieldsBeforeBlocksWithSideEffects(&child, statistics);
		}
	}
}

typedef ir::PTXOperand PTXOperand;

static void insertYield(block_iterator block)
{
	PTXInstruction* yield = new PTXInstruction(PTXInstruction::Call);

	yield->a = PTXOperand("_Zintrinsic_yield");

	block->instructions.insert(block->instructions.begin(), yield);
}

}


