/*! \file   SuperblockAnalysis.cpp
	\author Gregory Diamos <gdiamos@nvidia.com>
	\date   Monday August 15, 2011
	\brief  The source file for the superblock analysis class.
*/

// Ocelot Includes
#include <ocelot/analysis/interface/SuperblockAnalysis.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <unordered_set>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace analysis
{

SuperblockAnalysis::SuperblockAnalysis(ir::ControlFlowGraph& c, unsigned int b)
: _cfg(&c)
{
	typedef std::unordered_set<ir::ControlFlowGraph::iterator> BlockSet;
		
	// Find all trivial superblocks, assume they were freed up previously
	//  by tail duplication, unrolling, and peeling
	report("Finding superblocks.");
	report(" Seeding superblocks.");
	BlockSet superblocks;
	BlockSet visited;
	
	// Superblocks start from multiple-entry blocks
	for(ir::ControlFlowGraph::iterator block = _cfg->begin();
		block != _cfg->end(); ++block)
	{
		if(block->predecessors.size() > 1)
		{
			report("  Starting a new superblock at " << block->label());
			superblocks.insert(block);
			visited.insert(block);
		}
	}
	
	// Find successors with only one predecessor, these can be included
	report(" Adding successors.");
	for(BlockSet::iterator block = superblocks.begin();
		block != superblocks.end(); ++block)
	{
		report("  For superblock " << (*block)->label());
		Block superblock;
	
		for(ir::ControlFlowGraph::pointer_iterator successor =
			(*block)->successors.begin();
			successor != (*block)->successors.end(); ++successor)
		{
			if((*successor)->predecessors.size() == 1)
			{
				report("   Added successor " << (*successor)->label());
				superblock.insert(*successor);
				visited.insert(*successor);
			}
		}
		
		_blocks.push_back(superblock);
	}
	
	// Add regular blocks to fill out the program structure
	report(" Adding stragglers.");
	for(ir::ControlFlowGraph::iterator block = _cfg->begin();
		block != _cfg->end(); ++block)
	{
		if(visited.count(block) == 0)
		{
			report("  Adding normal block " << block->label());
			
			Block normalblock;
			normalblock.insert(block);
			_blocks.push_back(normalblock);
		}
	}
}

}


