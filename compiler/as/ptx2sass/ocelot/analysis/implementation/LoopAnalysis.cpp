/*! \file   LoopAnalysis.cpp
	\date   Thursday April 30, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the LoopAnalysis class.
*/

// Ocelot Includes
#include <ocelot/analysis/interface/LoopAnalysis.h>

#include <ocelot/analysis/interface/DominatorTree.h>

#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <algorithm>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace analysis
{

LoopAnalysis::Loop::Loop()
: parent(0)
{

}

LoopAnalysis::Loop::block_iterator LoopAnalysis::Loop::getHeader() const
{
	if(numberOfBlocks() > 0) return *block_begin();
	
	return block_iterator();
}

LoopAnalysis::Loop* LoopAnalysis::Loop::getParentLoop() const
{
	return parent;
}

LoopAnalysis::Loop::iterator LoopAnalysis::Loop::begin()
{
	return subLoops.begin();
}

LoopAnalysis::Loop::iterator LoopAnalysis::Loop::end()
{
	return subLoops.end();
}

LoopAnalysis::Loop::const_iterator LoopAnalysis::Loop::begin() const
{
	return subLoops.begin();
}

LoopAnalysis::Loop::const_iterator LoopAnalysis::Loop::end() const
{
	return subLoops.end();
}

size_t LoopAnalysis::Loop::size() const
{
	return subLoops.size();
}

bool LoopAnalysis::Loop::empty() const
{
	return subLoops.empty();
}

unsigned int LoopAnalysis::Loop::getLoopDepth() const
{
	if(parent == 0) return 0;
	
	return parent->getLoopDepth() + 1;
}

LoopAnalysis::Loop::block_pointer_iterator LoopAnalysis::Loop::block_begin()
{
	return blocks.begin();
}

LoopAnalysis::Loop::block_pointer_iterator LoopAnalysis::Loop::block_end()
{
	return blocks.end();
}

LoopAnalysis::Loop::const_block_pointer_iterator
	LoopAnalysis::Loop::block_begin() const
{
	return blocks.begin();
}

LoopAnalysis::Loop::const_block_pointer_iterator
	LoopAnalysis::Loop::block_end() const
{
	return blocks.end();
}

size_t LoopAnalysis::Loop::numberOfBlocks() const
{
	return blocks.size();
}

bool LoopAnalysis::Loop::contains(const_block_iterator block) const
{
	for(const_block_pointer_iterator b = block_begin(); b != block_end(); ++b)
	{
		if(*b == block) return true;
	}

	return false;
}

bool LoopAnalysis::Loop::contains(const Loop* loop) const
{
	if(loop == this) return true;
	if(loop == 0)    return false;

	return contains(loop->getParentLoop());
}

LoopAnalysis::Loop::BlockPointerVector LoopAnalysis::Loop::getExitBlocks()
{
	BlockPointerVector exitBlocks;
	
	for(auto block = block_begin(); block != block_end(); ++block)
	{
		for(auto successor = (*block)->successors.begin();
			successor != (*block)->successors.end(); ++successor)
		{
			if(std::find(block_begin(), block_end(), *successor) == block_end())
			{
				exitBlocks.push_back(*successor);
			}
		}
	}
	
	return exitBlocks;
}

LoopAnalysis::Loop::block_iterator LoopAnalysis::Loop::getExitBlock()
{
	BlockPointerVector exitBlocks = getExitBlocks();
	
	if(exitBlocks.size() == 1) return exitBlocks[0];
	
	return block_iterator();
}

LoopAnalysis::Loop::block_iterator LoopAnalysis::Loop::getLoopPreheader()
{
	block_iterator predecessor = getLoopPredecessor();
	
	if(predecessor == block_iterator() || predecessor->out_edges.size() > 1)
	{
		return block_iterator();
	}
	
	return predecessor;
}

LoopAnalysis::Loop::block_iterator LoopAnalysis::Loop::getLoopLatch()
{
	block_iterator header = getHeader();

	block_iterator latch = block_iterator();
	
	for(auto successor = header->successors.begin();
		successor != header->successors.end(); ++successor)
	{
		if(contains(*successor))
		{
			if(latch != block_iterator())
			{
				return block_iterator();
			}
			
			latch = *successor;
		}
	}

    return latch;
}

LoopAnalysis::Loop::block_iterator LoopAnalysis::Loop::getLoopPredecessor()
{
	block_iterator predecessor = block_iterator();
	
	block_iterator header = getHeader();
	
	for(auto successor = header->successors.begin();
		successor != header->successors.end(); ++successor)
	{
		if(!contains(*successor))
		{
			if(predecessor != block_iterator() && predecessor != *successor)
			{
				return block_iterator();
			}
			
			predecessor = *successor;
		}
	}

	return predecessor;
}

LoopAnalysis::LoopAnalysis()
: KernelAnalysis("LoopAnalysis", {"DominatorTreeAnalysis"})
{

}

void LoopAnalysis::analyze(ir::IRKernel& kernel)
{
	_blockToLoopMap.clear();
	
	analysis::Analysis* dominatorTreeAnalysis =
		getAnalysis("DominatorTreeAnalysis");
	assert(dominatorTreeAnalysis != 0);
	
	analysis::DominatorTree* dominatorTree =
		static_cast<analysis::DominatorTree*>(dominatorTreeAnalysis);

	_kernel = &kernel;

	report("Running loop analysis over kernel '" << kernel.name << "'");

	ir::ControlFlowGraph::BlockPointerVector dfsOrder =
		kernel.cfg()->pre_order_sequence();

	for(ir::ControlFlowGraph::pointer_iterator block = dfsOrder.begin();
		block != dfsOrder.end(); ++block)
	{
		_tryAddingLoop(*block, dominatorTree);
	}
}

LoopAnalysis::iterator LoopAnalysis::begin()
{
	return _loops.begin();
}

LoopAnalysis::iterator LoopAnalysis::end()
{
	return _loops.end();
}

LoopAnalysis::const_iterator LoopAnalysis::begin() const
{
	return _loops.begin();
}

LoopAnalysis::const_iterator LoopAnalysis::end() const
{
	return _loops.end();
}

bool LoopAnalysis::isContainedInLoop(const_block_iterator block)
{
	BlockToLoopMap::iterator loop = _blockToLoopMap.find(block);
	
	return loop != _blockToLoopMap.end();
} 

static bool isNotContainedIn(LoopAnalysis::Loop* subLoop,
	LoopAnalysis::Loop* loop)
{
	if(subLoop == 0)    return true;
	if(subLoop == loop) return false;
	return isNotContainedIn(subLoop->getParentLoop(), loop);
}

bool LoopAnalysis::_tryAddingLoop(ir::ControlFlowGraph::iterator bb,
	analysis::DominatorTree* dominatorTree)
{
	typedef std::vector<block_iterator> BlockQueue;
	
	report(" Checking for loops starting at " << bb->label() << ":");
	
	if(_blockToLoopMap.count(bb) != 0) return false;
	
	BlockQueue queued;
	
	// Scan the predecessors of the current block, if this block dominates a
	//  predecessor, it could be a loop header
	for(auto predecessor = bb->predecessors.begin();
		predecessor != bb->predecessors.end(); ++predecessor)
	{
		if(dominatorTree->dominates(bb, *predecessor))
		{
			queued.push_back(*predecessor);
		}
	}
	
	if(queued.empty()) return false;
	
	report("  Found one, discovering contained blocks/loops");
	
	iterator loop = _loops.insert(_loops.end(), Loop());
	loop->blocks.push_back(bb);
	
	_blockToLoopMap.insert(std::make_pair(bb, &*loop));
	
	block_iterator entryBlock = _kernel->cfg()->get_entry_block();
	
	// Add all blocks to the loop
	while(!queued.empty())
	{
		block_iterator block = queued.back();
		queued.pop_back();
		
		report("   checking predecessor " << block->label() << "...");
		if(!loop->contains(block) &&
			dominatorTree->dominates(entryBlock, block))
		{
			report("    adding it to the loop");
			
			Loop* subLoop = _getLoopAt(block);
			
			// fix loops that should be subloops of this loop
			if(subLoop != 0)
			{
				if(subLoop->getHeader() == block &&
					isNotContainedIn(subLoop, &*loop))
				{
					report("     but it is already a loop header that was "
						"already created, making it a subloop.");
			
					assert(subLoop->getParentLoop() != 0);
					assert(subLoop->getParentLoop() != &*loop);
					
					Loop::iterator subLoopIterator = std::find(
						subLoop->getParentLoop()->begin(),
						subLoop->getParentLoop()->end(), &*subLoop);
					assert(subLoopIterator != subLoop->getParentLoop()->end());
					
					subLoop->getParentLoop()->subLoops.erase(subLoopIterator);
					
					subLoop->parent = &*loop;
					loop->subLoops.push_back(subLoop);
				}
			}
			
			loop->blocks.push_back(block);
			
			for(auto predecessor = block->predecessors.begin();
				predecessor != block->predecessors.end(); ++predecessor)
			{
				queued.push_back(*predecessor);
			}
		}
	}
	
	// Find subloops
	report("   searching for subloops");
	for(block_pointer_iterator block = loop->block_begin();
		block != loop->block_end(); ++block)
	{
		if(_tryAddingLoop(*block, dominatorTree))
		{
			Loop* subLoop   = _getLoopAt(*block);
			subLoop->parent = &*loop;
			loop->subLoops.push_back(subLoop);
			report("    setting parent to " << bb->label());
		}
	}
	
	report("   registering blocks in the loop");
	
	// Register blocks in this loop
	for(block_pointer_iterator block = loop->block_begin();
		block != loop->block_end(); ++block)
	{
		_blockToLoopMap.insert(std::make_pair(*block, &*loop));
	}
	
	// Handle nested loops
	BlockToLoopMap containingLoops;
	
	Loop::LoopVector subLoops = loop->subLoops;
	
	for(Loop::iterator subLoop = subLoops.begin();
		subLoop != subLoops.end(); ++subLoop)
	{
		assert((*subLoop)->getParentLoop() == &*loop);
		
		BlockToLoopMap::iterator containingLoop = containingLoops.find(
			(*subLoop)->getHeader());
		
		if(containingLoop != containingLoops.end())
		{
			_moveSiblingLoopInto(*subLoop, containingLoop->second);
		}
		else
		{
			for(block_pointer_iterator block = (*subLoop)->block_begin();
				block != (*subLoop)->block_end(); ++block)
			{
				BlockToLoopMap::iterator subSubLoop = containingLoops.find(
					*block);
					
				if(subSubLoop != containingLoops.end() &&
					subSubLoop->second != *subLoop)
				{
					for(block_pointer_iterator
						block = subSubLoop->second->block_begin();
						block != subSubLoop->second->block_begin(); ++block)
					{
						containingLoops[*block] = *subLoop;
					}
					
					_moveSiblingLoopInto(*subLoop, subSubLoop->second);
				}
			}
		}
	}
	
	return true;
}
	
void LoopAnalysis::_moveSiblingLoopInto(Loop* to, Loop* from)
{
    Loop* toParent = to->getParentLoop();
    assert(toParent != 0 &&
    	from->getParentLoop() == toParent && to != from);

	Loop::iterator toIterator = std::find(toParent->subLoops.begin(),
		toParent->subLoops.end(), to);
    assert(toIterator != toParent->subLoops.end());
    
    toParent->subLoops.erase(toIterator);
    to->parent = 0;
    
    _insertLoopInto(to, from);
}
	
void LoopAnalysis::_insertLoopInto(Loop* loop, Loop* parent)
{
	Loop::block_iterator header = loop->getHeader();
	assert(parent->contains(header));

	for(Loop::iterator subLoop = parent->begin();
		subLoop != parent->end(); ++subLoop)
	{
		if((*subLoop)->contains(header))
		{
			_insertLoopInto(loop, *subLoop);
			return;
		}
	}
    
    parent->subLoops.push_back(loop);
    loop->parent = parent;
}

LoopAnalysis::Loop* LoopAnalysis::_getLoopAt(const_block_iterator block)
{
	BlockToLoopMap::iterator loop = _blockToLoopMap.find(block);
	
	if(loop != _blockToLoopMap.end())
	{
		return loop->second;
	}
	
	return 0;
}

}

