/*! \file   HammockGraphAnalysis.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday May 31, 2013
	\brief  The source file for the HammockGraphAnalysis class.
*/

// Ocelot Incudes
#include <ocelot/analysis/interface/HammockGraphAnalysis.h>

#include <ocelot/analysis/interface/PostdominatorTree.h>
#include <ocelot/analysis/interface/DominatorTree.h>

#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cassert>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace analysis
{

typedef HammockGraphAnalysis::Hammock  Hammock;
typedef HammockGraphAnalysis::iterator iterator;

typedef HammockGraphAnalysis::HammockList      HammockList;
typedef HammockGraphAnalysis::hammock_iterator hammock_iterator;

HammockGraphAnalysis::HammockGraphAnalysis()
: KernelAnalysis("HammockGraphAnalysis",
	{"DominatorTreeAnalysis", "PostDominatorTreeAnalysis"})
{

}

static void splitHammock(Hammock& hammock, DominatorTree* dominatorTree,
	PostdominatorTree* postDominatorTree);
	
void HammockGraphAnalysis::analyze(ir::IRKernel& kernel)
{
	report("Forming hammock graph for kernel " << kernel.name);
	
	_root.entry = kernel.cfg()->get_entry_block();
	_root.exit  = kernel.cfg()->get_exit_block();
	
	report("  creating new hammock (entry: " << _root.entry->label()
		<< ", exit: " << _root.exit->label() << ")");
	
	// Start with a single hammock containing the entire function
	for(auto block = kernel.cfg()->begin();
		block != kernel.cfg()->end(); ++block)
	{
		if(block == _root.entry) continue;
		if(block == _root.exit)  continue;
		
		report("   containing " << block->label());
	
		_root.children.push_back(Hammock(&_root, block, block));
	}
	
	auto dom = static_cast<DominatorTree*>(
		getAnalysis("DominatorTreeAnalysis"));
	auto pdom = static_cast<PostdominatorTree*>(
		getAnalysis("PostDominatorTreeAnalysis"));
	
	// Recursively split the hammock
	splitHammock(_root, dom, pdom);
}

const Hammock* HammockGraphAnalysis::getHammock(const_iterator block) const
{
	auto hammock = _blockToHammockMap.find(block);

	if(hammock != _blockToHammockMap.end()) return hammock->second;

	return 0;
}

const Hammock* HammockGraphAnalysis::getRoot() const
{
	return &_root;
}

Hammock::Hammock(Hammock* p, iterator e, iterator ex)
: parent(p), entry(e), exit(ex)
{

}

bool Hammock::isLeaf() const
{
	return entry == exit;
}

typedef std::unordered_map<iterator, hammock_iterator> BlockToHammockMap;

static bool expandHammock(iterator& entry, iterator& exit,
	iterator parentEntry, iterator parentExit,
	DominatorTree* dominatorTree, PostdominatorTree* postDominatorTree)
{
	auto dominator = dominatorTree->getDominator(entry);
	
	if(dominator == parentEntry || dominator == parentExit) return false;

	while(dominator->successors.size() < 2)
	{
		dominator = dominatorTree->getDominator(dominator);
		
		if(dominator == parentEntry || dominator == parentExit) return false;
	}

	auto postdominator = postDominatorTree->getPostDominator(dominator);
	
	if(!postDominatorTree->postDominates(postdominator, exit))
	{
		return false;
	}
	
	bool changed = entry != dominator || exit != postdominator;
	
	entry = dominator;
	exit  = postdominator;

	return changed;
}

static hammock_iterator createNewHammock(BlockToHammockMap& unvisited,
	hammock_iterator hammock,
	DominatorTree* dominatorTree, PostdominatorTree* postDominatorTree)
{
	// Find the maximal containing hammock
	auto entry = hammock->entry;
	auto exit  = hammock->entry;
	
	report(" finding hammock surrounding " << entry->label()
		<< " from parent (entry: " << hammock->parent->entry->label()
		<< ", exit: " << hammock->parent->exit->label() << ")");
	
	bool changed = true;
	
	while(changed)
	{
		changed = expandHammock(entry, exit, hammock->parent->entry,
			hammock->parent->exit, dominatorTree, postDominatorTree);
			
		if(unvisited.count(entry) == 0) return hammock;
	}
	
	if(entry == hammock->entry)
	{
		return hammock;
	}
	
	if(exit == hammock->entry)
	{
		return hammock;
	}
	
	// Create the new hammock
	report("  creating new hammock (entry: " << entry->label()
		<< ", exit: " << exit->label() << ")");
	
	// Remove the entry and exit from the parent
	auto entryHammock = unvisited.find(entry);
	assert(entryHammock != unvisited.end());
	
	auto newHammock = entryHammock->second;
	
	newHammock->entry = entry;
	newHammock->exit  = exit;

	unvisited.erase(entryHammock);
		
	// Move all contained children into the new hammock
	typedef std::list<hammock_iterator> HammockIteratorList;
	
	HammockIteratorList unvisitedIterators;
	
	for(auto unvisitedIterator = unvisited.begin();
		unvisitedIterator != unvisited.end(); ++unvisitedIterator)
	{
		unvisitedIterators.push_back(unvisitedIterator->second);
	}
	
	for(auto unvisitedIterator = unvisitedIterators.begin();
		unvisitedIterator != unvisitedIterators.end(); ++unvisitedIterator)
	{
		if((*unvisitedIterator)->entry == entry) continue;
		if((*unvisitedIterator)->entry == exit)  continue;

		if(!dominatorTree->dominates(entry, (*unvisitedIterator)->entry))
		{
			continue;
		}

		if(!postDominatorTree->postDominates(exit, (*unvisitedIterator)->entry))
		{
			continue;
		}

		report("   containing " << (*unvisitedIterator)->entry->label());
		
		(*unvisitedIterator)->parent = &*newHammock;
		
		newHammock->children.splice(newHammock->children.end(),
			newHammock->parent->children, *unvisitedIterator);
	
		auto oldHammock = unvisited.find((*unvisitedIterator)->entry);
		assert(oldHammock != unvisited.end());
		
		unvisited.erase(oldHammock);
	}
	
	return newHammock;
}

static void splitHammock(Hammock& hammock, DominatorTree* dominatorTree,
	PostdominatorTree* postDominatorTree)
{
	BlockToHammockMap unvisited;
	
	for(auto child = hammock.children.begin();
		child != hammock.children.end(); ++child)
	{
		unvisited.insert(std::make_pair(child->entry, child));
	}
	
	// Attempt to pull out hammocks until there is no change
	bool changed = true;
	
	while(changed)
	{
		changed = false;
	
		// check child blocks
		for(auto child = unvisited.begin(); child != unvisited.end(); ++child)
		{
			auto newHammock = createNewHammock(unvisited, child->second,
				dominatorTree, postDominatorTree);
		
			bool isNewHammockASubset = !newHammock->isLeaf();
			
			if(isNewHammockASubset)
			{
				splitHammock(*newHammock, dominatorTree, postDominatorTree);

				changed = true;
				break;
			}
		}
	}
}


}

