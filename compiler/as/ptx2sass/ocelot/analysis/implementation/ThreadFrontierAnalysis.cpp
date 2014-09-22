/*! \file   ThreadFrontierAnalysis.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Wednesday May 11, 2011
	\brief  The source file for the ThreadFrontierAnalysis class.
*/

#ifndef THREAD_FRONTIER_ANALYSIS_CPP_INCLUDED
#define THREAD_FRONTIER_ANALYSIS_CPP_INCLUDED

// Ocelot Incudes
#include <ocelot/analysis/interface/ThreadFrontierAnalysis.h>
#include <ocelot/analysis/interface/DominatorTree.h>

#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stack>
#include <unordered_set>
#include <set>
#include <algorithm>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0 

namespace analysis
{

ThreadFrontierAnalysis::ThreadFrontierAnalysis()
: KernelAnalysis("ThreadFrontierAnalysis",
	{"DominatorTreeAnalysis"})
{

}

void ThreadFrontierAnalysis::analyze(ir::IRKernel& kernel)
{
	_computePriorities(kernel);
	_computeFrontiers(kernel);
}

ThreadFrontierAnalysis::BlockVector ThreadFrontierAnalysis::getThreadFrontier(
	const_iterator block) const
{
	BlockMap::const_iterator frontier = _frontiers.find(block);
	assert(frontier != _frontiers.end());

	return frontier->second;
}

ThreadFrontierAnalysis::Priority ThreadFrontierAnalysis::getPriority(
	const_iterator block) const
{
	PriorityMap::const_iterator priority = _priorities.find(block);
	assert(priority != _priorities.end());

	return priority->second;
}

bool ThreadFrontierAnalysis::isInThreadFrontier(const_iterator block,
	const_iterator testedBlock) const
{
	BlockVector frontier = getThreadFrontier(block);
	
	return std::find(frontier.begin(), frontier.end(), testedBlock) !=
		frontier.end();
}

void ThreadFrontierAnalysis::_computePriorities(ir::IRKernel& kernel)
{
	typedef std::unordered_set<const_iterator> BlockSet;

	#if REPORT_BASE != 0
	kernel.cfg()->write(std::cout);	
	#endif
	
	report("Setting basic block priorities.");
	
	Node::NodeList edgeCoveringTree;
	NodeMap        nodeMap;

	node_iterator root = edgeCoveringTree.insert(edgeCoveringTree.end(),
		Node(kernel.cfg()->get_exit_block(),
		node_iterator(), node_iterator(), 0));

	root->parent = root;
	root->root   = root;

	// 1) Build the edge-covering tree
	_visitNode(nodeMap, root);
	
	// 2) Adjust priorites so that independent paths are separated
	report("Separating independent paths.");
	
	nodeMap.clear();
	Priority priority = 0;
	
	_separatePathPriorities(nodeMap, root, priority);
	
	// 3) Walk the tree, assign priorities
	root->assignPriorities(_priorities);
	
	// 4) Break ties
	_breakPriorityTies();
}

void ThreadFrontierAnalysis::_computeFrontiers(ir::IRKernel& kernel)
{
	typedef std::multimap<Priority, const_iterator,
		std::greater<Priority>> ReversePriorityMap;
	typedef std::unordered_set<const_iterator> BlockSet;

	ReversePriorityMap priorityToBlocks;

	report("Computing basic block thread frontiers.");
	
	// sort by priority (high to low)
	for(PriorityMap::iterator priority = _priorities.begin();
		priority != _priorities.end(); ++priority)
	{
		priorityToBlocks.insert(std::make_pair(priority->second,
			priority->first));
	}
	
	BlockSet outstandingWarps;
	
	// walk the list in priority order, track possibly divergent warps
	for(ReversePriorityMap::const_iterator
		priorityAndBlock = priorityToBlocks.begin();
		priorityAndBlock != priorityToBlocks.end(); ++priorityAndBlock)
	{
		const_iterator block = priorityAndBlock->second;

		// this block can no longer have a divergent warp
		outstandingWarps.erase(block);
		
		report(" " << block->label());
		
		BlockVector frontier;
		
		for(BlockSet::const_iterator b = outstandingWarps.begin();
			b != outstandingWarps.end(); ++b)
		{
			report("  " << (*b)->label());
			frontier.push_back(*b);	
		}
		
		_frontiers.insert(std::make_pair(block, frontier));
		
		// add block successors if they have not already been scheduled
		for(ir::BasicBlock::BlockPointerVector::const_iterator
			successor  = block->successors.begin();
			successor != block->successors.end(); ++successor)
		{
			if(getPriority(*successor) < priorityAndBlock->first)
			{
				outstandingWarps.insert(*successor);
			}
		}
	}
}

void ThreadFrontierAnalysis::_visitNode(NodeMap& nodes, node_iterator node)
{
	assert(nodes.count(node->block) == 0);
	
	report(" Visiting basic block '" << node->block->label() << "'");
	
	nodes.insert(std::make_pair(node->block, node));
	
	for(const_edge_iterator edge = node->block->in_edges.begin();
		edge != node->block->in_edges.end(); ++edge)
	{
		const_iterator predecessor = (*edge)->head;
		
		NodeMap::iterator predecessorNode = nodes.find(predecessor);
		
		if(predecessorNode == nodes.end())
		{
			node_iterator child = node->children.insert(node->children.end(),
				Node(predecessor, node, node->root, node->priority + 1));
			report("  adding child '" << child->block->label() << "'...");
			_visitNode(nodes, child);
			report(" Returned to basic block '" << node->block->label() << "'");
		}
		else
		{
			// swap if it results in a longer path through the subtree
			if(predecessorNode->second->priority < node->priority + 1)
			{
				// detect loops (can this be more efficient?)
				if(node->block != predecessor &&
					!node->isThisMyParent(predecessorNode->second))
				{
					report("  adding subtree rooted at '"
						<< predecessorNode->second->block->label() << "'");
			
					predecessorNode->second->updatePriority(node->priority + 1);
					node->children.splice(node->children.end(),
						predecessorNode->second->parent->children,
						predecessorNode->second);
					nodes[predecessor] = --node->children.end();
				}
			}
		}
	}
}

void ThreadFrontierAnalysis::_separatePathPriorities(NodeMap& nodes, node_iterator node,
	Priority& nextPriority)
{
	typedef std::list<node_iterator> NodePointerList;

	// If this block dominates an unvisited node, visit it now
	auto dom = static_cast<DominatorTree*>(
		getAnalysis("DominatorTreeAnalysis"));

	bool changed = true;

	while(changed)
	{
		changed = false;

		for(auto unvisited = nodes.begin(); unvisited != nodes.end(); ++unvisited)
		{
			if(!dom->dominates(node->block, unvisited->first)) continue;

			auto unvisitedNode = unvisited->second;

			nodes.erase(unvisited);

			changed = true;

			_separatePathPriorities(nodes, unvisitedNode, nextPriority);

			break;
		}
	}

	report(" Setting basic block '" << node->block->label() << "' priority "
		<< nextPriority);

	// Visit the node
	node->priority = nextPriority++;

	// sort children by edge type, fallthrough first	
	NodePointerList children;

	for(node_iterator child = node->children.begin();
		child != node->children.end(); ++child)
	{
		auto edge = child->block->get_edge(node->block);
		
		if(edge->isFallthrough())
		{
			children.push_front(child);
		}
		else
		{
			children.push_back(child);
		}
	}

	// register children as unvisited
	for(auto child = children.begin(); child != children.end(); ++child)
	{
		nodes.insert(std::make_pair((*child)->block, *child));
	}

	// update priorities of unvisited children
	for(auto child = children.begin(); child != children.end(); ++child)
	{
		auto visitedNode = nodes.find((*child)->block);

		if(visitedNode == nodes.end()) continue;
	
		nodes.erase(visitedNode);

		_separatePathPriorities(nodes, *child, nextPriority);
	}
}

void ThreadFrontierAnalysis::_breakPriorityTies()
{
	typedef std::multimap<Priority, const_iterator> PriorityMultiMap;
	
	PriorityMultiMap priorities;
	
	for(auto entry = _priorities.begin(); entry != _priorities.end(); ++entry)
	{
		priorities.insert(std::make_pair(entry->second, entry->first));
	}
	
	Priority priority = 0;
	
	for(auto entry = priorities.begin(); entry != priorities.end(); ++entry)
	{
		report(" Assigning basic block '" << entry->second->label()
			<< "' (" << entry->second->id << ") priority " << priority);	
		_priorities[entry->second] = priority++;
	}
}

ThreadFrontierAnalysis::Node::Node(const_iterator b, node_iterator p,
	node_iterator r, Priority pri)
: block(b), parent(p), root(r), priority(pri)
{

}

void ThreadFrontierAnalysis::Node::assignPriorities(PriorityMap& priorities)
{
	priorities[block] = priority;
	
	for(node_iterator child = children.begin();
		child != children.end(); ++child)
	{
		child->assignPriorities(priorities);
	}
}

void ThreadFrontierAnalysis::Node::updatePriority(Priority p)
{
	priority = p;

	for(node_iterator child = children.begin();
		child != children.end(); ++child)
	{
		child->updatePriority(p + 1);
	}
}

bool ThreadFrontierAnalysis::Node::isThisMyParent(node_iterator possibleParent)
{
	if(parent == possibleParent) return true;

	if(parent == root) return false;
	
	return parent->isThisMyParent(possibleParent);
}

}

#endif

