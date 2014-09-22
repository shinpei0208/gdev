/*! \file   ThreadFrontierAnalysis.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday May 9, 2011
	\brief  The header file for the ThreadFrontierAnalysis class.

	Note: This analysis was rewritten to use an edge-covering tree based
		algorithm for assigning thread priorities, it should be safe
		for commercial use now.
*/

#ifndef THREAD_FRONTIER_ANALYSIS_H_INCLUDED
#define THREAD_FRONTIER_ANALYSIS_H_INCLUDED

// Ocelot Incudes
#include <ocelot/analysis/interface/Analysis.h>

#include <ocelot/ir/interface/ControlFlowGraph.h>

namespace analysis
{

/*! \brief A class for determining thread frontiers for all basic blocks.

	Thread frontiers defines a scheduling order of threads and identifies
	re-convergence points.  The approach has two main steps:
		1) Assign a priority order to basic blocks.
		2) For each basic block, determine all other possible blocks where
			stalled threads may be waiting.

	The intersection of this set and a branch's target needs to be
	checked for re-convergence. See more info here [1].
	
	[1] - "SIMD Reconvergence at Thread Frontiers" by Diamos et al. 
*/
class ThreadFrontierAnalysis: public KernelAnalysis
{
public:
	typedef ir::ControlFlowGraph CFG;

	typedef CFG::ConstBlockPointerVector                  BlockVector;
	typedef CFG::const_iterator                           const_iterator;
	typedef CFG::const_edge_pointer_iterator              const_edge_iterator;
	typedef unsigned int                                  Priority;
	typedef std::unordered_map<const_iterator, Priority>  PriorityMap;

public:
	/*! \brief Create the analysis */
	ThreadFrontierAnalysis();

	/*! \brief Computes an up to date set of thread frontiers */
	void analyze(ir::IRKernel& kernel);

public:
	/*! \brief Get the blocks in the thread frontier of a specified block */
	BlockVector getThreadFrontier(const_iterator block) const;
	/*! \brief Get the scheduling priorty of a specified block */
	Priority getPriority(const_iterator block) const;
	/*! \brief Test if a block is in the thread frontier of another block */
	bool isInThreadFrontier(const_iterator block, const_iterator b) const;

private:
	typedef std::unordered_map<const_iterator, BlockVector> BlockMap;

	/*! \brief A node in the edge-covering tree */
	class Node
	{
	public:
		typedef std::list<Node>    NodeList;
		typedef NodeList::iterator node_iterator;
	
	public:
		Node(const_iterator block, node_iterator parent,
			node_iterator root, Priority priority);
	
	public:
		const_iterator block;
		NodeList       children;
		node_iterator  parent;
		node_iterator  root;
		
		Priority       priority;
		
	public:
		void updatePriority(Priority p);
		bool isThisMyParent(node_iterator possibleParent);
		
	public:
		void assignPriorities(PriorityMap&);
	};
	
	typedef Node::node_iterator node_iterator;
	typedef std::unordered_map<const_iterator, node_iterator> NodeMap;

private:
	void _computePriorities(ir::IRKernel& kernel);
	void _computeFrontiers(ir::IRKernel& kernel);

	void _visitNode(NodeMap& nodes, node_iterator node);
	void _separatePathPriorities(NodeMap& nodes, node_iterator node, Priority&);
	void _breakPriorityTies();

private:
	PriorityMap _priorities;
	BlockMap    _frontiers;
	
};

}

#endif

