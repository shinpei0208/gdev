/*! \file PostdominatorTree.h
	
	\author Andrew Kerr <arkerr@gatech.edu>
	
	\date 21 Jan 2009
	
	\brief computes a dominator tree from a control flow graph; a
		flag in the constructor permits reversing the edges to compute
		a postdominator tree
*/

#ifndef IR_POSTDOMINATORTREE_H_INCLUDED
#define IR_POSTDOMINATORTREE_H_INCLUDED

// Ocelot Includes
#include <ocelot/ir/interface/ControlFlowGraph.h>
#include <ocelot/analysis/interface/Analysis.h>

// Standard Library Includes
#include <vector>

namespace analysis 
{
	
	/*!
		A tree structure in which each node corresponds to a BasicBlock in the
		control flow graph such that each node's block is immediately 
		post-dominated by its parent. Each node is owned by its parent.
	*/
	class PostdominatorTree : public KernelAnalysis {
	public:
		typedef std::vector<int> IndexVector;
		typedef std::vector<IndexVector> IndexArrayVector;
		typedef ir::ControlFlowGraph::iterator block_iterator;
		typedef ir::ControlFlowGraph::BlockPointerVector BlockPointerVector;
		
	public:
		PostdominatorTree();
		~PostdominatorTree();
	
	public:
		void analyze(ir::IRKernel& kernel);
		
	public:
		/*! Writes a representation of the DominatorTree to an output stream */
		std::ostream& write(std::ostream& out);

		/*! Does a particular block post-dominate another block? */
		bool postDominates(block_iterator block, 
			block_iterator potentialSuccessor);

		/*! Given a block known to be in the control flow graph, 
			return the post dominator */
		block_iterator getPostDominator(block_iterator block);

		/*! Get the nearest common post-dominator of two blocks */
		block_iterator getCommonPostDominator(block_iterator block2,
			block_iterator block1);
	
		/*! Get the post dominance frontier of a block */
		BlockPointerVector getPostDominanceFrontier(block_iterator block);
	
	public:
		/*! Parent control flow graph */
		ir::ControlFlowGraph* cfg;
	
		/*! store of the basic blocks in the dominator tree in post-order */
		BlockPointerVector blocks;
	
		/*! nth element stores the immediate post-dominator 
			of node n or -1 if undefined */
		IndexVector p_dom;
	
		/*!  nth element stores a list of elements for which 
			n is the immediate post-dominator */
		IndexArrayVector dominated;
	
		/*!  nth element stores a list of elements in the post dominance
			frontier of n */
		IndexArrayVector frontiers;
	
		/*! Map from a BasicBlock pointer to an index into the blocks vector */
		ir::ControlFlowGraph::BlockMap blocksToIndex;	

	private:
		void computeDT();
		int intersect(int b1, int b2) const;
	};
	
}

#endif

