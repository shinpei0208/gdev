/*! \file PostdominatorTree.cpp
	
	\author Andrew Kerr <arkerr@gatech.edu>
	\date 21 Jan 2009
	
	\brief computes a dominator tree from a control flow graph; a
		flag in the constructor permits reversing the edges to compute
		a postdominator tree
*/

// Ocelot Includes
#include <ocelot/analysis/interface/PostdominatorTree.h>
#include <ocelot/ir/interface/Instruction.h>
#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/string.h>
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <unordered_set>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace analysis
{

PostdominatorTree::PostdominatorTree()
: KernelAnalysis("PostDominatorTreeAnalysis", {}) {
	
}

PostdominatorTree::~PostdominatorTree() {

}

void PostdominatorTree::analyze(ir::IRKernel& kernel) {
	// form a vector of the basic blocks in post-order
	report("Building post-dominator tree.");
	cfg = kernel.cfg();
	report(" Starting with post order sequence");
	// form a vector of the basic blocks in post-order
	ir::ControlFlowGraph::BlockPointerVector 
		post_order = cfg->reverse_topological_sequence();
	
	ir::ControlFlowGraph::reverse_pointer_iterator it = post_order.rbegin();
	ir::ControlFlowGraph::reverse_pointer_iterator end = post_order.rend();
	for (; it != end; ++it) {
		blocks.push_back(*it);
		blocksToIndex[*it] = (int)blocks.size()-1;
		p_dom.push_back(-1);
		report("  " << (*it)->label());
	}
	
	computeDT();
}

bool PostdominatorTree::postDominates(ir::ControlFlowGraph::iterator block, 
	ir::ControlFlowGraph::iterator potentialPredecessor) {
	int id = blocksToIndex[block];
	int predecessorId = blocksToIndex[potentialPredecessor];
	int endId = blocksToIndex[cfg->get_exit_block()];
	
	bool postDominates = false;
	
	int nextId = predecessorId;
	
	do
	{
		postDominates = nextId == id;
		nextId = p_dom[nextId];
	}
	while(endId != nextId && !postDominates);
	
	return postDominates || nextId == id;
}

ir::ControlFlowGraph::iterator PostdominatorTree::getPostDominator(
	ir::ControlFlowGraph::iterator block) {
	int n = blocksToIndex[block];
	return blocks[p_dom[n]];
}

ir::ControlFlowGraph::iterator PostdominatorTree::getCommonPostDominator(
	ir::ControlFlowGraph::iterator block1,
	ir::ControlFlowGraph::iterator block2) {

	int n1 = blocksToIndex[block1];
	int n2 = blocksToIndex[block2];
	
	int n = intersect(p_dom[n1], p_dom[n2]);
	
	return blocks[n];
}

/*! Computes the dominator tree from a CFG using algorithm __*/
void PostdominatorTree::computeDT() {
	int end_node = blocksToIndex[cfg->get_exit_block()];

	bool changed = true;
	p_dom[end_node] = end_node;

	report( " Computing tree" );

	while (changed) {
		changed = false;

		// post-order
		for (int b_ind = 0; b_ind < (int)blocks.size(); b_ind++) {
			if (b_ind == end_node)  continue;

			ir::ControlFlowGraph::iterator b = blocks[b_ind];
			assert(!b->successors.empty());
			int new_pdom = 0;
			bool processed = false;

			ir::ControlFlowGraph::pointer_iterator 
				succ_it = b->successors.begin();
			for (; succ_it != b->successors.end(); ++succ_it) {
				int p = blocksToIndex[*succ_it];
				assert(p<(int)p_dom.size());
				if (p_dom[p] != -1) {
					if( !processed ) {
						new_pdom = p;
						processed = true;
					}
					else {
						new_pdom = intersect(p, new_pdom);
					}
				}
			}
		
			if( processed ) {			
				if (p_dom[b_ind] != new_pdom) {
					p_dom[b_ind] = new_pdom;
					changed = true;
				}
			}
		}
	}
	
	dominated.resize(blocks.size());
	for (int n = 0; n < (int)blocks.size(); n++) {
		if (p_dom[n] >= 0) {
			dominated[p_dom[n]].push_back(n);
		}
	}
	
	report(" Computing frontiers")
	
	frontiers.resize(blocks.size());
	for (int b_ind = 0; b_ind < (int)blocks.size(); b_ind++) {
	
		ir::ControlFlowGraph::iterator block = blocks[b_ind];
			
		if(block->successors.size() < 2) continue;

		typedef std::unordered_set<int> BasicBlockSet;
		
		BasicBlockSet blocksWithThisBlockInTheirFrontier;
		
		for (auto successor : block->successors) {
			auto runner = successor;
			
			while (runner != getPostDominator(block)) {
				blocksWithThisBlockInTheirFrontier.insert(
					blocksToIndex[runner]);
				
				runner = getPostDominator(runner);
			}
		}
		
		for (auto frontierBlock : blocksWithThisBlockInTheirFrontier) {
			frontiers[b_ind].push_back(frontierBlock);
		}
	}
}

int PostdominatorTree::intersect(int b1, int b2) const {
	int finger1 = b1;
	int finger2 = b2;
	while (finger1 != finger2) {
		report( "finger1 " << finger1 << " finger2 " << finger2 );
		while (finger1 < finger2) {
			finger1 = p_dom[finger1];
		}
		while (finger2 < finger1) {
			finger2 = p_dom[finger2];
		}
	}
	return finger1;
}

PostdominatorTree::BlockPointerVector
	PostdominatorTree::getPostDominanceFrontier(block_iterator block)
{
	BlockPointerVector frontierBlocks;
	
	auto& frontier = frontiers[blocksToIndex[block]];
	
	for(auto blockId : frontier)
	{
		frontierBlocks.push_back(blocks[blockId]);
	}
	
	return frontierBlocks;
}

std::ostream& PostdominatorTree::write(std::ostream& out) {

	out << "digraph {\n";
	out << "  // basic blocks in post-order\n";

	for (int n = 0; n < (int)blocks.size(); n++) {
		out << "  bb_" << n << " [shape=record,label=\"{" 
			<< blocks[n]->label() << " | ";
		ir::ControlFlowGraph::InstructionList::iterator 
			instr_it = blocks[n]->instructions.begin();
		for (int j = 0; instr_it != blocks[n]->instructions.end(); 
			++instr_it, ++j) {
			out << (j > 0 ? " | " : "") 
			<< hydrazine::toGraphVizParsableLabel((*instr_it)->toString());
		}
		out << "}\"];\n";
	}

	out << "\n  // tree structure\n";
	for (int n = 0; n < (int)blocks.size(); n++) {
		if (p_dom[n] >= 0 && n != p_dom[n]) {
			out << "  bb_" << n << " -> bb_" << p_dom[n] << ";\n";
		}
	}

	out << "}\n";

	return out;
}

}

