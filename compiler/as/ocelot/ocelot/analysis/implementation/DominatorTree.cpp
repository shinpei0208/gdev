/*! \file DominatorTree.cpp
	\author Andrew Kerr <arkerr@gatech.edu>
	\date 21 Jan 2009
	\brief computes a dominator tree from a control flow graph; a
		flag in the constructor permits reversing the edges to compute
		a postdominator tree
*/

// Ocelot Includes
#include <ocelot/analysis/interface/DominatorTree.h>
#include <ocelot/ir/interface/IRKernel.h>
#include <ocelot/ir/interface/Instruction.h>

// Hydrazine Includes
#include <hydrazine/interface/string.h>
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace analysis
{

DominatorTree::DominatorTree()
: KernelAnalysis("DominatorTreeAnalysis", {}) {
}

DominatorTree::~DominatorTree() {
}

void DominatorTree::analyze(ir::IRKernel& kernel) {
	
	report("Building dominator tree.");
	cfg = kernel.cfg();
	report(" Starting with reverse post order sequence");
	// form a vector of the basic blocks in reverse post-order
	ir::ControlFlowGraph::BlockPointerVector 
		post_order = cfg->topological_sequence();
	
	ir::ControlFlowGraph::reverse_pointer_iterator it = post_order.rbegin();
	ir::ControlFlowGraph::reverse_pointer_iterator end = post_order.rend();
	for (; it != end; ++it) {
		blocks.push_back(*it);
		blocksToIndex[*it] = (int)blocks.size()-1;
		i_dom.push_back(-1);
		report("  " << (*it)->label());
	}
	
	computeDT();
}

std::ostream& DominatorTree::write(std::ostream& out) {

	out << "digraph {\n";
	out << "  // basic blocks in post-order\n";

	for (int n = 0; n < (int)blocks.size(); n++) {
		out << "  bb_" << n << " [shape=record,label=\"{" << 
			hydrazine::toGraphVizParsableLabel(blocks[n]->label()) << " | ";
		ir::ControlFlowGraph::InstructionList::iterator 
			instr_it = blocks[n]->instructions.begin();
		for (int j = 0; instr_it != blocks[n]->instructions.end(); 
			++instr_it, ++j) {
			std::string str = (*instr_it)->toString();
			out << (j > 0 ? " | " : "") 
				<< hydrazine::toGraphVizParsableLabel(str);
		}
		out << "}\"];\n";
	}

	out << "\n  // tree structure\n";
	for (int n = 0; n < (int)blocks.size(); n++) {
		if (i_dom[n] >= 0 && i_dom[n] != n) {
			out << "  bb_" << i_dom[n] << " -> bb_" << n << ";\n";
		}
	}

	out << "}\n";

	return out;
}

bool DominatorTree::dominates(ir::ControlFlowGraph::const_iterator block, 
	ir::ControlFlowGraph::const_iterator potentialSuccessor) {
	int id = blocksToIndex[block];
	int successorId = blocksToIndex[potentialSuccessor];
	int startId = blocksToIndex[cfg->get_entry_block()];
	
	bool dominates = false;
	
	int nextId = successorId;
	
	do
	{
		dominates = nextId == id;
		nextId = i_dom[nextId];
	}
	while(startId != nextId && !dominates);
	
	return dominates || nextId == id;
}

ir::ControlFlowGraph::iterator DominatorTree::getDominator(
	ir::ControlFlowGraph::const_iterator block) {
	int n = blocksToIndex[block];
	return blocks[i_dom[n]];
}

ir::ControlFlowGraph::iterator DominatorTree::getCommonDominator(
	ir::ControlFlowGraph::const_iterator block1,
	ir::ControlFlowGraph::const_iterator block2) {

	int n1 = blocksToIndex[block1];
	int n2 = blocksToIndex[block2];
	
	int n = intersect(i_dom[n1], i_dom[n2]);
	
	return blocks[n];
}


ir::ControlFlowGraph::BlockPointerVector DominatorTree::getDominatedBlocks(
	ir::ControlFlowGraph::const_iterator block) {
	ir::ControlFlowGraph::BlockPointerVector dominatedBlocks;
	
	int n = blocksToIndex[block];
	
	for(auto dominatedBlock = dominated[n].begin();
		dominatedBlock != dominated[n].end(); ++dominatedBlock) {
		
		dominatedBlocks.push_back(blocks[*dominatedBlock]);
	}
	
	return dominatedBlocks;
}

/*! Computes the dominator tree from a CFG using algorithm desrcibed in
	"A simple and fast dominance algorithm" by 
		Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy
 */
void DominatorTree::computeDT() {
	int start_node = blocksToIndex[cfg->get_entry_block()];

	bool changed = true;
	i_dom[start_node] = start_node;

	while (changed) {
		changed = false;

		// reverse post-order
		for (int b_ind = 0; b_ind < (int)blocks.size(); b_ind++) {
			if (b_ind == start_node) continue;
			
			ir::ControlFlowGraph::iterator b = blocks[b_ind];
			int new_idom = 0;
			bool processed = false;

			ir::ControlFlowGraph::pointer_iterator 
				pred_it = b->predecessors.begin();
			for (; pred_it != b->predecessors.end(); ++pred_it) {
				int p = blocksToIndex[*pred_it];
				if (i_dom[p] != -1) {
					if( !processed ) {
						new_idom = p;
						processed = true;
					}
					else {
						new_idom = intersect(p, new_idom);
					}
				}
			}
			
			if( processed ) {			
				if (i_dom[b_ind] != new_idom) {
					i_dom[b_ind] = new_idom;
					changed = true;
				}
			}
		}
	}

	dominated.resize(blocks.size());
	for (int n = 0; n < (int)blocks.size(); n++) {
		if (i_dom[n] >= 0) {
			dominated[i_dom[n]].push_back(n);
		}
	}
}

int DominatorTree::intersect(int b1, int b2) const {
	int finger1 = b1;
	int finger2 = b2;
	while (finger1 != finger2) {
		while (finger1 < finger2) {
			finger1 = i_dom[finger1];
		}
		while (finger2 < finger1) {
			finger2 = i_dom[finger2];
		}
	}
	return finger1;
}

}

//////////////////////////////////////////////////////////////////////////////////////////////////
