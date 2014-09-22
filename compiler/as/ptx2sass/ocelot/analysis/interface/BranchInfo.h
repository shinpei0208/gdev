/*! \file BranchInfo.cpp
	\date Aug 23, 2010
	\author Diogo Sampaio <dnsampaio@gmail.com>
	\brief The header file for the BranchInfo class
*/

#ifndef BRANCHINFO_H_INCLUDED
#define BRANCHINFO_H_INCLUDED

// Ocelot Includes
#include <ocelot/analysis/interface/DataflowGraph.h>
#include <ocelot/analysis/interface/DivergenceGraph.h>

namespace analysis
{

/* BranchInfo holds branch information for the divergence analysis */
class BranchInfo
{
public:
	typedef DataflowGraph::Block Block;
	typedef DirectionalGraph::node_type node_type;
	typedef DirectionalGraph::node_set node_set;
	typedef std::set<const Block*> touched_blocks;

	BranchInfo(const Block *block, const Block *postDominator,
                const DataflowGraph::Instruction &dfgInstruction,
                DivergenceGraph &divergGraph );

  bool operator<(const BranchInfo& x) const;
  bool operator<=(const BranchInfo& x) const;
  bool operator>(const BranchInfo& x) const;
  bool operator>=(const BranchInfo& x) const;

	/*!\brief Tells if a node is control dependent
		on a divergent branch instruction. */
	bool isTainted(const node_type &node) const;

	/*!\brief Compute the influence of the branch in variables and blocks */
	void populate();

	/*!\brief Returns the pointer to the block that
		holds the branch instruction */
	const Block *block() const {return _block;};
	/*!\brief Returns the pointer to the target block of the branch */
	const Block *branch() const {return _branch;};
	/*!\brief Returns the pointer to the fallthtrough block */
	const Block *fallThrough() const {return _fallThrough;};
	/*!\brief Returns the pointer to the block that postdominates block */
	const Block *postDominator() const {return _postDominator;};
	/*!\brief Returns the the predicate ID, in the kernel->DFG, in SSA form */
	const DirectionalGraph::node_type &predicate() const
		{return *_dfgInstruction.s.begin()->pointer;};
	/*!\brief Returns the reference to the branch instruction */
	const DataflowGraph::Instruction &instruction() const {return _dfgInstruction;};

private:
	/* Pointer to the block that holds the branch instruction, 
		the target block of the branch, the fallthrough block,
		the block that postdominates block */
	const Block *_block, *_branch, *_fallThrough, *_postDominator;
	/*!\brief Reference to the branch instruction */
	const DataflowGraph::Instruction &_dfgInstruction;
	/*!\brief Reference to divergence graph associated to the kernel */
	DivergenceGraph &_divergGraph;

	/*!\brief Blocks dependency from the branch and fallthrough sides */
	touched_blocks _branchBlocks, _fallThroughBlocks;
	/*!\brief Variables dependency from the branch and fallthrough sides */
	node_set _branchVariables, _fallThroughVariables;
	
private:
	/*!\brief Insert a variable and all it's
		successors to the dependency list */
	void _insertVariable(node_type &variable, node_set &variables);
	/*!\brief Walks through the blocks instruction,
		inserting the destination variables to the dependency list */
	void _taintVariables(const Block *block, node_set &variables);
	/*!\brief Walks through the blocks creating a list of
		dependency until reach of the postdominator block */
	void _taintBlocks(const Block *start, touched_blocks &blocks,
		node_set &variables);

};

}

#endif /* BRANCHINFO_H_ */

