/*! \file BranchInfo.cpp
	\date Aug 23, 2010
	\author Diogo Sampaio <dnsampaio@gmail.com>
	\brief The source file for the BranchInfo class, the class that holds
		informations of branches required to do the divergence analysis
*/

// Ocelot Includes
#include <ocelot/analysis/interface/BranchInfo.h>
#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/analysis/interface/DataflowGraph.h>
#include <ocelot/analysis/interface/DivergenceGraph.h>

namespace analysis 
{

BranchInfo::BranchInfo(const Block *block, const Block *postDominator,
    const DataflowGraph::Instruction &dfgInstruction,
    DivergenceGraph &divergGraph) :
	_block(block), _postDominator(postDominator),
        _dfgInstruction(dfgInstruction), _divergGraph( divergGraph)
{
	_fallThrough = &(*_block->fallthrough());
	const DataflowGraph::BlockPointerSet targets = _block->targets();
	DataflowGraph::BlockPointerSet::const_iterator
		targetBlock = targets.begin();
	DataflowGraph::BlockPointerSet::const_iterator
		endTargetBlock = targets.end();
	for (; targetBlock != endTargetBlock; targetBlock++) {
		DataflowGraph::BlockVector::const_iterator targetBlockI = *targetBlock;
		if (block->fallthrough() != targetBlockI) {
			_branch = &(*targetBlockI);
		}
	}
}

bool BranchInfo::operator<(const BranchInfo& x) const
{
  return _block->id() < x.block()->id();
}

bool BranchInfo::operator<=(const BranchInfo& x) const
{
  return _block->id() <= x.block()->id();
}

bool BranchInfo::operator>(const BranchInfo& x) const
{
  return _block->id() > x.block()->id();
}

bool BranchInfo::operator>=(const BranchInfo& x) const
{
  return _block->id() >= x.block()->id();
}

bool BranchInfo::isTainted(const DivergenceGraph::node_type &node) const
{
	return ((_branchVariables.find(node) != _branchVariables.end()) ||
	    (_fallThroughVariables.find(node) != _fallThroughVariables.end()));
}

void BranchInfo::populate()
{
	if(_fallThrough != _postDominator){
		_taintBlocks(_fallThrough, _fallThroughBlocks, _fallThroughVariables);
	}
	if(_branch != _postDominator){
		_taintBlocks(_branch, _branchBlocks, _branchVariables);
	}
}
;

void BranchInfo::_insertVariable(node_type &variable, node_set &variables)
{
	if ((variables.find(variable) != variables.end())
		|| (_divergGraph.isDivNode(variable))) 
	{
		return;
	}

	node_set newVariables;
	newVariables.insert(variable);

	while (newVariables.size() > 0)
	{
		node_type newVar = *newVariables.begin();

		variables.insert(newVar);
		newVariables.erase(newVar);

		node_set dependences = _divergGraph.getOutNodesSet(newVar);
		node_set::const_iterator depend = dependences.begin();
		node_set::const_iterator endDepend = dependences.end();

		for (; depend != endDepend; depend++) {
			if ((variables.find(*depend) == variables.end())
				&& (!_divergGraph.isDivNode(*depend))) {
				newVariables.insert(*depend);
			}
		}
	}
}

void BranchInfo::_taintVariables(const Block *block, node_set &variables)
{
	const DataflowGraph::InstructionVector instructions = block->instructions();
	DataflowGraph::InstructionVector::const_iterator ins = instructions.begin();
	DataflowGraph::InstructionVector::const_iterator
		endIns = instructions.end();

	for (; ins != endIns; ins++) {
		DataflowGraph::RegisterPointerVector::const_iterator
			destination = ins->d.begin();
		DataflowGraph::RegisterPointerVector::const_iterator
			endDestination = ins->d.end();
		for (; destination != endDestination; destination++) {
			_insertVariable(*destination->pointer, variables);
		}
	}
}

void BranchInfo::_taintBlocks(const Block *start,
	touched_blocks &blocks, node_set &variables)
{
	touched_blocks toCompute;
	toCompute.insert(start);

	while (toCompute.size() > 0) {
		const Block *block = *toCompute.begin();
		if (blocks.find(block) == blocks.end()) {
			blocks.insert(block);

			DataflowGraph::BlockPointerSet::const_iterator
				newBlock = block->targets().begin();
			DataflowGraph::BlockPointerSet::const_iterator
				endNewBlock = block->targets().end();

			for (; newBlock != endNewBlock; newBlock++) {
				toCompute.insert(&(*(*newBlock)));
			}
		}
		toCompute.erase(block);
		_taintVariables(block, variables);
	}
}

}

