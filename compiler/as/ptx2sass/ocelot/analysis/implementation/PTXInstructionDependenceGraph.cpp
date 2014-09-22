/*! \file   PTXInstructionDependenceGraph.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday June 29, 2013
	\file   The source file for the PTXInstructionDependenceGraph class.
*/

// Ocelot Includes
#include <ocelot/analysis/interface/PTXInstructionDependenceGraph.h>

#include <ocelot/ir/interface/IRKernel.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>

// Standard Library Includes
#include <unordered_set> 

#include <cassert> 

namespace analysis
{

typedef ir::PTXInstruction PTXInstruction;
typedef std::unordered_set<PTXInstruction*> InstructionSet;
typedef PTXInstructionDependenceGraph::iterator iterator;
typedef PTXInstructionDependenceGraph::InstructionPair InstructionPair;
typedef PTXInstructionDependenceGraph::DependenceMap DependenceMap;

PTXInstructionDependenceGraph::Node::Node(PTXInstruction* i)
: instruction(i)
{

}

static bool dependsOn(iterator next, iterator target, InstructionSet& visited,
	DependenceMap& savedDependencies);

bool PTXInstructionDependenceGraph::dependsOn(const PTXInstruction* instruction,
	const PTXInstruction* dependee)
{
	if(instruction == dependee) return false;
	
	auto dependeeNode = _instructionToNodes.find(dependee);
	assert(dependeeNode != _instructionToNodes.end());

	auto instructionNode = _instructionToNodes.find(instruction);
	assert(instructionNode != _instructionToNodes.end());
	
	InstructionSet visited;
	
	return analysis::dependsOn(instructionNode->second,
		dependeeNode->second, visited, _savedDependencies);
}

PTXInstructionDependenceGraph::iterator PTXInstructionDependenceGraph::begin()
{
	return _nodes.begin();
}

PTXInstructionDependenceGraph::const_iterator
	PTXInstructionDependenceGraph::begin() const
{
	return _nodes.begin();
}

PTXInstructionDependenceGraph::iterator PTXInstructionDependenceGraph::end()
{
	return _nodes.end();
}

PTXInstructionDependenceGraph::const_iterator
	PTXInstructionDependenceGraph::end() const
{
	return _nodes.end();
}

PTXInstructionDependenceGraph::iterator
	PTXInstructionDependenceGraph::getNode(const PTXInstruction* instruction)
{
	auto node = _instructionToNodes.find(instruction);
	
	if(node == _instructionToNodes.end()) return end();
	
	return node->second;
}

PTXInstructionDependenceGraph::const_iterator
	PTXInstructionDependenceGraph::getNode(
	const PTXInstruction* instruction) const
{
	auto node = _instructionToNodes.find(instruction);
	
	if(node == _instructionToNodes.end()) return end();
	
	return node->second;
}

static bool dependsOn(iterator next, iterator dependee, InstructionSet& visited,
	DependenceMap& savedDependencies)
{
	auto dependence = InstructionPair(next->instruction, dependee->instruction);

	auto savedDependence = savedDependencies.find(dependence);

	if(savedDependence != savedDependencies.end())
	{
		return savedDependence->second;
	}

	if(!visited.insert(next->instruction).second) return false;
	
	if(next == dependee) return true;
	
	for(auto successor : next->successors)
	{
		if(dependsOn(successor, dependee, visited, savedDependencies))
		{
			return true;
		}
		
		savedDependencies.insert(std::make_pair(
			InstructionPair(next->instruction, successor->instruction), true));
	}

	savedDependencies.insert(std::make_pair(
		InstructionPair(next->instruction, dependee->instruction), false));

	return false;
}

}


