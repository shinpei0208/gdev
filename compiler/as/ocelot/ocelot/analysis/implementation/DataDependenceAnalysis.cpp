/*! \file   DataDependenceAnalysis.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday June 29, 2013
	\file   The source file for the DataDependenceAnalysis class.
*/

// Ocelot Includes
#include <ocelot/analysis/interface/DataDependenceAnalysis.h>

#include <ocelot/analysis/interface/DataflowGraph.h>

#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <unordered_set>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace analysis
{

DataDependenceAnalysis::DataDependenceAnalysis()
: KernelAnalysis("DataDependenceAnalysis", {"DataflowGraphAnalysis"})
{
	
}

typedef ir::PTXInstruction PTXInstruction;
typedef DataflowGraph::iterator   dataflow_iterator;
typedef DataflowGraph::RegisterId Register;
typedef DataDependenceAnalysis::iterator iterator;
typedef DataDependenceAnalysis::Node     Node;
typedef DataDependenceAnalysis::NodeList NodeList;
typedef DataDependenceAnalysis::InstructionToNodeMap InstructionToNodeMap;

static void analyzeBlock(dataflow_iterator block, DataflowGraph* dfg,
	NodeList& nodes, InstructionToNodeMap& instructionToNodes);

void DataDependenceAnalysis::analyze(ir::IRKernel& kernel)
{
	auto dfg = static_cast<DataflowGraph*>(
		getAnalysis("DataflowGraphAnalysis"));

	report("Running data dependence analysis on kernel " << kernel.name);
	
	for(auto block = dfg->begin(); block != dfg->end(); ++block)
	{
		analyzeBlock(block, dfg, _nodes, _instructionToNodes);
	}
}

typedef std::unordered_set<dataflow_iterator> BlockSet;

static void chaseDownPredecessors(iterator node, Register value,
	DataflowGraph* dfg,
	dataflow_iterator block, NodeList& nodes,
	InstructionToNodeMap& instructionToNodes, BlockSet& visited)
{
	if(!visited.insert(block).second) return;
	
	assert(block->aliveIn().count(value) != 0);

	for(auto predecessor : block->predecessors())
	{
		if(predecessor->aliveOut().count(value) == 0) continue;
		
		bool foundAnyDefinitions = false;
		
		// check the body for a definition
		for(auto instruction = predecessor->instructions().rbegin();
			instruction != predecessor->instructions().rend(); ++instruction)
		{
			for(auto destination : instruction->d)
			{
				if(*destination.pointer == value)
				{
					auto producer = nodes.end();
					
					auto ptx = static_cast<PTXInstruction*>(instruction->i);
					
					auto existingNode = instructionToNodes.find(ptx);
		
					if(existingNode == instructionToNodes.end())
					{
						producer = nodes.insert(nodes.end(), Node(ptx));
						
						instructionToNodes.insert(
							std::make_pair(ptx, producer));
					}
					else
					{
						producer = existingNode->second;
					}
					
					report(" " << producer->instruction->toString() << " -> "
						<< node->instruction->toString());
					
					node->predecessors.push_back(producer);
					producer->successors.push_back(node);
					
					foundAnyDefinitions = true;
					break;
				}
			}
		}
		
		if(foundAnyDefinitions) continue;
		
		// if no definitions were found, recurse through predecessors
		chaseDownPredecessors(node, value, dfg, predecessor, nodes,
			instructionToNodes, visited);
	}
}

static void analyzeBlock(dataflow_iterator block, DataflowGraph* dfg,
	NodeList& nodes, InstructionToNodeMap& instructionToNodes)
{
	typedef std::unordered_map<Register, iterator> RegisterToProducerMap;
	
	RegisterToProducerMap lastProducers;
	
	for(auto& instruction : block->instructions())
	{
		// Create a node
		auto node = nodes.end();
		
		auto ptx = static_cast<PTXInstruction*>(instruction.i);
		
		auto existingNode = instructionToNodes.find(ptx);
		
		if(existingNode == instructionToNodes.end())
		{
			node = nodes.insert(nodes.end(), Node(ptx));
		
			instructionToNodes.insert(std::make_pair(ptx, node));
		}
		else
		{
			node = existingNode->second;
		}
		
		// Add predecessors
		for(auto source : instruction.s)
		{
			auto producer = lastProducers.find(*source.pointer);
		
			if(producer == lastProducers.end())
			{
				BlockSet visited;
			
				chaseDownPredecessors(node, *source.pointer, dfg, block, nodes,
					instructionToNodes, visited);
				continue;
			}
			
			report(" " << producer->second->instruction->toString() << " -> "
				<< node->instruction->toString());
			
			node->predecessors.push_back(producer->second);
			producer->second->successors.push_back(node);
		}
		
		// Update last producers
		for(auto destination : instruction.d)
		{	
			lastProducers[*destination.pointer] = node;
		}
	}
}

}


