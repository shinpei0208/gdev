/*! \file   MemoryDependenceAnalysis.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday June 29, 2013
	\file   The header file for the MemoryDependenceAnalysis class.
*/

// Ocelot Includes
#include <ocelot/analysis/interface/MemoryDependenceAnalysis.h>

#include <ocelot/analysis/interface/SimpleAliasAnalysis.h>

#include <ocelot/ir/interface/IRKernel.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <cassert>
#include <unordered_set>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace analysis
{

MemoryDependenceAnalysis::MemoryDependenceAnalysis()
: KernelAnalysis("MemoryDependenceAnalysis", {"SimpleAliasAnalysis"})
{

}

typedef ir::PTXInstruction PTXInstruction;
typedef std::unordered_set<PTXInstruction*> InstructionSet;

static InstructionSet getAllMemoryInstructions(ir::IRKernel&);
static InstructionSet getAllLoads(const InstructionSet&);
static InstructionSet getAllStores(const InstructionSet&);

void MemoryDependenceAnalysis::analyze(ir::IRKernel& kernel)
{
	auto memoryInstructions = getAllMemoryInstructions(kernel);
	
	auto loads  = getAllLoads(memoryInstructions);
	auto stores = getAllStores(memoryInstructions);
	
	auto aliasAnalysis = static_cast<SimpleAliasAnalysis*>(
		getAnalysis("SimpleAliasAnalysis"));

	for(auto store : stores)
	{
		auto node = _nodes.insert(_nodes.end(), Node(store));

		_instructionToNodes.insert(std::make_pair(store, node));
	}
	
	for(auto load : loads)
	{
		if(getNode(load) != end()) continue;
		
		auto node = _nodes.insert(_nodes.end(), Node(load));

		_instructionToNodes.insert(std::make_pair(load, node));
	}

	for(auto store : stores)
	{
		// TODO: filter out loads that always precede the store

		auto storeNode = getNode(store);

		for(auto load : loads)
		{
			if(load == store) continue;
			
			if(aliasAnalysis->canAlias(store, load))
			{
				auto loadNode = getNode(load);
				
				loadNode->predecessors.push_back(storeNode);
				storeNode->successors.push_back(loadNode);
			}
		}
	}
}

static InstructionSet getAllMemoryInstructions(ir::IRKernel& kernel)
{
	InstructionSet instructions;
	
	for(auto& block : *kernel.cfg())
	{
		for(auto instruction : block.instructions)
		{
			auto ptx = static_cast<PTXInstruction*>(instruction);
			
			if(!ptx->isMemoryInstruction()) continue;
			
			instructions.insert(ptx);
		}
	}
	
	return instructions;
}

static InstructionSet getAllLoads(const InstructionSet& memoryInstructions)
{
	InstructionSet loads;
	
	for(auto instruction : memoryInstructions)
	{
		if(!instruction->isLoad()) continue;
		
		loads.insert(instruction);
	}
	
	return loads;
}

static InstructionSet getAllStores(const InstructionSet& memoryInstructions)
{
	InstructionSet stores;
	
	for(auto instruction : memoryInstructions)
	{
		if(!instruction->isStore()) continue;
		
		stores.insert(instruction);
	}
	
	return stores;
}

}


