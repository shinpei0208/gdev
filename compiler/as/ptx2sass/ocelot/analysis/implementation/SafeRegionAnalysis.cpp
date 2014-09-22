/*! \file   SafeRegionAnalysis.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday May 4, 2012
	\brief  The source file for the SafeRegionAnalysis class.
*/

// Ocelot Incudes
#include <ocelot/analysis/interface/SafeRegionAnalysis.h>

#include <ocelot/ir/interface/IRKernel.h>

#include <ocelot/analysis/interface/CycleAnalysis.h>
#include <ocelot/analysis/interface/DependenceAnalysis.h>
#include <ocelot/analysis/interface/ControlDependenceAnalysis.h>
#include <ocelot/analysis/interface/HammockGraphAnalysis.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <unordered_set>
#include <cassert>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace analysis
{

SafeRegionAnalysis::SafeRegionAnalysis()
: KernelAnalysis("SafeRegionAnalysis",
	{"HammockGraphAnalysis", "DependenceAnalysis", "CycleAnalysis",
		"ControlDependenceAnalysis"})
{

}


typedef SafeRegionAnalysis::SafeRegion SafeRegion;
typedef SafeRegionAnalysis::SafeRegionMap SafeRegionMap;
typedef ir::ControlFlowGraph::iterator block_iterator;
typedef std::unordered_set<block_iterator> BlockSet;

static BlockSet getBlocksThatDependOnSideEffects(ir::IRKernel&, CycleAnalysis*,
	DependenceAnalysis*, ControlDependenceAnalysis*);
static void formSafeRegionsAroundHammocks(SafeRegion&,
	SafeRegionMap&, HammockGraphAnalysis*, const BlockSet&);

void SafeRegionAnalysis::analyze(ir::IRKernel& kernel)
{
	// Get analyses
	auto cycleAnalysis = static_cast<CycleAnalysis*>(
		getAnalysis("CycleAnalysis"));
	auto dependenceAnalysis = static_cast<DependenceAnalysis*>(
		getAnalysis("DependenceAnalysis"));
	auto controlDependenceAnalysis =
		static_cast<ControlDependenceAnalysis*>(
		getAnalysis("ControlDependenceAnalysis"));
		
	// Find basic blocks that cannot be contained in safe regions
	auto blocksThatDependOnSideEffects = getBlocksThatDependOnSideEffects(
		kernel, cycleAnalysis, dependenceAnalysis, controlDependenceAnalysis);

	// Find hammocks in the program
	auto hammockAnalysis = static_cast<HammockGraphAnalysis*>(
		getAnalysis("HammockGraphAnalysis"));
		
	// Form safe regions around hammocks that do not contain blocks with
	// side effects
	formSafeRegionsAroundHammocks(_root, _regions, hammockAnalysis,
		blocksThatDependOnSideEffects);
}

const SafeRegionAnalysis::SafeRegion* SafeRegionAnalysis::getRegion(
	const_iterator block) const
{
	auto region = _regions.find(block);
	assert(region != _regions.end());
	
	return region->second;
}

const SafeRegionAnalysis::SafeRegion* SafeRegionAnalysis::getRoot() const
{
	return &_root;
}

SafeRegion::SafeRegion(SafeRegion* p)
: parent(p), doesNotDependOnSideEffects(false)
{

}

bool SafeRegion::isLeaf() const
{
	return children.size() == 0;
}

static BlockSet getBlocksWithBranchSideEffects(ir::IRKernel& k,
	CycleAnalysis* c, DependenceAnalysis*,
	ControlDependenceAnalysis*);

static BlockSet getBlocksWithCallsToFuctionsThatObserveSideEffects(
	ir::IRKernel& k);

static BlockSet getBlocksThatDependOnSideEffects(ir::IRKernel& k,
	CycleAnalysis* cycleAnalysis,
	DependenceAnalysis* dependenceAnalysis,
	ControlDependenceAnalysis* controlDependenceAnalysis)
{
	auto blocksWithSideEffects = getBlocksWithBranchSideEffects(
		k, cycleAnalysis, dependenceAnalysis, controlDependenceAnalysis);

	auto blocksWithFunctionSideEffects =
		getBlocksWithCallsToFuctionsThatObserveSideEffects(k);

	blocksWithSideEffects.insert(blocksWithFunctionSideEffects.begin(),
		blocksWithFunctionSideEffects.end());

	return blocksWithSideEffects;
}

typedef ir::PTXInstruction PTXInstruction;
typedef std::unordered_set<PTXInstruction*> InstructionSet;

static BlockSet getBlocksWithBackwardsBranches(CycleAnalysis* c);
static InstructionSet getInstructionsThatCanObserveSideEffects(ir::IRKernel& k);
static BlockSet getBlocksWithBranchesThatDependOn(
	const BlockSet& blocksWithBackwardsBranches,
	const InstructionSet& instructionsThatCanObserveSideEffects,
	DependenceAnalysis* dependenceAnalysis,
	ControlDependenceAnalysis* controlDependenceAnalysis);

static BlockSet getBlocksWithBranchSideEffects(ir::IRKernel& kernel,
	CycleAnalysis* cycleAnalysis,
	DependenceAnalysis* dependenceAnalysis,
	ControlDependenceAnalysis* controlDependenceAnalysis)
{
	auto backwardsBranches = getBlocksWithBackwardsBranches(cycleAnalysis);
	
	auto instructionsThatCanObserveSideEffects =
		getInstructionsThatCanObserveSideEffects(kernel);
	
	return getBlocksWithBranchesThatDependOn(
		backwardsBranches, instructionsThatCanObserveSideEffects,
		dependenceAnalysis, controlDependenceAnalysis);
}

static BlockSet getBlocksWithCallsToFuctionsThatObserveSideEffects(
	ir::IRKernel& k)
{
	BlockSet blocks;

	report(" Getting functions that can observe side-effects");
	
	for(auto block = k.cfg()->begin(); block != k.cfg()->end(); ++block)
	{
		for(auto instruction : block->instructions)
		{
			auto ptxInstruction = static_cast<ir::PTXInstruction*>(instruction);
		
			// TODO: Check that the target can observe side effects
			if(ptxInstruction->isCall())
			{
				report("  " << ptxInstruction->toString());

				blocks.insert(block);
				break;
			}
		}
	}
	
	return blocks;
}

static PTXInstruction* getBranch(block_iterator block)
{
	if(block->instructions.empty()) return nullptr;	
	
	auto ptxBranch = static_cast<ir::PTXInstruction*>(
		block->instructions.back());
	
	if(!ptxBranch->isBranch()) return nullptr;
	
	return ptxBranch;
}

static BlockSet getBlocksWithBackwardsBranches(CycleAnalysis* cycleAnalysis)
{
	auto edges = cycleAnalysis->getAllBackEdges();
	
	report(" Getting blocks with backwards branches");
	
	BlockSet backwardsBranchBlocks;
	
	for(auto& edge : edges)
	{
		if(edge->type != ir::Edge::Branch) continue;
		
		auto block = edge->head;
		
		if(getBranch(block) == nullptr) continue;

		backwardsBranchBlocks.insert(block);
		
		report("  " << block->label());
	}
	
	return backwardsBranchBlocks;
}

static InstructionSet getInstructionsThatCanObserveSideEffects(ir::IRKernel& k)
{
	InstructionSet instructions;
	
	report(" Getting instructions that can observe side-effects");
	
	for(auto& block : *k.cfg())
	{
		for(auto instruction : block.instructions)
		{
			auto ptxInstruction = static_cast<ir::PTXInstruction*>(instruction);
		
			if(ptxInstruction->canObserveSideEffects())
			{
				report("  " << ptxInstruction->toString());
				
				instructions.insert(ptxInstruction);
			}
		}
	}
	
	return instructions;
}

static InstructionSet getControlDependentInstructions(
	const PTXInstruction* branch, const InstructionSet& instructions,
	analysis::ControlDependenceAnalysis* controlDependenceAnalysis)
{
	InstructionSet controlDependentInstructions;
	
	for(auto instruction : instructions)
	{
		if(controlDependenceAnalysis->dependsOn(branch, instruction))
		{
			controlDependentInstructions.insert(instruction);
		}
	}
	
	return controlDependentInstructions;
}

static BlockSet getBlocksWithBranchesThatDependOn(
	const BlockSet& blocksWithBackwardsBranches,
	const InstructionSet& instructionsThatCanObserveSideEffects,
	DependenceAnalysis* dependenceAnalysis,
	ControlDependenceAnalysis* controlDependenceAnalysis)
{
	BlockSet blocksWithDependentBranches;
	
	report(" Getting blocks with branches that can observe side-effects");
	
	for(auto blockWithBranch : blocksWithBackwardsBranches)
	{
		auto branch = getBranch(blockWithBranch);
		
		if(branch == nullptr) continue;
	
		auto controlDependentInstructions = getControlDependentInstructions(
			branch, instructionsThatCanObserveSideEffects,
			controlDependenceAnalysis);
	
		for(auto instruction : controlDependentInstructions)
		{
			if(dependenceAnalysis->dependsOn(instruction, branch))
			{
				report("  " << blockWithBranch->label());
				
				blocksWithDependentBranches.insert(blockWithBranch);
				break;
			}
		}
	}
	
	return blocksWithDependentBranches;
}

typedef HammockGraphAnalysis::Hammock Hammock;

static void constructSafeRegionsFromHammocks(SafeRegion& root,
	SafeRegionMap& regionMap, const Hammock* hammock);
static bool flagHammocksWithSideEffects(SafeRegion& region,
	const BlockSet& blocksWithSideEffects);

static void formSafeRegionsAroundHammocks(SafeRegion& root,
	SafeRegionMap& regionMap, HammockGraphAnalysis* hammockAnalysis,
	const BlockSet& blocksWithSideEffects)
{
	constructSafeRegionsFromHammocks(root, regionMap,
		hammockAnalysis->getRoot());
	
	flagHammocksWithSideEffects(root, blocksWithSideEffects);
}

static void addChild(SafeRegion& parent, block_iterator block,
	SafeRegionMap& regionMap);

static void constructSafeRegionsFromHammocks(SafeRegion& parent,
	SafeRegionMap& regionMap, const Hammock* hammock)
{
	if(hammock->isLeaf())
	{
		parent.block = hammock->entry;
	}
	else
	{
		addChild(parent, hammock->entry, regionMap);

		for(auto& hammockChild : hammock->children)
		{
			auto child = parent.children.insert(parent.children.end(),
				SafeRegion(&parent));
			
			constructSafeRegionsFromHammocks(*child, regionMap, &hammockChild);
		}
		
		//addChild(parent, hammock->exit, regionMap);
	}
}

static void addChild(SafeRegion& parent, block_iterator block,
	SafeRegionMap& regionMap)
{
	auto child = parent.children.insert(parent.children.end(),
		SafeRegion(&parent));

	child->block = block;
	
	regionMap.insert(std::make_pair(block, &*child));
}

static bool flagHammocksWithSideEffects(SafeRegion& region,
	const BlockSet& blocksWithSideEffects)
{
	if(region.isLeaf())
	{
		region.doesNotDependOnSideEffects =
			blocksWithSideEffects.count(region.block) == 0;
	}
	else
	{
		region.doesNotDependOnSideEffects = true;
	
		for(auto& child : region.children)
		{
			region.doesNotDependOnSideEffects &=
				flagHammocksWithSideEffects(child, blocksWithSideEffects);
		}
	}
	
	return region.doesNotDependOnSideEffects;
}

}


