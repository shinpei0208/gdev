/*! \file SubkernelFormationPass.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday July 27, 2010
	\brief The source file for the SubkernelFormationPass class.

	Notes:
	The motivation here is fourfold:
		 1) Create natural boundaries for context switching between threads.
		 2) Enforce that all threads in a CTA execute loosely in lock-step to 
		 	improve data/instruction locality.
		 3) Coarsen regions between barriers to amortize their cost.
		 4) Create well-defined regions for performing local optimizations.  
		 	Reduce the complexity of subsequent operations with complexities 
		 	that depend on the kernel size.
*/

#ifndef SUBKERNEL_FORMATION_PASS_CPP_INCLUDED
#define SUBKERNEL_FORMATION_PASS_CPP_INCLUDED

// Standard Library Includes
#include <queue>

// Ocelot Includes
#include <ocelot/transforms/interface/SubkernelFormationPass.h>
#include <ocelot/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace transforms
{

SubkernelFormationPass::SubkernelFormationPass(unsigned int e)
	: ModulePass({"DataflowGraphAnalysis"}, "SubkernelFormationPass"), 
	_expectedRegionSize(e)
{
	report("Constructing SubkernelFormationPass() - expected region size: "
		<< e);
}

void SubkernelFormationPass::runOnModule(ir::Module& m)
{
	report("Running SubkernelFormationPass - expected region size: "
		<< _expectedRegionSize);
	// This pass requires all kernels to be loaded
	assert(m.loaded());

	ExtractKernelsPass pass(_expectedRegionSize);
	
	pass.initialize(m);

	for(ir::Module::KernelMap::const_iterator kernel = m.kernels().begin(); 
		kernel != m.kernels().end(); ++kernel)
	{
		pass.runOnKernel(*kernel->second);
	}

	pass.finalize();
	
	report(" Adding subkernels to the module.");
	for(ExtractKernelsPass::KernelVectorMap::const_iterator 
		kernel = pass.kernels.begin(); 
		kernel != pass.kernels.end(); ++kernel)
	{
		report("  Replacing kernel " << kernel->first->name);
		m.removeKernel(kernel->first->name);
		for(KernelVector::const_iterator 
			subkernel = kernel->second.begin();
			subkernel != kernel->second.end(); ++subkernel)
		{
			report("   with " << (*subkernel)->name);
			m.insertKernel(*subkernel);
		}
	}
}


void SubkernelFormationPass::setExpectedRegionSize(unsigned int s)
{
	_expectedRegionSize = s;
}

SubkernelFormationPass::ExtractKernelsPass::ExtractKernelsPass(
	unsigned int regionSize) : 
	KernelPass({"DataflowGraphAnalysis"}, "ExtractKernelsPass"), 
	_expectedRegionSize(regionSize)
{

}

void SubkernelFormationPass::ExtractKernelsPass::initialize(const ir::Module& m)
{
	
}

analysis::DataflowGraph& SubkernelFormationPass::ExtractKernelsPass::_dfg()
{
	Analysis* dfg_structure = getAnalysis("DataflowGraphAnalysis");
	assert(dfg_structure != 0);

	return *static_cast<analysis::DataflowGraph*>(dfg_structure);
}

typedef std::unordered_set<ir::ControlFlowGraph::iterator> BlockSet;

static unsigned int createRestorePoint(ir::PTXKernel& newKernel,
	ir::PTXKernel& ptx, ir::ControlFlowGraph::iterator newBlock, 
	ir::ControlFlowGraph::iterator oldBlock, 
	ir::ControlFlowGraph::edge_iterator newEdge,
	const analysis::DataflowGraph::IteratorMap& cfgToDfgMap,
	BlockSet& savedBlocks, analysis::DataflowGraph& dfg)
{
	report("   Creating restore point for block '" << newBlock->label() << "'");

	analysis::DataflowGraph::IteratorMap::const_iterator
		dfgBlock = cfgToDfgMap.find(oldBlock);

	if(dfgBlock->second->aliveIn().empty()) return 0;
	
	ir::ControlFlowGraph::edge_iterator splitEdge = newKernel.cfg()->split_edge(
		newEdge, ir::BasicBlock(newKernel.cfg()->newId())).second;

	report("    restoring block wth id '" << splitEdge->head->id << "'");
	savedBlocks.insert(splitEdge->head);

	ir::PTXInstruction* move = new ir::PTXInstruction(ir::PTXInstruction::Mov);

	move->a = std::move(ir::PTXOperand(ir::PTXOperand::Address,
		ir::PTXOperand::u32, "_Zocelot_spill_area"));
	move->d = std::move(ir::PTXOperand(ir::PTXOperand::Register, 
		ir::PTXOperand::u32, dfg.newRegister()));
	move->type = ir::PTXOperand::u32;
	
	unsigned int offset = 0;

	splitEdge->head->instructions.push_back(move);

	for(analysis::DataflowGraph::RegisterSet::const_iterator
		reg = dfgBlock->second->aliveIn().begin();
		reg != dfgBlock->second->aliveIn().end(); ++reg)
	{
		report("    restoring r" << reg->id 
			<< " (" << ir::PTXOperand::bytes(reg->type) << " bytes)");
		
		ir::PTXInstruction* load =
			new ir::PTXInstruction(ir::PTXInstruction::Ld);
		
		load->addressSpace = ir::PTXInstruction::Local;
		load->type = reg->type;

		load->a = std::move(ir::PTXOperand(ir::PTXOperand::Indirect,
			ir::PTXOperand::u32, move->d.reg, offset));
		load->d = std::move(ir::PTXOperand(ir::PTXOperand::Register,
			reg->type, reg->id));
	
		splitEdge->head->instructions.push_back(load);
	
		offset += ir::PTXOperand::bytes(reg->type);
	}
	
	if(splitEdge->type == ir::Edge::Branch)
	{
		ir::PTXInstruction* branch = new ir::PTXInstruction(
			ir::PTXInstruction::Bra);
		branch->uni = true;
		branch->d = std::move(ir::PTXOperand(splitEdge->tail->label()));
		splitEdge->head->instructions.push_back(branch);
	}
	
	return offset;
}

typedef std::unordered_map<std::string,
	ir::ControlFlowGraph::iterator> BlockLabelMap;

static unsigned int createSavePoint(ir::PTXKernel& newKernel,
	ir::PTXKernel& ptx, BlockLabelMap& saveBlocks,
	ir::ControlFlowGraph::iterator oldBlock,
	ir::ControlFlowGraph::edge_iterator newEdge,
	const analysis::DataflowGraph::IteratorMap& cfgToDfgMap,
	analysis::DataflowGraph& dfg)
{
	report("   Creating save point for edge '" << newEdge->head->label()
		<< "' -> '" << oldBlock->label() << "'");
	
	// Possibly update the branch target
	if(newEdge->type == ir::Edge::Branch)
	{
		assert(!newEdge->head->instructions.empty());
		ir::PTXInstruction* branch = static_cast<ir::PTXInstruction*>(
			newEdge->head->instructions.back());
	
		branch->d.identifier += "_save";
	}

	BlockLabelMap::iterator existingBlock = saveBlocks.find(
		oldBlock->label() + "_save");
	
	if(existingBlock != saveBlocks.end())
	{
		ir::Edge edge(newEdge->head, existingBlock->second, newEdge->type);
	
		newKernel.cfg()->remove_edge(newEdge);
		newKernel.cfg()->insert_edge(edge);
	
		return 0;
	}
	
	analysis::DataflowGraph::IteratorMap::const_iterator
		dfgBlock = cfgToDfgMap.find(oldBlock);
	assert(dfgBlock != cfgToDfgMap.end());
	
	ir::ControlFlowGraph::edge_iterator splitEdge = newKernel.cfg()->split_edge(
		newEdge, ir::BasicBlock(newKernel.cfg()->newId())).second;

	unsigned int offset = 0;
	
	if(!dfgBlock->second->aliveIn().empty())
	{
		ir::PTXInstruction* move = new ir::PTXInstruction(
			ir::PTXInstruction::Mov);

		move->a = std::move(ir::PTXOperand(ir::PTXOperand::Address,
			ir::PTXOperand::u32, "_Zocelot_spill_area"));
		move->d = std::move(ir::PTXOperand(ir::PTXOperand::Register, 
			ir::PTXOperand::u32, dfg.newRegister()));
		move->type = ir::PTXOperand::u32;
		
		splitEdge->head->instructions.push_back(move);

		for(analysis::DataflowGraph::RegisterSet::const_iterator
			reg = dfgBlock->second->aliveIn().begin();
			reg != dfgBlock->second->aliveIn().end(); ++reg)
		{
			report("    saving r" << reg->id
				<< " (" << ir::PTXOperand::bytes(reg->type) << " bytes)");
		
			ir::PTXInstruction* store =
				new ir::PTXInstruction(ir::PTXInstruction::St);
		
			store->addressSpace = ir::PTXInstruction::Local;
			store->type = reg->type;

			store->d = std::move(ir::PTXOperand(ir::PTXOperand::Indirect,
				ir::PTXOperand::u32, move->d.reg, offset));
			store->a = std::move(ir::PTXOperand(ir::PTXOperand::Register,
				reg->type, reg->id));
	
			splitEdge->head->instructions.push_back(store);
	
			offset += ir::PTXOperand::bytes(reg->type);
		}
	}
	
	ir::PTXInstruction* move = new ir::PTXInstruction(ir::PTXInstruction::Mov);

	move->a = std::move(ir::PTXOperand(ir::PTXOperand::Address,
		ir::PTXOperand::u32, "_Zocelot_resume_point"));
	move->d = std::move(ir::PTXOperand(ir::PTXOperand::Register, 
		ir::PTXOperand::u32, dfg.newRegister()));
	move->type = ir::PTXOperand::u32;

	splitEdge->head->instructions.push_back(move);

	ir::PTXInstruction* store =
		new ir::PTXInstruction(ir::PTXInstruction::St);
	
	store->addressSpace = ir::PTXInstruction::Local;
	store->type = ir::PTXOperand::u32;

	store->d = std::move(ir::PTXOperand(ir::PTXOperand::Indirect,
		ir::PTXOperand::u32, move->d.reg));
	store->a = std::move(ir::PTXOperand(ir::PTXOperand::Immediate,
		ir::PTXOperand::u32));
	store->a.imm_uint = oldBlock->id;

	splitEdge->head->instructions.push_back(store);

	report("   Creating tail call to next kernel containing block "
		<< oldBlock->id);
	
	ir::PTXInstruction* call = new ir::PTXInstruction(ir::PTXInstruction::Call);
	call->a = std::move(ir::PTXOperand(ir::PTXOperand::FunctionName, ""));
	call->tailCall = true;
	call->branchTargetInstruction = oldBlock->id;
	
	splitEdge->head->instructions.push_back(call);
	
	saveBlocks.insert(std::make_pair(oldBlock->label() + "_save",
		splitEdge->head));
	
	return offset;
}

static ir::ControlFlowGraph::iterator createRegion(
	ir::PTXKernel& newKernel, unsigned int& spillRegionSize, ir::PTXKernel& ptx,
	const BlockSet& region, const BlockSet& inEdges, 
	ir::ControlFlowGraph::iterator entry,
	const analysis::DataflowGraph::IteratorMap& cfgToDfgMap,
	const BlockSet& alreadyAdded, BlockSet& savedBlocks,
	analysis::DataflowGraph& dfg)
{
	typedef std::vector<ir::Edge> EdgeVector;
	typedef std::unordered_map<ir::ControlFlowGraph::iterator, 
		ir::ControlFlowGraph::iterator> BlockMap;
	report("  Creating sub-kernel " << newKernel.name);
	
	ir::ControlFlowGraph::iterator exit = ptx.cfg()->get_exit_block();
	
	ir::BasicBlock::EdgePointerVector deletedEdges;

	EdgeVector    newEdges;
	BlockMap      oldToNewBlockMap;
	BlockLabelMap saveBlocks;

	// create the map of old to new blocks
	for(BlockSet::const_iterator block = region.begin(); 
		block != region.end(); ++block)
	{
		ir::ControlFlowGraph::iterator newBlock = newKernel.cfg()->insert_block(
			ir::BasicBlock((*block)->id, (*block)->instructions));
		
		oldToNewBlockMap.insert(std::make_pair(*block, newBlock));
	}
	
	newKernel.cfg()->computeNewBlockId();
	
	bool fallthroughToExit = false;
	
	for(BlockSet::const_iterator block = region.begin(); 
		block != region.end(); ++block)
	{
		report("   Adding block " << (*block)->label());

		for(ir::ControlFlowGraph::edge_pointer_iterator 
			edge = (*block)->out_edges.begin(); 
			edge != (*block)->out_edges.end(); ++edge)
		{
			ir::ControlFlowGraph::iterator tail = (*edge)->tail;

			// skip edges that exit the main kernel
			if(tail == ptx.cfg()->get_exit_block())
			{
				BlockMap::iterator head = oldToNewBlockMap.find((*edge)->head);
				assert(head != oldToNewBlockMap.end());
			
				newKernel.cfg()->insert_edge(ir::Edge(head->second, 
					newKernel.cfg()->get_exit_block(), ir::Edge::Branch));
				deletedEdges.push_back(*edge);

				continue;
			}

			// skip edges that are contained in the region
			if(region.count(tail) != 0)
			{
				BlockMap::iterator tail = oldToNewBlockMap.find((*edge)->tail);
				BlockMap::iterator head = oldToNewBlockMap.find((*edge)->head);
				assert(tail != oldToNewBlockMap.end());
				assert(head != oldToNewBlockMap.end());
			
				newKernel.cfg()->insert_edge(ir::Edge(head->second, 
					tail->second, (*edge)->type));

				continue;
			}

			// edges that leave the region are directed to the save point
			BlockMap::iterator newHead = oldToNewBlockMap.find((*edge)->head);
			assert(newHead != oldToNewBlockMap.end());
			ir::ControlFlowGraph::edge_iterator newEdge;

			if((*edge)->type == ir::Edge::FallThrough)
			{
				if(fallthroughToExit)
				{
					ir::ControlFlowGraph::edge_iterator exitEdge =
						newKernel.cfg()->insert_edge(ir::Edge(
						newHead->second, newKernel.cfg()->get_exit_block(), 
						ir::Edge::Branch));

					ir::PTXInstruction* branch = new ir::PTXInstruction(
						ir::PTXInstruction::Bra);
					branch->uni = true;
					branch->d = std::move(ir::PTXOperand(
						(*edge)->tail->label()));

					ir::ControlFlowGraph::EdgePair splitEdge = 
						newKernel.cfg()->split_edge(exitEdge,
						ir::BasicBlock(newKernel.cfg()->newId()));
					
					splitEdge.second->type = ir::Edge::Branch;
					splitEdge.second->head->instructions.push_back(branch);
					
					splitEdge.first->type = ir::Edge::FallThrough;
					
					newEdge = splitEdge.second;
				}
				else
				{
					fallthroughToExit = true;
					newEdge = newKernel.cfg()->insert_edge(ir::Edge(
						newHead->second, newKernel.cfg()->get_exit_block(),
						(*edge)->type));
				}
			}
			else
			{
				newEdge = newKernel.cfg()->insert_edge(ir::Edge(newHead->second, 
					newKernel.cfg()->get_exit_block(), (*edge)->type));
			}

			spillRegionSize = std::max(spillRegionSize, 
				createSavePoint(newKernel, ptx, saveBlocks,
				tail, newEdge, cfgToDfgMap, dfg));
			
			// skip edges that go to an already created subkernel
			if(alreadyAdded.count(tail) != 0) continue;

			if(exit == ptx.cfg()->get_exit_block())
			{
				report("   Creating exit point");
				exit = ptx.cfg()->insert_block(
					ir::BasicBlock(ptx.cfg()->newId()));			
			}
			
			ir::ControlFlowGraph::iterator head = (*edge)->head;
			report("    Redirecting edge from block_" << head->id 
				<< " -> block_" << tail->id << " to context save point" );
			
			deletedEdges.push_back(*edge);
			newEdges.push_back(ir::Edge(exit, tail, ir::Edge::Branch));
		}
	}
	
	BlockSet restoredBlocks;
	
	report("   Merging entry points");
	for(BlockSet::const_iterator block = inEdges.begin(); 
		block != inEdges.end(); ++block)
	{
		report("    From block " << (*block)->id);
		for(ir::ControlFlowGraph::edge_pointer_iterator 
			edge = (*block)->out_edges.begin(); 
			edge != (*block)->out_edges.end(); ++edge)
		{
			// skip edges not entering the region
			ir::ControlFlowGraph::iterator tail = (*edge)->tail;
			if(region.count(tail) == 0) continue;
			
			// skip edges targetting the same block
			if(!restoredBlocks.insert(tail).second) continue;
			
			BlockMap::iterator newTail = oldToNewBlockMap.find(tail);
			assert(newTail != oldToNewBlockMap.end());
			
			ir::ControlFlowGraph::edge_iterator newEdge = 
				newKernel.cfg()->insert_edge(ir::Edge(
				newKernel.cfg()->get_entry_block(),
				newTail->second, ir::Edge::Branch));
			
			spillRegionSize = std::max(spillRegionSize, 
				createRestorePoint(newKernel, ptx, newTail->second, 
				tail, newEdge, cfgToDfgMap, savedBlocks, dfg));
		}
	}

	if(entry == ptx.cfg()->get_entry_block())
	{
		BlockMap::iterator newTail = oldToNewBlockMap.find(
			entry->get_fallthrough_edge()->tail);
		assert(newTail != oldToNewBlockMap.end());

		newKernel.cfg()->insert_edge(ir::Edge(
			newKernel.cfg()->get_entry_block(),
			newTail->second, ir::Edge::FallThrough));
	}
	
	// Move all edges from the entry that do not enter the region to the exit
	for(ir::ControlFlowGraph::edge_pointer_iterator 
		edge = entry->out_edges.begin(); 
		edge != entry->out_edges.end(); ++edge)
	{
		if(region.count((*edge)->tail) != 0) continue;

		if(exit == ptx.cfg()->get_exit_block())
		{
			report("   Creating exit point" );
			exit = ptx.cfg()->insert_block(ir::BasicBlock(ptx.cfg()->newId()));			
		}
		
		report("   Moving edge from block " << (*edge)->head->id << " to block " 
			<< (*edge)->tail->id << " to next region");
		deletedEdges.push_back(*edge);
		newEdges.push_back(ir::Edge(exit, (*edge)->tail, ir::Edge::Branch));
	}

	for(ir::ControlFlowGraph::edge_pointer_iterator edge = deletedEdges.begin();
		edge != deletedEdges.end(); ++edge)
	{
		ptx.cfg()->remove_edge(*edge);
	}

	for(EdgeVector::iterator edge = newEdges.begin();
		edge != newEdges.end(); ++edge)
	{
		ptx.cfg()->insert_edge(*edge);
	}

	report("   Kernel exit point is " << exit->label());

	return exit;
}

class KernelAndId
{
public:
	KernelAndId(unsigned int i, ir::PTXKernel* k) : id(i), kernel(k)
	{
	
	}

public:
	unsigned int   id;
	ir::PTXKernel* kernel;
};

typedef std::unordered_map<ir::BasicBlock::Id, KernelAndId> IdToSubkernelMap;

static void updateTailCallTargets(
	const SubkernelFormationPass::KernelVector& splitKernels,
	const IdToSubkernelMap& idToKernelMap)
{
	for(SubkernelFormationPass::KernelVector::const_iterator
		subkernel = splitKernels.begin();
		subkernel != splitKernels.end(); ++subkernel)
	{
		for(ir::ControlFlowGraph::iterator block = (*subkernel)->cfg()->begin();
			block != (*subkernel)->cfg()->end(); ++block)
		{
			if(block->instructions.empty()) continue;
			
			ir::PTXInstruction* call = static_cast<ir::PTXInstruction*>(
				block->instructions.back());
		
			if(call->opcode != ir::PTXInstruction::Call) continue;
			if(!call->tailCall) continue;
			
			IdToSubkernelMap::const_iterator 
				kernelName = idToKernelMap.find(call->branchTargetInstruction);
			assertM(kernelName != idToKernelMap.end(),
				"Could not find kernel containing block with id " 
				<< call->branchTargetInstruction);
			
			call->a.identifier = kernelName->second.kernel->name;
			call->branchTargetInstruction = kernelName->second.id;
		}
	}
}

static void createScheduler(ir::PTXKernel& kernel,
	ir::PTXKernel& originalKernel, const BlockSet& savedBlocks,
	analysis::DataflowGraph& dfg)
{
	typedef ir::BasicBlock::EdgePointerVector EdgePointerVector;
	typedef std::vector<ir::Edge> EdgeVector;
	
	if(kernel.cfg()->get_entry_block()->out_edges.size() == 1)
	{
		kernel.cfg()->get_entry_block()->out_edges[0]->type
			= ir::Edge::FallThrough;
		return;
	}
	
	report("  Creating scheduler for subkernel '" << kernel.name << "'");
	
	BlockSet targets;
	
	ir::ControlFlowGraph::iterator scheduler = kernel.cfg()->insert_block(
		ir::BasicBlock(kernel.cfg()->newId()));

	ir::PTXInstruction* move = new ir::PTXInstruction(ir::PTXInstruction::Mov);

	move->a = std::move(ir::PTXOperand(ir::PTXOperand::Address,
		ir::PTXOperand::u32, "_Zocelot_resume_point"));
	move->d = std::move(ir::PTXOperand(ir::PTXOperand::Register, 
		ir::PTXOperand::u32, dfg.newRegister()));
	move->type = ir::PTXOperand::u32;
	
	scheduler->instructions.push_back(move);

	ir::PTXInstruction* load =
		new ir::PTXInstruction(ir::PTXInstruction::Ld);
	
	load->addressSpace = ir::PTXInstruction::Local;
	load->type = ir::PTXOperand::u32;

	load->a = std::move(ir::PTXOperand(ir::PTXOperand::Indirect,
		ir::PTXOperand::u32, move->d.reg));
	load->d = std::move(ir::PTXOperand(ir::PTXOperand::Register,
		ir::PTXOperand::u32, dfg.newRegister()));

	scheduler->instructions.push_back(load);

	// Create edges to and from the scheduler
	ir::BasicBlock::EdgePointerVector edges = kernel.cfg()->begin()->out_edges;
	
	for(ir::ControlFlowGraph::const_edge_pointer_iterator edge = edges.begin(); 
		edge != edges.end(); ++edge)
	{
		ir::Edge newEdge(scheduler, (*edge)->tail, (*edge)->type);

		report("   scheduling path to block '" << newEdge.tail->label()
			<< "' from block '" << newEdge.head->label() << "'");
		report("      - former edge type [" << (*edge)->type << "]");
		report("      - new edge type    [" << newEdge.type  << "]");

		kernel.cfg()->remove_edge(*edge);
		
		// don't add multiple paths to the same block
		if(!targets.insert(newEdge.tail).second) continue;

		kernel.cfg()->insert_edge(newEdge);
	}

	kernel.cfg()->insert_edge(ir::Edge(kernel.cfg()->begin(), scheduler, 
		ir::Edge::FallThrough));

	assert(scheduler->out_edges.size() > 1);
	
	// unroll the scheduler into a chain of 2-path schedulers
	while(scheduler->out_edges.size() > 2)
	{
		ir::ControlFlowGraph::edge_iterator target = scheduler->out_edges[0];
		
		ir::PTXInstruction* compare = new ir::PTXInstruction(
			ir::PTXInstruction::SetP);
		
		compare->a = load->d;
		compare->b = std::move(ir::PTXOperand(ir::PTXOperand::Immediate,
			ir::PTXOperand::u32));

		if(savedBlocks.count(target->tail) == 0)
		{
			compare->b.imm_uint = target->tail->id;
		}
		else if(target->tail->has_fallthrough_edge())
		{
			compare->b.imm_uint = 
				target->tail->get_fallthrough_edge()->tail->id;
		}
		else
		{
			compare->b.imm_uint = target->tail->get_branch_edge()->tail->id;
		}

		compare->d = std::move(ir::PTXOperand(ir::PTXOperand::Register,
			ir::PTXOperand::pred, dfg.newRegister()));
		compare->comparisonOperator = ir::PTXInstruction::Eq;
		compare->type = ir::PTXOperand::u32;
	
		scheduler->instructions.push_back(compare);
	
		ir::PTXInstruction* branch = new ir::PTXInstruction(
			ir::PTXInstruction::Bra);
	
		branch->d = std::move(ir::PTXOperand(target->tail->label()));
		branch->pg = compare->d;
	
		scheduler->instructions.push_back(branch);

		ir::BasicBlock::Id id = kernel.cfg()->newId();

		ir::ControlFlowGraph::iterator newScheduler =
			kernel.cfg()->insert_block(ir::BasicBlock(id));
			
		ir::Edge newFallthrough(scheduler, newScheduler, ir::Edge::FallThrough);
		
		ir::ControlFlowGraph::const_edge_pointer_iterator 
			edge = scheduler->out_edges.begin();
		
		EdgePointerVector killedEdges;
		EdgeVector newEdges;
		
		for(std::advance(edge, 1); edge != scheduler->out_edges.end(); ++edge)
		{
			ir::Edge replacement(newScheduler, (*edge)->tail, (*edge)->type);
			
			newEdges.push_back(replacement);
			killedEdges.push_back(*edge);
		}
		
		for(EdgePointerVector::iterator edge = killedEdges.begin();
			edge != killedEdges.end(); ++edge)
		{
			kernel.cfg()->remove_edge(*edge);
		}

		for(EdgeVector::iterator edge = newEdges.begin();
			edge != newEdges.end(); ++edge)
		{
			kernel.cfg()->insert_edge(*edge);
		}
		
		kernel.cfg()->insert_edge(newFallthrough);
		
		scheduler = newScheduler;
	}

	assert(scheduler->out_edges.size() > 1);
	
	// Add compare and branch instructions
	ir::ControlFlowGraph::edge_iterator target      = scheduler->out_edges[0];
	ir::ControlFlowGraph::edge_iterator fallthrough = scheduler->out_edges[1];
	
	// add dummy block to enable the scheduler to branch anywhere
	ir::ControlFlowGraph::iterator trampoline = kernel.cfg()->insert_block(
		ir::BasicBlock(kernel.cfg()->newId()));
		
	ir::ControlFlowGraph::iterator fallthroughTarget = fallthrough->tail;
	
	kernel.cfg()->insert_edge(ir::Edge(scheduler,
		trampoline, ir::Edge::FallThrough));
	kernel.cfg()->insert_edge(ir::Edge(trampoline,
		fallthrough->tail, ir::Edge::Branch));
	kernel.cfg()->remove_edge(fallthrough);
	
	ir::PTXInstruction *trampolineBra = 
		new ir::PTXInstruction(ir::PTXInstruction::Bra);
	trampolineBra->d = std::move(ir::PTXOperand(fallthroughTarget->label()));
	trampoline->instructions.push_back(trampolineBra);
	
	ir::PTXInstruction* compare = new ir::PTXInstruction(
		ir::PTXInstruction::SetP);
		
	compare->a = load->d;
	compare->b = std::move(ir::PTXOperand(ir::PTXOperand::Immediate,
		ir::PTXOperand::u32));
		
	if(savedBlocks.count(target->tail) == 0)
	{
		compare->b.imm_uint = target->tail->id;
	}
	else if(target->tail->has_fallthrough_edge())
	{
		compare->b.imm_uint = target->tail->get_fallthrough_edge()->tail->id;
	}
	else
	{
		compare->b.imm_uint = target->tail->get_branch_edge()->tail->id;
	}
	
	compare->d = std::move(ir::PTXOperand(ir::PTXOperand::Register,
		ir::PTXOperand::pred, dfg.newRegister()));
	compare->comparisonOperator = ir::PTXInstruction::Eq;
	compare->type = ir::PTXOperand::u32;
	
	scheduler->instructions.push_back(compare);
	
	ir::PTXInstruction* branch = new ir::PTXInstruction(
		ir::PTXInstruction::Bra);
	
	branch->d = std::move(ir::PTXOperand(target->tail->label()));
	branch->pg = compare->d;
	
	scheduler->instructions.push_back(branch);
}

static void addVariables(ir::PTXKernel& subkernel, const ir::IRKernel& kernel,
	unsigned int spillRegionSize, bool needsTransition)
{
	subkernel.arguments  = kernel.arguments;
	subkernel.parameters = kernel.parameters;
	subkernel.locals     = kernel.locals;
	
	if(!needsTransition) return;
	
	ir::PTXStatement resume(ir::PTXStatement::Local);
		
	resume.type = ir::PTXOperand::u32;
	resume.name = "_Zocelot_resume_point";
	
	subkernel.locals.insert(std::make_pair(resume.name, ir::Local(resume)));
	
	ir::PTXStatement spillRegion(ir::PTXStatement::Local);
		
	spillRegion.type = ir::PTXOperand::b8;
	spillRegion.name = "_Zocelot_spill_area";
	spillRegion.array.stride.push_back(spillRegionSize);
	
	subkernel.locals.insert(std::make_pair(spillRegion.name,
		ir::Local(spillRegion)));
		
	report("  Spill region size is " << spillRegionSize);
}

/* algorithm
    1) start at a kernel entry point that dominates all remaining blocks
    2) create a strongly connected subgraph with N instructions and no barriers
      a) This is a new kernel
    3) For all edges leaving the graph
      a) save all live registers
      b) save the target block's id
      c) create a new scheduler block includes an indirect branch to each 
         of the targets
      d) redirect each edge to the kernel exit point
      e) create a new kernel rooted in the new scheduler block, goto 1
*/
void SubkernelFormationPass::ExtractKernelsPass::runOnKernel(ir::IRKernel& k)
{
	typedef std::vector<ir::ControlFlowGraph::iterator> BlockQueue;
	
	report(" Splitting kernel '" << k.name << "' into sub-kernel regions.");
	report("  - expected region size: " << _expectedRegionSize);
	
	BlockQueue queue;
	KernelVector splitKernels;
	BlockSet region;
	BlockSet encountered;
	BlockSet alreadyAdded;
	BlockSet inEdges;
	BlockSet savedBlocks;
	IdToSubkernelMap idToKernelMap;

	unsigned int currentRegionSize = 0;
	unsigned int kernelId = 0;
	unsigned int spillRegionSize = 0;
		
	std::string originalName = k.name;
	
	ir::PTXKernel& ptx = static_cast<ir::PTXKernel&>(k);

	analysis::DataflowGraph::IteratorMap cfgToDfgMap = _dfg().getCFGtoDFGMap();
	
	// This is the new kernel entry point
	splitKernels.push_back(new ir::PTXKernel(k.name, false, k.module));

	ir::PTXKernel* newKernel = splitKernels.back();

	report(" New kernel name is " << newKernel->name);

	// Start at the entry block	
	ir::ControlFlowGraph::iterator entry = ptx.cfg()->get_entry_block();
	queue.push_back(entry);
	
	while(true)
	{
		// The front of the queue becomes part of the region
		ir::ControlFlowGraph::iterator block = queue.back();
		queue.pop_back();

		// Remove this block from the entering list (if it exists)
		inEdges.erase(block);
		
		// Keep track of all blocks entering the region
		for(ir::ControlFlowGraph::const_edge_pointer_iterator 
			edge = block->in_edges.begin(); 
			edge != block->in_edges.end(); ++edge)
		{
			// skip the entry block
			if((*edge)->head == ptx.cfg()->get_entry_block()) continue;

			// skip the current block
			if((*edge)->head == block) continue;

			// skip blocks in the region
			if(region.count((*edge)->head)) continue;
			
			inEdges.insert((*edge)->head);
		}
		
		// Edges leaving the region become candidates for the next subkernel
		for(ir::ControlFlowGraph::const_edge_pointer_iterator 
			edge = block->out_edges.begin(); 
			edge != block->out_edges.end(); ++edge)
		{
			ir::ControlFlowGraph::iterator tail = (*edge)->tail;

			// skip the exit block
			if(tail == ptx.cfg()->get_exit_block()) continue;
			
			// skip blocks that have already been added
			if(alreadyAdded.count(tail) != 0) continue;
			
			// skip already encountered blocks
			if(!encountered.insert(tail).second) continue;
			
			// push successor blocks
			report("  Queuing up sucessor of block " 
				<< block->id << ", block " << tail->id);
			queue.push_back(tail);
		}

		// Add the block to the new kernel (skip the entry)
		if(block == entry) continue;

		currentRegionSize += block->instructions.size();
		region.insert(block);
		report("  Adding block with id " << block->id 
			<< " to kernel " << newKernel->name);
		idToKernelMap.insert(std::make_pair(block->id,
			KernelAndId(kernelId, newKernel)));
		
		// create a new region if there are enough blocks
		if(currentRegionSize < _expectedRegionSize && !queue.empty()) continue;
		
		entry = createRegion(*newKernel, spillRegionSize, ptx, region,
			inEdges, entry, cfgToDfgMap, alreadyAdded, savedBlocks, _dfg());
		alreadyAdded.insert(region.begin(), region.end());
		
		createScheduler(*newKernel, ptx, savedBlocks, _dfg());

		// restart with a new kernel if there are any blocks left
		if(entry == ptx.cfg()->get_exit_block()) break;
	
		std::stringstream name;
		name << originalName << "_split_" << kernelId++;

		splitKernels.push_back(new ir::PTXKernel(name.str(), true, k.module));
		encountered.clear();
		inEdges.clear();
		region.clear();
		queue.clear();
		savedBlocks.clear();
		queue.push_back(entry);
		currentRegionSize = 0;
		
		newKernel = splitKernels.back();
		report(" New kernel name is " << newKernel->name);
	}

	// Add variables
	for(KernelVector::iterator kernel = splitKernels.begin();
		kernel != splitKernels.end(); ++kernel)
	{
		addVariables(**kernel, k, spillRegionSize, splitKernels.size() > 1);
	}
	
	// Rename 
	updateTailCallTargets(splitKernels, idToKernelMap);

	kernels.insert(std::make_pair(splitKernels.front(),
		std::move(splitKernels)));
}

void SubkernelFormationPass::ExtractKernelsPass::finalize()
{
	
}

}

#endif

