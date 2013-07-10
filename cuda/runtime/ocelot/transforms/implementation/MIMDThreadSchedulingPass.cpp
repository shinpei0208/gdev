/*! \file   MIMDThreadSchedulingPass.cpp
	\date   Friday February 18, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the MIMDThreadSchedulingPass class.
*/

#ifndef MIMD_THREAD_SCHEDULING_PASS_CPP_INCLUDED
#define MIMD_THREAD_SCHEDULING_PASS_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/transforms/interface/MIMDThreadSchedulingPass.h>

#include <ocelot/ir/interface/PTXKernel.h>

#include <ocelot/analysis/interface/DominatorTree.h>
#include <ocelot/analysis/interface/PostdominatorTree.h>
#include <ocelot/analysis/interface/DataflowGraph.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace transforms
{

MIMDThreadSchedulingPass::MIMDThreadSchedulingPass()
	: KernelPass(Analysis::DataflowGraphAnalysis
		| Analysis::PostDominatorTreeAnalysis
		| Analysis::DominatorTreeAnalysis, 
		"MIMDThreadSchedulingPass")
{

}

void MIMDThreadSchedulingPass::initialize(const ir::Module& m)
{
	// none needed, intentionally empty
}

static void addPredicateInitialValue(ir::ControlFlowGraph::iterator dom,
	ir::Instruction::RegisterType reg)
{
	ir::PTXInstruction* setp = new ir::PTXInstruction(ir::PTXInstruction::SetP);
	
	setp->d = ir::PTXOperand(ir::PTXOperand::Register,
		ir::PTXOperand::pred, reg);
	setp->type = ir::PTXOperand::u32;
	setp->a = ir::PTXOperand(0ULL);
	setp->b = ir::PTXOperand(0ULL);
	setp->comparisonOperator = ir::PTXInstruction::Ne;
	
	dom->instructions.push_front(setp);
	
	report("   initializing predicate register " << reg << " in block "
		<< dom->id << " with '" << setp->toString() << "'");
}

typedef std::unordered_set<ir::ControlFlowGraph::iterator> BlockSet;

static void addYield(ir::PTXKernel& kernel, BlockSet& barriers,
	ir::ControlFlowGraph::iterator block,
	ir::ControlFlowGraph::iterator successor,
	ir::ControlFlowGraph::iterator pdom, ir::Instruction::RegisterType reg)
{
	ir::PTXInstruction* barrier = static_cast<ir::PTXInstruction*>(
		block->instructions.back());
	block->instructions.pop_back();

	ir::PTXInstruction* setp = new ir::PTXInstruction(ir::PTXInstruction::SetP);
	
	setp->d = ir::PTXOperand(ir::PTXOperand::Register,
		ir::PTXOperand::pred, reg);
	setp->type = ir::PTXOperand::u32;
	setp->a = ir::PTXOperand(0);
	setp->b = ir::PTXOperand(0);
	setp->comparisonOperator = ir::PTXInstruction::Eq;
	
	block->instructions.push_front(setp);

	setp = new ir::PTXInstruction(ir::PTXInstruction::SetP);
	
	setp->d = ir::PTXOperand(ir::PTXOperand::Register,
		ir::PTXOperand::pred, reg);
	setp->type = ir::PTXOperand::u32;
	setp->a = ir::PTXOperand(0);
	setp->b = ir::PTXOperand(0);
	setp->comparisonOperator = ir::PTXInstruction::Ne;
	
	successor->instructions.push_front(setp);
	
	ir::ControlFlowGraph::iterator scheduler = kernel.cfg()->split_block(pdom,
		pdom->instructions.begin(), ir::Edge::FallThrough);

	report("   created scheduler block " << scheduler->id
		<< " from pdom block " << pdom->id);

	if(!barriers.insert(pdom).second)
	{
		report("   removing barrier from " << scheduler->id << " "
			<< scheduler->instructions.front()->toString());
		delete scheduler->instructions.front();
		scheduler->instructions.pop_front();
	}

	std::swap(pdom, scheduler);

	scheduler->instructions.push_back(barrier);
		
	ir::PTXInstruction* branchToResumePoint
		= new ir::PTXInstruction(ir::PTXInstruction::Bra);

	branchToResumePoint->pg = ir::PTXOperand(ir::PTXOperand::Register,
		ir::PTXOperand::pred, reg);
	branchToResumePoint->d = ir::PTXOperand(successor->label());
	
	scheduler->instructions.push_back(branchToResumePoint);
	
	ir::PTXInstruction* branchToScheduler
		= new ir::PTXInstruction(ir::PTXInstruction::Bra);

	branchToScheduler->d = ir::PTXOperand(scheduler->label());
	
	block->instructions.push_back(branchToScheduler);	
	
	kernel.cfg()->insert_edge(ir::Edge(block, scheduler, ir::Edge::Branch));
	kernel.cfg()->insert_edge(ir::Edge(scheduler, successor, ir::Edge::Branch));

	for(ir::ControlFlowGraph::edge_pointer_iterator
		edge = scheduler->in_edges.begin();
		edge != scheduler->in_edges.end(); ++edge)
	{
		if((*edge)->type != ir::Edge::Branch) continue;
		assert(!(*edge)->head->instructions.empty());

		ir::PTXInstruction& ptx = *static_cast<ir::PTXInstruction*>(
			(*edge)->head->instructions.back());
		assert(ptx.opcode == ir::PTXInstruction::Bra);
		
		ptx.d.identifier = scheduler->label();
	}
}

void sinkBarrier(ir::PTXKernel& kernel, BlockSet& barriers,
	ir::ControlFlowGraph::iterator block, ir::ControlFlowGraph::iterator dom,
	ir::ControlFlowGraph::iterator pdom, analysis::DataflowGraph* dfg)
{
	bool split = true;
	while(split)
	{
		split = false;
		report(" Sinking all barriers in block " << block->id);
	
		for(ir::ControlFlowGraph::InstructionList::iterator
			instruction = block->instructions.begin();
			instruction != block->instructions.end(); ++instruction)
		{
			const ir::PTXInstruction& ptx =
				*static_cast<const ir::PTXInstruction*>(*instruction);
			if(ptx.opcode == ir::PTXInstruction::Bar)
			{
				ir::ControlFlowGraph::iterator
					successor = kernel.cfg()->split_block(block,
					++instruction, ir::Edge::Branch);
				
				kernel.cfg()->remove_edge(block->get_edge(successor));
				
				report("  for barrier at instruction " << (std::distance(
					block->instructions.begin(), instruction)));
				report("   created resume block " << successor->id);
				
				ir::Instruction::RegisterType
					predicate = dfg->newRegister();
				
				addPredicateInitialValue(dom, predicate);
				addYield(kernel, barriers, block, successor, pdom, predicate);

				block = successor;
				split = true;
				break;
			}
		}		
	}
	
}

void MIMDThreadSchedulingPass::runOnKernel(ir::IRKernel& k)
{
	assertM(k.ISA == ir::Instruction::PTX,
		"This pass is valid for PTX kernels only.");

	ir::PTXKernel* kernel = static_cast<ir::PTXKernel*>(&k);

	bool changed = true;
	BlockSet barriers;

	Analysis* pdom_structure = getAnalysis(Analysis::PostDominatorTreeAnalysis);
	assert(pdom_structure != 0);

	analysis::PostdominatorTree* pdom_tree = static_cast<
		analysis::PostdominatorTree*>(pdom_structure);

	Analysis* dom_structure = getAnalysis(Analysis::DominatorTreeAnalysis);
	assert(dom_structure != 0);

	analysis::DominatorTree* dom_tree = static_cast<
		analysis::DominatorTree*>(dom_structure);

	Analysis* dfg_structure = getAnalysis(Analysis::DataflowGraphAnalysis);
	assert(dfg_structure != 0);

	analysis::DataflowGraph* dfg = static_cast<
		analysis::DataflowGraph*>(dfg_structure);

	while(changed)
	{
		changed = false;
		report("Applying pass...");
		ir::ControlFlowGraph::BlockPointerVector schedulingPoints;
		ir::ControlFlowGraph::BlockPointerVector postDominators;
		ir::ControlFlowGraph::BlockPointerVector dominators;

		// identify scheduling points
		//  - blocks with barriers
		for(ir::ControlFlowGraph::iterator block = kernel->cfg()->begin();
			block != kernel->cfg()->end(); ++block)
		{
			// skip blocks that are post dominators of the entry point	
			if(pdom_tree->postDominates(
				block, kernel->cfg()->get_entry_block()))
			{
				report(" Skipping block " << block->id);
				continue;
			}
	
			for(ir::ControlFlowGraph::InstructionList::const_iterator
				instruction = block->instructions.begin();
				instruction != block->instructions.end(); ++instruction)
			{
				const ir::PTXInstruction& ptx =
					*static_cast<const ir::PTXInstruction*>(*instruction);
				if(ptx.opcode == ir::PTXInstruction::Bar)
				{
					ir::ControlFlowGraph::iterator
						pdom = pdom_tree->getPostDominator(block);
					ir::ControlFlowGraph::iterator
						dom = dom_tree->getDominator(block);
					report(" Found barrier " << ptx.toString()
						<< " in block " << block->id << " (postdominator "
						<< pdom->id << ") (dominator " << dom->id << ")");
					schedulingPoints.push_back(block);
					postDominators.push_back(pdom);
					dominators.push_back(dom);
					changed = true;
					break;
				}
			}
		}
	
		ir::ControlFlowGraph::pointer_iterator block = schedulingPoints.begin();
		ir::ControlFlowGraph::pointer_iterator dom   = dominators.begin();
		ir::ControlFlowGraph::pointer_iterator pdom  = postDominators.begin();
		for( ; block != schedulingPoints.end(); ++block, ++dom, ++pdom)
		{
			sinkBarrier(*kernel, barriers, *block, *dom, *pdom, dfg);
		}
	}
}

void MIMDThreadSchedulingPass::finalize()
{
	// none needed, intentionall empty
}

}

#endif

