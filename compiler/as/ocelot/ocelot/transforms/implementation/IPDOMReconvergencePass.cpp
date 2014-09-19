/*! \file   IPDOMReconvergencePass.cpp
	\date   Monday May 9, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the IPDOMReconvergence class.
*/

#ifndef IPDOM_RECONVERGENCE_PASS_CPP_INCLUDED
#define IPDOM_RECONVERGENCE_PASS_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/transforms/interface/IPDOMReconvergencePass.h>

#include <ocelot/ir/interface/IRKernel.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>

#include <ocelot/analysis/interface/PostdominatorTree.h>
#include <ocelot/analysis/interface/DominatorTree.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <unordered_map>
#include <set>
#include <cassert>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace transforms
{

IPDOMReconvergencePass::IPDOMReconvergencePass()
: KernelPass({"PostDominatorTreeAnalysis",
	"DominatorTreeAnalysis"}, "IPDOMReconvergencePass")
{

}

void IPDOMReconvergencePass::initialize(const ir::Module& m)
{
	instructions.clear();
}

void IPDOMReconvergencePass::runOnKernel(ir::IRKernel& k)
{
	typedef std::unordered_map<ir::ControlFlowGraph::InstructionList::iterator, 
		ir::ControlFlowGraph::InstructionList::iterator > InstructionMap;
	typedef std::unordered_map<ir::ControlFlowGraph::InstructionList::iterator,
		unsigned int> InstructionIdMap;

	Analysis* pdom_structure = getAnalysis("PostDominatorTreeAnalysis");
	assert(pdom_structure != 0);
	
	analysis::PostdominatorTree* pdom_tree
		= static_cast<analysis::PostdominatorTree*>(pdom_structure);
	
	// visit basic blocks and add reconverge instructions
	ir::ControlFlowGraph::BlockPointerVector
		bb_sequence = k.cfg()->executable_sequence();
	
	InstructionMap reconvergeTargets;
	
	report(" Adding reconverge instructions");
	// Create reconverge instructions
	for(ir::ControlFlowGraph::pointer_iterator bb_it = bb_sequence.begin(); 
		bb_it != bb_sequence.end(); ++bb_it) 
	{
		ir::ControlFlowGraph::InstructionList::iterator 
			i_it = (*bb_it)->instructions.begin();
		for(; i_it != (*bb_it)->instructions.end(); ++i_it) 
		{
			ir::PTXInstruction &ptx_instr = static_cast<
				ir::PTXInstruction&>(**i_it);
			if(ptx_instr.opcode == ir::PTXInstruction::Bra && !ptx_instr.uni) 
			{
				ir::ControlFlowGraph::iterator 
					pdom = pdom_tree->getPostDominator(*bb_it);

				report( "  Getting post dominator block " << pdom->label()
					<< " of instruction " << ptx_instr.toString() );
				if(pdom->instructions.empty() ||
					static_cast<ir::PTXInstruction*>(
					pdom->instructions.front())->opcode
					!= ir::PTXInstruction::Reconverge)
				{
					pdom->instructions.push_front(ir::PTXInstruction(
						ir::PTXInstruction::Reconverge).clone());
				}
				reconvergeTargets.insert(std::make_pair(i_it, 
					pdom->instructions.begin()));
			}
		}
	}

	InstructionIdMap ids;

	report(" Packing instructions into a vector");
	size_t lastPC = 0;
	for (ir::ControlFlowGraph::pointer_iterator bb_it = bb_sequence.begin(); 
		bb_it != bb_sequence.end(); ++bb_it) {
		int n = 0;
		
		report("  " << (*bb_it)->label());
		
		for (ir::ControlFlowGraph::InstructionList::iterator 
			i_it = (*bb_it)->instructions.begin(); 
			i_it != (*bb_it)->instructions.end(); ++i_it, ++n) {
			ir::PTXInstruction& ptx = static_cast<ir::PTXInstruction&>(**i_it);
			if (ptx.opcode == ir::PTXInstruction::Reconverge 
				|| i_it == (*bb_it)->instructions.begin()) 
			{
				ids.insert(std::make_pair(i_it, instructions.size()));
			}
			ptx.pc = instructions.size();
			lastPC = ptx.pc;
			instructions.push_back(ptx);
			report("   [" << lastPC << "] - " << ptx.toString());
		}
		
	}

	std::set< int > targets;	// set of branch targets

	report( "\n\n    Updating branch targets and reconverge points" );
	unsigned int id = 0;
	for (ir::ControlFlowGraph::pointer_iterator bb_it = bb_sequence.begin();
		bb_it != bb_sequence.end(); ++bb_it) 
	{
		for (ir::ControlFlowGraph::InstructionList::iterator 
			i_it = (*bb_it)->instructions.begin(); 
			i_it != (*bb_it)->instructions.end(); ++i_it, ++id) 
		{
			ir::PTXInstruction& ptx = static_cast<ir::PTXInstruction&>(**i_it);				
						
			if (ptx.opcode == ir::PTXInstruction::Bra) 
			{
				report( "  Instruction " << ptx.toString() );
				if (!ptx.uni) {
					InstructionMap::iterator 
						reconverge = reconvergeTargets.find(i_it);
					assert(reconverge != reconvergeTargets.end());
					InstructionIdMap::iterator 
						target = ids.find(reconverge->second);
					assert(target != ids.end());
					instructions[id].reconvergeInstruction = target->second;
					report("   reconverge at " << target->second);
				}
				
				auto target = (*bb_it)->get_branch_edge()->tail;
				
				while(target->instructions.empty())
				{
					target = target->get_fallthrough_edge()->tail;
				}
				
				InstructionIdMap::iterator branch = ids.find(
					target->instructions.begin());
				assert(branch != ids.end());
				instructions[id].branchTargetInstruction = branch->second;
				report("   target at " << branch->second);
			}
		}
	}

}

void IPDOMReconvergencePass::finalize()
{

}

}

#endif

