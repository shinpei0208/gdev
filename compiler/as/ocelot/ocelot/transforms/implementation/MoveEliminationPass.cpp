/*!	\file   MoveEliminationPass.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Wednesday January 23, 2013
	\brief  The source file for the MoveEliminationPass class.
*/


// Ocelot Includes
#include <ocelot/transforms/interface/MoveEliminationPass.h>

#include <ocelot/analysis/interface/DataflowGraph.h>

#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace transforms
{

MoveEliminationPass::MoveEliminationPass()
: KernelPass({"DataflowGraphAnalysis"}, "MoveEliminationPass")
{
	
}

typedef analysis::DataflowGraph::instruction_iterator instruction_iterator;
typedef std::vector<instruction_iterator> InstructionVector;

static InstructionVector getMoves(analysis::DataflowGraph* dfg);
static bool canEliminate(instruction_iterator move);
static void eliminate(instruction_iterator move);

void MoveEliminationPass::runOnKernel(ir::IRKernel& k)
{
	report("Eliminating moves in kernel " << k.name << "");
	
	auto dfg = static_cast<analysis::DataflowGraph*>(
		getAnalysis("DataflowGraphAnalysis"));
	assert(dfg != 0);

	dfg->convertToSSAType(analysis::DataflowGraph::Minimal);

	auto moves = getMoves(dfg);
	
	bool eliminatedAny = false;
	
	report(" Eliminating moves");
	
	for(auto move = moves.begin(); move != moves.end(); ++move)
	{
		if(canEliminate(*move))
		{
			report("  " << (*move)->i->toString());
			eliminate(*move);
			eliminatedAny = true;
		}
	}
	
	if(eliminatedAny)
	{
		invalidateAnalysis("DataflowGraphAnalysis");
	}
	
	report("finished...");
}

static ir::PTXInstruction* toPTX(instruction_iterator instruction)
{
	return static_cast<ir::PTXInstruction*>(instruction->i);
}

static bool isMove(ir::PTXInstruction* ptx)
{
	return ptx->opcode == ir::PTXInstruction::Mov;
}

static InstructionVector getMoves(analysis::DataflowGraph* dfg)
{
	InstructionVector moves;

	report(" Enumerating moves....");

	for(auto block = dfg->begin(); block != dfg->end(); ++block)
	{
		for(auto instruction = block->instructions().begin();
			instruction != block->instructions().end(); ++instruction)
		{
			auto ptx = toPTX(instruction);
			
			if(isMove(ptx))
			{
				report("  " << ptx->toString());
				moves.push_back(instruction);
			}
		}
	}
	
	return moves;
}

static bool sourcesRegister(instruction_iterator move)
{
	auto ptx = toPTX(move);
	
	return ptx->a.isRegister() && !ptx->a.isVector();
}

typedef analysis::DataflowGraph::iterator block_iterator;
typedef std::unordered_set<ir::ControlFlowGraph::BasicBlock::Id> BlockSet;

static bool allPreviousDefinitionsDominateMove(instruction_iterator move,
	block_iterator block, BlockSet& visited)
{
	if(!visited.insert(block->id()).second) return true;
	
	assert(move->d.size() == 1);

	auto destination = *move->d.front().pointer;
	
	// If the value is defined by a phi with multiple sources, then the
	//  previous definition does not dominate the move
	for(auto phi = block->phis().begin(); phi != block->phis().end(); ++phi)
	{
		if(phi->d == destination)
		{
			return false;
		}
	}
	
	// Check all predecessors with the value live out
	for(auto predecessor = block->predecessors().begin();
		predecessor != block->predecessors().end(); ++predecessor)
	{
		if((*predecessor)->aliveOut().count(destination) == 0) continue;
		
		if(!allPreviousDefinitionsDominateMove(move, *predecessor, visited))
		{
			return false;
		}
	}
	
	return true;
}

static bool allDefinitionsDominateMove(instruction_iterator move)
{
	BlockSet visited;
	
	return allPreviousDefinitionsDominateMove(move, move->block, visited);
}

static bool isPredicated(instruction_iterator move)
{
	auto ptx = toPTX(move);
	
	return ptx->pg.condition != ir::PTXOperand::PT;
}

static bool isVectorMove(instruction_iterator move)
{
	return move->d.size() > 1;
}

static bool typesMatch(instruction_iterator move)
{
	auto ptx = toPTX(move);
	
	return ir::PTXOperand::valid(ptx->d.type, ptx->a.type);
}

static bool canEliminate(instruction_iterator move)
{
	// Does the move source a register?
	if(!sourcesRegister(move)) return false;

	// Does the move only have a single destination
	if(isVectorMove(move)) return false;

	// Is the move predicated?
	if(isPredicated(move)) return false;

	// Do all definitions of the move source dominate the move?
	if(!allDefinitionsDominateMove(move)) return false;
	
	// Does the source type match the destination type?
	if(!typesMatch(move)) return false;

	return true;
}

static void propagateMoveSourceToUsersInBlock(instruction_iterator move,
	instruction_iterator position, block_iterator block, BlockSet& visited)
{
	// early exit for visited blocks
	if(!visited.insert(block->id()).second) return;

	assert(move->d.size() == 1);
	assert(move->s.size() == 1);

	auto destination = *move->d.front().pointer;
	auto moveSource  = *move->s.front().pointer;
	
	// We can skip PHIs because the use of a PHI would make the removal illegal

	// replace uses in the block
	for(; position != block->instructions().end(); ++position)
	{
		for(auto source = position->s.begin();
			source != position->s.end(); ++source)
		{
			if(*source->pointer == destination)
			{
				*source->pointer = moveSource;
			}
		}
	}
	
	// replace in successors
	for(auto successor = block->successors().begin();
		successor != block->successors().end(); ++successor)
	{
		if((*successor)->aliveIn().count(destination) == 0) continue;
		
		propagateMoveSourceToUsersInBlock(move,
			(*successor)->instructions().begin(), *successor, visited);
	}

}

static void propagateMoveSourceToUsers(instruction_iterator move)
{
	BlockSet visited;
		
	propagateMoveSourceToUsersInBlock(move, move, move->block, visited);
}

static void eliminate(instruction_iterator move)
{
	propagateMoveSourceToUsers(move);
	
	// erase the move
	auto dfg = move->block->dfg();
	
	dfg->erase(move->block, move);
}

}

