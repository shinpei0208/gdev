/*!	\file   ConstantPropagationPass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday November 9, 2012
	\brief  The header file for the ConstantPropagationPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/ConstantPropagationPass.h>

#include <ocelot/analysis/interface/DataflowGraph.h>

#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0
#define REPORT_PTX  0

namespace transforms
{

ConstantPropagationPass::ConstantPropagationPass()
: KernelPass({"DataflowGraphAnalysis"}, "ConstantPropagationPass")
{

}

typedef analysis::DataflowGraph::iterator iterator;
typedef std::unordered_set<iterator>      BlockSet;

static void eliminateRedundantInstructions(analysis::DataflowGraph& dfg,
	BlockSet& blocks, iterator block);

void ConstantPropagationPass::runOnKernel(ir::IRKernel& k)
{
	report("Running constant propagation on kernel " << k.name);
	
	Analysis* dfgAnalysis = getAnalysis("DataflowGraphAnalysis");
	assert(dfgAnalysis != 0);
	
	analysis::DataflowGraph& dfg =
		*static_cast<analysis::DataflowGraph*>(dfgAnalysis);
	
	dfg.convertToSSAType(analysis::DataflowGraph::Minimal);
	
	assert(dfg.ssa() == analysis::DataflowGraph::Minimal);
	
	BlockSet blocks;
	
	report(" Starting by scanning all basic blocks");
	
	for(iterator block = dfg.begin(); block != dfg.end(); ++block)
	{
		report("  Queueing up BB_" << block->id());
		blocks.insert(block);
	}
	
	while(!blocks.empty())
	{
		iterator block = *blocks.begin();
		blocks.erase(blocks.begin());
	
		eliminateRedundantInstructions(dfg, blocks, block);
	}

	report("Finished running constant propagation on kernel " << k.name);
	reportE(REPORT_PTX, k);

}

typedef analysis::DataflowGraph::InstructionVector InstructionVector;

static bool canRemoveInstruction(iterator block,
	InstructionVector::iterator instruction);
static bool propagateValueToSuccessors(analysis::DataflowGraph& dfg,
	BlockSet& blocks, InstructionVector::iterator instruction);

static void eliminateRedundantInstructions(analysis::DataflowGraph& dfg,
	BlockSet& blocks, iterator block)
{
	typedef std::vector<unsigned int> KillList;
	
	KillList killList;
	
	report("  Propagating constants through instructions in BB_" << block->id());
	unsigned int index = 0;
	for(auto instruction = block->instructions().begin();
		instruction != block->instructions().end(); ++instruction)
	{
		if(!propagateValueToSuccessors(dfg, blocks, instruction))
		{
			++index;
			continue;
		}
		
		if(canRemoveInstruction(block, instruction))
		{
			report("    value is not used, removed it.");
			killList.push_back(index);
			
			// schedule the block for more work
			// TODO: do this when we consider values in multiple blocks
		}
		else
		{
			++index;
		}
	}
	
	for(KillList::iterator killed = killList.begin();
		killed != killList.end(); ++killed)
	{
		dfg.erase(block, *killed);
	}
}

static bool canRemoveInstruction(iterator block,
	InstructionVector::iterator instruction)
{
	ir::PTXInstruction& ptx = *static_cast<ir::PTXInstruction*>(
		instruction->i);

	if(ptx.hasSideEffects())
	{
		report("    can't remove because it has side effects.");
		return false;
	}
	
	for(auto reg = instruction->d.begin(); reg != instruction->d.end(); ++reg)
	{
		// the reg is alive outside the block
		if(block->aliveOut().count(*reg) != 0)
		{
			report("    can't remove because it is live out of the block");
			return false;
		}
		
		auto next = instruction;
		for(++next; next != block->instructions().end(); ++next)
		{
			for(auto source = next->s.begin();
				source != next->s.end(); ++source)
			{
				// found a user in the block
				if(*source->pointer == *reg->pointer)
				{
					report("    can't remove because " << next->i->toString()
						<< " uses the value.");
					return false;
				}
			}
		}
	}
	
	// There are no users and the instruction has no side effects
	return true;
}

static bool isOutputConstant(InstructionVector::iterator instruction)
{
	auto ptx = static_cast<ir::PTXInstruction*>(instruction->i);

	if(ptx->hasSideEffects()) return false;

	// The output is not constant if there are register inputs
	if(!instruction->s.empty()) return false;

	ir::PTXOperand* operands[] = {&ptx->a, &ptx->b, &ptx->c, &ptx->pg};

	for(unsigned int i = 0; i < 4; ++i)
	{
		if(operands[i]->addressMode == ir::PTXOperand::Invalid) continue;

		if(operands[i]->isRegister() &&
			operands[i]->type == ir::PTXOperand::pred)
		{
			if(operands[i]->condition == ir::PTXOperand::PT ||
				operands[i]->condition == ir::PTXOperand::nPT)
			{
				continue;
			}
		}

		if(operands[i]->addressMode != ir::PTXOperand::Immediate)
		{
			report("    operand " << operands[i]->toString()
				<< " is not constant.");
			return false;
		}
	}

	return true;
}

static void replaceOperand(ir::PTXOperand& operand,
	ir::Instruction::RegisterType registerId,
	const ir::PTXOperand& immediate)
{
	if(!operand.array.empty())
	{
		for(auto sub = operand.array.begin();
			sub != operand.array.end(); ++sub)
		{
			replaceOperand(*sub, registerId, immediate);
		}

		return;
	}

	if(!operand.isRegister()) return;

	if(operand.reg != registerId) return;;
	
	int offset = 0;
	
	if(operand.addressMode == ir::PTXOperand::Indirect)
	{
		offset = operand.offset;
	}

	operand = immediate;
	
	operand.imm_uint += offset;
}

static void replaceOperand(ir::PTXInstruction& ptx,
	ir::Instruction::RegisterType registerId,
	const ir::PTXOperand& immediate)
{
	ir::PTXOperand* operands[] = {&ptx.a, &ptx.b, &ptx.c, &ptx.pq,
		&ptx.d};

	unsigned int sources = 4;

	if(ptx.isStore()) ++sources;
	
	for(unsigned int i = 0; i < sources; ++i)
	{
		replaceOperand(*operands[i], registerId, immediate);
	}
}

static uint64_t getMask(const ir::PTXOperand& operand)
{
	uint64_t mask = (1ULL << (operand.bytes() * 8)) - 1;

	if(mask == 0) mask = 0xffffffffffffffffULL;

	return mask;
}

static uint64_t getValue(const ir::PTXOperand& operand)
{
	uint64_t value = operand.imm_uint;

	uint64_t mask = getMask(operand);

	return value & mask;
}

static double getDouble(const ir::PTXOperand& operand)
{
	if(operand.type == ir::PTXOperand::f32)
	{
		return operand.imm_single;
	}

	return operand.imm_float;
}

static void setValue(ir::PTXOperand& operand, uint64_t value)
{
	uint64_t mask = getMask(operand);

	operand.imm_uint = value & mask;
}

static void setValue(ir::PTXOperand& operand, double value)
{
	if(operand.type == ir::PTXOperand::f32)
	{
		operand.imm_single = value;
	}
	else
	{
		operand.imm_float = value;
	}
}

static ir::PTXOperand computeCvtValue(const ir::PTXInstruction& ptx);
static bool computeSetPValue(ir::PTXOperand&, const ir::PTXInstruction& ptx);

static bool computeValue(ir::PTXOperand& result, const ir::PTXInstruction& ptx)
{
	result = ir::PTXOperand(ir::PTXOperand::Immediate, ptx.d.type);

	switch(ptx.opcode)
	{
	case ir::PTXInstruction::Add:
	{
		if(ir::PTXOperand::isFloat(ptx.type))
		{
			double a = getDouble(ptx.a);
			double b = getDouble(ptx.b);

			setValue(result, a + b);
			break;
		}
		
		uint64_t a = getValue(ptx.a);
		uint64_t b = getValue(ptx.b);

		setValue(result, a + b);
		break;
	}
	case ir::PTXInstruction::And:
	{
		uint64_t a = getValue(ptx.a);
		uint64_t b = getValue(ptx.b);

		setValue(result, a & b);
		break;
	}
	case ir::PTXInstruction::Cvt:
	{
		result = computeCvtValue(ptx);
		break;
	}
	case ir::PTXInstruction::Mov:
	{
		setValue(result, getValue(ptx.a));
		break;
	}
	case ir::PTXInstruction::Mul:
	{
		if(!(ptx.modifier & ir::PTXInstruction::Lo)) return false;

		if(ir::PTXOperand::isFloat(ptx.type))
		{
			double a = getDouble(ptx.a);
			double b = getDouble(ptx.b);

			setValue(result, a * b);
			break;
		}
		
		uint64_t a = getValue(ptx.a);
		uint64_t b = getValue(ptx.b);

		setValue(result, a * b);
		break;
	}
	case ir::PTXInstruction::Neg:
	{
		if(ir::PTXOperand::isFloat(ptx.type))
		{
			double a = getDouble(ptx.a);

			setValue(result, -a);
			break;
		}
		
		uint64_t a = getValue(ptx.a);

		setValue(result, -a);
		break;
	}
	case ir::PTXInstruction::SetP:
	{
		return computeSetPValue(result, ptx);
		break;
	}
	case ir::PTXInstruction::Shr:
	{
		uint64_t a = getValue(ptx.a);
		uint64_t b = getValue(ptx.b);

		setValue(result, a >> b);
		break;
	}
	case ir::PTXInstruction::Sub:
	{
		if(ir::PTXOperand::isFloat(ptx.type))
		{
			double a = getDouble(ptx.a);
			double b = getDouble(ptx.b);

			setValue(result, a - b);
			break;
		}
		
		uint64_t a = getValue(ptx.a);
		uint64_t b = getValue(ptx.b);

		setValue(result, a - b);
		break;
	}
	default:
	{
		return false;
	}
	}

	return true;
}

static void updateUses(iterator block, ir::Instruction::RegisterType registerId,
	const ir::PTXOperand& value, BlockSet& visited)
{
	typedef analysis::DataflowGraph::RegisterPointerVector
		RegisterPointerVector;

	if(!visited.insert(block).second) return;

	// phi uses
	bool replacedPhi = false;
	bool anyPhis     = false;
	ir::Instruction::RegisterType newRegisterId = 0;
	
	for(auto phi = block->phis().begin(); phi != block->phis().end(); ++phi)
	{
		if(phi->s.size() != 1)
		{
			for(auto source = phi->s.begin(); source != phi->s.end(); ++source)
			{
				if(source->id == registerId)
				{
					anyPhis = true;
					report("    could not remove " << phi->toString());
					break;
				}
			}
									
			continue;
		}
		
		for(auto source = phi->s.begin(); source != phi->s.end(); ++source)
		{
			if(source->id == registerId)
			{
				newRegisterId = phi->d.id;
				block->phis().erase(phi);
			    	
				auto livein = block->aliveIn().find(registerId);
			    
			    assert(livein != block->aliveIn().end());
				block->aliveIn().erase(livein);
				
				report("    removed " << phi->toString());
				replacedPhi = true;
				break;
			}
		}

		if(replacedPhi)
		{
			break;
		}
	}

	if(replacedPhi)
	{
		BlockSet visited;
		
		updateUses(block, newRegisterId, value, visited);
	}
	
	// local uses
	for(auto instruction = block->instructions().begin();
		instruction != block->instructions().end(); ++instruction)
	{
		auto ptx = static_cast<ir::PTXInstruction*>(instruction->i);
	
		RegisterPointerVector newSources;
	
		for(auto source = instruction->s.begin(); source !=
			instruction->s.end(); ++source)
		{
			if(*source->pointer == registerId)
			{
				report("    updated use by '" << ptx->toString()
					<< "', of r" << registerId); 
		
				replaceOperand(*ptx, registerId, value);
			}
			else
			{
				newSources.push_back(*source);
			}
		}

		instruction->s = std::move(newSources);
	}
		
	if(!anyPhis)
	{
		auto livein = block->aliveIn().find(registerId);
   
		if(livein != block->aliveIn().end())
		{
			block->aliveIn().erase(livein);
																
			report("    removed from live-in set of block " <<
				block->id());
		}
	}

	auto liveout = block->aliveOut().find(registerId);

	if(liveout == block->aliveOut().end()) return;
	
	// uses by successors
	bool anyUsesBySuccessors = false;
	
	for(auto successor = block->successors().begin();
		successor != block->successors().end(); ++successor)
	{
		auto livein = (*successor)->aliveIn().find(registerId);
		
		if(livein == (*successor)->aliveIn().end()) continue;

		updateUses(*successor, registerId, value, visited);
		
		livein = (*successor)->aliveIn().find(registerId);
		
		if(livein == (*successor)->aliveIn().end()) continue;

		anyUsesBySuccessors = true;
	}

	if(!anyUsesBySuccessors)
	{
		report("    removed from live-out set of BB_" << block->id());
		block->aliveOut().erase(liveout);
	}	
}

static bool propagateValueToSuccessors(analysis::DataflowGraph& dfg,
	BlockSet& blocks, InstructionVector::iterator instruction)
{
	if(!isOutputConstant(instruction))
	{
		return false;
	}
	
	auto ptx = static_cast<ir::PTXInstruction*>(instruction->i);
	
	if(ptx->isLoad())
	{
		return false;
	}

	// TODO support instructions with multiple destinations
	if(instruction->d.size() != 1)
	{
		return false;
	}

	report("   checking " << instruction->i->toString());
	
	// get the value 	
	ir::PTXOperand value;

	bool success = computeValue(value, *ptx);

	if(!success)
	{
		report("    could not determine the resulting value.");
		return false;
	}

	// send it to successors	
	auto registerId = *instruction->d.back().pointer;

	auto block = instruction->block;

	BlockSet visited;
	updateUses(block, registerId, value, visited);

	return true;
}

static bool isTrivialCvt(const ir::PTXInstruction& ptx)
{
	if(ptx.a.type == ptx.type) return true;

	if(ptx.modifier != 0) return false; 

	if(ir::PTXOperand::isInt(ptx.type))
	{
		if(ir::PTXOperand::isInt(ptx.type))
		{
			if(ir::PTXOperand::bytes(ptx.type) <=
				ir::PTXOperand::bytes(ptx.a.type))
			{
				return true;
			}
			
			if(!ir::PTXOperand::isSigned(ptx.type) &&
				!ir::PTXOperand::isSigned(ptx.a.type))
			{
				return true;
			}
		}
	}

	return false;
}

static ir::PTXOperand computeCvtValue(const ir::PTXInstruction& ptx)
{
	ir::PTXOperand result(ir::PTXOperand::Immediate, ptx.d.type);

	if(isTrivialCvt(ptx))
	{
		setValue(result, getValue(ptx.a));
	}

	return result;
}

static bool computeSetPValue(ir::PTXOperand& result,
	const ir::PTXInstruction& ptx)
{
	result = ir::PTXOperand(ir::PTXOperand::Immediate, ptx.d.type);

	uint64_t a = getValue(ptx.a);
	uint64_t b = getValue(ptx.b);

	switch(ptx.comparisonOperator)
	{
	case ir::PTXInstruction::Eq:
	{
		setValue(result, (uint64_t)(a == b));
		break;
	}
	default:
	{
		return false;
	}
	}

	return true;
}

}

