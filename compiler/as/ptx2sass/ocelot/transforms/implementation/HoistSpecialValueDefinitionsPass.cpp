/*	\file   HoistSpecialValueDefinitionsPass.cpp
	\date   Monday January 30, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the HoistSpecialValueDefinitionsPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/HoistSpecialValueDefinitionsPass.h>

#include <ocelot/analysis/interface/DominatorTree.h>

#include <ocelot/ir/interface/Module.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace transforms
{

HoistSpecialValueDefinitionsPass::HoistSpecialValueDefinitionsPass()
: KernelPass({"DominatorTreeAnalysis"},
	"HoistSpecialValueDefinitionsPass"),
  hoistSpecialRegisters(true), hoistSpecialMemoryOperations(true)
{

}

void HoistSpecialValueDefinitionsPass::initialize(const ir::Module& m)
{
	// empty
}

ir::Instruction::RegisterType computeNextRegister(ir::IRKernel& k)
{
	ir::PTXKernel& ptx = static_cast<ir::PTXKernel&>(k);
	
	return ptx.getUnusedRegister();
}

void HoistSpecialValueDefinitionsPass::runOnKernel(ir::IRKernel& k)
{
	report("Running Hoist-Special-Value-Definitions-Pass...");
	_nextRegister = computeNextRegister(k);
	
	report("  Finding all uses of special values...");
	for(ir::ControlFlowGraph::iterator block = k.cfg()->begin();
		block != k.cfg()->end(); ++block)
	{
		_findAllVariableUses(k, block);
	}
	
	report("  Hoisting all uses...");
	_hoistAllVariableUses(k);
	
	report("  Finding all uses of special address spaces...");
	for(ir::ControlFlowGraph::iterator block = k.cfg()->begin();
		block != k.cfg()->end(); ++block)
	{
		_findAllAddressSpaceUses(k, block);
	}
	
	report("  Hoisting all uses...");
	_hoistAllAddressSpaceUses(k);
}	

void HoistSpecialValueDefinitionsPass::finalize()
{
	for(VariableMap::iterator variable = _variables.begin();
		variable != _variables.end(); ++variable)
	{
		delete variable->second;
	}
	
	_variables.clear();
	
	for(AddressSpaceMap::iterator space = _addressSpaces.begin();
		space != _addressSpaces.end(); ++space)
	{
		delete space->second;
	}
	
	_addressSpaces.clear();
		
	invalidateAnalysis("DataflowGraphAnalysis");
}

HoistSpecialValueDefinitionsPass::VariableDescriptor::VariableDescriptor(
	bool isRegister)
: _isRegister(isRegister)
{

}

HoistSpecialValueDefinitionsPass::VariableDescriptor::~VariableDescriptor()
{

}

bool HoistSpecialValueDefinitionsPass::VariableDescriptor::isRegister() const
{
	return _isRegister;
}

HoistSpecialValueDefinitionsPass::VariableUse::VariableUse(
	ir::ControlFlowGraph::iterator b,
	ir::ControlFlowGraph::instruction_iterator i,
	ir::PTXOperand* o)
: block(b), instruction(i), operand(o)
{

}

HoistSpecialValueDefinitionsPass::MemoryVariable::MemoryVariable(
	const std::string& n, ir::PTXInstruction::AddressSpace s)
: VariableDescriptor(false), name(n), space(s)
{

}

HoistSpecialValueDefinitionsPass::SpecialRegister::SpecialRegister(
	ir::PTXOperand::SpecialRegister s)
: VariableDescriptor(true), specialRegister(s)
{

}

static bool isOperandASpecialRegister(ir::PTXOperand& operand)
{
	return operand.addressMode == ir::PTXOperand::Special;
}

static bool isOperandAVariable(const ir::PTXInstruction& ptx,
	const ir::PTXOperand& operand)
{
	if(operand.addressMode == ir::PTXOperand::Address)
	{
		if(ptx.opcode == ir::PTXInstruction::Cvta)
		{
			return true;
		}
	}
	
	return false;
}

static bool isGlobalLocal(const ir::IRKernel& kernel,
	const ir::PTXInstruction& ptx,
	const ir::PTXOperand& operand)
{
	const ir::Global* global = kernel.module->getGlobal(operand.identifier);
	if(global != 0)
	{
		return global->space() == ir::PTXInstruction::Local;
	}
	
	return false;
}

static bool isOperandAnAddress(const ir::IRKernel& kernel,
	const ir::PTXInstruction& ptx,
	const ir::PTXOperand& operand)
{
	return (ptx.isMemoryInstruction() || ptx.opcode == ir::PTXInstruction::Cvta)
		&& (operand.addressMode == ir::PTXOperand::Address ||
		operand.addressMode == ir::PTXOperand::Indirect) && 
		!(ptx.addressSpace == ir::PTXInstruction::Local &&
		isGlobalLocal(kernel, ptx, operand));
}

static inline ir::PTXInstruction*
	toPTX(ir::ControlFlowGraph::instruction_iterator i)
{
	return static_cast<ir::PTXInstruction*>(*i);
}

void HoistSpecialValueDefinitionsPass::_findAllVariableUses(
	ir::IRKernel& kernel, ir::ControlFlowGraph::iterator block)
{
	// search all instructions for a match
	for(ir::ControlFlowGraph::instruction_iterator
		instruction = block->instructions.begin();
		instruction != block->instructions.end(); ++instruction)
	{
		ir::PTXInstruction& ptx = *toPTX(instruction);
		
		ir::PTXOperand* operands[] = {&ptx.a, &ptx.b, &ptx.c, &ptx.d};
		
		for(unsigned int i = 0; i < 4; ++i)
		{
			ir::PTXOperand& operand = *operands[i];
			
			if(hoistSpecialRegisters)
			{
				if(isOperandASpecialRegister(operand))
				{
					_addSpecialVariableUse(kernel, block,
						instruction, &operand);
				}
			}
			
			if(hoistSpecialMemoryOperations)
			{
				if(isOperandAVariable(ptx, operand))
				{
					_addMemoryVariableUse(kernel, block, instruction, &operand);
				}
			}
		}
	}
}

void HoistSpecialValueDefinitionsPass::_findAllAddressSpaceUses(
	ir::IRKernel& kernel, ir::ControlFlowGraph::iterator block)
{
	// search all instructions for a match
	for(ir::ControlFlowGraph::instruction_iterator
		instruction = block->instructions.begin();
		instruction != block->instructions.end(); ++instruction)
	{
		ir::PTXInstruction& ptx = *toPTX(instruction);
		
		ir::PTXOperand* operands[] = { &ptx.a, &ptx.b, &ptx.c, &ptx.d };
		
		for(unsigned int i = 0; i < 4; ++i)
		{
			ir::PTXOperand& operand = *operands[i];
					
			if(hoistSpecialMemoryOperations)
			{
				if( ptx.addressSpace != ir::PTXInstruction::Global &&
					ptx.addressSpace != ir::PTXInstruction::Generic &&
					ptx.addressSpace != ir::PTXInstruction::Param)
				{
					if(isOperandAnAddress(kernel, ptx, operand))
					{
						_addAddressSpaceUse(kernel, block,
							instruction, &operand);
					}
				}
			}
		}
	}
}

void HoistSpecialValueDefinitionsPass::_hoistAllVariableUses(ir::IRKernel& k)
{
	Analysis* a = getAnalysis("DominatorTreeAnalysis");
	assert(a != 0);
	
	analysis::DominatorTree* dominatorTree =
		static_cast<analysis::DominatorTree*>(a);
	
	for(VariableMap::iterator variable = _variables.begin();
		variable != _variables.end(); ++variable)
	{
		variable->second->hoistAllUses(this, k, dominatorTree, _nextRegister);
	}
}

void HoistSpecialValueDefinitionsPass::_hoistAllAddressSpaceUses(
	ir::IRKernel& k)
{
	Analysis* a = getAnalysis("DominatorTreeAnalysis");
	assert(a != 0);
	
	analysis::DominatorTree* dominatorTree =
		static_cast<analysis::DominatorTree*>(a);
		
	for(AddressSpaceMap::iterator space = _addressSpaces.begin();
		space != _addressSpaces.end(); ++space)
	{
		space->second->hoistAllUses(this, k, dominatorTree, _nextRegister);
	}
}

std::string getName(const ir::PTXOperand& operand)
{
	if(operand.addressMode == ir::PTXOperand::Special)
	{
		return operand.toString();
	}
	
	return operand.identifier;
}

void HoistSpecialValueDefinitionsPass::_addSpecialVariableUse(
	ir::IRKernel& kernel, ir::ControlFlowGraph::iterator block,
	ir::ControlFlowGraph::instruction_iterator ptx, ir::PTXOperand* operand)
{
	std::string name = getName(*operand);

	VariableMap::iterator variable = _variables.find(name);
	
	if(variable == _variables.end())
	{
		variable = _variables.insert(std::make_pair(name,
			new SpecialRegister(operand->special))).first;
	}
	
	report("   Found use of special register: " << toPTX(ptx)->toString()
		<< " (" << block->label() << ")");
	
	variable->second->uses.push_back(VariableUse(block, ptx, operand));	
}

static ir::PTXInstruction::AddressSpace getSpace(ir::IRKernel& kernel,
	ir::PTXInstruction& ptx, ir::PTXOperand& operand)
{
	if(ptx.isMemoryInstruction())
	{
		return ptx.addressSpace;
	}
	
	if(operand.addressMode == ir::PTXOperand::Address)
	{
		ir::Kernel::LocalMap::iterator local =
			kernel.locals.find(operand.identifier);
		if(local != kernel.locals.end())
		{
			return local->second.space;
		}
		
		ir::Kernel::ParameterMap::iterator arg =
			kernel.parameters.find(operand.identifier);
		if(arg != kernel.parameters.end())
		{
			return ir::PTXInstruction::Param;
		}
		
		if(kernel.getParameter(operand.identifier) != 0)
		{
			return ir::PTXInstruction::Param;
		}
		
		const ir::Global* global = kernel.module->getGlobal(operand.identifier);
		if(global != 0)
		{
			return global->space();
		}
	}
	
	return ir::PTXInstruction::Generic;
}

void HoistSpecialValueDefinitionsPass::_addMemoryVariableUse(
	ir::IRKernel& kernel, ir::ControlFlowGraph::iterator block,
	ir::ControlFlowGraph::instruction_iterator ptx, ir::PTXOperand* operand)
{
	std::string name = getName(*operand);

	VariableMap::iterator variable = _variables.find(name);
	
	if(variable == _variables.end())
	{
		variable = _variables.insert(std::make_pair(name,
			new MemoryVariable(name,
				getSpace(kernel, *toPTX(ptx), *operand)))).first;
	}
	
	report("   Found use of special address: " << toPTX(ptx)->toString()
		<< " (" << block->label() << ")");
	
	variable->second->uses.push_back(VariableUse(block, ptx, operand));	
}

void HoistSpecialValueDefinitionsPass::_addAddressSpaceUse(ir::IRKernel& kernel,
	ir::ControlFlowGraph::iterator block,
	ir::ControlFlowGraph::instruction_iterator ptx,
	ir::PTXOperand* operand)
{
	AddressSpaceMap::iterator space = _addressSpaces.find(
		toPTX(ptx)->addressSpace);
	
	if(space == _addressSpaces.end())
	{
		space = _addressSpaces.insert(std::make_pair(toPTX(ptx)->addressSpace,
			new AddressSpace(toPTX(ptx)->addressSpace))).first;
	}
	
	report("   Found use of non-global address space: "
		<< ir::PTXInstruction::toString(toPTX(ptx)->addressSpace)
		<< " (" << toPTX(ptx)->toString() << ")"
		<< " (" << block->label() << ")");
	
	space->second->uses.push_back(VariableUse(block, ptx, operand));	
}

static void insertBeforeTerminator(ir::ControlFlowGraph::iterator block,
	ir::PTXInstruction* instruction)
{
	bool hasTerminator = false;
	
	if(!block->instructions.empty())
	{
		ir::PTXInstruction* possibleTerminator =
			static_cast<ir::PTXInstruction*>(block->instructions.back());
	
		hasTerminator = possibleTerminator->isBranch();
	}
	
	ir::ControlFlowGraph::instruction_iterator position =
		block->instructions.end();

	if(hasTerminator) --position;
	
	block->instructions.insert(position, instruction);
}

ir::ControlFlowGraph::iterator
	HoistSpecialValueDefinitionsPass::VariableDescriptor::getDominatorOfAllUses(
	KernelPass* pass, ir::IRKernel& k, analysis::DominatorTree*& dominatorTree)
{
	ir::ControlFlowGraph::iterator dominator = uses.front().block;

	for(VariableUseVector::iterator use = ++uses.begin();
		use != uses.end(); ++use)
	{
		dominator = dominatorTree->getCommonDominator(dominator, use->block);
	}
	
	report("    to " << dominator->label());
	
	// Create a new block if the entry block is the dominator
	if(dominator == k.cfg()->get_entry_block())
	{
		dominator = k.cfg()->split_edge(
			dominator->get_fallthrough_edge(),
			ir::BasicBlock(k.cfg()->newId())).first->tail;
			
		// invalidate the dominator tree
		pass->invalidateAnalysis("DominatorTreeAnalysis");
		
		Analysis* a = pass->getAnalysis("DominatorTreeAnalysis");
		assert(a != 0);
	
		dominatorTree = static_cast<analysis::DominatorTree*>(a);		
	}
	
	return dominator;
}

void HoistSpecialValueDefinitionsPass::MemoryVariable::hoistAllUses(
	KernelPass* pass, ir::IRKernel& k, analysis::DominatorTree*& dominatorTree,
	ir::Instruction::RegisterType& nextRegister)
{
	assert(!uses.empty());

	if(uses.size() == 1) return;
	
	report("   Hoisting uses of " << name);
	
	// Find a block that dominates all uses
	ir::ControlFlowGraph::iterator dominator =
		getDominatorOfAllUses(pass, k, dominatorTree);
			
	// Insert a definition of the memory
	ir::PTXInstruction* cvta = 0;
	
	if(space != ir::PTXInstruction::Global)
	{
		cvta = new ir::PTXInstruction(ir::PTXInstruction::Cvta);
		cvta->addressSpace = space;
	}
	else
	{
		cvta = new ir::PTXInstruction(ir::PTXInstruction::Mov);
	}
	
	cvta->type = ir::PTXOperand::u64;
	
	cvta->d = ir::PTXOperand(ir::PTXOperand::Register, ir::PTXOperand::u64,
		nextRegister++);
	cvta->a = *uses.front().operand;
	
	report("    to " << dominator->label() << " (" << cvta->toString() << ")");	
	
	insertBeforeTerminator(dominator, cvta);
	
	report("     handling users...");
	
	// Convert all uses into generic accesses to this location
	for(VariableUseVector::iterator use = uses.begin();
		use != uses.end(); ++use)
	{
		report("      converting '"
			<< toPTX(use->instruction)->toString() << "' to");
		
		if(toPTX(use->instruction)->isMemoryInstruction())
		{
			toPTX(use->instruction)->addressSpace = ir::PTXInstruction::Generic;
		}
		else if(toPTX(use->instruction)->opcode == ir::PTXInstruction::Cvta)
		{
			toPTX(use->instruction)->opcode = ir::PTXInstruction::Mov;
		}
		
		*use->operand = cvta->d;
		
		report("       '" << toPTX(use->instruction)->toString() << "'.");
	}
}

void HoistSpecialValueDefinitionsPass::SpecialRegister::hoistAllUses(
	KernelPass* pass, ir::IRKernel& k, analysis::DominatorTree*& dominatorTree,
	ir::Instruction::RegisterType& nextRegister)
{
	assert(!uses.empty());

	if(uses.size() == 1) return;
	
	report("   Hoisting uses of " << ir::PTXOperand::toString(specialRegister));
	
	// Find a block that dominates all uses
	ir::ControlFlowGraph::iterator dominator =
		getDominatorOfAllUses(pass, k, dominatorTree);
		
	// Insert a definition of the register
	ir::PTXInstruction* cvt = new ir::PTXInstruction(ir::PTXInstruction::Cvt);
	
	cvt->type = toPTX(uses.front().instruction)->type;
	cvt->d = ir::PTXOperand(ir::PTXOperand::Register, cvt->type,
		nextRegister++);
	cvt->a = *uses.front().operand;
	
	report("    to " << dominator->label() << " (" << cvt->toString() << ")");	
	
	insertBeforeTerminator(dominator, cvt);
	
	report("     handling users...");
	
	// Convert all uses into register reads
	for(VariableUseVector::iterator use = uses.begin();
		use != uses.end(); ++use)
	{
		report("      converting '"
			<< toPTX(use->instruction)->toString() << "' to");
		
		*use->operand = cvt->d;
		
		report("       '" << toPTX(use->instruction)->toString() << "'.");
	}
}


HoistSpecialValueDefinitionsPass::AddressSpace::AddressSpace(
	ir::PTXInstruction::AddressSpace s)
: VariableDescriptor(false), space(s)
{

}

void HoistSpecialValueDefinitionsPass::AddressSpace::hoistAllUses(
	KernelPass* pass, ir::IRKernel& k, analysis::DominatorTree*& dominatorTree,
	ir::Instruction::RegisterType& nextRegister)
{
	assert(!uses.empty());

	if(uses.size() == 1) return;
	
	report("   Hoisting uses of " << ir::PTXInstruction::toString(space));
	
	// Find a block that dominates all uses
	ir::ControlFlowGraph::iterator dominator =
		getDominatorOfAllUses(pass, k, dominatorTree);
	
	// Insert a definition of the address space
	ir::PTXInstruction* cvta = new ir::PTXInstruction(ir::PTXInstruction::Cvta);
	
	cvta->type = ir::PTXOperand::u64;
	cvta->addressSpace = toPTX(uses.front().instruction)->addressSpace;
	cvta->d = ir::PTXOperand(ir::PTXOperand::Register, ir::PTXOperand::u64,
		nextRegister++);
	cvta->a = ir::PTXOperand(0ULL, ir::PTXOperand::u64);
	
	report("    to " << dominator->label() << " (" << cvta->toString() << ")");	
	
	insertBeforeTerminator(dominator, cvta);
	
	report("     handling users...");
	
	// Convert all uses into generic accesses to this location
	for(VariableUseVector::iterator use = uses.begin();
		use != uses.end(); ++use)
	{
		report("      converting '"
			<< toPTX(use->instruction)->toString() << "' to");
		
		{
			toPTX(use->instruction)->addressSpace = ir::PTXInstruction::Generic;
			
			ir::PTXOperand offset;
			
			if(use->operand->addressMode == ir::PTXOperand::Address)
			{
				ir::PTXInstruction* mov = new ir::PTXInstruction(
					ir::PTXInstruction::Mov);
				
				mov->type = ir::PTXOperand::u64;
				mov->d = ir::PTXOperand(ir::PTXOperand::Register,
					ir::PTXOperand::u64, nextRegister++);
				mov->a = *use->operand;
			
				use->block->instructions.insert(use->instruction, mov);
				
				report("       '" << mov->toString() << "'.");
			
				offset = mov->d;
			}
			else
			{
				offset = *use->operand;
			}
			
			if(offset.addressMode == ir::PTXOperand::Immediate &&
				offset.imm_uint == 0)
			{
				*use->operand = cvta->d;
			}
			else
			{
				// add the base of the address space
				ir::PTXInstruction* add = new ir::PTXInstruction(
					ir::PTXInstruction::Add);

				add->type = ir::PTXOperand::u64;
				add->d = ir::PTXOperand(ir::PTXOperand::Register,
					ir::PTXOperand::u64, nextRegister++);
				add->a = offset;
				add->b = cvta->d;
				
				report("       '" << add->toString() << "'.");
				
				use->block->instructions.insert(use->instruction, add);
			
				*use->operand = add->d;
			}
		}
		
		if(toPTX(use->instruction)->opcode == ir::PTXInstruction::Cvta)
		{
			toPTX(use->instruction)->opcode = ir::PTXInstruction::Mov;
		}
		
		report("       '" << toPTX(use->instruction)->toString() << "'.");
	}
}

}

