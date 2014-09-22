/*	\file   HoistSpecialValueDefinitionsPass.h
	\date   Monday January 30, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the HoistSpecialValueDefinitionsPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

#include <ocelot/ir/interface/PTXKernel.h>

// Forward Declarations
namespace analysis { class DominatorTree; }

namespace transforms
{

/*! \brief Hoists definitions of special values to a dominating block

	Special values are known to be constant for the lifetime of a kernel.  This
	pass exploits this fact to hoist the definition of special values to create
	a single definition.  This pass reduces the number of special value accesses
	at the cost of increased register pressure.
	
	It is also possible to apply this to memory base addresses for special
	memory spaces.  This effectively converts all special memory accesses into
	global accesss.

*/
class HoistSpecialValueDefinitionsPass : public KernelPass
{
public:
	/*! \brief The default constructor */
	HoistSpecialValueDefinitionsPass();

public:
	/*! \brief Should this pass hoist special register accesses? */
	bool hoistSpecialRegisters;
	
	/*! \brief Should this pass hoist non-global base addresses? */
	bool hoistSpecialMemoryOperations;

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);

	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		

	/*! \brief Finalize the pass */
	void finalize();
	
private:
	/*! \brief Describes the use of a single variable/special */
	class VariableUse
	{
	public:
		VariableUse(ir::ControlFlowGraph::iterator b,
			ir::ControlFlowGraph::instruction_iterator i,
			ir::PTXOperand* o);
	
	public:
		ir::ControlFlowGraph::iterator             block;
		ir::ControlFlowGraph::instruction_iterator instruction;
		ir::PTXOperand*                            operand;
	};

	typedef std::vector<VariableUse> VariableUseVector;

	/*! \brief Describes all uses of a variable/special */
	class VariableDescriptor
	{
	public:
		VariableDescriptor(bool isRegister);
		virtual ~VariableDescriptor();
	
	public:
		virtual void hoistAllUses(KernelPass* pass, ir::IRKernel& k,
			analysis::DominatorTree*& t, ir::Instruction::RegisterType& r) = 0;
	
	public:
		ir::ControlFlowGraph::iterator getDominatorOfAllUses(KernelPass* pass,
			ir::IRKernel& k, analysis::DominatorTree*& t);
		bool isRegister() const;
	
	public:
		VariableUseVector uses;
	
	private:
		bool _isRegister;
	};
	
	/*! \brief Describes a variable stored in memory */
	class MemoryVariable : public VariableDescriptor
	{
	public:
		MemoryVariable(const std::string& name,
			ir::PTXInstruction::AddressSpace space);
	
	public:
		void hoistAllUses(KernelPass* pass, ir::IRKernel& k,
			analysis::DominatorTree*& t, ir::Instruction::RegisterType& r);

	public:
		std::string                      name;
		ir::PTXInstruction::AddressSpace space;
	};
	
	/*! \brief Describes a special register */
	class SpecialRegister : public VariableDescriptor
	{
	public:
		SpecialRegister(ir::PTXOperand::SpecialRegister specialRegister);
	
	public:
		void hoistAllUses(KernelPass* pass, ir::IRKernel& k,
			analysis::DominatorTree*& t, ir::Instruction::RegisterType& r);
	
	public:
		ir::PTXOperand::SpecialRegister specialRegister;
	};
	
	/*! \brief Describes an address space */
	class AddressSpace : public VariableDescriptor
	{
	public:
		AddressSpace(ir::PTXInstruction::AddressSpace space);
	
	public:
		void hoistAllUses(KernelPass* pass, ir::IRKernel& k,
			analysis::DominatorTree*& t, ir::Instruction::RegisterType& r);
	
	public:
		ir::PTXInstruction::AddressSpace space;
	};

	typedef std::unordered_map<std::string, VariableDescriptor*> VariableMap;
	typedef std::map<ir::PTXInstruction::AddressSpace,
		VariableDescriptor*> AddressSpaceMap;

private:
	void _findAllVariableUses(ir::IRKernel& k,
		ir::ControlFlowGraph::iterator b);
	void _findAllAddressSpaceUses(ir::IRKernel& k,
		ir::ControlFlowGraph::iterator b);
	
	void _hoistAllVariableUses(ir::IRKernel& k);
	void _hoistAllAddressSpaceUses(ir::IRKernel& k);
	
	void _addSpecialVariableUse(ir::IRKernel& kernel,
		ir::ControlFlowGraph::iterator block,
		ir::ControlFlowGraph::instruction_iterator ptx,
		ir::PTXOperand* operand);
	void _addMemoryVariableUse(ir::IRKernel& kernel,
		ir::ControlFlowGraph::iterator block,
		ir::ControlFlowGraph::instruction_iterator ptx,
		ir::PTXOperand* operand);
	void _addAddressSpaceUse(ir::IRKernel& kernel,
		ir::ControlFlowGraph::iterator block,
		ir::ControlFlowGraph::instruction_iterator ptx,
		ir::PTXOperand* operand);
	
private:
	AddressSpaceMap               _addressSpaces;
	VariableMap                   _variables;
	ir::Instruction::RegisterType _nextRegister;

};

}

