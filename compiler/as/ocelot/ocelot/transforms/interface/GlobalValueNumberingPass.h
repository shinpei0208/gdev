/*! \file   GlobalValueNumberingPass.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday July 10, 2012
	\brief  The header file for the GlobalValueNumberingPass class.
*/

#pragma once

// Ocelot Includes
#include <ocelot/transforms/interface/Pass.h>

#include <ocelot/analysis/interface/DataflowGraph.h>

namespace transforms
{

/*! \brief Perform global value numbering on a PTX kernel.
	
	Based on the LLVM implementation in GVN.cpp.
 */
class GlobalValueNumberingPass : public KernelPass
{
public:
	/*! \brief Create the pass, create dependencies */
	GlobalValueNumberingPass(bool eliminateInstructions =true);

public:
	/*! \brief Initialize the pass using a specific module */
	void initialize(const ir::Module& m);
	/*! \brief Run the pass on a specific kernel in the module */
	void runOnKernel(ir::IRKernel& k);		
	/*! \brief Finalize the pass */
	void finalize();

public:
	/*! \brief Should the pass attempt to eliminate redundant instructions,
		or just renumber values */
	bool eliminateInstructions;

private:
	typedef unsigned int Number;
	typedef analysis::DataflowGraph::instruction_iterator InstructionIterator;
	typedef analysis::DataflowGraph::phi_iterator         PhiIterator;
	typedef analysis::DataflowGraph::iterator             BlockIterator;
	typedef analysis::DataflowGraph::BlockPointerVector   BlockPointerVector;
	typedef ir::Instruction::RegisterType                 Register;

	typedef analysis::DataflowGraph::register_pointer_iterator
		RegisterPointerIterator;

	class Expression
	{
	public:
		ir::PTXInstruction::Opcode opcode;
		ir::PTXOperand::DataType   type;
		
		Number arguments[5];
		
	public:
		Expression(ir::PTXInstruction::Opcode
			oc = ir::PTXInstruction::Invalid_Opcode, 
			ir::PTXOperand::DataType t = ir::PTXOperand::TypeSpecifier_invalid);
	
	public:
		bool operator==(const Expression& eq) const;
				
	};
	
	class Immediate
	{
	public:
		ir::PTXOperand::DataType type;
		ir::PTXU64               value;
		
	public:
		Immediate(ir::PTXOperand::DataType type, ir::PTXU64 value);

	public:
		bool operator==(const Immediate& imm) const;
	};
	
	class Indirect
	{
	public:
		Register reg;
		int      offset;
		
	public:
		Indirect(Register reg, int offset);

	public:
		bool operator==(const Indirect& imm) const;
	};
	
	class SpecialValue
	{
	public:
		ir::PTXOperand::SpecialRegister special;
		ir::PTXOperand::VectorIndex     vectorIndex;
		
	public:
		SpecialValue(ir::PTXOperand::SpecialRegister special,
			ir::PTXOperand::VectorIndex vectorIndex);

	public:
		bool operator==(const SpecialValue& imm) const;
	};

	class ExpressionHash
	{
	public:
		inline size_t operator()(const Expression& e) const
		{
			return e.opcode ^ e.type ^ e.arguments[0] ^
				e.arguments[1] ^ e.arguments[2] ^ e.arguments[3] ^
				e.arguments[4];
		}
	};

	class ImmediateHash
	{
	public:
		inline size_t operator()(const Immediate& i) const
		{
			return (i.type << 16) ^ i.value;
		}
	};

	class IndirectHash
	{
	public:
		inline size_t operator()(const Indirect& i) const
		{
			return (i.reg) ^ (i.offset << 16);
		}
	};

	class SpecialValueHash
	{
	public:
		inline size_t operator()(const SpecialValue& s) const
		{
			return (s.special << 4) ^ s.vectorIndex;
		}
	};

	class GeneratingInstruction
	{
	public:
		GeneratingInstruction(const InstructionIterator& it);
		GeneratingInstruction(const PhiIterator& it);
		GeneratingInstruction(bool valid);
		
	public:
		std::string toString() const;
		
	public:
		InstructionIterator instruction;
		PhiIterator         phi;

	public:
		bool isPhi;
		bool valid;
	};

	typedef std::list<GeneratingInstruction> GeneratingInstructionList;
	typedef std::vector<InstructionIterator> InstructionVector;

	typedef std::unordered_map<ir::Instruction::RegisterType, Number>
		ValueToNumberMap;
	typedef std::unordered_map<Expression, Number, ExpressionHash>
		ExpressionToNumberMap;
	typedef std::unordered_map<Immediate, Number, ImmediateHash>
		ImmediateToNumberMap;
	typedef std::unordered_map<Indirect, Number, IndirectHash>
		IndirectToNumberMap;
	typedef std::unordered_map<SpecialValue, Number, SpecialValueHash>
		SpecialValueToNumberMap;
	typedef std::unordered_map<Number, GeneratingInstructionList>
		NumberToGeneratingInstructionMap;

	enum PredefinedNumbers
	{
		UnsetNumber   = (Number)-1,
		InvalidNumber = (Number)-2
	};

private:
	bool _numberThenMergeIdenticalValues(ir::IRKernel& k);

	BlockPointerVector _depthFirstTraversal(ir::IRKernel& k);
	bool _processBlock(const BlockIterator& block);
	void _clearValueAssignments();

private:
	bool _processPhi(const PhiIterator& phi);
	bool _processInstruction(const InstructionIterator& instruction);
	bool _processLoad(const InstructionIterator& instruction);
	void _processEliminatedInstructions();
	
private:
	bool _isSimpleLoad(const InstructionIterator& instruction);
	
	Number _getNextNumber();
	Number _lookupExistingOrCreateNewNumber(
		const InstructionIterator& instruction);
	Number _lookupExistingOrCreateNewNumber(const PhiIterator& phi);
	Number _lookupExistingOrCreateNewNumber(const ir::PTXOperand& operand);
	
	void _setGeneratingInstruction(Number n,
		const InstructionIterator& instruction);
	void _setGeneratingInstruction(Number n, const PhiIterator& phi);
	GeneratingInstruction _findGeneratingInstruction(Number n,
		const InstructionIterator& instruction);
	void _eliminateInstruction(
		const GeneratingInstruction& generatingInstruction,
		const InstructionIterator& instruction);
	void _updateDataflow(const BlockIterator& instruction,
		const RegisterPointerIterator& replacedValue,
		const RegisterPointerIterator& generatedValue);
	bool _couldAliasStore(
		const GeneratingInstruction& generatingInstruction,
		const InstructionIterator& instruction);
	
	Expression _createExpression(const InstructionIterator& instruction);
	Immediate  _createImmediate(const ir::PTXOperand& operand);
	
private:
	ValueToNumberMap                 _numberedValues;
	ExpressionToNumberMap            _numberedExpressions;
	ImmediateToNumberMap             _numberedImmediates;
	IndirectToNumberMap              _numberedIndirects;
	SpecialValueToNumberMap          _numberedSpecials;
	Number                           _nextNumber;
	NumberToGeneratingInstructionMap _generatingInstructions;
	InstructionVector                _eliminatedInstructions;

};

}



