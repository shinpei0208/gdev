/*! \file PTXToLLVMTranslator.h
	\date Wednesday July 29, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the PTXToLLVMTranslator class
*/

#ifndef PTX_TO_LLVM_TRANSLATOR_H_INCLUDED
#define PTX_TO_LLVM_TRANSLATOR_H_INCLUDED

// Ocelot Includes
#include <ocelot/translator/interface/Translator.h>

#include <ocelot/transforms/interface/Pass.h>

#include <ocelot/ir/interface/LLVMInstruction.h>
#include <ocelot/ir/interface/PTXKernel.h>

#include <ocelot/analysis/interface/DataflowGraph.h>

// Forward Declarations
namespace ir
{
	class PTXInstruction;
	class Instruction;
	class LLVMKernel;
	class PTXOperand;
	class ExternalFunctionSet;
}

namespace translator
{

/*! \brief A translator from PTX to LLVM */
class PTXToLLVMTranslator : public Translator
{
public:
	PTXToLLVMTranslator( OptimizationLevel l = NoOptimization,
		const ir::ExternalFunctionSet* s = 0 );
	~PTXToLLVMTranslator();

public:
	ir::Kernel* translate( const ir::Kernel* i );

public:
	ir::Kernel* translatedKernel() const;

protected:
	typedef std::vector< analysis::DataflowGraph::Register > 
		RegisterVector;
	typedef std::vector< std::string > StringVector;
	typedef std::unordered_set< std::string > StringSet;

protected:
	ir::LLVMKernel*                _llvmKernel;
	unsigned int                   _tempRegisterCount;
	unsigned int                   _tempCCRegisterCount;
	unsigned int                   _tempBlockCount;
	bool                           _usesTextures;
	StringSet                      _usedExternalCalls;
	RegisterVector                 _uninitialized;
	const ir::PTXKernel*           _ptx;
	analysis::DataflowGraph*       _dfg;
	const ir::ExternalFunctionSet* _externals;

protected:
	static ir::LLVMInstruction::DataType _translate( 
		ir::PTXOperand::DataType type );
	static ir::LLVMInstruction::AtomicOperation _translate( 
		ir::PTXInstruction::AtomicOperation operation,
		ir::PTXOperand::DataType type );
	static void _doubleWidth( ir::LLVMInstruction::DataType& v );
	static ir::LLVMInstruction::Comparison _translate( 
		ir::PTXInstruction::CmpOp, bool isInt, bool isSigned );
	
	static ir::LLVMInstruction::Type _getCtaContextType();
	static ir::LLVMInstruction::Operand _context();

protected:
	void _debug( const analysis::DataflowGraph::Block& b );
	void _debug( const analysis::DataflowGraph::Instruction& i );
	void _reportReads( const analysis::DataflowGraph::Instruction& i );
	void _reportWrites( const analysis::DataflowGraph::Instruction& i );
	void _check( ir::PTXInstruction::AddressSpace space,
		const ir::LLVMInstruction::Operand& address, 
		unsigned int bytes, bool isArgument, bool isGlobalLocal,
		unsigned int statement );
	void _addMemoryCheckingDeclarations();
	void _insertDebugSymbols();
	
protected:
	void _yield( unsigned int type, const ir::LLVMInstruction::Operand&
		continuation = ir::LLVMInstruction::Operand( "",
		ir::LLVMInstruction::Type( 
		ir::LLVMInstruction::I32, ir::LLVMInstruction::Type::Element ) ) );

	ir::LLVMInstruction::Operand _translate( const ir::PTXOperand& o );
	void _swapAllExceptName( ir::LLVMInstruction::Operand& o, 
		const ir::PTXOperand& i );

protected:
	void _translateInstructions();
	void _newBlock( const std::string& name );
	void _translate( const analysis::DataflowGraph::Instruction& i, 
		const analysis::DataflowGraph::Block& block );
	void _translate( const ir::PTXInstruction& i, 
		const analysis::DataflowGraph::Block& block );
	void _translateAbs( const ir::PTXInstruction& i );
	void _translateAdd( const ir::PTXInstruction& i );
	void _translateAddC( const ir::PTXInstruction& i );
	void _translateAnd( const ir::PTXInstruction& i );
	void _translateAtom( const ir::PTXInstruction& i );
	void _translateBar( const ir::PTXInstruction& i );
	void _translateBfe( const ir::PTXInstruction& i );
	void _translateBfi( const ir::PTXInstruction& i );
	void _translateBfind( const ir::PTXInstruction& i );
	void _translateBra( const ir::PTXInstruction& i, 
		const analysis::DataflowGraph::Block& block );
	void _translateBrev( const ir::PTXInstruction& i );
	void _translateBrkpt( const ir::PTXInstruction& i );
	void _translateCall( const ir::PTXInstruction& i, 
		const analysis::DataflowGraph::Block& block );
	void _translateClz( const ir::PTXInstruction& i );
	void _translateCNot( const ir::PTXInstruction& i );
	void _translateCopySign( const ir::PTXInstruction& i );
	void _translateCos( const ir::PTXInstruction& i );
	void _translateCvt( const ir::PTXInstruction& i );
	void _translateCvta( const ir::PTXInstruction& i );
	void _translateDiv( const ir::PTXInstruction& i );
	void _translateEx2( const ir::PTXInstruction& i );
	void _translateExit( const ir::PTXInstruction& i );
	void _translateIsspacep( const ir::PTXInstruction& i );
	void _translateLd( const ir::PTXInstruction& i );
	void _translateLdu( const ir::PTXInstruction& i );
	void _translateLg2( const ir::PTXInstruction& i );
	void _translateMad24( const ir::PTXInstruction& i );
	void _translateMad( const ir::PTXInstruction& i );
	void _translateMax( const ir::PTXInstruction& i );
	void _translateMembar( const ir::PTXInstruction& i );
	void _translateMin( const ir::PTXInstruction& i );
	void _translateMov( const ir::PTXInstruction& i );
	void _translateMul24( const ir::PTXInstruction& i );
	void _translateMul( const ir::PTXInstruction& i );
	void _translateNeg( const ir::PTXInstruction& i );
	void _translateNot( const ir::PTXInstruction& i );
	void _translateOr( const ir::PTXInstruction& i );
	void _translatePmevent( const ir::PTXInstruction& i );
	void _translatePopc( const ir::PTXInstruction& i );
	void _translatePrmt( const ir::PTXInstruction& i );
	void _translateRcp( const ir::PTXInstruction& i );
	void _translateRed( const ir::PTXInstruction& i );
	void _translateRem( const ir::PTXInstruction& i );
	void _translateRet( const ir::PTXInstruction& i, 
		const analysis::DataflowGraph::Block& block );
	void _translateRsqrt( const ir::PTXInstruction& i );
	void _translateSad( const ir::PTXInstruction& i );
	void _translateSelP( const ir::PTXInstruction& i );
	void _translateSet( const ir::PTXInstruction& i );
	void _translateSetP( const ir::PTXInstruction& i );
	void _translateShl( const ir::PTXInstruction& i );
	void _translateShr( const ir::PTXInstruction& i );
	void _translateSin( const ir::PTXInstruction& i );
	void _translateSlCt( const ir::PTXInstruction& i );
	void _translateSqrt( const ir::PTXInstruction& i );
	void _translateSt( const ir::PTXInstruction& i );
	void _translateSub( const ir::PTXInstruction& i );
	void _translateSubC( const ir::PTXInstruction& i );
	void _translateSuld( const ir::PTXInstruction& i );
	void _translateSuq( const ir::PTXInstruction& i );
	void _translateSured( const ir::PTXInstruction& i );
	void _translateSust( const ir::PTXInstruction& i );
	void _translateTestP( const ir::PTXInstruction& i );
	void _translateTex( const ir::PTXInstruction& i );
	void _translateTxq( const ir::PTXInstruction& i );
	void _translateTrap( const ir::PTXInstruction& i );
	void _translateVote( const ir::PTXInstruction& i );
	void _translateXor( const ir::PTXInstruction& i );

protected:
	void _bitcast( const ir::PTXInstruction& i );
	void _bitcast( const ir::PTXOperand& d, const ir::PTXOperand& s );
	void _bitcast( const ir::LLVMInstruction::Operand& d, 
		const ir::LLVMInstruction::Operand& s, bool isSigned = false );
	void _convert( const ir::LLVMInstruction::Operand& d, 
		ir::PTXOperand::DataType dType, 
		const ir::LLVMInstruction::Operand& s, 
		ir::PTXOperand::DataType sType, int modifier = 0 );
	void _flushToZero( const ir::LLVMInstruction::Operand& d, 
		const ir::LLVMInstruction::Operand& a );
	void _saturate( const ir::LLVMInstruction::Operand& d, 
		const ir::LLVMInstruction::Operand& a );
	void _floatToIntSaturate( const ir::LLVMInstruction::Operand& d, 
		const ir::LLVMInstruction::Operand& ftoint,
		const ir::LLVMInstruction::Operand& f, bool isSigned );

	void _trunc( const ir::LLVMInstruction::Operand& d, 
		const ir::LLVMInstruction::Operand& a );
	void _nearbyint( const ir::LLVMInstruction::Operand& d, 
		const ir::LLVMInstruction::Operand& a );
	void _floor( const ir::LLVMInstruction::Operand& d, 
		const ir::LLVMInstruction::Operand& a );
	void _ceil( const ir::LLVMInstruction::Operand& d, 
		const ir::LLVMInstruction::Operand& a );

protected:
	std::string _tempRegister();
	std::string _loadSpecialRegister( 
		ir::PTXOperand::SpecialRegister s, 
		ir::PTXOperand::VectorIndex index );
	ir::LLVMInstruction::Operand _getMemoryBasePointer( 
	 	ir::PTXInstruction::AddressSpace space,
	 	bool isGlobalLocal, bool isArgument );
	ir::LLVMInstruction::Operand _getMemoryExtent( 
	 	ir::PTXInstruction::AddressSpace space );
	ir::LLVMInstruction::Operand _getAddressableVariablePointer( 
		ir::PTXInstruction::AddressSpace space,
		const ir::PTXOperand& o );
	ir::LLVMInstruction::Operand _getAddressableGlobalPointer( 
		const ir::PTXOperand& o );
	ir::LLVMInstruction::Operand _getLoadOrStorePointer( 
		const ir::PTXOperand& o, 
		ir::PTXInstruction::AddressSpace space, 
		ir::LLVMInstruction::DataType type, unsigned int vector );
	
	ir::LLVMInstruction::Operand _destination( 
		const ir::PTXInstruction& i, bool pd = false );
	ir::LLVMInstruction::Operand _destinationCC( 
		const ir::PTXInstruction& i );
	ir::LLVMInstruction::Operand _conditionCodeRegister( 
		const ir::PTXOperand& op );
	void _add( const ir::LLVMInstruction& i );
	
	void _initializeRegisters();

protected:
	void _addStackAllocations();
	void _addTextureCalls();
	void _addSurfaceCalls();
	void _addQueryCalls();
	void _addAtomicCalls();
	void _addMathCalls();
	void _addLLVMIntrinsics();
	void _addUtilityCalls();
	void _addKernelPrefix();
	void _addGlobalDeclarations();
	void _addExternalFunctionDeclarations();
	void _addKernelSuffix();
};

}

#endif

