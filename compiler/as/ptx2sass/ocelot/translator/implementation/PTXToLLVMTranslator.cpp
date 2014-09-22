/*! \file PTXToLLVMTranslator.cpp
	\date Wednesday July 29, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The source file for the PTXToLLVMTranslator class
	\comment : Written with subdued haste
*/

#ifndef PTX_TO_LLVM_TRANSLATOR_CPP_INCLUDED
#define PTX_TO_LLVM_TRANSLATOR_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/translator/interface/PTXToLLVMTranslator.h>
#include <ocelot/ir/interface/ExternalFunctionSet.h>
#include <ocelot/ir/interface/LLVMInstruction.h>
#include <ocelot/ir/interface/LLVMKernel.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <ocelot/ir/interface/Module.h>
#include <ocelot/executive/interface/LLVMExecutableKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <climits>
#include <limits>

// Preprocessor Macros
#ifdef __i386__
#define USE_VECTOR_INSTRUCTIONS 0
#else
#define USE_VECTOR_INSTRUCTIONS 1
#endif

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace translator
{

PTXToLLVMTranslator::PTXToLLVMTranslator( OptimizationLevel l,
	const ir::ExternalFunctionSet* s ) 
	: Translator( ir::Instruction::PTX, ir::Instruction::LLVM, l, 
		{"DataflowGraphAnalysis"},
		"PTXToLLVMTranslator" ),
	_llvmKernel( 0 ), _tempRegisterCount( 0 ), _tempCCRegisterCount( 0 ),
	_tempBlockCount( 0 ), _usesTextures( false ), _externals( s )
{

}

PTXToLLVMTranslator::~PTXToLLVMTranslator()
{

}

ir::Kernel* PTXToLLVMTranslator::translate( const ir::Kernel* k )
{
	report( "Translating PTX kernel " << k->name );

	assertM( k->ISA == ir::Instruction::PTX, 
		"Kernel must a PTXKernel to translate to an LLVMKernel" );
	
	_tempRegisterCount = 0;
	_tempBlockCount = 0;
	_uninitialized.clear();
	_usedExternalCalls.clear();
	_usesTextures = false;
	
	_dfg = 0;
	
	_ptx = static_cast< const ir::PTXKernel* >( k );
	
	_llvmKernel = new ir::LLVMKernel( *k );

	analysis::Analysis* analysis = getAnalysis( "DataflowGraphAnalysis" );
	assert( analysis != 0 );
	
	_dfg = static_cast< analysis::DataflowGraph* >( analysis );
	
	_dfg->convertToSSAType( analysis::DataflowGraph::Minimal );
	
	_translateInstructions();
	_initializeRegisters();
	_addStackAllocations();
	_addKernelPrefix();
	_addKernelSuffix();
	_addGlobalDeclarations();
	_addExternalFunctionDeclarations();
		
	return _llvmKernel;
}

ir::Kernel* PTXToLLVMTranslator::translatedKernel() const
{
	return _llvmKernel;
}


ir::LLVMInstruction::DataType PTXToLLVMTranslator::_translate( 
	ir::PTXOperand::DataType type )
{
	switch( type )
	{
		case ir::PTXOperand::pred:
		{
			return ir::LLVMInstruction::I1;
			break;
		}			
		case ir::PTXOperand::b8: /* fall through */
		case ir::PTXOperand::u8: /* fall through */
		case ir::PTXOperand::s8:
		{
			return ir::LLVMInstruction::I8;
			break;
		}
		case ir::PTXOperand::b16: /* fall through */
		case ir::PTXOperand::s16: /* fall through */
		case ir::PTXOperand::u16:
		{
			return ir::LLVMInstruction::I16;
			break;
		}
		case ir::PTXOperand::b32: /* fall through */
		case ir::PTXOperand::u32: /* fall through */
		case ir::PTXOperand::s32:
		{
			return ir::LLVMInstruction::I32;
			break;
		}
		case ir::PTXOperand::b64: /* fall through */
		case ir::PTXOperand::s64: /* fall through */
		case ir::PTXOperand::u64:
		{
			return ir::LLVMInstruction::I64;
			break;
		}
		case ir::PTXOperand::f32:
		{
			return ir::LLVMInstruction::F32;
			break;
		}
		case ir::PTXOperand::f64:
		{
			return ir::LLVMInstruction::F64;
			break;
		}
		default:
		{
			assertM( false, "PTXOperand datatype " 
				+ ir::PTXOperand::toString( type ) 
				+ " not supported." );
		}
	}
	return ir::LLVMInstruction::InvalidDataType;
}

ir::LLVMInstruction::AtomicOperation PTXToLLVMTranslator::_translate( 
	ir::PTXInstruction::AtomicOperation operation,
	ir::PTXOperand::DataType type )
{
	switch( operation )
	{
		case ir::PTXInstruction::AtomicAnd:
		{
			return ir::LLVMInstruction::AtomicAnd;
		}
		case ir::PTXInstruction::AtomicOr:
		{
			return ir::LLVMInstruction::AtomicOr;
			break;
		}
		case ir::PTXInstruction::AtomicXor:
		{
			return ir::LLVMInstruction::AtomicXor;
			break;
		}
		case ir::PTXInstruction::AtomicCas:
		{
			assertM( false, "Cannot translate atomic CAS." );
			break;
		}
		case ir::PTXInstruction::AtomicExch:
		{
			return ir::LLVMInstruction::AtomicXchg;
			break;
		}
		case ir::PTXInstruction::AtomicAdd:
		{
			return ir::LLVMInstruction::AtomicAdd;
			break;
		}
		case ir::PTXInstruction::AtomicInc:
		{
			assertM( false, "Cannot translate atomic Inc." );
			break;
		}
		case ir::PTXInstruction::AtomicDec:
		{
			assertM( false, "Cannot translate atomic Dec." );
			break;
		}
		case ir::PTXInstruction::AtomicMin:
		{
			if( ir::PTXOperand::isSigned( type ) )
			{
				return ir::LLVMInstruction::AtomicMin;
			}
			else
			{
				return ir::LLVMInstruction::AtomicUmin;
			}
			break;
		}
		case ir::PTXInstruction::AtomicMax:
		{
			if( ir::PTXOperand::isSigned( type ) )
			{
				return ir::LLVMInstruction::AtomicMax;
			}
			else
			{
				return ir::LLVMInstruction::AtomicUmax;
			}
			break;
		}
		case ir::PTXInstruction::AtomicOperation_Invalid:
		{
			assertM( false, "Invalid atomic operation" );
		}
	}
	
	return ir::LLVMInstruction::InvalidAtomicOperation;
}	

void PTXToLLVMTranslator::_doubleWidth( ir::LLVMInstruction::DataType& t )
{
	switch( t )
	{
		case ir::LLVMInstruction::I1:
		{
			assertM( false, "Cannot double i1" );
			break;
		}
		case ir::LLVMInstruction::I8:
		{
			t = ir::LLVMInstruction::I16;
			break;
		}
		case ir::LLVMInstruction::I16:
		{
			t = ir::LLVMInstruction::I32;
			break;
		}
		case ir::LLVMInstruction::I32:
		{
			t = ir::LLVMInstruction::I64;
			break;
		}
		case ir::LLVMInstruction::I64:
		{
			t = ir::LLVMInstruction::I128;
			break;
		}
		case ir::LLVMInstruction::I128:
		{
			assertM( false, "Cannot double i128" );
			break;
		}
		case ir::LLVMInstruction::F32:
		{
			t = ir::LLVMInstruction::F64;
			break;
		}
		case ir::LLVMInstruction::F64:
		{
			t = ir::LLVMInstruction::F128;
			break;
		}
		case ir::LLVMInstruction::F128:
		{
			assertM( false, "Cannot double fp128" );
			break;
		}
		case ir::LLVMInstruction::InvalidDataType:
		{
			assertM( false, "Cannot double invalid data type" );
			break;
		}
	}
}

ir::LLVMInstruction::Comparison PTXToLLVMTranslator::_translate( 
	ir::PTXInstruction::CmpOp op, bool isInt, bool isSigned )
{
	if( isInt )
	{
		if( isSigned )
		{
			switch( op )
			{
				case ir::PTXInstruction::Eq: 
					return ir::LLVMInstruction::Eq; break;
				case ir::PTXInstruction::Ne: 
					return ir::LLVMInstruction::Ne; break;
				case ir::PTXInstruction::Lo: /* fall through */
				case ir::PTXInstruction::Lt: 
					return ir::LLVMInstruction::Slt; break;
				case ir::PTXInstruction::Ls: /* fall through */
				case ir::PTXInstruction::Le: 
					return ir::LLVMInstruction::Sle; break;
				case ir::PTXInstruction::Hi: /* fall through */
				case ir::PTXInstruction::Gt: 
					return ir::LLVMInstruction::Sgt; break;
				case ir::PTXInstruction::Hs: /* fall through */
				case ir::PTXInstruction::Ge: 
					return ir::LLVMInstruction::Sge; break;
				default: assertM( false, "Invalid comparison " 
					<< ir::PTXInstruction::toString( op ) 
					<< " for integer operand." );
			}
		}
		else
		{
			switch( op )
			{
				case ir::PTXInstruction::Eq: 
					return ir::LLVMInstruction::Eq; break;
				case ir::PTXInstruction::Ne: 
					return ir::LLVMInstruction::Ne; break;
				case ir::PTXInstruction::Lo: /* fall through */
				case ir::PTXInstruction::Lt: 
					return ir::LLVMInstruction::Ult; break;
				case ir::PTXInstruction::Ls: /* fall through */
				case ir::PTXInstruction::Le: 
					return ir::LLVMInstruction::Ule; break;
				case ir::PTXInstruction::Hi: /* fall through */
				case ir::PTXInstruction::Gt: 
					return ir::LLVMInstruction::Ugt; break;
				case ir::PTXInstruction::Hs: /* fall through */
				case ir::PTXInstruction::Ge: 
					return ir::LLVMInstruction::Uge; break;
				default: assertM( false, "Invalid comparison " 
					<< ir::PTXInstruction::toString( op ) 
					<< " for integer operand." );
			}
		}
	}
	else
	{
		switch( op )
		{
			case ir::PTXInstruction::Eq: 
				return ir::LLVMInstruction::Oeq; break;
			case ir::PTXInstruction::Ne: 
				return ir::LLVMInstruction::One; break;
			case ir::PTXInstruction::Lt: 
				return ir::LLVMInstruction::Olt; break;
			case ir::PTXInstruction::Le: 
				return ir::LLVMInstruction::Ole; break;
			case ir::PTXInstruction::Gt: 
				return ir::LLVMInstruction::Ogt; break;
			case ir::PTXInstruction::Ge: 
				return ir::LLVMInstruction::Oge; break;
			case ir::PTXInstruction::Lo: 
				return ir::LLVMInstruction::Olt; break;
			case ir::PTXInstruction::Ls: 
				return ir::LLVMInstruction::Ole; break;
			case ir::PTXInstruction::Hi: 
				return ir::LLVMInstruction::Ogt; break;
			case ir::PTXInstruction::Hs: 
				return ir::LLVMInstruction::Oge; break;
			case ir::PTXInstruction::Equ: 
				return ir::LLVMInstruction::Ueq; break;
			case ir::PTXInstruction::Neu: 
				return ir::LLVMInstruction::Une; break;
			case ir::PTXInstruction::Ltu: 
				return ir::LLVMInstruction::Ult; break;
			case ir::PTXInstruction::Leu: 
				return ir::LLVMInstruction::Ule; break;
			case ir::PTXInstruction::Gtu: 
				return ir::LLVMInstruction::Ugt; break;
			case ir::PTXInstruction::Geu: 
				return ir::LLVMInstruction::Uge; break;
			case ir::PTXInstruction::Num: 
				return ir::LLVMInstruction::Ord; break;
			case ir::PTXInstruction::Nan: 
				return ir::LLVMInstruction::Uno; break;
			default: assertM( false, "Invalid comparison " 
				<< ir::PTXInstruction::toString( op ) 
				<< " for floating point operand." );
		}
	}
	
	return ir::LLVMInstruction::True;
}

ir::LLVMInstruction::Type PTXToLLVMTranslator::_getCtaContextType()
{
	ir::LLVMInstruction::Type context;
	
	context.category = ir::LLVMInstruction::Type::Structure;
	context.members.resize( 11 );

	context.members[0].category = ir::LLVMInstruction::Type::Structure;
	context.members[0].label    = "%Dimension";

	context.members[1] = context.members[0];
	context.members[2] = context.members[0];
	context.members[3] = context.members[0];

	context.members[4].category = ir::LLVMInstruction::Type::Pointer;
	context.members[4].type     = ir::LLVMInstruction::I8;

	context.members[5]  = context.members[4];
	context.members[6]  = context.members[4];
	context.members[7]  = context.members[4];	
	context.members[8]  = context.members[4];
	context.members[9]  = context.members[4];

	context.members[10].category = ir::LLVMInstruction::Type::Element;
	context.members[10].type = ir::LLVMInstruction::I32;
	
	return context;
}

ir::LLVMInstruction::Operand PTXToLLVMTranslator::_context()
{
	ir::LLVMInstruction::Operand context;
	
	context.type.category = ir::LLVMInstruction::Type::Pointer;
	context.type.members.resize(1);
	context.type.members[0].category 
		= ir::LLVMInstruction::Type::Structure;
	context.type.members[0].label = "%LLVMContext";
	context.name = "%__ctaContext";
	
	return context;
}

void PTXToLLVMTranslator::_debug( const analysis::DataflowGraph::Block& b )
{
	if( optimizationLevel != DebugOptimization
		&& optimizationLevel != ReportOptimization ) return;
			
	ir::LLVMCall call;
	
	call.name = "@__ocelot_debug_block";
	
	call.parameters.resize( 2 );

	call.parameters[0] = _context();

	call.parameters[1].type.type = ir::LLVMInstruction::I32;
	call.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[1].constant = true;
	call.parameters[1].i32 = b.id();
	
	_add( call );
}

void PTXToLLVMTranslator::_debug( 
	const analysis::DataflowGraph::Instruction& i )
{
	if( optimizationLevel != DebugOptimization
		&& optimizationLevel != ReportOptimization ) return;

	ir::LLVMCall call;

	call.name = "@__ocelot_debug_instruction";
	
	call.parameters.resize( 2 );

	call.parameters[0] = _context();

	call.parameters[1].type.type = ir::LLVMInstruction::I64;
	call.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[1].constant = true;
	call.parameters[1].i64 = (long long unsigned int)i.i;

	_add( call );
}

void PTXToLLVMTranslator::_reportReads( 
	const analysis::DataflowGraph::Instruction& i )
{
	if( optimizationLevel != DebugOptimization ) return;

	ir::LLVMCall call;
	
	call.parameters.resize( 2 );
	
	call.parameters[0] = _context();
	
	for( analysis::DataflowGraph::RegisterPointerVector::const_iterator 
		r = i.s.begin(); r != i.s.end(); ++r )
	{
		call.parameters[1].type.type = _translate( r->type );
		call.parameters[1].type.category 
			= ir::LLVMInstruction::Type::Element;

		std::stringstream stream;
		stream << "%r" << *r->pointer;

		call.parameters[1].name = stream.str();

		switch( r->type )
		{
			case ir::PTXOperand::s8:
			{
				call.name = "@__ocelot_register_read_s8";
				break;
			}
			case ir::PTXOperand::s16:
			{
				call.name = "@__ocelot_register_read_s16";
				break;
			}
			case ir::PTXOperand::s32:
			{
				call.name = "@__ocelot_register_read_s32";
				break;
			}
			case ir::PTXOperand::s64:
			{
				call.name = "@__ocelot_register_read_s64";
				break;
			}
			case ir::PTXOperand::pred: /* fall through */
			{
				ir::LLVMZext extend;
				
				extend.a = call.parameters[1];
				call.parameters[1].name = _tempRegister();
				call.parameters[1].type.type = ir::LLVMInstruction::I8;
				extend.d = call.parameters[1];
				
				_add( extend );
			}
			case ir::PTXOperand::u8: /* fall through */
			case ir::PTXOperand::b8:
			{
				call.name = "@__ocelot_register_read_u8";
				break;
			}
			case ir::PTXOperand::u16: /* fall through */
			case ir::PTXOperand::b16:
			{
				call.name = "@__ocelot_register_read_u16";
				break;
			}
			case ir::PTXOperand::u32: /* fall through */
			case ir::PTXOperand::b32:
			{
				call.name = "@__ocelot_register_read_u32";
				break;
			}
			case ir::PTXOperand::u64: /* fall through */
			case ir::PTXOperand::b64:
			{
				call.name = "@__ocelot_register_read_u64";
				break;
			}
			case ir::PTXOperand::f32:
			{
				call.name = "@__ocelot_register_read_f32";
				break;
			}
			case ir::PTXOperand::f64:
			{
				call.name = "@__ocelot_register_read_f64";
				break;
			}
			default: assertM(false, "Invalid data type " 
				<< ir::PTXOperand::toString( r->type ) << ".");
		}

		_add( call );
	}
}

void PTXToLLVMTranslator::_reportWrites( 
	const analysis::DataflowGraph::Instruction& i )
{
	if( optimizationLevel != DebugOptimization ) return;

	ir::LLVMCall call;
	
	call.parameters.resize( 2 );

	call.parameters[0] = _context();
	
	for( analysis::DataflowGraph::RegisterPointerVector::const_iterator 
		r = i.d.begin(); r != i.d.end(); ++r )
	{
		call.parameters[1].type.type = _translate( r->type );
		call.parameters[1].type.category 
			= ir::LLVMInstruction::Type::Element;

		std::stringstream stream;
		stream << "%r" << *r->pointer;

		call.parameters[1].name = stream.str();

		switch( r->type )
		{
			case ir::PTXOperand::s8:
			{
				call.name = "@__ocelot_register_write_s8";
				break;
			}
			case ir::PTXOperand::s16:
			{
				call.name = "@__ocelot_register_write_s16";
				break;
			}
			case ir::PTXOperand::s32:
			{
				call.name = "@__ocelot_register_write_s32";
				break;
			}
			case ir::PTXOperand::s64:
			{
				call.name = "@__ocelot_register_write_s64";
				break;
			}
			case ir::PTXOperand::pred: /* fall through */
			{
				ir::LLVMZext extend;
				
				extend.a = call.parameters[1];
				call.parameters[1].name = _tempRegister();
				call.parameters[1].type.type = ir::LLVMInstruction::I8;
				extend.d = call.parameters[1];
				
				_add( extend );
			}
			case ir::PTXOperand::u8: /* fall through */
			case ir::PTXOperand::b8:
			{
				call.name = "@__ocelot_register_write_u8";
				break;
			}
			case ir::PTXOperand::u16: /* fall through */
			case ir::PTXOperand::b16:
			{
				call.name = "@__ocelot_register_write_u16";
				break;
			}
			case ir::PTXOperand::u32: /* fall through */
			case ir::PTXOperand::b32:
			{
				call.name = "@__ocelot_register_write_u32";
				break;
			}
			case ir::PTXOperand::u64: /* fall through */
			case ir::PTXOperand::b64:
			{
				call.name = "@__ocelot_register_write_u64";
				break;
			}
			case ir::PTXOperand::f32:
			{
				call.name = "@__ocelot_register_write_f32";
				break;
			}
			case ir::PTXOperand::f64:
			{
				call.name = "@__ocelot_register_write_f64";
				break;
			}
			default: assertM(false, "Invalid data type " 
				<< ir::PTXOperand::toString( r->type ) << ".");
		}

		_add( call );
	}
}

void PTXToLLVMTranslator::_check( ir::PTXInstruction::AddressSpace space,
	const ir::LLVMInstruction::Operand& address, unsigned int bytes,
	bool isArgument, bool isGlobalLocal, unsigned int statement )
{
	if( optimizationLevel != MemoryCheckOptimization 
		&& optimizationLevel != DebugOptimization
		&& optimizationLevel != ReportOptimization) return;

	ir::LLVMCall call;

	switch( space )
	{
		case ir::PTXInstruction::Shared:
		{
			call.name = "@__ocelot_check_shared_memory_access";
			break;
		}
		case ir::PTXInstruction::Global:
		{
			call.name = "@__ocelot_check_global_memory_access";
			break;
		}
		case ir::PTXInstruction::Local:
		{
			call.name = "@__ocelot_check_local_memory_access";
			break;
		}
		case ir::PTXInstruction::Const:
		{
			call.name = "@__ocelot_check_constant_memory_access";
			break;
		}
		case ir::PTXInstruction::Param:
		{
			if( isArgument )
			{
				call.name = "@__ocelot_check_argument_memory_access";
			}
			else
			{
				call.name = "@__ocelot_check_param_memory_access";
			}
			break;
		}
		default: 
		{
			call.name = "@__ocelot_check_generic_memory_access";
		}
	}
	
	call.parameters.resize( 4 );

	call.parameters[0] = _context();

	call.parameters[1].name = _tempRegister();
	call.parameters[1].type.type = ir::LLVMInstruction::I64;
	call.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	
	call.parameters[2].type.type = ir::LLVMInstruction::I32;
	call.parameters[2].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[2].constant = true;
	call.parameters[2].i32 = bytes;
	
	call.parameters[3].type.type = ir::LLVMInstruction::I32;
	call.parameters[3].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[3].constant = true;
	call.parameters[3].i32 = statement;
	
	ir::LLVMPtrtoint convert;
	
	convert.a = address;
	convert.d = call.parameters[1];
	
	_add( convert );
	_add( call );
}

void PTXToLLVMTranslator::_addMemoryCheckingDeclarations()
{
	if( optimizationLevel != MemoryCheckOptimization
		&& optimizationLevel != ReportOptimization
		&& optimizationLevel != DebugOptimization ) return;		
	
	ir::LLVMStatement check( ir::LLVMStatement::FunctionDeclaration );

	check.label = "__ocelot_check_global_memory_access";
	check.linkage = ir::LLVMStatement::InvalidLinkage;
	check.convention = ir::LLVMInstruction::DefaultCallingConvention;
	check.visibility = ir::LLVMStatement::Default;
	
	check.parameters.resize( 4 );

	check.parameters[0].type.category = ir::LLVMInstruction::Type::Pointer;
	check.parameters[0].type.members.resize(1);
	check.parameters[0].type.members[0].category 
		= ir::LLVMInstruction::Type::Structure;
	check.parameters[0].type.members[0].label = "%LLVMContext";

	check.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	check.parameters[1].type.type = ir::LLVMInstruction::I64;

	check.parameters[2].type.category = ir::LLVMInstruction::Type::Element;
	check.parameters[2].type.type = ir::LLVMInstruction::I32;

	check.parameters[3].type.category = ir::LLVMInstruction::Type::Element;
	check.parameters[3].type.type = ir::LLVMInstruction::I32;

	_llvmKernel->push_front( check );		

	check.label = "__ocelot_check_shared_memory_access";
	_llvmKernel->push_front( check );		

	check.label = "__ocelot_check_constant_memory_access";
	_llvmKernel->push_front( check );		

	check.label = "__ocelot_check_local_memory_access";
	_llvmKernel->push_front( check );
	
	check.label = "__ocelot_check_param_memory_access";
	_llvmKernel->push_front( check );
	
	check.label = "__ocelot_check_argument_memory_access";
	_llvmKernel->push_front( check );
	
	check.label = "__ocelot_check_generic_memory_access";
	_llvmKernel->push_front( check );
}

void PTXToLLVMTranslator::_insertDebugSymbols()
{
	if( optimizationLevel != DebugOptimization
		&& optimizationLevel != ReportOptimization ) return;		

	ir::LLVMStatement block( ir::LLVMStatement::FunctionDeclaration );

	block.label = "__ocelot_debug_block";
	block.linkage = ir::LLVMStatement::InvalidLinkage;
	block.convention = ir::LLVMInstruction::DefaultCallingConvention;
	block.visibility = ir::LLVMStatement::Default;
	
	block.parameters.resize(2);

	block.parameters[0].type.category = ir::LLVMInstruction::Type::Pointer;
	block.parameters[0].type.members.resize(1);
	block.parameters[0].type.members[0].category 
		= ir::LLVMInstruction::Type::Structure;
	block.parameters[0].type.members[0].label = "%LLVMContext";

	block.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	block.parameters[1].type.type = ir::LLVMInstruction::I32;

	_llvmKernel->push_front( block );

	ir::LLVMStatement instruction( ir::LLVMStatement::FunctionDeclaration );

	instruction.label = "__ocelot_debug_instruction";
	instruction.linkage = ir::LLVMStatement::InvalidLinkage;
	instruction.convention = ir::LLVMInstruction::DefaultCallingConvention;
	instruction.visibility = ir::LLVMStatement::Default;
	
	instruction.parameters.resize( 2 );
	instruction.parameters[0].type.category 
		= ir::LLVMInstruction::Type::Pointer;
	instruction.parameters[0].type.members.resize(1);
	instruction.parameters[0].type.members[0].category 
		= ir::LLVMInstruction::Type::Structure;
	instruction.parameters[0].type.members[0].label = "%LLVMContext";

	instruction.parameters[1].type.category 
		= ir::LLVMInstruction::Type::Element;
	instruction.parameters[1].type.type = ir::LLVMInstruction::I64;

	_llvmKernel->push_front( instruction );

	if( optimizationLevel != DebugOptimization ) return;		

	instruction.parameters.resize( 2 );
	instruction.parameters[0].type.category 
		= ir::LLVMInstruction::Type::Pointer;
	instruction.parameters[0].type.members.resize(1);
	instruction.parameters[0].type.members[0].category 
		= ir::LLVMInstruction::Type::Structure;
	instruction.parameters[0].type.members[0].label = "%LLVMContext";

	instruction.parameters[1].type.type = ir::LLVMInstruction::I8;
	instruction.label = "__ocelot_register_write_u8";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_write_s8";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_read_u8";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_read_s8";
	_llvmKernel->push_front( instruction );

	instruction.parameters[1].type.type = ir::LLVMInstruction::I16;
	instruction.label = "__ocelot_register_write_u16";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_write_s16";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_read_u16";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_read_s16";
	_llvmKernel->push_front( instruction );

	instruction.parameters[1].type.type = ir::LLVMInstruction::I32;
	instruction.label = "__ocelot_register_write_u32";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_write_s32";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_read_u32";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_read_s32";
	_llvmKernel->push_front( instruction );

	instruction.parameters[1].type.type = ir::LLVMInstruction::I64;
	instruction.label = "__ocelot_register_write_u64";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_write_s64";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_read_u64";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_read_s64";
	_llvmKernel->push_front( instruction );

	instruction.parameters[1].type.type = ir::LLVMInstruction::F32;
	instruction.label = "__ocelot_register_write_f32";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_read_f32";
	_llvmKernel->push_front( instruction );
	instruction.parameters[1].type.type = ir::LLVMInstruction::F64;
	instruction.label = "__ocelot_register_write_f64";
	_llvmKernel->push_front( instruction );
	instruction.label = "__ocelot_register_read_f64";
	_llvmKernel->push_front( instruction );
}
		
void PTXToLLVMTranslator::_yield( unsigned int type,
	const ir::LLVMInstruction::Operand& continuation )
{
	ir::LLVMBitcast bitcast;
	
	bitcast.a = _getMemoryBasePointer( ir::PTXInstruction::Local,
		false, false );

	bitcast.d.type = ir::LLVMInstruction::Type( 
		continuation.type.type, ir::LLVMInstruction::Type::Pointer );
	bitcast.d.name = _tempRegister();

	_add( bitcast );

	ir::LLVMStore store;

	store.d = bitcast.d;
	store.a = ir::LLVMInstruction::Operand( (ir::LLVMI32) type );
	store.a.type.type = continuation.type.type;
	
	_add( store );
	
	if( type == executive::LLVMExecutableKernel::TailCall 
		|| type == executive::LLVMExecutableKernel::NormalCall )
	{
		ir::LLVMGetelementptr get;
	
		get.a = bitcast.d;
		get.d = ir::LLVMInstruction::Operand( _tempRegister(), 
			ir::LLVMInstruction::Type( continuation.type.type,
			ir::LLVMInstruction::Type::Pointer ) );
		get.indices.push_back( 1 );
	
		_add( get );
	
		store.d = get.d;
		store.a = continuation;
	
		_add( store );
	}
}

ir::LLVMInstruction::Operand PTXToLLVMTranslator::_translate( 
	const ir::PTXOperand& o )
{
	ir::LLVMInstruction::Operand op( o.identifier );
	op.constant = o.addressMode == ir::PTXOperand::Immediate;
	
	op.type.type = _translate( o.type );
	
	if( o.vec == ir::PTXOperand::v1 )
	{
		op.type.category = ir::LLVMInstruction::Type::Element;
	}
	else
	{
		op.type.category = ir::LLVMInstruction::Type::Vector;
	}
	
	switch( o.addressMode )
	{
		case ir::PTXOperand::Register: /* fall through */
		case ir::PTXOperand::Indirect:
		{
			std::stringstream stream;
			stream << "%r" << o.reg;
			op.name = stream.str();
			break;
		}
		case ir::PTXOperand::Immediate:
		{
			switch( o.type )
			{
				case ir::PTXOperand::s8:  /* fall through */
				case ir::PTXOperand::s16: /* fall through */
				case ir::PTXOperand::s32: /* fall through */
				case ir::PTXOperand::s64: /* fall through */
				case ir::PTXOperand::u8:  /* fall through */
				case ir::PTXOperand::u16: /* fall through */
				case ir::PTXOperand::u32: /* fall through */
				case ir::PTXOperand::u64: /* fall through */
				case ir::PTXOperand::b8:  /* fall through */
				case ir::PTXOperand::b16: /* fall through */
				case ir::PTXOperand::b32: /* fall through */
				case ir::PTXOperand::b64:
				{
					op.i64 = o.imm_uint;
					break;
				}
				case ir::PTXOperand::f32:
				{
					op.f32 = o.imm_single;
					break;
				}
				case ir::PTXOperand::f64:
				{
					op.f64 = o.imm_float;
					break;
				}
				default:
				{
					assertM( false, "PTXOperand datatype " 
						+ ir::PTXOperand::toString( o.type ) 
						+ " not supported for immediate operand." );
				}				
			}
			break;
		}
		case ir::PTXOperand::Address:
		{
			assertM( false, 
				"Addressable variables require context" 
				<< " sensitive translation.  This is a bug in Ocelot." );
			break;
		}
		case ir::PTXOperand::Label:
		{
			assertM( false, "PTXOperand datatype " 
				+ ir::PTXOperand::toString( o.type ) 
				+ " not supported." );
			break;
		}
		case ir::PTXOperand::ArgumentList:
		{
			assertM( false, "Argument Lists not supported yet ." );
			break;
		}
		case ir::PTXOperand::FunctionName:
		{
			assertM( false, "Function Names not supported yet ." );
			break;
		}
		case ir::PTXOperand::Special:
		{
			op.name = _loadSpecialRegister( o.special, o.vIndex );
			break;
		}
		case ir::PTXOperand::BitBucket:
		{
			op.name = _tempRegister();
			break;
		}
		case ir::PTXOperand::Invalid:
		{
			assertM( false, "Cannot translate invalid PTX operand." );
		}
	}
	
	op.type.vector = o.vec;
	
	return op;
}

void PTXToLLVMTranslator::_swapAllExceptName( 
	ir::LLVMInstruction::Operand& o, const ir::PTXOperand& i )
{
	std::string temp = o.name;
	o = _translate( i );
	o.name = temp;
}

void PTXToLLVMTranslator::_translateInstructions()
{
	for( analysis::DataflowGraph::const_iterator 
		block = ++_dfg->begin(); 
		block != _dfg->end(); ++block )
	{
		_newBlock( block->label() );
		report( "  Translating Phi Instructions" );
		for( analysis::DataflowGraph::PhiInstructionVector::const_iterator 
			phi = block->phis().begin(); 
			phi != block->phis().end(); ++phi )
		{
			ir::LLVMPhi p;
			analysis::DataflowGraph::RegisterVector::const_iterator 
				s = phi->s.begin();
			for( ; s != phi->s.end(); ++s )
			{			
				ir::LLVMPhi::Node node;
				
				auto producer = block->producer( *s );
				
				if(producer == _dfg->end())
				{
					node.label = "%$OcelotRegisterInitializerBlock";
					_uninitialized.push_back( *s );
					node.reg = s->id;

					std::stringstream stream;
					stream << "%ri" << s->id;
				
					node.operand.name = stream.str();
					node.operand.type.category 
						= ir::LLVMInstruction::Type::Element;
					node.operand.type.type = _translate( s->type );
				
					p.nodes.push_back( node );
					continue;
				}
				else
				{
					node.label = "%" + producer->label();
				}
				
				node.reg = s->id;

				std::stringstream stream;
				stream << "%r" << s->id;
				
				node.operand.name = stream.str();
				node.operand.type.category 
					= ir::LLVMInstruction::Type::Element;
				node.operand.type.type = _translate( s->type );
				
				p.nodes.push_back( node );
			}

			assert( !p.nodes.empty() );
			
			std::stringstream stream;
			stream << "%r" << phi->d.id;
			p.d.name = stream.str();
			p.d.type.category = ir::LLVMInstruction::Type::Element;
			p.d.type.type = _translate( phi->d.type );
			
			_add( p );
		}

		_debug( *block );

		report( "  Translating Instructions" );
		for( analysis::DataflowGraph::InstructionVector::const_iterator 
			instruction = block->instructions().begin();
			instruction != block->instructions().end(); ++instruction )
		{
			_translate( *instruction, *block );
		}
		
		if( block->targets().empty() )
		{
			if( block->fallthrough() != _dfg->end() )
			{
				ir::LLVMBr branch;
			
				branch.iftrue = "%" + block->fallthrough()->label();
				_add( branch );
			}
			else
			{
				_add( ir::LLVMRet() );
			}
		}
	}
}

void PTXToLLVMTranslator::_newBlock( const std::string& name )
{
	report( " Translating basic block: " << name );
	_llvmKernel->push_back( ir::LLVMStatement( name ) );
}

void PTXToLLVMTranslator::_translate( 
	const analysis::DataflowGraph::Instruction& i, 
	const analysis::DataflowGraph::Block& block )
{
	_debug( i );
	_reportReads( i );
	_translate( static_cast<ir::PTXInstruction&>(*i.i), block );
	_reportWrites( i );
}

void PTXToLLVMTranslator::_translate( const ir::PTXInstruction& i, 
	const analysis::DataflowGraph::Block& block )
{
	report( "   Translating: " << i.toString() );
	assertM( i.valid() == "", "Instruction " << i.toString() 
		<< " is not valid: " << i.valid() );
	switch( i.opcode )
	{
	case ir::PTXInstruction::Abs:      _translateAbs( i );         break;
	case ir::PTXInstruction::Add:      _translateAdd( i );         break;
	case ir::PTXInstruction::AddC:     _translateAddC( i );        break;
	case ir::PTXInstruction::And:      _translateAnd( i );         break;
	case ir::PTXInstruction::Atom:     _translateAtom( i );        break;
	case ir::PTXInstruction::Bar:      _translateBar( i );         break;
	case ir::PTXInstruction::Bfe:      _translateBfe( i );         break;
	case ir::PTXInstruction::Bfi:      _translateBfi( i );         break;
	case ir::PTXInstruction::Bfind:    _translateBfind( i );       break;
	case ir::PTXInstruction::Bra:      _translateBra( i, block );  break;
	case ir::PTXInstruction::Brev:     _translateBrev( i );        break;
	case ir::PTXInstruction::Brkpt:    _translateBrkpt( i );       break;
	case ir::PTXInstruction::Call:     _translateCall( i, block ); break;
	case ir::PTXInstruction::Clz:      _translateClz( i );         break;
	case ir::PTXInstruction::CNot:     _translateCNot( i );        break;
	case ir::PTXInstruction::CopySign: _translateCopySign( i );    break;
	case ir::PTXInstruction::Cos:      _translateCos( i );         break;
	case ir::PTXInstruction::Cvt:      _translateCvt( i );         break;
	case ir::PTXInstruction::Cvta:     _translateCvta( i );        break;
	case ir::PTXInstruction::Div:      _translateDiv( i );         break;
	case ir::PTXInstruction::Ex2:      _translateEx2( i );         break;
	case ir::PTXInstruction::Exit:     _translateExit( i );        break;
	case ir::PTXInstruction::Isspacep: _translateIsspacep( i );    break;
	case ir::PTXInstruction::Fma:      _translateMad( i );         break;
	case ir::PTXInstruction::Ld:       _translateLd( i );          break;
	case ir::PTXInstruction::Ldu:      _translateLdu( i );         break;
	case ir::PTXInstruction::Lg2:      _translateLg2( i );         break;
	case ir::PTXInstruction::Mad24:    _translateMad24( i );       break;
	case ir::PTXInstruction::Mad:      _translateMad( i );         break;
	case ir::PTXInstruction::Max:      _translateMax( i );         break;
	case ir::PTXInstruction::Membar:   _translateMembar( i );      break;
	case ir::PTXInstruction::Min:      _translateMin( i );         break;
	case ir::PTXInstruction::Mov:      _translateMov( i );         break;
	case ir::PTXInstruction::Mul24:    _translateMul24( i );       break;
	case ir::PTXInstruction::Mul:      _translateMul( i );         break;
	case ir::PTXInstruction::Neg:      _translateNeg( i );         break;
	case ir::PTXInstruction::Not:      _translateNot( i );         break;
	case ir::PTXInstruction::Or:       _translateOr( i );          break;
	case ir::PTXInstruction::Pmevent:  _translatePmevent( i );     break;
	case ir::PTXInstruction::Popc:     _translatePopc( i );        break;
	case ir::PTXInstruction::Prmt:     _translatePrmt( i );        break;
	case ir::PTXInstruction::Rcp:      _translateRcp( i );         break;
	case ir::PTXInstruction::Red:      _translateRed( i );         break;
	case ir::PTXInstruction::Rem:      _translateRem( i );         break;
	case ir::PTXInstruction::Ret:      _translateRet( i, block );  break;
	case ir::PTXInstruction::Rsqrt:    _translateRsqrt( i );       break;
	case ir::PTXInstruction::Sad:      _translateSad( i );         break;
	case ir::PTXInstruction::SelP:     _translateSelP( i );        break;
	case ir::PTXInstruction::Set:      _translateSet( i );         break;
	case ir::PTXInstruction::SetP:     _translateSetP( i );        break;
	case ir::PTXInstruction::Shl:      _translateShl( i );         break;
	case ir::PTXInstruction::Shr:      _translateShr( i );         break;
	case ir::PTXInstruction::Sin:      _translateSin( i );         break;
	case ir::PTXInstruction::SlCt:     _translateSlCt( i );        break;
	case ir::PTXInstruction::Sqrt:     _translateSqrt( i );        break;
	case ir::PTXInstruction::St:       _translateSt( i );          break;
	case ir::PTXInstruction::Sub:      _translateSub( i );         break;
	case ir::PTXInstruction::SubC:     _translateSubC( i );        break;
	case ir::PTXInstruction::Suld:     _translateSuld( i );        break;
	case ir::PTXInstruction::Suq:      _translateSuq( i );         break;
	case ir::PTXInstruction::Sured:    _translateSured( i );       break;
	case ir::PTXInstruction::Sust:     _translateSust( i );        break;
	case ir::PTXInstruction::TestP:    _translateTestP( i );       break;
	case ir::PTXInstruction::Tex:      _translateTex( i );         break;
	case ir::PTXInstruction::Txq:      _translateTxq( i );         break;
	case ir::PTXInstruction::Trap:     _translateTrap( i );        break;
	case ir::PTXInstruction::Vote:     _translateVote( i );        break;
	case ir::PTXInstruction::Xor:      _translateXor( i );         break;
	default:
	{
		assertM( false, "Opcode " 
			<< ir::PTXInstruction::toString( i.opcode ) 
			<< " not supported." );
		break;
	}
	}
}

void PTXToLLVMTranslator::_translateAbs( const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand destination = _destination( i );
	
	if( ir::PTXOperand::isFloat( i.type ) )
	{
		ir::LLVMFcmp compare;
		
		compare.comparison = ir::LLVMInstruction::Olt;
		compare.d.type.category = ir::LLVMInstruction::Type::Element;
		compare.d.type.type = ir::LLVMInstruction::I1;
		compare.d.name = _tempRegister();
		compare.a = _translate( i.a );
		compare.b = compare.a;
		compare.b.constant = true;
		
		if( ir::PTXOperand::f64 == i.a.type )
		{
			compare.b.f64 = 0;
		}
		else
		{
			compare.b.f32 = 0;
		}
		
		_add( compare );
		
		ir::LLVMFsub sub;
		
		sub.d = destination;
		sub.d.name = _tempRegister();
		sub.a = compare.a;
		sub.a.constant = true;
		
		if( ir::PTXOperand::f64 == i.a.type )
		{
			sub.a.f64 = 0;
		}
		else
		{
			sub.a.f32 = 0;
		}			
		
		sub.b = compare.a;
		
		_add( sub );
		
		ir::LLVMSelect select;
		
		if( i.modifier & ir::PTXInstruction::ftz )
		{
			select.d = ir::LLVMInstruction::Operand( _tempRegister(), 
				destination.type );
		}
		else
		{
			select.d = destination;
		}
		
		select.condition = compare.d;
		select.a = sub.d;
		select.b = compare.a;
		
		_add( select );
		
		if( i.modifier & ir::PTXInstruction::ftz )
		{
			_flushToZero( destination, select.d );
		}
	}
	else
	{
		ir::LLVMIcmp compare;
		
		compare.comparison = ir::LLVMInstruction::Slt;
		compare.d.type.category = ir::LLVMInstruction::Type::Element;
		compare.d.type.type = ir::LLVMInstruction::I1;
		compare.d.name = _tempRegister();
		compare.a = _translate( i.a );
		compare.b = compare.a;
		compare.b.constant = true;
		
		if( ir::PTXOperand::s64 == i.a.type )
		{
			compare.b.i64 = 0;
		}
		else
		{
			compare.b.i32 = 0;
		}
		
		_add( compare );
		
		ir::LLVMSub sub;
		
		sub.d = destination;
		sub.d.name = _tempRegister();
		sub.a = compare.a;
		sub.a.constant = true;
		
		if( ir::PTXOperand::s64 == i.a.type )
		{
			sub.a.i64 = 0;
		}
		else
		{
			sub.a.i32 = 0;
		}			
		
		sub.b = compare.a;
		
		_add( sub );
		
		ir::LLVMSelect select;
		
		select.d = destination;
		select.condition = compare.d;
		select.a = sub.d;
		select.b = compare.a;
		
		_add( select );
	}
}

void PTXToLLVMTranslator::_translateAdd( const ir::PTXInstruction& i )
{
	if( ir::PTXOperand::isFloat( i.type ) )
	{
		ir::LLVMFadd add;
		
		ir::LLVMInstruction::Operand result = _destination( i );

		add.a = _translate( i.a );
		add.b = _translate( i.b );

		if( i.modifier & ir::PTXInstruction::sat
			|| i.modifier & ir::PTXInstruction::ftz )
		{
			add.d = add.a;
			add.d.name = _tempRegister();
		}
		else
		{
			add.d = result;
		}
	
		_add( add );	
		
		if( i.modifier & ir::PTXInstruction::sat )
		{
			if( i.modifier & ir::PTXInstruction::ftz )
			{
				ir::LLVMInstruction::Operand temp =
					ir::LLVMInstruction::Operand( _tempRegister(),
					add.d.type );
				_saturate( temp, add.d );
				_flushToZero( result, temp );
			}
			else
			{
				_saturate( result, add.d );
			}
		}
		else if( i.modifier & ir::PTXInstruction::ftz )
		{
			_flushToZero( result, add.d );
		}
	}
	else
	{
		if( i.modifier & ir::PTXInstruction::sat )
		{
			assert( i.type == ir::PTXOperand::s32 );
			
			ir::LLVMSext extendA;
			ir::LLVMSext extendB;
							
			extendA.a = _translate( i.a );
			extendA.d.type.type = ir::LLVMInstruction::I64;
			extendA.d.type.category = ir::LLVMInstruction::Type::Element;
			extendA.d.name = _tempRegister();
			
			_add( extendA );
			
			extendB.a = _translate( i.b );
			extendB.d.type.type = ir::LLVMInstruction::I64;
			extendB.d.type.category = ir::LLVMInstruction::Type::Element;
			extendB.d.name = _tempRegister();
			
			_add( extendB );
			
			ir::LLVMAdd add;
			
			add.a = extendA.d;
			add.b = extendB.d;
			add.d.name = _tempRegister();
			add.d.type.type = ir::LLVMInstruction::I64;
			add.d.type.category = ir::LLVMInstruction::Type::Element;
			
			_add( add );
			
			ir::LLVMIcmp compare;
			
			compare.d.name = _tempRegister();
			compare.d.type.type = ir::LLVMInstruction::I1;
			compare.d.type.category = ir::LLVMInstruction::Type::Element;
			compare.comparison = ir::LLVMInstruction::Slt;
			compare.a = add.d;
			compare.b.type.type = ir::LLVMInstruction::I64;
			compare.b.type.category = ir::LLVMInstruction::Type::Element;
			compare.b.constant = true;
			compare.b.i64 = INT_MIN;

			_add( compare );
			
			ir::LLVMSelect select;
			
			select.d.name = _tempRegister();
			select.d.type.type = ir::LLVMInstruction::I64;
			select.d.type.category = ir::LLVMInstruction::Type::Element;

			select.condition = compare.d;
			select.a = compare.b;
			select.b = compare.a;
			
			_add( select );
			
			compare.d.name = _tempRegister();
			compare.comparison = ir::LLVMInstruction::Sgt;
			compare.b.i64 = INT_MAX;
			compare.a = select.d;
			
			_add( compare );

			select.condition = compare.d;
			select.a = compare.b;
			select.b = compare.a;
			select.d.name = _tempRegister();

			_add( select );
			
			ir::LLVMTrunc truncate;
			
			truncate.a = select.d;
			truncate.d = _destination( i );
			
			_add( truncate );
		}
		else if ( i.carry & ir::PTXInstruction::CC )
		{
			ir::LLVMInstruction::Operand a = _translate( i.a );	
			ir::LLVMInstruction::Operand b = _translate( i.b );
						
			ir::LLVMAdd add;

			add.d = _destination( i );
			add.a = a;
			add.b = b;
			
			_add( add );
			
			ir::LLVMInstruction::Operand carry = _translate( i.pq );
			ir::LLVMInstruction::Operand lessThanA = 
				ir::LLVMInstruction::Operand( _tempRegister(),
					ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
					ir::LLVMInstruction::Type::Element ) );
			ir::LLVMInstruction::Operand lessThanB = 
				ir::LLVMInstruction::Operand( _tempRegister(),
					ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
					ir::LLVMInstruction::Type::Element ) );
			ir::LLVMInstruction::Operand lessThanEither = 
				ir::LLVMInstruction::Operand( _tempRegister(),
					ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
					ir::LLVMInstruction::Type::Element ) );
			
			ir::LLVMIcmp compare;
		
			compare.comparison = ir::LLVMInstruction::Ult;
			compare.d = lessThanA;
			compare.a = add.d;
			compare.b = a;
		
			_add( compare );
			
			compare.d = lessThanB;
			compare.b = b;
			
			_add( compare );
			
			ir::LLVMOr Or;
			
			Or.d = lessThanEither;
			Or.a = lessThanA;
			Or.b = lessThanB;
			
			_add( Or );
		
			ir::LLVMSelect select;
			
			select.d = carry;
			select.condition = lessThanEither;
			select.a = ir::LLVMInstruction::Operand( (ir::LLVMI32) 1 );
			select.b = ir::LLVMInstruction::Operand( (ir::LLVMI32) 0 );
			
			_add( select );
		}
		else
		{
			ir::LLVMAdd add;

			add.d = _destination( i );
			add.a = _translate( i.a );
			add.b = _translate( i.b );
			
			_add( add );
		}
	}		
}

void PTXToLLVMTranslator::_translateAddC( const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand destination = _destination( i );
	ir::LLVMInstruction::Operand a = _translate( i.a );	
	ir::LLVMInstruction::Operand b = _translate( i.b );
			
	ir::LLVMAdd add;
	
	add.d = destination;
	add.d.name = _tempRegister();
	add.a = a;
	add.b = b;
	
	_add( add );
	
	add.a = add.d;
	add.d = destination;
	add.b = _translate( i.c );
	
	_add( add );

	if( i.carry & ir::PTXInstruction::CC )
	{
		ir::LLVMInstruction::Operand carry = _translate( i.pq );
		ir::LLVMInstruction::Operand lessThanA = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
		ir::LLVMInstruction::Operand lessThanB = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
		ir::LLVMInstruction::Operand lessThanEither = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
		
		ir::LLVMIcmp compare;
	
		compare.comparison = ir::LLVMInstruction::Ult;
		compare.d = lessThanA;
		compare.a = destination;
		compare.b = a;
	
		_add( compare );
		
		compare.d = lessThanB;
		compare.b = b;
		
		_add( compare );
		
		ir::LLVMOr Or;
		
		Or.d = lessThanEither;
		Or.a = lessThanA;
		Or.b = lessThanB;
		
		_add( Or );
	
		ir::LLVMSelect select;
		
		select.d = carry;
		select.condition = lessThanEither;
		select.a = ir::LLVMInstruction::Operand( (ir::LLVMI32) 1 );
		select.b = ir::LLVMInstruction::Operand( (ir::LLVMI32) 0 );
		
		_add( select );					
	}
}

void PTXToLLVMTranslator::_translateAnd( const ir::PTXInstruction& i )
{						
	ir::LLVMAnd And;
	
	And.d = _destination( i );
	And.a = _translate( i.a );
	And.b = _translate( i.b );

	_add( And );
}

void PTXToLLVMTranslator::_translateAtom( const ir::PTXInstruction& i )
{
	if( i.addressSpace != ir::PTXInstruction::Shared )
	{
		if( i.atomicOperation == ir::PTXInstruction::AtomicCas )
		{
			ir::LLVMCmpxchg atom;
			
			atom.d = _destination( i );
			atom.a = _getLoadOrStorePointer( i.a, i.addressSpace, 
				_translate( i.type ), i.vec );
			atom.b = _translate( i.b );
			atom.c = _translate( i.c );
			
			_add( atom );		
		}
		else if( i.atomicOperation != ir::PTXInstruction::AtomicInc && 
			i.atomicOperation != ir::PTXInstruction::AtomicDec )
		{
			ir::LLVMAtomicrmw atom;
			
			atom.d = _destination( i );
			atom.a = _getLoadOrStorePointer( i.a, i.addressSpace, 
				_translate( i.type ), i.vec );
			atom.b = _translate( i.b );
			atom.operation = _translate( i.atomicOperation, i.type );
			
			_add( atom );
		}
		else
		{
			ir::LLVMCall call;
	
			call.d = _destination( i );
			call.parameters.resize( 2 );

			switch( i.atomicOperation )
			{
				case ir::PTXInstruction::AtomicInc:
				{
					call.name = "@__ocelot_atomic_inc";
					break;
				}
				case ir::PTXInstruction::AtomicDec: 
				{
					call.name = "@__ocelot_atomic_dec";
					break;
				}
				
				default: break;
			}

			call.parameters[0] = _translate( i.a );

			switch( i.type )
			{
				case ir::PTXOperand::b32: /* fall through */
				case ir::PTXOperand::u32: /* fall through */
				case ir::PTXOperand::s32:
				{
					call.name += "_32";
					break;
				}
				case ir::PTXOperand::s64: /* fall through */
				case ir::PTXOperand::u64: /* fall through */
				case ir::PTXOperand::b64:
				{
					call.name += "_64";
					break;
				}
				default: assertM(false, "Invalid type.");
			}
			
			call.parameters[1] = _translate( i.b );

			_add( call );
		}
	}
	else
	{
		// Shared atomics do not need to be atomic because threads in
		// a CTA are serialized
		ir::LLVMLoad load;
		
		load.a = _getLoadOrStorePointer( i.a, i.addressSpace, 
			_translate( i.type ), i.vec );
		load.d = _destination( i );
		
		_add( load );
		
		ir::LLVMStore store;
		
		store.d = load.a;
		
		switch( i.atomicOperation )
		{
			case ir::PTXInstruction::AtomicAnd:
			{
				ir::LLVMAnd land;
				
				land.d.name          = _tempRegister();
				land.d.type.category = ir::LLVMInstruction::Type::Element;
				land.d.type.type     = _translate( i.type );
				land.a               = _translate( i.b );
				land.b               = load.d;
				
				_add( land );
				
				store.a = land.d;
				
				break;
			}
			case ir::PTXInstruction::AtomicOr:
			{
				ir::LLVMOr lor;
				
				lor.d.name          = _tempRegister();
				lor.d.type.category = ir::LLVMInstruction::Type::Element;
				lor.d.type.type     = _translate( i.type );
				lor.a               = _translate( i.b );
				lor.b               = load.d;
				
				_add( lor );
				
				store.a = lor.d;
				break;
			}
			case ir::PTXInstruction::AtomicXor:
			{
				ir::LLVMXor lxor;
				
				lxor.d.name          = _tempRegister();
				lxor.d.type.category = ir::LLVMInstruction::Type::Element;
				lxor.d.type.type     = _translate( i.type );
				lxor.a               = _translate( i.b );
				lxor.b               = load.d;
				
				_add( lxor );
				
				store.a = lxor.d;
				
				break;
			}
			case ir::PTXInstruction::AtomicCas:
			{
				ir::LLVMIcmp cmp;
				
				cmp.d.name          = _tempRegister();
				cmp.d.type.category = ir::LLVMInstruction::Type::Element;
				cmp.d.type.type     = ir::LLVMInstruction::I1;
			
				cmp.comparison      = ir::LLVMInstruction::Eq;
				
				cmp.a               = load.d;
				cmp.b               = _translate( i.b );
				
				_add( cmp );
				
				ir::LLVMSelect select;
				
				select.condition       = cmp.d;
				select.a               = _translate( i.c );
				select.b               = load.d;
				select.d.name          = _tempRegister();
				select.d.type.category = ir::LLVMInstruction::Type::Element;
				select.d.type.type     = _translate( i.type );
				
				_add( select );

				store.a = select.d;
				
				break;
			}
			case ir::PTXInstruction::AtomicExch:
			{
				store.a = _translate( i.b );		
				break;
			}
			case ir::PTXInstruction::AtomicAdd:
			{
				ir::LLVMAdd add;
				
				add.d.name          = _tempRegister();
				add.d.type.category = ir::LLVMInstruction::Type::Element;
				add.d.type.type     = _translate( i.type );
				add.a               = _translate( i.b );
				add.b               = load.d;
				
				_add( add );
				
				store.a = add.d;
				break;
			}
			case ir::PTXInstruction::AtomicInc:
			{
				ir::LLVMIcmp cmp;
				
				cmp.d.name          = _tempRegister();
				cmp.d.type.category = ir::LLVMInstruction::Type::Element;
				cmp.d.type.type     = ir::LLVMInstruction::I1;
			
				cmp.comparison      = ir::LLVMInstruction::Slt;
				
				cmp.a               = load.d;
				cmp.b               = _translate( i.b );
				
				_add( cmp );
				
				ir::LLVMAdd add;
				
				add.d.name          = _tempRegister();
				add.d.type.category = ir::LLVMInstruction::Type::Element;
				add.d.type.type     = _translate( i.type );
				add.a               = load.d;
				add.b.constant      = true;
				add.b.type.category = ir::LLVMInstruction::Type::Element;
				add.b.type.type     = _translate( i.type );
				add.b.i32           = 1;
				
				_add( add );
				
				ir::LLVMSelect select;
				
				select.condition       = cmp.d;
				select.a               = add.d;
				select.b.constant      = true;
				select.b.type.category = ir::LLVMInstruction::Type::Element;
				select.b.type.type     = _translate( i.type );
				select.b.i32           = 0;
				select.d.name          = _tempRegister();
				select.d.type.category = ir::LLVMInstruction::Type::Element;
				select.d.type.type     = _translate( i.type );
				
				_add( select );

				store.a = select.d;
				break;
			}
			case ir::PTXInstruction::AtomicDec: 
			{
				ir::LLVMIcmp cmp;
				
				cmp.d.name          = _tempRegister();
				cmp.d.type.category = ir::LLVMInstruction::Type::Element;
				cmp.d.type.type     = ir::LLVMInstruction::I1;
			
				cmp.comparison      = ir::LLVMInstruction::Sgt;
				
				cmp.a               = load.d;
				cmp.b               = _translate( i.b );
				
				_add( cmp );
				
				ir::LLVMSub sub;
				
				sub.d.name          = _tempRegister();
				sub.d.type.category = ir::LLVMInstruction::Type::Element;
				sub.d.type.type     = _translate( i.type );
				sub.a               = load.d;
				sub.b.constant      = true;
				sub.b.type.category = ir::LLVMInstruction::Type::Element;
				sub.b.type.type     = _translate( i.type );
				sub.b.i32           = 1;
				
				_add( sub );
				
				ir::LLVMSelect select;
				
				select.condition       = cmp.d;
				select.a               = cmp.b;
				select.b               = sub.d;
				select.d.name          = _tempRegister();
				select.d.type.category = ir::LLVMInstruction::Type::Element;
				select.d.type.type     = _translate( i.type );
				
				_add( select );

				store.a = select.d;
				break;
			}
			case ir::PTXInstruction::AtomicMin:
			{
				ir::LLVMIcmp cmp;
				
				cmp.d.name          = _tempRegister();
				cmp.d.type.category = ir::LLVMInstruction::Type::Element;
				cmp.d.type.type     = ir::LLVMInstruction::I1;
			
				if( ir::PTXOperand::isSigned( i.type ) )
				{
					cmp.comparison      = ir::LLVMInstruction::Slt;
				}
				else
				{
					cmp.comparison      = ir::LLVMInstruction::Ult;
				}
				
				cmp.a               = load.d;
				cmp.b               = _translate( i.b );
				
				_add( cmp );
				
				ir::LLVMSelect select;
				
				select.condition       = cmp.d;
				select.a               = cmp.a;
				select.b               = cmp.b;
				select.d.name          = _tempRegister();
				select.d.type.category = ir::LLVMInstruction::Type::Element;
				select.d.type.type     = _translate( i.type );
				
				_add( select );

				store.a = select.d;
				break;
			}
			case ir::PTXInstruction::AtomicMax:
			{
				ir::LLVMIcmp cmp;
				
				cmp.d.name          = _tempRegister();
				cmp.d.type.category = ir::LLVMInstruction::Type::Element;
				cmp.d.type.type     = ir::LLVMInstruction::I1;
				
				if( ir::PTXOperand::isSigned( i.type ) )
				{
					cmp.comparison      = ir::LLVMInstruction::Sgt;
				}
				else
				{
					cmp.comparison      = ir::LLVMInstruction::Ugt;
				}
				
				cmp.a               = load.d;
				cmp.b               = _translate( i.b );
				
				_add( cmp );
				
				ir::LLVMSelect select;
				
				select.condition       = cmp.d;
				select.a               = cmp.a;
				select.b               = cmp.b;
				select.d.name          = _tempRegister();
				select.d.type.category = ir::LLVMInstruction::Type::Element;
				select.d.type.type     = _translate( i.type );
				
				_add( select );

				store.a = select.d;
				break;
			}
			default: break;
		}
		
		_add( store );
	}
}

void PTXToLLVMTranslator::_translateBar( const ir::PTXInstruction& i )
{
	assertM(false, "All barriers should have been removed.");
}

void PTXToLLVMTranslator::_translateBfe( const ir::PTXInstruction& i )
{
    ir::LLVMCall call;
    
    if( i.type == ir::PTXOperand::u32 || i.type == ir::PTXOperand::s32 )
    {
	    call.name = "@__ocelot_bfe_b32";
    }
    else
    {
        call.name = "@__ocelot_bfe_b64";
    }
    
    ir::LLVMInstruction::Operand sign;    
    sign.constant = true;
    sign.type.category = ir::LLVMInstruction::Type::Element;
    sign.type.type = ir::LLVMInstruction::I1;
	
    if( ir::PTXOperand::isSigned( i.type ) )
    {
    sign.i1 = 1;
    }
    else
    {
    sign.i1 = 0;
    }
	
    call.d = _destination( i );
	
    call.parameters.push_back( _translate( i.a ) );
    call.parameters.push_back( _translate( i.b ) );
    call.parameters.push_back( _translate( i.c ) );
    call.parameters.push_back( sign );
	
    _add( call );
}
void PTXToLLVMTranslator::_translateBfi( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	if( i.type == ir::PTXOperand::b32 )
	{
		call.name = "@__ocelot_bfi_b32";
	}
	else
	{
		call.name = "@__ocelot_bfi_b64";
	}
	
	call.d = _destination( i );
	
	call.parameters.push_back( _translate( i.pq ) );
	call.parameters.push_back( _translate( i.a ) );
	call.parameters.push_back( _translate( i.b ) );
	call.parameters.push_back( _translate( i.c ) );
	
	_add( call );
}

void PTXToLLVMTranslator::_translateBfind( const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand a = _translate( i.a );

	if( ir::PTXOperand::isSigned( i.type ) )
	{
		ir::LLVMIcmp compare;
		
		compare.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
		
		compare.comparison = ir::LLVMInstruction::Slt;
		compare.a = a;
		compare.b.constant = true;
		compare.b.type = a.type;
		compare.b.i64 = 0;
		
		_add( compare );
		
		ir::LLVMSub negate;
		
		negate.d = ir::LLVMInstruction::Operand( _tempRegister(), a.type );
		negate.a.constant = true;
		negate.a.type = a.type;
		negate.a.i64 = 0;
		negate.b = a;
		
		_add( negate );
		
		ir::LLVMSelect select;
		
		select.condition = compare.d;
		select.a = negate.d;
		select.b = a;
		select.d = ir::LLVMInstruction::Operand( _tempRegister(), a.type );
		
		_add( select );
		
		a = select.d;
	}

	ir::LLVMCall call;
	
	if( ir::PTXOperand::bytes( i.type ) == 4 )
	{
		call.name = "@__ocelot_bfind_b32";
	}
	else
	{
		call.name = "@__ocelot_bfind_b64";
	}
	
	call.d = _destination( i );
	call.parameters.resize( 2 );
	call.parameters[0] = a;
	call.parameters[1].constant = true;
	call.parameters[1].i1 = i.shiftAmount;
	call.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[1].type.type = ir::LLVMInstruction::I1;
	
	_add( call );
}

void PTXToLLVMTranslator::_translateBra( const ir::PTXInstruction& i, 
	const analysis::DataflowGraph::Block& block )
{
	ir::LLVMBr branch;
	
	if( block.targets().empty() )
	{
		branch.iftrue = "%" + block.fallthrough()->label();
	}
	else
	{
		branch.iftrue = "%" + (*block.targets().begin())->label();
		if( block.fallthrough() != _dfg->end() )
		{
			if( (*block.targets().begin()) != block.fallthrough() )
			{
				if( ir::PTXOperand::PT != i.pg.condition 
					&& ir::PTXOperand::nPT != i.pg.condition )
				{
					branch.condition = _translate( i.pg );
				}
				else
				{
					branch.condition.type.category 
						= ir::LLVMInstruction::Type::Element;
					branch.condition.type.type = ir::LLVMInstruction::I1;
					branch.condition.constant = true;
		
					if( ir::PTXOperand::PT == i.pg.condition )
					{
						branch.condition.i1 = true;
					}
					else
					{
						branch.condition.i1 = false;
					}
				}
				branch.iffalse = "%" + block.fallthrough()->label();
			}
			if( i.pg.condition == ir::PTXOperand::InvPred )
			{
				std::swap( branch.iftrue, branch.iffalse );
			}
		}
		
	}
	_add( branch );
}

void PTXToLLVMTranslator::_translateBrev( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	if( i.type == ir::PTXOperand::b32 )
	{
		call.name = "@__ocelot_brev_b32";
	}
	else
	{
		call.name = "@__ocelot_brev_b64";
	}
	
	call.d = _destination( i );
	call.parameters.resize( 1 );
	call.parameters[0] = _translate( i.a );
	
	_add( call );
}

void PTXToLLVMTranslator::_translateBrkpt( const ir::PTXInstruction& i )
{
	assertM( false, "Opcode " 
		<< ir::PTXInstruction::toString( i.opcode ) 
		<< " not supported." );
}

void PTXToLLVMTranslator::_translateCall( const ir::PTXInstruction& i,
	const analysis::DataflowGraph::Block& block )
{
	if( i.tailCall )
	{
		if( i.a.addressMode == ir::PTXOperand::FunctionName )
		{
			if( i.reentryPoint == -1 )
			{
				_yield( executive::LLVMExecutableKernel::BarrierCall );
			}
			else
			{
				_yield( executive::LLVMExecutableKernel::TailCall, 
					ir::LLVMInstruction::Operand( 
					(ir::LLVMI32) i.reentryPoint ) );
			}
		}
		else
		{
			_yield( executive::LLVMExecutableKernel::TailCall,
				_translate( i.a ) );
		}
		
		if( !block.targets().empty() )
		{
			ir::LLVMBr branch;
		
			branch.iftrue = "%" + (*block.targets().begin())->label();
		
			_add( branch );
		}
	}
	else
	{
		ir::ExternalFunctionSet::ExternalFunction* external = 0;

		if( i.a.addressMode == ir::PTXOperand::FunctionName
			&& _externals != 0 )
		{
			external = _externals->find( i.a.identifier );
		}
		
		if( external != 0 )
		{
			_usedExternalCalls.insert( external->name() );
		
			ir::LLVMCall call;

			call.name = "@" + external->name();
			
			assert( i.d.array.size() < 2 );

			if( i.d.array.size() == 1 )
			{
				if( i.d.array[ 0 ].addressMode
					!= ir::PTXOperand::BitBucket )
				{
					call.d = _translate( i.d.array[ 0 ] );
				}
				else
				{
					ir::Module::FunctionPrototypeMap::const_iterator
						prototype = _ptx->module->prototypes().find(
							external->name() );
					assert( prototype != _ptx->module->prototypes().end() );
					
					call.d = ir::LLVMInstruction::Operand( _tempRegister(),
						ir::LLVMInstruction::Type( _translate(
							prototype->second.returnArguments[ 0 ].type ),
						ir::LLVMInstruction::Type::Element ) );
				}
			}
			
			for( ir::PTXOperand::Array::const_iterator
				operand = i.b.array.begin();
				operand != i.b.array.end(); ++operand )
			{
				if( operand->addressMode != ir::PTXOperand::BitBucket )
				{
					call.parameters.push_back( _translate( *operand ) );
				}
				else
				{
					assertM(false, "Bit bucked function "
						"parameters not supported.");
				}
			}
			
			_add( call );

			return;
		}

		ir::LLVMBr branch;
	
		std::string yieldLabel = "Ocelot_yield_" + block.label();
	
		branch.iftrue = "%" + yieldLabel;
		if( block.fallthrough() != _dfg->end() )
		{
			if( (*block.targets().begin()) != block.fallthrough() )
			{
				if( ir::PTXOperand::PT != i.pg.condition 
					&& ir::PTXOperand::nPT != i.pg.condition )
				{
					branch.condition = _translate( i.pg );
				}
				else
				{
					branch.condition.type.category 
						= ir::LLVMInstruction::Type::Element;
					branch.condition.type.type = ir::LLVMInstruction::I1;
					branch.condition.constant = true;
		
					if( ir::PTXOperand::PT == i.pg.condition )
					{
						branch.condition.i1 = true;
					}
					else
					{
						branch.condition.i1 = false;
					}
				}
				branch.iffalse = "%" + block.fallthrough()->label();
			}
			if( i.pg.condition == ir::PTXOperand::InvPred )
			{
				std::swap( branch.iftrue, branch.iffalse );
			}
		}

		_add( branch );
		
		_newBlock( yieldLabel );
		
		if( i.a.addressMode == ir::PTXOperand::Register )
		{
			ir::LLVMInstruction::Operand 
				functionPointer = _translate( i.a );
	
			if( i.a.type != ir::PTXOperand::u32 )
			{
				ir::LLVMInstruction::Operand temp( _tempRegister(), 
					ir::LLVMInstruction::Type( ir::LLVMInstruction::I32, 
					ir::LLVMInstruction::Type::Element ) );
		
				_convert( temp, ir::PTXOperand::u32,
					functionPointer, i.a.type );
			
				functionPointer = temp;
			}
		
			_yield( executive::LLVMExecutableKernel::NormalCall,
				functionPointer );			
		}
		else
		{
			_yield( executive::LLVMExecutableKernel::NormalCall,
				ir::LLVMInstruction::Operand( 
				(ir::LLVMI32) i.reentryPoint ) );
		}
		
		branch.iftrue = "%" + (*block.targets().begin())->label();
		branch.condition.type.category 
			= ir::LLVMInstruction::Type::InvalidCategory;
		
		_add(branch);
	}

}

void PTXToLLVMTranslator::_translateClz( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	if( i.type == ir::PTXOperand::b32 )
	{
		call.name = "@llvm.ctlz.i32";
		call.d = _destination( i );
	}
	else
	{
		call.name = "@llvm.ctlz.i64";
		call.d = ir::LLVMInstruction::Operand( _tempRegister(), 
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I64, 
			ir::LLVMInstruction::Type::Element ) );
	}
	
	call.parameters.resize( 1 );
	call.parameters[0] = _translate( i.a );
	
	_add( call );
	
	if( i.type != ir::PTXOperand::b32 )
	{
		ir::LLVMTrunc truncate;
		
		truncate.d = _destination( i );
		truncate.a = call.d;
		
		_add( truncate );
	}
}

void PTXToLLVMTranslator::_translateCNot( const ir::PTXInstruction& i )
{
	ir::LLVMIcmp cmp;

	cmp.d = ir::LLVMInstruction::Operand( _tempRegister(),
		ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
		ir::LLVMInstruction::Type::Element ) );
	cmp.a = _translate( i.a );
	cmp.comparison = ir::LLVMInstruction::Eq;
	cmp.b = ir::LLVMInstruction::Operand( (ir::LLVMI64) 0 );
	cmp.b.type.type = cmp.a.type.type;
	
	_add( cmp );

	ir::LLVMSelect select;
	
	select.d = _destination( i );
	select.condition = cmp.d;
	select.a = cmp.a;
	select.a.constant = true;
	select.a.i64 = 1;
	select.b = select.a;		
	select.b.i64 = 0;

	_add( select );
}

void PTXToLLVMTranslator::_translateCopySign( const ir::PTXInstruction& i )
{
	ir::LLVMBitcast castA;
	ir::LLVMBitcast castB;
	
	castA.a = _translate( i.a );
	castB.a = _translate( i.b );
	
	ir::LLVMAnd land;
			
	if( i.type == ir::PTXOperand::f32 )
	{
		castA.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I32,
			ir::LLVMInstruction::Type::Element ) );
		castB.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I32,
			ir::LLVMInstruction::Type::Element ) );

		land.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I32,
			ir::LLVMInstruction::Type::Element ) );
		land.a = castA.d;			
		land.b = ir::LLVMInstruction::Operand( (ir::LLVMI32) 0x80000000 );
	}
	else
	{
		castA.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I64,
			ir::LLVMInstruction::Type::Element ) );
		castB.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I64,
			ir::LLVMInstruction::Type::Element ) );

		land.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I64,
			ir::LLVMInstruction::Type::Element ) );
		land.a = castA.d;			
		land.b = ir::LLVMInstruction::Operand(
			(ir::LLVMI64) 0x8000000000000000ULL );
	}

	land.b.type.type = castA.d.type.type;

	_add( castA );
	_add( castB );
	_add( land );		

	ir::LLVMAnd land2;
	
	land2.d = ir::LLVMInstruction::Operand( _tempRegister(),
		ir::LLVMInstruction::Type( _translate( i.type ),
		ir::LLVMInstruction::Type::Element ) );
	land2.a = _translate( i.b );

	if( i.type == ir::PTXOperand::f32 )
	{
		land2.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I32,
			ir::LLVMInstruction::Type::Element ) );
		land2.a = castB.d;			
		land2.b = ir::LLVMInstruction::Operand( (ir::LLVMI32) 0x7fffffff );
	}
	else
	{
		land2.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I64,
			ir::LLVMInstruction::Type::Element ) );
		land2.a = castB.d;			
		land2.b = ir::LLVMInstruction::Operand(
			(ir::LLVMI64) 0x7fffffffffffffffULL );
	}

	land2.b.type.type = castB.d.type.type;

	_add( land2 );
	
	ir::LLVMOr lor;
	
	lor.d = ir::LLVMInstruction::Operand( _tempRegister(),
		ir::LLVMInstruction::Type( castB.d.type.type,
		ir::LLVMInstruction::Type::Element ) );
	lor.a = land.d;
	lor.b = land2.d;
	
	_add( lor );
	
	ir::LLVMBitcast castD;
	
	castD.d = _destination( i );
	castD.a = lor.d;
	
	_add( castD );
}

void PTXToLLVMTranslator::_translateCos( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;

	call.name = "@llvm.cos.f32";
	
	if( i.modifier & ir::PTXInstruction::ftz )
	{
		call.d.name          = _tempRegister();
		call.d.type.type     = ir::LLVMInstruction::F32;
		call.d.type.category = ir::LLVMInstruction::Type::Element;
	}
	else
	{
		call.d = _destination( i );
	}
	
	call.parameters.resize( 1 );
	call.parameters[0] = _translate( i.a );
			
	_add( call );

	if( i.modifier & ir::PTXInstruction::ftz )
	{
		_flushToZero( _destination( i ), call.d );
	}
}

void PTXToLLVMTranslator::_translateCvt( const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand destination;
	ir::LLVMInstruction::Operand source = _translate( i.a );
	
	if( _translate( i.d.type ) != _translate( i.type ) )
	{
		destination.name = _tempRegister();
		destination.type.category = ir::LLVMInstruction::Type::Element;
		destination.type.type = _translate( i.type );
	}
	else
	{
		destination = _translate( i.d );
	}

	ir::PTXOperand::DataType sourceType = i.a.type;
	
	if( i.a.relaxedType != ir::PTXOperand::TypeSpecifier_invalid )
	{
		sourceType = i.a.relaxedType;
		ir::LLVMInstruction::Operand temp( _tempRegister(), 
			ir::LLVMInstruction::Type( _translate( sourceType ), 
			ir::LLVMInstruction::Type::Element ) );

		_bitcast( temp, source );

		source = temp;
	}

	_convert( destination, i.type, source, sourceType, i.modifier );

	if( _translate( i.d.type ) != _translate( i.type ) )
	{
		_bitcast( _translate( i.d ), destination, 
			ir::PTXOperand::isSigned( i.type ) );
	}
}

void PTXToLLVMTranslator::_translateCvta( const ir::PTXInstruction& i )
{
	if( ir::PTXInstruction::Global == i.addressSpace )
	{
		_translateMov( i );
		return;
	}
	
	if( i.toAddrSpace )
	{
		ir::LLVMInstruction::Operand base = _getMemoryBasePointer( 
		 	i.addressSpace, false, false );
		
		ir::LLVMSub sub;
		
		sub.a = _translate( i.a );
		sub.d = _destination( i );
	
		ir::LLVMPtrtoint toint;
		
		toint.a = base;
		toint.d = ir::LLVMInstruction::Operand( _tempRegister(), sub.d.type );
		
		_add( toint );
		
		sub.b = toint.d;
		
		_add( sub );
	}
	else
	{
		switch( i.addressSpace )
		{
		case ir::PTXInstruction::Shared: /* fall through */
		case ir::PTXInstruction::Local:
		{
			assertM( i.addressSpace != ir::PTXInstruction::Local
				|| i.a.addressMode != ir::PTXOperand::Address
				|| !i.a.isGlobalLocal, 
				"Taking the address of a globally local "
				"value is not supported." );
		
			ir::LLVMPtrtoint toint;
		
			toint.a = _getLoadOrStorePointer( i.a, 
				i.addressSpace, _translate( i.type ), i.vec );

			toint.d = _destination( i );

			_add( toint );

			break;
		}
		default: assertM(false, "Invalid address space for cvta.");
		}
	}
}

void PTXToLLVMTranslator::_translateDiv( const ir::PTXInstruction& i )
{
	if( ir::PTXOperand::isFloat( i.type ) )
	{
		ir::LLVMFdiv div;
		
		ir::LLVMInstruction::Operand result = _destination( i );
		
		if( i.modifier & ir::PTXInstruction::ftz )
		{
			div.d = ir::LLVMInstruction::Operand( _tempRegister(), 
				ir::LLVMInstruction::Type( ir::LLVMInstruction::F32, 
				ir::LLVMInstruction::Type::Element ) );
			div.a = ir::LLVMInstruction::Operand( _tempRegister(), 
				ir::LLVMInstruction::Type( ir::LLVMInstruction::F32, 
				ir::LLVMInstruction::Type::Element ) );
			div.b = ir::LLVMInstruction::Operand( _tempRegister(), 
				ir::LLVMInstruction::Type( ir::LLVMInstruction::F32, 
				ir::LLVMInstruction::Type::Element ) );
			
			_flushToZero( div.a, _translate( i.a ) );
			_flushToZero( div.b, _translate( i.b ) );
		}
		else
		{
			div.d = result;
			div.a = _translate( i.a );
			div.b = _translate( i.b );
		}
			
		_add( div );
	
		if( i.modifier & ir::PTXInstruction::ftz )
		{
			_flushToZero( result, div.d );
		}
	}
	else if( ir::PTXOperand::isSigned( i.type ) )
	{
		ir::LLVMSdiv div;
		
		div.d = _destination( i );
		div.a = _translate( i.a );
		div.b = _translate( i.b );
		
		_add( div );
	}
	else
	{
		ir::LLVMUdiv div;
		
		div.d = _destination( i );
		div.a = _translate( i.a );
		div.b = _translate( i.b );
		
		_add( div );
	}
}

void PTXToLLVMTranslator::_translateEx2( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;

	#ifdef _WIN32
	call.name = "@llvm.pow.f32";
	#else
	call.name = "@llvm.exp2.f32";
	#endif
	
	if( i.modifier & ir::PTXInstruction::ftz )
	{
		call.d.name          = _tempRegister();
		call.d.type.type     = ir::LLVMInstruction::F32;
		call.d.type.category = ir::LLVMInstruction::Type::Element;
	}
	else
	{
		call.d = _destination( i );
	}
	
	#if _WIN32
	call.parameters.resize( 2 );
	call.parameters[0] = _translate( ir::PTXOperand(2.0f) );
	call.parameters[1] = _translate( i.a );
	#else
	call.parameters.resize( 1 );
	call.parameters[0] = _translate( i.a );
	#endif
	
	_add( call );

	if( i.modifier & ir::PTXInstruction::ftz )
	{
		_flushToZero( _destination( i ), call.d );
	}
}

void PTXToLLVMTranslator::_translateExit( const ir::PTXInstruction& i )
{
	_yield( executive::LLVMExecutableKernel::ExitCall );

	ir::LLVMBr branch;
	branch.iftrue = "%" + (--_dfg->end())->label();
	
	_add( branch );
}

void PTXToLLVMTranslator::_translateIsspacep( const ir::PTXInstruction& i )
{
	switch( i.addressSpace ) 
	{
	case ir::PTXInstruction::Shared: // fall through
	case ir::PTXInstruction::Local:
	{
		ir::LLVMInstruction::Operand base = _getMemoryBasePointer( 
		 	i.addressSpace, false, false );
		ir::LLVMInstruction::Operand extent = _getMemoryExtent(
			i.addressSpace );
			
		ir::LLVMInstruction::Operand baseInt = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I64, 
				ir::LLVMInstruction::Type::Element ) );
		ir::LLVMInstruction::Operand extentInt = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I64, 
				ir::LLVMInstruction::Type::Element ) );
		
		ir::LLVMPtrtoint ptrToInt;
	
		ptrToInt.a = base;
		ptrToInt.d = baseInt;
		
		_add( ptrToInt );
		
		_bitcast( extentInt, extent );

		ir::LLVMInstruction::Operand boundInt = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I64, 
				ir::LLVMInstruction::Type::Element ) );

		ir::LLVMAdd add;
		
		add.a = extentInt;
		add.b = baseInt;
		
		add.d = boundInt;

		_add( add );

		ir::LLVMInstruction::Operand geThanBase = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );

		ir::LLVMInstruction::Operand ltBound = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
				
		ir::LLVMInstruction::Operand a = _translate( i.a );
		
		ir::LLVMIcmp icmp;
				
		icmp.d = geThanBase;
		icmp.a = a;
		icmp.b = baseInt;
		icmp.comparison = ir::LLVMInstruction::Uge;
		
		_add( icmp );
		
		icmp.d = ltBound;
		icmp.b = boundInt;
		icmp.comparison = ir::LLVMInstruction::Ult;
		
		_add( icmp );
		
		ir::LLVMAnd land;
		
		land.d = _destination( i );
		land.a = ltBound;
		land.b = geThanBase;
		
		_add( land );
		
		break;
	}
	case ir::PTXInstruction::Global:
	{
		ir::LLVMInstruction::Operand base = _getMemoryBasePointer( 
		 	ir::PTXInstruction::Shared, false, false );
		ir::LLVMInstruction::Operand extent = _getMemoryExtent(
			ir::PTXInstruction::Shared );
			
		ir::LLVMInstruction::Operand baseInt = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I64, 
				ir::LLVMInstruction::Type::Element ) );
		ir::LLVMInstruction::Operand extentInt = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I64, 
				ir::LLVMInstruction::Type::Element ) );
		
		ir::LLVMPtrtoint ptrToInt;
	
		ptrToInt.a = base;
		ptrToInt.d = baseInt;
		
		_add( ptrToInt );
		
		_bitcast( extentInt, extent );

		ir::LLVMInstruction::Operand boundInt = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I64, 
				ir::LLVMInstruction::Type::Element ) );

		ir::LLVMAdd add;
		
		add.a = extentInt;
		add.b = baseInt;
		
		add.d = boundInt;

		_add( add );

		ir::LLVMInstruction::Operand geThanBase = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );

		ir::LLVMInstruction::Operand ltBound = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
				
		ir::LLVMInstruction::Operand isShared = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
				
		ir::LLVMInstruction::Operand a = _translate( i.a );
		
		ir::LLVMIcmp icmp;
				
		icmp.d = geThanBase;
		icmp.a = a;
		icmp.b = baseInt;
		icmp.comparison = ir::LLVMInstruction::Uge;
		
		_add( icmp );
		
		icmp.d = ltBound;
		icmp.b = boundInt;
		icmp.comparison = ir::LLVMInstruction::Ult;
		
		_add( icmp );
		
		ir::LLVMAnd land;
		
		land.d = isShared;
		land.a = geThanBase;
		land.b = ltBound;
		
		_add( land );
		
		// is the allocation local?
		base   = _getMemoryBasePointer( ir::PTXInstruction::Local,
			false, false );
		extent = _getMemoryExtent( ir::PTXInstruction::Local );
			
		baseInt.name = _tempRegister();
		extentInt.name = _tempRegister();
		boundInt.name = _tempRegister();
		
		ptrToInt.a = base;
		ptrToInt.d = baseInt;
		
		_add( ptrToInt );
		
		_bitcast( extentInt, extent );

		add.a = extentInt;
		add.b = baseInt;
		
		add.d = boundInt;

		_add( add );

		geThanBase.name = _tempRegister();
		ltBound.name = _tempRegister();
				
		ir::LLVMInstruction::Operand isLocal = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
				
		icmp.d = geThanBase;
		icmp.a = a;
		icmp.b = baseInt;
		icmp.comparison = ir::LLVMInstruction::Uge;
		
		_add( icmp );
		
		icmp.d = ltBound;
		icmp.b = boundInt;
		icmp.comparison = ir::LLVMInstruction::Ult;
		
		_add( icmp );
					
		land.d = isLocal;
		land.a = geThanBase;
		land.b = ltBound;
		
		_add( land );

		ir::LLVMInstruction::Operand isLocalOrShared = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
		
		ir::LLVMOr lor;
		
		lor.d = isLocalOrShared;
		lor.a = isLocal;
		lor.b = isShared;
		
		_add( lor );
		
		ir::LLVMXor lnot;
		
		lnot.d = _destination( i );
		lnot.a = isLocalOrShared;
		lnot.b.type = isLocalOrShared.type;
		lnot.b.constant = true;
		lnot.b.i64 = -1;
	
		_add( lnot );
		
		break;
	}
	default: assertM(false, "invalid address space"); break;
	}
}

void PTXToLLVMTranslator::_translateLd( const ir::PTXInstruction& i )
{
	#if(USE_VECTOR_INSTRUCTIONS == 1)
	ir::LLVMLoad load;
	
	if( i.d.vec != ir::PTXOperand::v1 )
	{
		load.d = _translate( i.d.array.front() );
		load.d.type.category = ir::LLVMInstruction::Type::Vector;
		load.d.type.vector = i.d.vec;
		load.d.type.type = _translate( i.type );
		load.d.name = _tempRegister();
	}
	else
	{
		load.d = _destination( i );
		load.d.type.type = _translate( i.type );
	}

	load.a = _getLoadOrStorePointer( i.a, i.addressSpace, 
		_translate( i.type ), i.vec );
	
	if( i.volatility == ir::PTXInstruction::Volatile )
	{
		load.isVolatile = true;
	}
	
	load.alignment = i.vec * ir::PTXOperand::bytes( i.type );
	
	if( i.d.array.empty() )
	{
		if( _translate( i.d.type ) != _translate( i.type ) )
		{
			ir::LLVMInstruction::Operand temp = load.d;
			temp.type.type = _translate( i.d.type );
			load.d.name = _tempRegister();
			_check( i.addressSpace, load.a, load.alignment,
				i.a.isArgument, i.a.isGlobalLocal, i.statementIndex );
			_add( load );
			_convert( temp, i.d.type, load.d, i.type );				
		}
		else
		{
			_check( i.addressSpace, load.a, load.alignment,
				i.a.isArgument, i.a.isGlobalLocal, i.statementIndex );
			_add( load );
		}
	}
	else
	{
		_check( i.addressSpace, load.a, load.alignment,
				i.a.isArgument, i.a.isGlobalLocal, i.statementIndex );
		_add( load );
	}
	
	for( ir::PTXOperand::Array::const_iterator 
		destination = i.d.array.begin(); 
		destination != i.d.array.end(); ++destination )
	{
		ir::LLVMInstruction::Operand target = _translate( *destination );
		
		ir::LLVMExtractelement extract;
		
		extract.d = target;
		extract.d.type.type = load.d.type.type;
		extract.a = load.d;
		extract.b.type.type = ir::LLVMInstruction::I32;
		extract.b.type.category = ir::LLVMInstruction::Type::Element;
		extract.b.constant = true;
		extract.b.i32 = std::distance( i.d.array.begin(), destination );
		
		if( destination->type != i.type )
		{
			ir::LLVMInstruction::Operand temp = target;
			extract.d.name = _tempRegister();
			_add( extract );
			_convert( temp, destination->type, extract.d, i.type );				
		}
		else
		{
			_add( extract );
		}
	}
	#else
	ir::LLVMLoad load;
	
	if( i.volatility == ir::PTXInstruction::Volatile )
	{
		load.isVolatile = true;
	}

	ir::LLVMInstruction::Operand address = _getLoadOrStorePointer( i.a, 
		i.addressSpace, _translate( i.type ), ir::PTXOperand::v1 );		
			
	if( i.d.array.empty() )
	{
		load.d = _destination( i );
		load.d.type.type = _translate( i.type );
		load.a = address;
		load.alignment = ir::PTXOperand::bytes( i.type );

		if( _translate( i.d.type ) != _translate( i.type ) )
		{
			ir::LLVMInstruction::Operand temp = load.d;
			temp.type.type = _translate( i.d.type );
			load.d.name = _tempRegister();
			_check( i.addressSpace, load.a, load.alignment,
				i.a.isArgument, i.a.isGlobalLocal, i.statementIndex );
			_add( load );
			_convert( temp, i.d.type, load.d, i.type );				
		}
		else
		{
			_check( i.addressSpace, load.a, load.alignment,
				i.a.isArgument, i.a.isGlobalLocal, i.statementIndex );
			_add( load );
		}
	}
	else
	{
		unsigned int index = 0;
		for( ir::PTXOperand::Array::const_iterator 
			destination = i.d.array.begin(); 
			destination != i.d.array.end(); ++destination, ++index )
		{
			ir::LLVMGetelementptr get;
		
			get.a = address;
			get.d = get.a;
			get.d.name = _tempRegister();
			get.indices.push_back( index );
		
			_add( get );
		
			load.d = _translate( *destination );
			load.d.type.type = _translate( i.type );
			load.alignment = ir::PTXOperand::bytes( i.type );
			load.a = get.d;
			_check( i.addressSpace, load.a, load.alignment,
					i.a.isArgument, i.a.isGlobalLocal, i.statementIndex );

			if( _translate( i.d.type ) != _translate( i.type ) )
			{
				ir::LLVMInstruction::Operand temp = load.d;
				temp.type.type = _translate( i.d.type );
				load.d.name = _tempRegister();
				_add( load );
				_convert( temp, i.d.type, load.d, i.type );				
			}
			else
			{
				_add( load );
			}
		}
	}
	#endif
}

void PTXToLLVMTranslator::_translateLdu( const ir::PTXInstruction& i )
{
	_translateLd( i );
}

void PTXToLLVMTranslator::_translateLg2( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	ir::LLVMInstruction::Operand destination;
	
	#ifdef _WIN32
	
	//   log2f(float x) = logf(x) * 1.44269504088896340736f
	call.name = "@llvm.log.f32";
	call.parameters.resize( 1 );
	call.d               = _tempRegister();
	call.d.type.type     = ir::LLVMInstruction::F32;
	call.d.type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[0]   = _translate( i.a );

	_add( call );

	ir::LLVMFmul mul;

	mul.d = _destination( i );
	mul.d.name = _tempRegister();
	mul.a = call.d;
	mul.b = _translate( ir::PTXOperand( 1.44269504088896340736f ) );

	_add( mul );

	destination = mul.d;
		
	#else
	
	call.name = "@llvm.log2.f32";	
	call.parameters.resize( 1 );
	
	if( i.modifier & ir::PTXInstruction::ftz )
	{
		call.d.name          = _tempRegister();
		call.d.type.type     = ir::LLVMInstruction::F32;
		call.d.type.category = ir::LLVMInstruction::Type::Element;

		call.parameters[0] = ir::LLVMInstruction::Operand( _tempRegister(),
			call.d.type );

		_flushToZero( call.parameters[0], _translate( i.a ) );
	}
	else
	{
		call.d = _destination( i );		
		call.parameters[0] = _translate( i.a );
	}
	
	_add( call );
	
	destination = call.d;
	#endif
	
	if( i.modifier & ir::PTXInstruction::ftz )
	{
		_flushToZero( _destination( i ), destination );
	}
	
}

void PTXToLLVMTranslator::_translateMad24( const ir::PTXInstruction& i )
{
	assertM( !( i.modifier & ir::PTXInstruction::sat ), 
		"No support for saturation in mad24" );
	assertM( !( i.modifier & ir::PTXInstruction::hi ), 
		"No support for hi multiply in mad24" );
	
	ir::LLVMInstruction::Operand destination = _destination( i );
	
	ir::LLVMMul multiply;
	
	multiply.d = destination;
	multiply.d.type.type = ir::LLVMInstruction::I64;
	multiply.d.name = _tempRegister();
	multiply.a = _translate( i.a );
	multiply.b = _translate( i.b );
	
	_add( multiply );
	
	ir::LLVMInstruction::Operand c = _translate( i.c );
	
	if( ir::PTXOperand::isSigned( i.c.type ) )
	{
		ir::LLVMSext extend;
		
		extend.d = c;
		extend.d.name = _tempRegister();
		extend.d.type = ir::LLVMInstruction::I64;
		extend.a = c;
		
		c = extend.d;
		_add( extend );
	}
	else
	{
		ir::LLVMZext extend;
		
		extend.d = c;
		extend.d.name = _tempRegister();
		extend.d.type = ir::LLVMInstruction::I64;
		extend.a = c;
		
		c = extend.d;
		_add( extend );
	}
	
	ir::LLVMAdd add;
	
	add.d = destination;
	add.d.name = _tempRegister();
	add.a = multiply.d;
	add.b = c;
	
	ir::LLVMTrunc truncate;
	
	truncate.d = destination;
	truncate.a = add.d;

	_add( truncate );
}

void PTXToLLVMTranslator::_translateMad( const ir::PTXInstruction& i )
{
	if( ir::PTXOperand::isFloat( i.type ) )
	{
		ir::LLVMFmul mul;
		ir::LLVMFadd add;

		ir::LLVMInstruction::Operand result = _destination( i );

		if( i.modifier & ir::PTXInstruction::ftz )
		{
			mul.a = ir::LLVMInstruction::Operand( _tempRegister(), 
				ir::LLVMInstruction::Type( ir::LLVMInstruction::F32, 
				ir::LLVMInstruction::Type::Element ) );
			mul.b = ir::LLVMInstruction::Operand( _tempRegister(), 
				ir::LLVMInstruction::Type( ir::LLVMInstruction::F32, 
				ir::LLVMInstruction::Type::Element ) );
			
			_flushToZero( mul.a, _translate( i.a ) );
			_flushToZero( mul.b, _translate( i.b ) );
		}
		else
		{
			mul.a = _translate( i.a );
			mul.b = _translate( i.b );
		}
		
		if( i.modifier & ir::PTXInstruction::sat
			|| i.modifier & ir::PTXInstruction::ftz )
		{
			add.d = mul.a;
			add.d.name = _tempRegister();
		}
		else
		{
			add.d = result;
		}
		
		mul.d = add.d;
		mul.d.name = _tempRegister();

		_add( mul );

		add.a = mul.d;
		add.b = _translate( i.c );
		
		_add( add );

		if( i.modifier & ir::PTXInstruction::sat )
		{
			if( i.modifier & ir::PTXInstruction::ftz )
			{
				ir::LLVMInstruction::Operand temp =
					ir::LLVMInstruction::Operand( _tempRegister(),
					add.d.type );
				_saturate( temp, add.d );
				_flushToZero( result, temp );
			}
			else
			{
				_saturate( result, add.d );
			}
		}
		else if( i.modifier & ir::PTXInstruction::ftz )
		{
			_flushToZero( result, add.d );
		}
	}
	else
	{
		if( i.modifier & ir::PTXInstruction::wide )
		{
			ir::LLVMInstruction::Operand extendedA = _translate( i.a );
			ir::LLVMInstruction::Operand extendedB = _translate( i.b );
			
			if( ir::PTXOperand::isSigned( i.a.type ) )
			{
				if( i.a.addressMode != ir::PTXOperand::Immediate )
				{
					ir::LLVMSext sextA;
				
					sextA.a = extendedA;
					_doubleWidth( extendedA.type.type );
					extendedA.name = _tempRegister();
					sextA.d = extendedA;
				
					_add( sextA );
				}
				else
				{
					_doubleWidth( extendedA.type.type );
				}
				
				if( i.b.addressMode != ir::PTXOperand::Immediate )
				{
					ir::LLVMSext sextB;
				
					sextB.a = extendedB;
					_doubleWidth( extendedB.type.type );
					extendedB.name = _tempRegister();
					sextB.d = extendedB;
				
					_add( sextB );
				}
				else
				{
					_doubleWidth( extendedB.type.type );
				}
			}
			else
			{
				if( i.a.addressMode != ir::PTXOperand::Immediate )
				{
					ir::LLVMZext sextA;
				
					sextA.a = extendedA;
					_doubleWidth( extendedA.type.type );
					extendedA.name = _tempRegister();
					sextA.d = extendedA;
				
					_add( sextA );

				}
				else
				{
					_doubleWidth( extendedA.type.type );
				}
				
				if( i.b.addressMode != ir::PTXOperand::Immediate )
				{
					ir::LLVMZext sextB;
				
					sextB.a = extendedB;
					_doubleWidth( extendedB.type.type );
					extendedB.name = _tempRegister();
					sextB.d = extendedB;
				
					_add( sextB );
				}
				else
				{
					_doubleWidth( extendedB.type.type );
				}
			}
			
			ir::LLVMMul mul;
			ir::LLVMAdd add;
			
			add.d = _destination( i );
			
			mul.d = add.d;
			mul.d.name = _tempRegister();	
			mul.a = extendedA;
			mul.b = extendedB;
		
			_add( mul );
			
			add.a = _translate( i.c );
			add.b = mul.d;
			
			_add( add );
		}
		else if( i.modifier & ir::PTXInstruction::lo )
		{
			ir::LLVMMul mul;
			ir::LLVMAdd add;
			
			add.d = _destination( i );
			
			mul.d = add.d;
			mul.d.name = _tempRegister();	
			mul.a = _translate( i.a );
			mul.b = _translate( i.b );
		
			_add( mul );
			
			add.a = _translate( i.c );
			add.b = mul.d;
			
			_add( add );
		}
		else
		{
			if( ir::PTXOperand::s64 == i.type )
			{
				ir::LLVMCall call;
				ir::LLVMAdd add;
				
				call.name = "@__ocelot_mul_hi_s64";
			
				add.d = _destination( i );
				call.d = add.d;
				call.d.name = _tempRegister();
				call.parameters.push_back( _translate( i.a ) );
				call.parameters.push_back( _translate( i.b ) );
				
				_add( call );
				
				add.a = call.d;
				add.b = _translate( i.c );
				
				_add( add );
			}
			else if( ir::PTXOperand::u64 == i.type )
			{
				ir::LLVMCall call;
				ir::LLVMAdd add;
				
				call.name = "@__ocelot_mul_hi_u64";
			
				add.d = _destination( i );
				call.d = add.d;
				call.d.name = _tempRegister();
				call.parameters.push_back( _translate( i.a ) );
				call.parameters.push_back( _translate( i.b ) );
				
				_add( call );
				
				add.a = call.d;
				add.b = _translate( i.c );
				
				_add( add );
			}
			else if( i.modifier & ir::PTXInstruction::sat )
			{
				assert( i.type == ir::PTXOperand::s32 );
		
				ir::LLVMSext extendA;
				ir::LLVMSext extendB;
				ir::LLVMSext extendC;
						
				extendA.a = _translate( i.a );
				extendA.d.type.type = ir::LLVMInstruction::I64;
				extendA.d.type.category = 
					ir::LLVMInstruction::Type::Element;
				extendA.d.name = _tempRegister();
		
				_add( extendA );
		
				extendB.a = _translate( i.b );
				extendB.d.type.type = ir::LLVMInstruction::I64;
				extendB.d.type.category = 
					ir::LLVMInstruction::Type::Element;
				extendB.d.name = _tempRegister();
		
				_add( extendB );

				extendC.a = _translate( i.c );
				extendC.d.type.type = ir::LLVMInstruction::I64;
				extendC.d.type.category = 
					ir::LLVMInstruction::Type::Element;
				extendC.d.name = _tempRegister();
		
				_add( extendC );
		
				ir::LLVMMul mul;
		
				mul.a = extendA.d;
				mul.b = extendB.d;
				mul.d.name = _tempRegister();
				mul.d.type.type = ir::LLVMInstruction::I64;
				mul.d.type.category = ir::LLVMInstruction::Type::Element;
				
				_add( mul );
				
				ir::LLVMAshr shift;
			
				shift.d.name = _tempRegister();
				shift.d.type.type = ir::LLVMInstruction::I64;
				shift.d.type.category = ir::LLVMInstruction::Type::Element;
				shift.a = mul.d;
				shift.b.constant = true;
				shift.b.type.category = ir::LLVMInstruction::Type::Element;
				shift.b.type.type = ir::LLVMInstruction::I32;
				shift.b.i32 = 32;
			
				_add( shift );
		
				ir::LLVMAdd add;
				
				add.a = shift.d;
				add.b = extendC.d;
				add.d.name = _tempRegister();
				add.d.type.type = ir::LLVMInstruction::I64;
				add.d.type.category = ir::LLVMInstruction::Type::Element;
				
				_add( add );
				
				ir::LLVMIcmp compare;
		
				compare.d.name = _tempRegister();
				compare.d.type.type = ir::LLVMInstruction::I1;
				compare.d.type.category = 
					ir::LLVMInstruction::Type::Element;
				compare.comparison = ir::LLVMInstruction::Slt;
				compare.a = add.d;
				compare.b.type.type = ir::LLVMInstruction::I64;
				compare.b.type.category = 
					ir::LLVMInstruction::Type::Element;
				compare.b.constant = true;
				compare.b.i64 = INT_MIN;

				_add( compare );
		
				ir::LLVMSelect select;
		
				select.d.name = _tempRegister();
				select.d.type.type = ir::LLVMInstruction::I64;
				select.d.type.category = ir::LLVMInstruction::Type::Element;

				select.condition = compare.d;
				select.a = compare.b;
				select.b = compare.a;
		
				_add( select );
		
				compare.d.name = _tempRegister();
				compare.comparison = ir::LLVMInstruction::Sgt;
				compare.b.i64 = INT_MAX;
				compare.a = select.d;
		
				_add( compare );

				select.condition = compare.d;
				select.a = compare.b;
				select.b = compare.a;
				select.d.name = _tempRegister();

				_add( select );
		
				ir::LLVMTrunc truncate;
		
				truncate.a = select.d;
				truncate.d = _destination( i );
		
				_add( truncate );
			}
			else
			{
				ir::LLVMInstruction::Operand 
					destination = _destination( i );
				ir::LLVMInstruction::Operand extendedA = _translate( i.a );
				ir::LLVMInstruction::Operand extendedB = _translate( i.b );
			
				if( ir::PTXOperand::isSigned( i.a.type ) )
				{
					if( i.a.addressMode != ir::PTXOperand::Immediate )
					{
						ir::LLVMSext sextA;
				
						sextA.a = extendedA;
						_doubleWidth( extendedA.type.type );
						extendedA.name = _tempRegister();
						sextA.d = extendedA;
				
						_add( sextA );
					}
					else
					{
						_doubleWidth( extendedA.type.type );
					}
				
					if( i.b.addressMode != ir::PTXOperand::Immediate )
					{
						ir::LLVMSext sextB;
				
						sextB.a = extendedB;
						_doubleWidth( extendedB.type.type );
						extendedB.name = _tempRegister();
						sextB.d = extendedB;
				
						_add( sextB );
					}
					else
					{
						_doubleWidth( extendedB.type.type );
					}
				
				}
				else
				{
					if( i.a.addressMode != ir::PTXOperand::Immediate )
					{
						ir::LLVMZext sextA;
				
						sextA.a = extendedA;
						_doubleWidth( extendedA.type.type );
						extendedA.name = _tempRegister();
						sextA.d = extendedA;
				
						_add( sextA );

					}
					else
					{
						_doubleWidth( extendedA.type.type );
					}
				
					if( i.b.addressMode != ir::PTXOperand::Immediate )
					{
						ir::LLVMZext sextB;
				
						sextB.a = extendedB;
						_doubleWidth( extendedB.type.type );
						extendedB.name = _tempRegister();
						sextB.d = extendedB;
				
						_add( sextB );
					}
					else
					{
						_doubleWidth( extendedB.type.type );
					}
				}
			
				ir::LLVMMul mul;
					
				mul.d = extendedA;
				mul.d.name = _tempRegister();
				mul.a = extendedA;
				mul.b = extendedB;
		
				_add( mul );
			
				ir::LLVMInstruction::Operand 
					shiftedDestination = destination;
				shiftedDestination.name = _tempRegister();
				_doubleWidth( shiftedDestination.type.type );
			
				if( ir::PTXOperand::isSigned( i.a.type ) )
				{
					ir::LLVMAshr shift;
				
					shift.d = shiftedDestination;
					shift.a = mul.d;
					shift.b.constant = true;
					shift.b.type.category = 
						ir::LLVMInstruction::Type::Element;
					shift.b.type.type = ir::LLVMInstruction::I32;
					shift.b.i32 = ir::PTXOperand::bytes( i.a.type ) * 8;
				
					_add( shift );
				}
				else
				{
					ir::LLVMLshr shift;
				
					shift.d = shiftedDestination;
					shift.a = mul.d;
					shift.b.constant = true;
					shift.b.type.category = 
						ir::LLVMInstruction::Type::Element;
					shift.b.type.type = ir::LLVMInstruction::I32;
					shift.b.i32 = ir::PTXOperand::bytes( i.a.type ) * 8;
				
					_add( shift );
				}
			
				ir::LLVMTrunc truncate;
			
				truncate.d = destination;
				truncate.d.name = _tempRegister();
				truncate.a = shiftedDestination;
				
				_add( truncate );
				
				ir::LLVMAdd add;
				
				add.d = destination;
				add.a = truncate.d;
				add.b = _translate( i.c );
				
				_add( add );
			}
		}
	}
}

void PTXToLLVMTranslator::_translateMax( const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand destination = _destination( i );
	ir::LLVMInstruction::Operand comparison;
	
	comparison.type.category = ir::LLVMInstruction::Type::Element;
	comparison.type.type = ir::LLVMInstruction::I1;
	comparison.name = _tempRegister();
	
	if( ir::PTXOperand::isFloat( i.a.type ) )
	{
		ir::LLVMFcmp compare;
		
		compare.d = comparison;
		compare.a = _translate( i.a );
		compare.b = _translate( i.b );
		compare.comparison = ir::LLVMInstruction::Ogt;
		
		ir::LLVMFcmp isNan;
		
		isNan.comparison = ir::LLVMInstruction::Uno;
		isNan.a.type = compare.a.type;
		isNan.a.constant = true;
		isNan.a.i64 = 0;
		isNan.b = compare.b;
		isNan.d = ir::LLVMInstruction::Operand( _tempRegister(),
			compare.d.type );
		
		_add( isNan );
		
		ir::LLVMSelect selectNan;
		
		selectNan.condition = isNan.d;
		selectNan.d = ir::LLVMInstruction::Operand( _tempRegister(),
			destination.type );
		selectNan.a = compare.a;
		selectNan.b = compare.b;
		
		_add( selectNan );
		
		ir::LLVMSelect select; 
		
		select.condition = compare.d;
		select.a = compare.a;
		select.b = selectNan.d;
		
		if( i.modifier & ir::PTXInstruction::ftz )
		{
			select.d = ir::LLVMInstruction::Operand( _tempRegister(),
				destination.type );
		}
		else
		{
			select.d = destination;
		}

		_add( compare );
		_add( select );
		
		if( i.modifier & ir::PTXInstruction::ftz )
		{
			_flushToZero( destination, select.d );
		}
	}
	else
	{
		ir::LLVMIcmp compare;
		
		compare.d = comparison;
		compare.a = _translate( i.a );
		compare.b = _translate( i.b );
		
		if( ir::PTXOperand::isSigned( i.type ) )
		{
			compare.comparison = ir::LLVMInstruction::Sgt;
		}
		else
		{
			compare.comparison = ir::LLVMInstruction::Ugt;
		}
		
		ir::LLVMSelect select; 
		
		select.condition = compare.d;
		select.a = compare.a;
		select.b = compare.b;
		select.d = destination;
		
		_add( compare );
		_add( select );
	}
	
}

void PTXToLLVMTranslator::_translateMembar( const ir::PTXInstruction& i )
{
	// This is a nop right now
}

void PTXToLLVMTranslator::_translateMin( const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand destination = _destination( i );
	ir::LLVMInstruction::Operand comparison;
	
	comparison.type.category = ir::LLVMInstruction::Type::Element;
	comparison.type.type = ir::LLVMInstruction::I1;
	comparison.name = _tempRegister();
	
	if( ir::PTXOperand::isFloat( i.a.type ) )
	{
		ir::LLVMFcmp compare;
		
		compare.d = comparison;
		compare.a = _translate( i.a );
		compare.b = _translate( i.b );
		compare.comparison = ir::LLVMInstruction::Olt;
		
		ir::LLVMFcmp isNan;
		
		isNan.comparison = ir::LLVMInstruction::Uno;
		isNan.a.type = compare.a.type;
		isNan.a.constant = true;
		isNan.a.i64 = 0;
		isNan.b = compare.b;
		isNan.d = ir::LLVMInstruction::Operand( _tempRegister(),
			compare.d.type );
		
		_add( isNan );
		
		ir::LLVMSelect selectNan;
		
		selectNan.condition = isNan.d;
		selectNan.d = ir::LLVMInstruction::Operand( _tempRegister(),
			destination.type );
		selectNan.a = compare.a;
		selectNan.b = compare.b;
		
		_add( selectNan );
		
		ir::LLVMSelect select; 
		
		select.condition = compare.d;
		select.a = compare.a;
		select.b = selectNan.d;

		if( i.modifier & ir::PTXInstruction::ftz )
		{
			select.d = ir::LLVMInstruction::Operand( _tempRegister(),
				destination.type );
		}
		else
		{
			select.d = destination;
		}

		_add( compare );
		_add( select );
		
		if( i.modifier & ir::PTXInstruction::ftz )
		{
			_flushToZero( destination, select.d );
		}
	}
	else
	{
		ir::LLVMIcmp compare;
		
		compare.d = comparison;
		compare.a = _translate( i.a );
		compare.b = _translate( i.b );
		
		if( ir::PTXOperand::isSigned( i.type ) )
		{
			compare.comparison = ir::LLVMInstruction::Slt;
		}
		else
		{
			compare.comparison = ir::LLVMInstruction::Ult;
		}
		
		ir::LLVMSelect select; 
		
		select.condition = compare.d;
		select.a = compare.a;
		select.b = compare.b;
		select.d = destination;
		
		_add( compare );
		_add( select );
	}
	
}

void PTXToLLVMTranslator::_translateMov( const ir::PTXInstruction& i )
{
	switch( i.d.vec )
	{
		case ir::PTXOperand::v1:
		{
			switch( i.a.vec )
			{
				case ir::PTXOperand::v1:
				{
					if( i.a.addressMode == ir::PTXOperand::Address
						|| i.a.addressMode == ir::PTXOperand::FunctionName
						|| i.a.addressMode == ir::PTXOperand::Indirect )
					{
						if( i.addressSpace == ir::PTXInstruction::Global )
						{
							ir::LLVMPtrtoint toint;
			
							toint.a = _getAddressableGlobalPointer( i.a );

							if( i.a.offset == 0 )
							{
								toint.d = _destination( i );
			
								_add( toint );
							}
							else
							{
								toint.d = ir::LLVMInstruction::Operand( 
									_tempRegister(),
									ir::LLVMInstruction::Type( 
									_translate( i.type ), 
									ir::LLVMInstruction::Type::Element ) );
								
								_add( toint );
								
								ir::LLVMAdd add;
								
								add.a          = toint.d;
								add.d          = _destination( i );
								add.b.constant = true;
								add.b.type     = add.a.type;
								add.b.i64      = i.a.offset;
								
								_add( add );
							}
						}
						else if( i.a.addressMode == ir::PTXOperand::Address &&
							i.addressSpace == ir::PTXInstruction::Local &&
							i.a.isGlobalLocal)
						{
							ir::LLVMPtrtoint cast;
		
							cast.d = _destination( i );
							cast.a = _getAddressableVariablePointer(
								i.addressSpace, i.a);
							
							_add( cast );
						}
						else
						{
							ir::LLVMBitcast cast;
		
							cast.d = _destination( i );
							cast.a.type.category = cast.d.type.category;
							cast.a.type.type = cast.d.type.type;
							cast.a.constant = true;

							if( i.a.addressMode == ir::PTXOperand::Address )
							{
								cast.a.i64 = i.a.offset;
							}
							else
							{
								cast.a.i64 = i.reentryPoint;
							}

							_add( cast );
						}
					}
					else if( i.d.type == i.a.type 
						|| i.type == ir::PTXOperand::b32 
						|| i.type == ir::PTXOperand::b64 
						|| i.type == ir::PTXOperand::b16 
						|| i.type == ir::PTXOperand::b8 )
					{
						_bitcast( i );
					}
					else
					{
						_translateCvt( i );
					}
					break;
				}
				case ir::PTXOperand::v2:
				{
					assertM( i.a.addressMode != ir::PTXOperand::Address, 
						"Addressable variables not supported" 
						<< " for vector moves." );
					
					ir::LLVMInstruction::Operand temp;
					
					temp.name = _tempRegister();
					temp.type.category 
						= ir::LLVMInstruction::Type::Element;
					temp.type.type = ir::LLVMInstruction::getIntOfSize( 
						ir::PTXOperand::bytes( i.d.type ) * 8 );
					
					_bitcast( temp, _translate( i.a.array[ 1 ] ) );
					
					ir::LLVMShl shift;
					
					shift.d = temp;
					shift.d.name = _tempRegister();
					shift.a = temp;
					
					shift.b.type.category 
						= ir::LLVMInstruction::Type::Element;
					shift.b.type.type = ir::LLVMInstruction::I32;
					shift.b.constant = true;
					shift.b.i32 = ir::PTXOperand::bytes( 
						i.a.array[ 1 ].type ) * 8;
					
					_add( shift );
					
					temp.name = _tempRegister();
					_bitcast( temp, _translate( i.a.array[ 0 ] ) );
					
					ir::LLVMOr combine;
					
					combine.d = temp;
					combine.d.name = _tempRegister();
					combine.a = temp;
					combine.b = shift.d;
					
					_add( combine );
					
					_bitcast( _destination( i ), combine.d );
					
					break;
				}
				case ir::PTXOperand::v4:
				{
					assertM( i.a.addressMode != ir::PTXOperand::Address, 
						"Addressable variables not supported" 
						<< " for vector moves." );
					assertM( false, 
						"Vector move from v" << i.a.vec << " to v" 
						<< i.d.vec << " not implemented." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::v2:
		{
			switch( i.a.vec )
			{
				case ir::PTXOperand::v1:
				{
					assertM( i.a.addressMode != ir::PTXOperand::Address, 
						"Addressable variables not supported" 
						<< " for vector moves." );
					ir::LLVMInstruction::Operand temp;
					
					temp.name = _tempRegister();
					temp.type.category 
						= ir::LLVMInstruction::Type::Element;
					temp.type.type = ir::LLVMInstruction::getIntOfSize( 
						ir::PTXOperand::bytes( i.a.type ) * 8 );
					
					_bitcast( temp, _translate( i.a ) );
					
					ir::LLVMTrunc truncate;
					
					truncate.a = temp;
					truncate.d.name = _tempRegister();
					truncate.d.type.category 
						= ir::LLVMInstruction::Type::Element;
					truncate.d.type.type 
						= ir::LLVMInstruction::getIntOfSize( 
						ir::PTXOperand::bytes( i.d.array[0].type ) * 8 );
					
					_add( truncate );
					_bitcast( _translate( i.d.array[0] ), truncate.d );
					
					ir::LLVMLshr shift;
					
					shift.a = temp;
					shift.d = temp;
					shift.d.name = _tempRegister();
					
					shift.b.type.category 
						= ir::LLVMInstruction::Type::Element;
					shift.b.type.type = ir::LLVMInstruction::I32;
					shift.b.constant = true;
					shift.b.i32 = ir::PTXOperand::bytes( 
						i.d.array[ 0 ].type ) * 8;
					
					_add( shift );
					
					truncate.a = shift.d;
					truncate.d.name = _tempRegister();
					
					_add( truncate );
					_bitcast( _translate( i.d.array[1] ), truncate.d );
					
					break;
				}
				case ir::PTXOperand::v2:
				{
					assertM( i.a.addressMode != ir::PTXOperand::Address, 
						"Addressable variables not supported" 
						<< " for vector moves." );
					assertM( false, 
						"Vector move from v" << i.a.vec << " to v" 
						<< i.d.vec << " not implemented." );
					break;
				}
				case ir::PTXOperand::v4:
				{
					assertM( i.a.addressMode != ir::PTXOperand::Address, 
						"Addressable variables not supported" 
						<< " for vector moves." );
					assertM( false, 
						"Vector move from v" << i.a.vec << " to v" 
						<< i.d.vec << " not implemented." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::v4:
		{
			switch( i.a.vec )
			{
				case ir::PTXOperand::v1:
				{
					assertM( i.a.addressMode != ir::PTXOperand::Address, 
						"Addressable variables not supported" 
						<< " for vector moves." );
					assertM( false, 
						"Vector move from v" << i.a.vec << " to v" 
						<< i.d.vec << " not implemented." );
					break;
				}
				case ir::PTXOperand::v2:
				{
					assertM( i.a.addressMode != ir::PTXOperand::Address, 
						"Addressable variables not supported" 
						<< " for vector moves." );
					assertM( false, 
						"Vector move from v" << i.a.vec << " to v" 
						<< i.d.vec << " not implemented." );
					break;
				}
				case ir::PTXOperand::v4:
				{
					assertM( i.a.addressMode != ir::PTXOperand::Address, 
						"Addressable variables not supported" 
						<< " for vector moves." );
					assertM( false, 
						"Vector move from v" << i.a.vec << " to v" 
						<< i.d.vec << " not implemented." );
					break;
				}
			}
			break;
		}
	}
}

void PTXToLLVMTranslator::_translateMul24( const ir::PTXInstruction& i )
{
	if( i.modifier & ir::PTXInstruction::lo )
	{
		// 24-bit lo is the same as 32-bit lo
		ir::LLVMMul mul;
				
		mul.d = _destination( i );
		mul.a = _translate( i.a );
		mul.b = _translate( i.b );
	
		_add( mul );
	}
	else
	{
		assertM( false, "No support for hi 24-bit multiply" );
	}
}

void PTXToLLVMTranslator::_translateMul( const ir::PTXInstruction& i )
{
	if( ir::PTXOperand::isFloat( i.type ) )
	{
		ir::LLVMFmul mul;
	
		ir::LLVMInstruction::Operand result = _destination( i );

		if( i.modifier & ir::PTXInstruction::ftz )
		{
			mul.a = ir::LLVMInstruction::Operand( _tempRegister(), 
				ir::LLVMInstruction::Type( ir::LLVMInstruction::F32, 
				ir::LLVMInstruction::Type::Element ) );
			mul.b = ir::LLVMInstruction::Operand( _tempRegister(), 
				ir::LLVMInstruction::Type( ir::LLVMInstruction::F32, 
				ir::LLVMInstruction::Type::Element ) );
			
			_flushToZero( mul.a, _translate( i.a ) );
			_flushToZero( mul.b, _translate( i.b ) );
		}
		else
		{
			mul.a = _translate( i.a );
			mul.b = _translate( i.b );
		}
		
		if( i.modifier & ir::PTXInstruction::sat
			|| i.modifier & ir::PTXInstruction::ftz )
		{
			mul.d = ir::LLVMInstruction::Operand( _tempRegister(), 
				ir::LLVMInstruction::Type( ir::LLVMInstruction::F32, 
				ir::LLVMInstruction::Type::Element ) );
		}
		else
		{
			mul.d = result;
		}

		_add( mul );
		
		if( i.modifier & ir::PTXInstruction::sat )
		{
			if( i.modifier & ir::PTXInstruction::ftz )
			{
				ir::LLVMInstruction::Operand temp =
					ir::LLVMInstruction::Operand( _tempRegister(),
					mul.d.type );
				_saturate( temp, mul.d );
				_flushToZero( result, temp );
			}
			else
			{
				_saturate( result, mul.d );
			}
		}
		else if( i.modifier & ir::PTXInstruction::ftz )
		{
			_flushToZero( result, mul.d );
		}
	}
	else
	{
		if( i.modifier & ir::PTXInstruction::wide )
		{
			ir::LLVMInstruction::Operand extendedA = _translate( i.a );
			ir::LLVMInstruction::Operand extendedB = _translate( i.b );
			
			if( ir::PTXOperand::isSigned( i.type ) )
			{
				if( i.a.addressMode != ir::PTXOperand::Immediate )
				{
					ir::LLVMSext sextA;
				
					sextA.a = extendedA;
					_doubleWidth( extendedA.type.type );
					extendedA.name = _tempRegister();
					sextA.d = extendedA;
				
					_add( sextA );
				}
				else
				{
					_doubleWidth( extendedA.type.type );
				}
				
				if( i.b.addressMode != ir::PTXOperand::Immediate )
				{
					ir::LLVMSext sextB;
				
					sextB.a = extendedB;
					_doubleWidth( extendedB.type.type );
					extendedB.name = _tempRegister();
					sextB.d = extendedB;
				
					_add( sextB );
				}
				else
				{
					_doubleWidth( extendedB.type.type );
				}
			}
			else
			{
				if( i.a.addressMode != ir::PTXOperand::Immediate )
				{
					ir::LLVMZext sextA;
				
					sextA.a = extendedA;
					_doubleWidth( extendedA.type.type );
					extendedA.name = _tempRegister();
					sextA.d = extendedA;
				
					_add( sextA );

				}
				else
				{
					_doubleWidth( extendedA.type.type );
				}
				
				if( i.b.addressMode != ir::PTXOperand::Immediate )
				{
					ir::LLVMZext sextB;
				
					sextB.a = extendedB;
					_doubleWidth( extendedB.type.type );
					extendedB.name = _tempRegister();
					sextB.d = extendedB;
				
					_add( sextB );
				}
				else
				{
					_doubleWidth( extendedB.type.type );
				}
			}
			
			ir::LLVMMul mul;
					
			mul.d = _destination( i );
			mul.a = extendedA;
			mul.b = extendedB;
		
			_add( mul );
		}
		else if( i.modifier & ir::PTXInstruction::lo )
		{
			ir::LLVMMul mul;
					
			mul.d = _destination( i );
			mul.a = _translate( i.a );
			mul.b = _translate( i.b );
		
			_add( mul );
		}
		else
		{
			if( ir::PTXOperand::s64 == i.type )
			{
				ir::LLVMCall call;
				call.name = "@__ocelot_mul_hi_s64";
			
				call.d = _destination( i );
				call.parameters.push_back( _translate( i.a ) );
				call.parameters.push_back( _translate( i.b ) );
				
				_add( call );
			}
			else if( ir::PTXOperand::u64 == i.type )
			{
				ir::LLVMCall call;
				call.name = "@__ocelot_mul_hi_u64";
			
				call.d = _destination( i );
				call.parameters.push_back( _translate( i.a ) );
				call.parameters.push_back( _translate( i.b ) );
				
				_add( call );
			}
			else
			{
				ir::LLVMInstruction::Operand 
					destination = _destination( i );
				ir::LLVMInstruction::Operand extendedA = _translate( i.a );
				ir::LLVMInstruction::Operand extendedB = _translate( i.b );
			
				if( ir::PTXOperand::isSigned( i.a.type ) )
				{
					if( i.a.addressMode != ir::PTXOperand::Immediate )
					{
						ir::LLVMSext sextA;
				
						sextA.a = extendedA;
						_doubleWidth( extendedA.type.type );
						extendedA.name = _tempRegister();
						sextA.d = extendedA;
				
						_add( sextA );
					}
					else
					{
						_doubleWidth( extendedA.type.type );
					}
				
					if( i.b.addressMode != ir::PTXOperand::Immediate )
					{
						ir::LLVMSext sextB;
				
						sextB.a = extendedB;
						_doubleWidth( extendedB.type.type );
						extendedB.name = _tempRegister();
						sextB.d = extendedB;
				
						_add( sextB );
					}
					else
					{
						_doubleWidth( extendedB.type.type );
					}
				}
				else
				{
					if( i.a.addressMode != ir::PTXOperand::Immediate )
					{
						ir::LLVMZext sextA;
				
						sextA.a = extendedA;
						_doubleWidth( extendedA.type.type );
						extendedA.name = _tempRegister();
						sextA.d = extendedA;
				
						_add( sextA );

					}
					else
					{
						_doubleWidth( extendedA.type.type );
					}
				
					if( i.b.addressMode != ir::PTXOperand::Immediate )
					{
						ir::LLVMZext sextB;
				
						sextB.a = extendedB;
						_doubleWidth( extendedB.type.type );
						extendedB.name = _tempRegister();
						sextB.d = extendedB;
				
						_add( sextB );
					}
					else
					{
						_doubleWidth( extendedB.type.type );
					}
				}
			
				ir::LLVMMul mul;
					
				mul.d = extendedA;
				mul.d.name = _tempRegister();
				mul.a = extendedA;
				mul.b = extendedB;
		
				_add( mul );
			
				ir::LLVMInstruction::Operand 
					shiftedDestination = destination;
				shiftedDestination.name = _tempRegister();
				_doubleWidth( shiftedDestination.type.type );
			
				if( ir::PTXOperand::isSigned( i.a.type ) )
				{
					ir::LLVMAshr shift;
				
					shift.d = shiftedDestination;
					shift.a = mul.d;
					shift.b.constant = true;
					shift.b.type.category = 
						ir::LLVMInstruction::Type::Element;
					shift.b.type.type = ir::LLVMInstruction::I32;
					shift.b.i32 = ir::PTXOperand::bytes( i.a.type ) * 8;
				
					_add( shift );
				}
				else
				{
					ir::LLVMLshr shift;
				
					shift.d = shiftedDestination;
					shift.a = mul.d;
					shift.b.constant = true;
					shift.b.type.category = 
						ir::LLVMInstruction::Type::Element;
					shift.b.type.type = ir::LLVMInstruction::I32;
					shift.b.i32 = ir::PTXOperand::bytes( i.a.type ) * 8;
				
					_add( shift );
				}
			
				ir::LLVMTrunc truncate;
			
				truncate.d = destination;
				truncate.a = shiftedDestination;
			
				_add( truncate );
			}
		}
	}
}

void PTXToLLVMTranslator::_translateNeg( const ir::PTXInstruction& i )
{
	if( ir::PTXOperand::isFloat( i.type ) )
	{
		ir::LLVMFsub sub;

		sub.b = _translate( i.a );
		sub.a.type = sub.b.type;
		sub.a.constant = true;
		sub.a.i64 = 0;
	
		if( i.modifier & ir::PTXInstruction::ftz )
		{
			sub.d = ir::LLVMInstruction::Operand( _tempRegister(),
				sub.b.type );
		}
		else
		{
			sub.d = _destination( i );
		}
	
		_add( sub );
		
		if( i.modifier & ir::PTXInstruction::ftz )
		{
			_flushToZero( _destination( i ), sub.d );
		}
	}
	else
	{
		ir::LLVMSub sub;
	
		sub.d = _destination( i );
		sub.b = _translate( i.a );
		sub.a = sub.b;
		sub.a.constant = true;
		sub.a.i64 = 0;
	
		_add( sub );
	}
}

void PTXToLLVMTranslator::_translateNot( const ir::PTXInstruction& i )
{
	ir::LLVMXor Not;
	
	Not.d = _destination( i );
	Not.a = _translate( i.a );
	Not.b = Not.a;
	Not.b.constant = true;
	Not.b.i64 = -1;
	
	_add( Not );
}

void PTXToLLVMTranslator::_translateOr( const ir::PTXInstruction& i )
{
	ir::LLVMOr Or;
	
	Or.d = _destination( i );
	Or.a = _translate( i.a );
	Or.b = _translate( i.b );

	_add( Or );
}

void PTXToLLVMTranslator::_translatePmevent( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	call.name = "@pmevent";
	
	call.parameters.resize( 1 );
	call.parameters[0] = _translate( i.a );
	
	_add( call );
}

void PTXToLLVMTranslator::_translatePopc( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	if( i.type == ir::PTXOperand::b32 )
	{
		call.name = "@llvm.ctpop.i32";
		call.d = _destination( i );
	}
	else
	{
		call.name = "@llvm.ctpop.i64";
		call.d = ir::LLVMInstruction::Operand( _tempRegister(), 
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I64, 
			ir::LLVMInstruction::Type::Element ) );
	}
	
	call.parameters.resize( 1 );
	call.parameters[0] = _translate( i.a );
	
	_add( call );
	
	if( i.type != ir::PTXOperand::b32 )
	{
		ir::LLVMTrunc truncate;
		
		truncate.d = _destination( i );
		truncate.a = call.d;
		
		_add( truncate );
	}
}

void PTXToLLVMTranslator::_translatePrmt( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;

	switch( i.permuteMode )
	{
		case ir::PTXInstruction::DefaultPermute:
		{
			call.name = "@__ocelot_prmt";
			break;
		}
		case ir::PTXInstruction::ForwardFourExtract:
		{
			call.name = "@__ocelot_prmt_f4e";
			break;
		}
		case ir::PTXInstruction::BackwardFourExtract:
		{
			call.name = "@__ocelot_prmt_b4e";
			break;
		}
		case ir::PTXInstruction::ReplicateEight:
		{
			call.name = "@__ocelot_prmt_rc8";
			break;
		}
		case ir::PTXInstruction::EdgeClampLeft:
		{
			call.name = "@__ocelot_prmt_ecl";
			break;
		}
		case ir::PTXInstruction::EdgeClampRight:
		{
			call.name = "@__ocelot_prmt_ecr";
			break;
		}
		case ir::PTXInstruction::ReplicateSixteen:
		{
			call.name = "@__ocelot_prmt_rc16";
			break;
		}
	}
	
	call.d = _destination( i );
	call.parameters.push_back( _translate( i.a ) );
	call.parameters.push_back( _translate( i.b ) );
	call.parameters.push_back( _translate( i.c ) );

	_add( call );
}

void PTXToLLVMTranslator::_translateRcp( const ir::PTXInstruction& i )
{
	ir::LLVMFdiv div;
	
	div.b = _translate( i.a );
	div.a.type = div.b.type;
	div.a.constant = true;
	
	if( i.a.type == ir::PTXOperand::f32 )
	{
		div.a.f32 = 1.0f;
	}
	else
	{
		div.a.f64 = 1.0;
	}
	
	if( i.modifier & ir::PTXInstruction::ftz
		&& i.type == ir::PTXOperand::f32 )
	{
		div.d = ir::LLVMInstruction::Operand( _tempRegister(), div.a.type );
	}
	else
	{
		div.d = _destination( i );
	}
	
	_add( div );

	if( i.modifier & ir::PTXInstruction::ftz
		&& i.type == ir::PTXOperand::f32 )
	{
		_flushToZero( _destination( i ), div.d );
	}
}

void PTXToLLVMTranslator::_translateRed( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	call.name = "@reduction";
	
	call.parameters.resize( 4 );
	call.parameters[0].type.type = ir::LLVMInstruction::I32;
	call.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[0].i32 = i.addressSpace;

	call.parameters[1].type.type = ir::LLVMInstruction::I32;
	call.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[1].i32 = i.reductionOperation;

	call.parameters[2] = _translate( i.a );
	call.parameters[3] = _translate( i.b );
	
	_add( call );
}

void PTXToLLVMTranslator::_translateRem( const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand destination = _destination( i );
	
	if( ir::PTXOperand::isSigned( i.type ) )
	{
		ir::LLVMSrem rem;
		
		rem.d = destination;
		rem.a = _translate( i.a );
		rem.b = _translate( i.b );
		
		_add( rem );
	}
	else
	{
		ir::LLVMUrem rem;
		
		rem.d = destination;
		rem.a = _translate( i.a );
		rem.b = _translate( i.b );
		
		_add( rem );			
	}
}

void PTXToLLVMTranslator::_translateRet( const ir::PTXInstruction& i, 
	const analysis::DataflowGraph::Block& block )
{
    _yield( executive::LLVMExecutableKernel::ReturnCall );

    if( !block.targets().empty() )
	{
		ir::LLVMBr branch;
	
		branch.iftrue = "%" + (*block.targets().begin())->label();
		_add( branch );
	}
}

void PTXToLLVMTranslator::_translateRsqrt( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	if( i.type == ir::PTXOperand::f32 )
	{
		call.name = "@llvm.sqrt.f32";
	}
	else
	{
		call.name = "@llvm.sqrt.f64";
	}

	call.d = ir::LLVMInstruction::Operand( 
		_tempRegister(), ir::LLVMInstruction::Type( _translate( i.type ), 
		ir::LLVMInstruction::Type::Element ) );

	call.parameters.resize( 1 );
	
	if( i.modifier & ir::PTXInstruction::ftz )
	{
		call.parameters[0] = ir::LLVMInstruction::Operand( 
			_tempRegister(), call.d.type );
		
		_flushToZero( call.parameters[0], _translate( i.a ) );
	}
	else
	{
		call.parameters[0] = _translate( i.a );
	}

	_add( call );
	
	ir::LLVMFdiv divide;
	
	if( i.modifier & ir::PTXInstruction::ftz )
	{
		divide.d.type = _translate( i.d.type );
		divide.d.type.category = ir::LLVMInstruction::Type::Element;
		divide.d.name = _tempRegister();
	}
	else
	{			
		divide.d = _destination( i );
	}
	
	divide.a.type = call.d.type;
	divide.a.type.category = ir::LLVMInstruction::Type::Element;
	divide.a.constant = true;

	if( i.type == ir::PTXOperand::f32 )
	{
		divide.a.f32 = 1.0f;
	}
	else
	{
		divide.a.f64 = 1.0;
	}
	
	divide.b = call.d;		

	_add( divide );		

	if( i.modifier & ir::PTXInstruction::ftz )
	{
		_flushToZero( _destination( i ), divide.d );
	}
}

void PTXToLLVMTranslator::_translateSad( const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand destination = _destination( i );
	
	if( ir::PTXOperand::isFloat( i.type ) )
	{
		ir::LLVMFcmp compare;
		
		compare.d.name = _tempRegister();
		compare.d.type.category = ir::LLVMInstruction::Type::Element;
		compare.d.type.type = ir::LLVMInstruction::I1;
		compare.a = _translate( i.a );
		compare.b = _translate( i.b );
		compare.comparison = ir::LLVMInstruction::Olt;
		
		_add( compare );
		
		ir::LLVMSub subtract;
		
		subtract.d = destination;
		subtract.d.name = _tempRegister();
		subtract.a = compare.b;
		subtract.b = compare.a;
		
		_add( subtract );

		ir::LLVMSub subtract1;
		
		subtract1.d = destination;
		subtract1.d.name = _tempRegister();
		subtract1.a = compare.a;
		subtract1.b = compare.b;
		
		_add( subtract1 );
		
		ir::LLVMSelect select;
		
		select.condition = compare.d;
		select.d = destination;
		select.d.name = _tempRegister();
		select.a = subtract.d;
		select.b = subtract1.d;
		
		_add( select );
		
		ir::LLVMFadd add;
		
		add.d = destination;
		add.a = _translate( i.c );
		add.b = select.d;
		
		_add( add );
	}
	else
	{
		ir::LLVMIcmp compare;
		
		compare.d.name = _tempRegister();
		compare.d.type.category = ir::LLVMInstruction::Type::Element;
		compare.d.type.type = ir::LLVMInstruction::I1;
		compare.a = _translate( i.a );
		compare.b = _translate( i.b );
		
		if( ir::PTXOperand::isSigned( i.type ) )
		{
			compare.comparison = ir::LLVMInstruction::Slt;
		}
		else
		{
			compare.comparison = ir::LLVMInstruction::Ult;			
		}
		
		_add( compare );
		
		ir::LLVMSub subtract;
		
		subtract.d = destination;
		subtract.d.name = _tempRegister();
		subtract.a = compare.b;
		subtract.b = compare.a;
		
		_add( subtract );

		ir::LLVMSub subtract1;
		
		subtract1.d = destination;
		subtract1.d.name = _tempRegister();
		subtract1.a = compare.a;
		subtract1.b = compare.b;
		
		_add( subtract1 );
		
		ir::LLVMSelect select;
		
		select.condition = compare.d;
		select.d = destination;
		select.d.name = _tempRegister();
		select.a = subtract.d;
		select.b = subtract1.d;
		
		_add( select );
		
		ir::LLVMAdd add;
		
		add.d = destination;
		add.a = _translate( i.c );
		add.b = select.d;
		
		_add( add );		
	}
}

void PTXToLLVMTranslator::_translateSelP( const ir::PTXInstruction& i )
{
	ir::LLVMSelect select;
	
	select.d = _destination( i );
	select.a = _translate( i.a );
	select.b = _translate( i.b );
	select.condition = _translate( i.c );
	
	_add( select );
}

void PTXToLLVMTranslator::_translateSet( const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand d = _destination( i );

	ir::LLVMInstruction::Operand comparison; 
	comparison.name = _tempRegister();
	comparison.type.category = ir::LLVMInstruction::Type::Element;
	comparison.type.type = ir::LLVMInstruction::I1;

	if( ir::PTXOperand::isFloat( i.a.type ) )
	{
		ir::LLVMFcmp fcmp;
		
		fcmp.d = comparison;
		fcmp.a = _translate( i.a );
		fcmp.b = _translate( i.b );
		fcmp.comparison = _translate( i.comparisonOperator, false, false );
		
		_add( fcmp );
	}
	else
	{
		ir::LLVMIcmp icmp;
				
		icmp.d = comparison;
		icmp.a = _translate( i.a );
		icmp.b = _translate( i.b );
		icmp.comparison = _translate( i.comparisonOperator, true, 
			ir::PTXOperand::isSigned( i.a.type ) );
		
		_add( icmp );		
	}
	
	if( i.c.addressMode == ir::PTXOperand::Register )
	{
		ir::LLVMInstruction::Operand c = _translate( i.c );
	
		if( i.c.condition == ir::PTXOperand::InvPred )
		{
			ir::LLVMXor Not;
			
			Not.d = c;
			Not.d.name = _tempRegister();
			Not.a = c;
			Not.b = c;
			Not.b.constant = true;
			Not.b.i1 = 1;
		
			_add( Not );

			c = Not.d;
		}
				
		switch( i.booleanOperator )
		{
			case ir::PTXInstruction::BoolAnd:
			{
				ir::LLVMAnd And;
				
				And.d = comparison;
				And.d.name = _tempRegister();
				And.a = c;
				And.b = comparison;
				
				_add( And );
				
				comparison.name = And.d.name;
				
				break;
			}
			case ir::PTXInstruction::BoolOr:
			{
				ir::LLVMOr Or;
				
				Or.d = comparison;
				Or.d.name = _tempRegister();
				Or.a = c;
				Or.b = comparison;
				
				_add( Or );
				
				comparison.name = Or.d.name;
				
				break;
			}
			case ir::PTXInstruction::BoolXor:
			{
				ir::LLVMXor Xor;
				
				Xor.d = comparison;
				Xor.d.name = _tempRegister();
				Xor.a = c;
				Xor.b = comparison;
				
				_add( Xor );
				
				comparison.name = Xor.d.name;

				break;
			}
			default:
			{
				break;
			}
		}
	}

	ir::LLVMSelect select;
	
	select.condition = comparison;
	select.d = d;
	select.a = d;
	select.b = d;
	select.a.constant = true;
	select.b.constant = true;
	
	if( ir::PTXOperand::f64 == i.type )
	{
		select.a.f64 = 1.0;
		select.b.f64 = 0.0;
	}
	else if( ir::PTXOperand::f32 == i.type )
	{
		select.a.f32 = 1.0f;
		select.b.f32 = 0.0f;		
	}
	else
	{
		select.a.i64 = -1;
		select.b.f64 = 0;		
	}
	
	_add( select );
}

void PTXToLLVMTranslator::_translateSetP( const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand d = _destination( i );
	ir::LLVMInstruction::Operand tempD = d;

	if( i.c.addressMode == ir::PTXOperand::Register )
	{
		tempD.name = _tempRegister();
	}

	if( ir::PTXOperand::isFloat( i.a.type ) )
	{
		ir::LLVMFcmp fcmp;
		
		fcmp.d = tempD;
		fcmp.a = _translate( i.a );
		fcmp.b = _translate( i.b );
		fcmp.comparison = _translate( i.comparisonOperator, false, false );
		
		_add( fcmp );
	}
	else
	{
		ir::LLVMIcmp icmp;
				
		icmp.d = tempD;
		icmp.a = _translate( i.a );
		icmp.b = _translate( i.b );
		icmp.comparison = _translate( i.comparisonOperator, true, 
			ir::PTXOperand::isSigned( i.type ) );
		
		_add( icmp );
	}
	
	ir::LLVMInstruction::Operand pd = d;
	ir::LLVMInstruction::Operand pq;
	ir::LLVMXor Not;

	if( i.pq.addressMode != ir::PTXOperand::Invalid )
	{
		pq = _translate( i.pq );

		if( i.c.addressMode == ir::PTXOperand::Register )
		{
			Not.d = tempD;
			Not.d.name = _tempRegister();
		}
		else
		{
			Not.d = pq;
		}

		Not.a = tempD;
		Not.b.type.category = ir::LLVMInstruction::Type::Element;
		Not.b.type.type = ir::LLVMInstruction::I1;
		Not.b.constant = true;
		Not.b.i1 = true;
	
		_add( Not );
	}
	
	if( i.c.addressMode == ir::PTXOperand::Register )
	{
				
		switch( i.booleanOperator )
		{
			case ir::PTXInstruction::BoolAnd:
			{
				ir::LLVMAnd And;
				
				And.d = pd;
				And.a = _translate( i.c );
				And.b = tempD;
				
				_add( And );
				
				if( i.pq.addressMode != ir::PTXOperand::Invalid )
				{
					And.d = pq;
					And.b = Not.d;
				
					_add( And );
				}
				break;
			}
			case ir::PTXInstruction::BoolOr:
			{
				ir::LLVMOr Or;
				
				Or.d = pd;
				Or.a = _translate( i.c );
				Or.b = tempD;
				
				_add( Or );
				
				if( i.pq.addressMode != ir::PTXOperand::Invalid )
				{
					Or.d = pq;
					Or.b = Not.d;
				
					_add( Or );
				}
				break;
			}
			case ir::PTXInstruction::BoolXor:
			{
				ir::LLVMXor Xor;
				
				Xor.d = pd;
				Xor.a = _translate( i.c );
				Xor.b = tempD;
				
				_add( Xor );

				if( i.pq.addressMode != ir::PTXOperand::Invalid )
				{
					Xor.d = pq;
					Xor.b = Not.d;
				
					_add( Xor );
				}
				break;
			}
			default:
			{
				break;
			}
		}
	}
}

void PTXToLLVMTranslator::_translateShl( const ir::PTXInstruction& i )
{
	ir::LLVMShl shift;
	
	shift.d = _destination( i );
	shift.a = _translate( i.a );
	
	if(ir::PTXOperand::bytes(i.b.type) > ir::PTXOperand::bytes(i.a.type))
	{
    ir::LLVMIcmp compare;
		ir::LLVMTrunc truncate;
	
    compare.d.name = _tempRegister();
    compare.d.type.type = ir::LLVMInstruction::I1;
    compare.d.type.category = ir::LLVMInstruction::Type::Element;
    compare.comparison = ir::LLVMInstruction::Ult;
    compare.a = _translate( i.b );
    compare.b.type.type = ir::LLVMInstruction::I32;
    compare.b.type.category = ir::LLVMInstruction::Type::Element;
    compare.b.constant = true;
    compare.b.i32 = USHRT_MAX;

    _add(compare);
	
    ir::LLVMSelect select;

    select.d.name = _tempRegister();
    select.d.type.type = ir::LLVMInstruction::I32;
    select.d.type.category = ir::LLVMInstruction::Type::Element;

    select.condition = compare.d;
    select.a = compare.a;
    select.b = compare.b;

    _add(select);

		truncate.a = select.d;
    
    truncate.d = ir::LLVMInstruction::Operand( _tempRegister(),
			shift.a.type );

		shift.b = truncate.d;
		
		_add( truncate );
	}
	else if( ir::PTXOperand::bytes(i.b.type) 
		< ir::PTXOperand::bytes(i.a.type) )
	{
		ir::LLVMZext extend;
		
		extend.a = _translate( i.b );
		extend.d = ir::LLVMInstruction::Operand( _tempRegister(),
			shift.a.type );
		
		shift.b = extend.d;
		
		_add( extend );
	}
	else
	{
		shift.b = _translate( i.b );
	}

	_add( shift );
}

void PTXToLLVMTranslator::_translateShr( const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand a = _translate( i.a );
	ir::LLVMInstruction::Operand b;

	if(ir::PTXOperand::bytes(i.b.type) > ir::PTXOperand::bytes(i.a.type))
	{
		ir::LLVMTrunc truncate;
    ir::LLVMIcmp compare;	
  	
    compare.d.name = _tempRegister();
    compare.d.type.type = ir::LLVMInstruction::I1;
    compare.d.type.category = ir::LLVMInstruction::Type::Element;
    compare.comparison = ir::LLVMInstruction::Ult;
    compare.a = _translate( i.b );
    compare.b.type.type = ir::LLVMInstruction::I32;
    compare.b.type.category = ir::LLVMInstruction::Type::Element;
    compare.b.constant = true;
    compare.b.i32 = USHRT_MAX;

    _add(compare);
	
    ir::LLVMSelect select;

    select.d.name = _tempRegister();
    select.d.type.type = ir::LLVMInstruction::I32;
    select.d.type.category = ir::LLVMInstruction::Type::Element;

    select.condition = compare.d;
    select.a = compare.a;
    select.b = compare.b;

    _add(select);
		
		truncate.a  = select.d;
    truncate.d = ir::LLVMInstruction::Operand( _tempRegister(),
			a.type );

		b = truncate.d;
		
		_add( truncate );
	}
	else if( ir::PTXOperand::bytes(i.b.type) 
		< ir::PTXOperand::bytes(i.a.type) )
	{
		if( ir::PTXOperand::isSigned( i.type ) )
		{
			ir::LLVMSext extend;
		
			extend.a = _translate( i.b );
			extend.d = ir::LLVMInstruction::Operand( _tempRegister(),
				a.type );
		
			b = extend.d;
		
			_add( extend );
		}
		else
		{
			ir::LLVMZext extend;
		
			extend.a = _translate( i.b );
			extend.d = ir::LLVMInstruction::Operand( _tempRegister(),
				a.type );
		
			b = extend.d;
		
			_add( extend );
		}
	}
	else
	{
		b = _translate( i.b );
	}

	if( ir::PTXOperand::isSigned( i.type ) )
	{
		ir::LLVMAshr shift;
		
		shift.d = _destination( i );
		shift.a = a;
		shift.b = b;
		
		_add( shift );
	}
	else
	{
		ir::LLVMLshr shift;
		
		shift.d = _destination( i );
		shift.a = a;
		shift.b = b;
		
		_add( shift );
	}
}

void PTXToLLVMTranslator::_translateSin( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;

	call.name = "@llvm.sin.f32";
	
	if( i.modifier & ir::PTXInstruction::ftz )
	{
		call.d.name          = _tempRegister();
		call.d.type.type     = ir::LLVMInstruction::F32;
		call.d.type.category = ir::LLVMInstruction::Type::Element;
	}
	else
	{
		call.d = _destination( i );
	}
	
	call.parameters.resize( 1 );
	call.parameters[0] = _translate( i.a );
	
	_add( call );

	if( i.modifier & ir::PTXInstruction::ftz )
	{
		_flushToZero( _destination( i ), call.d );
	}
}

void PTXToLLVMTranslator::_translateSlCt( const ir::PTXInstruction& i )
{

	ir::LLVMInstruction::Operand comparison;
	
	comparison.type.category = ir::LLVMInstruction::Type::Element;
	comparison.type.type = ir::LLVMInstruction::I1;
	comparison.name = _tempRegister();

	if( ir::PTXOperand::isFloat( i.c.type ) )
	{
		ir::LLVMFcmp compare;
		
		compare.d = comparison;
		compare.a = _translate( i.c );
		compare.b = compare.a;
		compare.b.constant = true;
		compare.b.f32 = 0;
		compare.comparison = ir::LLVMInstruction::Oge;
		
		_add( compare );
	}
	else
	{
		ir::LLVMIcmp compare;
		
		compare.d = comparison;
		compare.a = _translate( i.c );
		compare.b = compare.a;
		compare.b.constant = true;
		compare.b.i32 = 0;
		compare.comparison = ir::LLVMInstruction::Sge;
		
		_add( compare );
	}
	
	ir::LLVMSelect select;
	select.d = _destination( i );
	select.condition = comparison;
	select.a = _translate( i.a );
	select.b = _translate( i.b );
	
	_add( select );
}

void PTXToLLVMTranslator::_translateSqrt( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	if( i.a.type == ir::PTXOperand::f64 )
	{
		call.name = "@llvm.sqrt.f64";
	}
	else
	{
		call.name = "@llvm.sqrt.f32";
	}
	
	call.parameters.resize( 1 );
	
	if( i.modifier & ir::PTXInstruction::ftz 
		|| i.modifier & ir::PTXInstruction::approx )
	{
		call.d.name          = _tempRegister();
		call.d.type.category = ir::LLVMInstruction::Type::Element;
		call.d.type.type     = ir::LLVMInstruction::F32;
		call.parameters[0] = ir::LLVMInstruction::Operand( 
			_tempRegister(), call.d.type );
		
		_flushToZero( call.parameters[0], _translate( i.a ) );
	}
	else
	{
		call.parameters[0] = _translate( i.a );
		call.d = _destination( i );
	}
	
	_add( call );
			
	if( i.modifier & ir::PTXInstruction::ftz
		|| i.modifier & ir::PTXInstruction::approx )
	{
		_flushToZero( _destination( i ), call.d );
	}
}

void PTXToLLVMTranslator::_translateSt( const ir::PTXInstruction& i )
{
	#if(USE_VECTOR_INSTRUCTIONS == 1)
	ir::LLVMStore store;

	if( i.vec == ir::PTXOperand::v1 )
	{
		if( _translate( i.type ) != _translate( i.a.type ) )
		{
			ir::LLVMInstruction::Operand temp = _translate( i.a );
			temp.name = _tempRegister();
			temp.type.type = _translate( i.type );
			_convert( temp, i.type, _translate( i.a ), i.a.type );
			store.a = temp;		
		}
		else
		{
			store.a = _translate( i.a );
		}
	}
	else
	{
		store.a = _translate( i.a.array.front() );
		store.a.type.vector = i.vec;
		store.a.type.category = ir::LLVMInstruction::Type::Vector;
		store.a.type.type = _translate( i.type );
	}

	store.d = _getLoadOrStorePointer( i.d, i.addressSpace, 
		_translate( i.type ), i.vec );
	
	if( i.volatility == ir::PTXInstruction::Volatile )
	{
		store.isVolatile = true;
	}
	
	store.alignment = i.vec * ir::PTXOperand::bytes( i.type );

	if( i.vec != ir::PTXOperand::v1 )
	{
		store.a = _translate( i.a.array.front() );
		store.a.type.vector = i.vec;
		store.a.type.category = ir::LLVMInstruction::Type::Vector;
		store.a.type.type = _translate( i.type );

		ir::PTXOperand::Array::const_iterator 
			source = i.a.array.begin();

		ir::LLVMInsertelement insertOne;
	
		insertOne.d = store.a;
		insertOne.d.name = _tempRegister();
		insertOne.a = store.a;
		insertOne.a.constant = true;
		
		ir::LLVMInstruction::Value value;
		value.i64 = 0;
		
		insertOne.a.values.assign( i.vec, value );
		
		if( i.type != source->type )
		{
			ir::LLVMInstruction::Operand temp = _translate( *source );
			temp.name = _tempRegister();
			temp.type.type = _translate( i.type );
			_convert( temp, i.type, _translate( *source ), source->type );
			insertOne.b = temp;
		}
		else
		{
			insertOne.b = _translate( *source );
		}

		insertOne.c.type.category = ir::LLVMInstruction::Type::Element;
		insertOne.c.type.type = ir::LLVMInstruction::I32;
		insertOne.c.constant = true;
		insertOne.c.i32 = 0;
		
		_add( insertOne );
		
		std::string currentSource = insertOne.d.name;
		
		for( ++source; source != i.a.array.end(); ++source )
		{
			ir::LLVMInsertelement insert;

			insert.d = store.a;
			insert.d.name = _tempRegister();
			if( ++ir::PTXOperand::Array::const_iterator( source ) 
				== i.a.array.end() )
			{
				store.a.name = insert.d.name;
			}
			insert.a = store.a;
			insert.a.name = currentSource;
			if( i.type != source->type )
			{
				ir::LLVMInstruction::Operand temp = _translate( *source );
				temp.name = _tempRegister();
				temp.type.type = _translate( i.type );
				_convert( temp, i.type, _translate( *source ), 
					source->type );
				insert.b = temp;
			}
			else
			{
				insert.b = _translate( *source );
			}
			insert.c.type.category = ir::LLVMInstruction::Type::Element;
			insert.c.type.type = ir::LLVMInstruction::I32;
			insert.c.constant = true;
			insert.c.i32 = std::distance( i.a.array.begin(), source );
			
			_add( insert );
			currentSource = insert.d.name;
		}
	}
	
	_check( i.addressSpace, store.d, store.alignment,
		i.d.isArgument, i.d.isGlobalLocal, i.statementIndex );
	_add( store );
	#else
	ir::LLVMStore store;

	if( i.volatility == ir::PTXInstruction::Volatile )
	{
		store.isVolatile = true;
	}

	store.alignment = ir::PTXOperand::bytes( i.type );
	ir::LLVMInstruction::Operand address = _getLoadOrStorePointer( 
		i.d, i.addressSpace, _translate( i.type ), ir::PTXOperand::v1 );

	if( i.vec == ir::PTXOperand::v1 )
	{
		store.d = address;

		if( _translate( i.type ) != _translate( i.a.type ) )
		{
			ir::LLVMInstruction::Operand temp = _translate( i.a );
			temp.name = _tempRegister();
			temp.type.type = _translate( i.type );
			_convert( temp, i.type, _translate( i.a ), i.a.type );
			store.a = temp;
		}
		else
		{
			store.a = _translate( i.a );
		}

		_check( i.addressSpace, store.d, 
			store.alignment, i.d.isArgument, i.d.isGlobalLocal,
			i.statementIndex );
		_add( store );
	}
	else
	{
		unsigned int index = 0;
		for( ir::PTXOperand::Array::const_iterator 
			source = i.a.array.begin(); 
			source != i.a.array.end(); ++source, ++index)
		{
			ir::LLVMGetelementptr get;
		
			get.a = address;
			get.d = get.a;
			get.d.name = _tempRegister();
			get.indices.push_back( index );
		
			_add( get );

			store.d = get.d;

			if( _translate( i.type ) != _translate( source->type ) )
			{
				ir::LLVMInstruction::Operand temp = _translate( *source );
				temp.name = _tempRegister();
				temp.type.type = _translate( i.type );
				_convert( temp, i.type, _translate( *source ), source->type );
				store.a = temp;
			}
			else
			{
				store.a = _translate( *source );
			}

			_check( i.addressSpace, store.d, 
				store.alignment, i.d.isArgument, i.d.isGlobalLocal,
				i.statementIndex );
			_add( store );
		}
	}
	#endif
}

void PTXToLLVMTranslator::_translateSub( const ir::PTXInstruction& i )
{
	if( ir::PTXOperand::isFloat( i.type ) )
	{
		ir::LLVMFsub sub;
	
		ir::LLVMInstruction::Operand result = _destination( i );

		sub.a = _translate( i.a );
		sub.b = _translate( i.b );
	
		if( i.modifier & ir::PTXInstruction::sat
			|| i.modifier & ir::PTXInstruction::ftz )
		{
			sub.d = sub.a;
			sub.d.name = _tempRegister();
		}
		else
		{
			sub.d = result;
		}

		_add( sub );
		
		if( i.modifier & ir::PTXInstruction::sat )
		{
			if( i.modifier & ir::PTXInstruction::ftz )
			{
				ir::LLVMInstruction::Operand temp =
					ir::LLVMInstruction::Operand( _tempRegister(),
					sub.d.type );
				_saturate( temp, sub.d );
				_flushToZero( result, temp );
			}
			else
			{
				_saturate( result, sub.d );
			}
		}
		else if( i.modifier & ir::PTXInstruction::ftz )
		{
			_flushToZero( result, sub.d );
		}
	}
	else
	{
		if( i.modifier & ir::PTXInstruction::sat )
		{
			assert( i.type == ir::PTXOperand::s32 );
			
			ir::LLVMSext extendA;
			ir::LLVMSext extendB;
							
			extendA.a = _translate( i.a );
			extendA.d.type.type = ir::LLVMInstruction::I64;
			extendA.d.type.category = ir::LLVMInstruction::Type::Element;
			extendA.d.name = _tempRegister();
			
			_add( extendA );
			
			extendB.a = _translate( i.b );
			extendB.d.type.type = ir::LLVMInstruction::I64;
			extendB.d.type.category = ir::LLVMInstruction::Type::Element;
			extendB.d.name = _tempRegister();
			
			_add( extendB );
			
			ir::LLVMSub sub;
			
			sub.a = extendA.d;
			sub.b = extendB.d;
			sub.d.name = _tempRegister();
			sub.d.type.type = ir::LLVMInstruction::I64;
			sub.d.type.category = ir::LLVMInstruction::Type::Element;
			
			_add( sub );
			
			ir::LLVMIcmp compare;
			
			compare.d.name = _tempRegister();
			compare.d.type.type = ir::LLVMInstruction::I1;
			compare.d.type.category = ir::LLVMInstruction::Type::Element;
			compare.comparison = ir::LLVMInstruction::Slt;
			compare.a = sub.d;
			compare.b.type.type = ir::LLVMInstruction::I64;
			compare.b.type.category = ir::LLVMInstruction::Type::Element;
			compare.b.constant = true;
			compare.b.i64 = INT_MIN;

			_add( compare );
			
			ir::LLVMSelect select;
			
			select.d.name = _tempRegister();
			select.d.type.type = ir::LLVMInstruction::I64;
			select.d.type.category = ir::LLVMInstruction::Type::Element;

			select.condition = compare.d;
			select.a = compare.b;
			select.b = compare.a;
			
			_add( select );
			
			compare.d.name = _tempRegister();
			compare.comparison = ir::LLVMInstruction::Sgt;
			compare.b.i64 = INT_MAX;
			compare.a = select.d;
			
			_add( compare );

			select.condition = compare.d;
			select.a = compare.b;
			select.b = compare.a;
			select.d.name = _tempRegister();

			_add( select );
			
			ir::LLVMTrunc truncate;
			
			truncate.a = select.d;
			truncate.d = _destination( i );
			
			_add( truncate );
		}
		else if( i.carry & ir::PTXInstruction::CC )
		{
			ir::LLVMInstruction::Operand a = _translate( i.a );	
			ir::LLVMInstruction::Operand b = _translate( i.b );
			
			ir::LLVMSub negate;
			
			negate.d = ir::LLVMInstruction::Operand(
				_tempRegister(), b.type );
			negate.a = ir::LLVMInstruction::Operand( (ir::LLVMI32) 0 );
			negate.b = b;
			
			_add( negate );
		
			ir::LLVMAdd add;

			add.d = _destination( i );
			add.a = a;
			add.b = negate.d;
			
			_add( add );
			
			ir::LLVMInstruction::Operand carry = _translate( i.pq );
			ir::LLVMInstruction::Operand lessThanA = 
				ir::LLVMInstruction::Operand( _tempRegister(),
					ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
					ir::LLVMInstruction::Type::Element ) );
			ir::LLVMInstruction::Operand lessThanB = 
				ir::LLVMInstruction::Operand( _tempRegister(),
					ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
					ir::LLVMInstruction::Type::Element ) );
			ir::LLVMInstruction::Operand lessThanEither = 
				ir::LLVMInstruction::Operand( _tempRegister(),
					ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
					ir::LLVMInstruction::Type::Element ) );
			
			ir::LLVMIcmp compare;
		
			compare.comparison = ir::LLVMInstruction::Ult;
			compare.d = lessThanA;
			compare.a = add.d;
			compare.b = a;
		
			_add( compare );
			
			compare.d = lessThanB;
			compare.b = negate.d;
			
			_add( compare );
			
			ir::LLVMOr Or;
			
			Or.d = lessThanEither;
			Or.a = lessThanA;
			Or.b = lessThanB;
			
			_add( Or );
		
			ir::LLVMSelect select;
			
			select.d = carry;
			select.condition = lessThanEither;
			select.a = ir::LLVMInstruction::Operand( (ir::LLVMI32) 1 );
			select.b = ir::LLVMInstruction::Operand( (ir::LLVMI32) 0 );
			
			_add( select );
		}
		else
		{
			ir::LLVMSub sub;

			sub.d = _destination( i );
			sub.a = _translate( i.a );
			sub.b = _translate( i.b );
			
			_add( sub );
		}
	}
}

void PTXToLLVMTranslator::_translateSubC( const ir::PTXInstruction& i )
{	
	ir::LLVMInstruction::Operand destination = _destination( i );
	ir::LLVMInstruction::Operand a = _translate( i.a );	
	ir::LLVMInstruction::Operand b = _translate( i.b );
			
	ir::LLVMSub negate;
	
	negate.d = ir::LLVMInstruction::Operand(_tempRegister(), b.type );
	negate.a = ir::LLVMInstruction::Operand( (ir::LLVMI32) 0 );
	negate.b = b;
	
	_add( negate );
	
	ir::LLVMAdd add;
	
	add.d = destination;
	add.d.name = _tempRegister();
	add.a = a;
	add.b = negate.d;
	
	_add( add );
	
	add.a = add.d;
	add.d = destination;
	add.d.name = _tempRegister();
	add.b = _translate( i.c );
	
	_add( add );

	ir::LLVMSub sub;

	sub.d = destination;
	sub.a = add.d;
	sub.b = ir::LLVMInstruction::Operand( (ir::LLVMI32) 1 );

	_add( sub );

	if( i.carry & ir::PTXInstruction::CC )
	{
		ir::LLVMInstruction::Operand carry = _translate( i.pq );
		ir::LLVMInstruction::Operand lessThanA = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
		ir::LLVMInstruction::Operand lessThanB = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
		ir::LLVMInstruction::Operand lessThanEither = 
			ir::LLVMInstruction::Operand( _tempRegister(),
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
				ir::LLVMInstruction::Type::Element ) );
		
		ir::LLVMIcmp compare;
	
		compare.comparison = ir::LLVMInstruction::Ult;
		compare.d = lessThanA;
		compare.a = destination;
		compare.b = a;
	
		_add( compare );
		
		compare.d = lessThanB;
		compare.b = negate.d;
		
		_add( compare );
		
		ir::LLVMOr Or;
		
		Or.d = lessThanEither;
		Or.a = lessThanA;
		Or.b = lessThanB;
		
		_add( Or );
	
		ir::LLVMSelect select;
		
		select.d = carry;
		select.condition = lessThanEither;
		select.a = ir::LLVMInstruction::Operand( (ir::LLVMI32) 1 );
		select.b = ir::LLVMInstruction::Operand( (ir::LLVMI32) 0 );
		
		_add( select );					
	}
}

void PTXToLLVMTranslator::_translateSuld( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	call.name = "@__ocelot_suld";

	_add( call );
}

void PTXToLLVMTranslator::_translateSuq( const ir::PTXInstruction& i )
{
	_translateTxq( i );
}

void PTXToLLVMTranslator::_translateSured( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	call.name = "@__ocelot_sured";

	_add( call );
}	

void PTXToLLVMTranslator::_translateSust( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	call.name = "@__ocelot_sust";

	_add( call );	
}	

void PTXToLLVMTranslator::_translateTestP( const ir::PTXInstruction& i )
{
	switch( i.floatingPointMode )
	{
	case ir::PTXInstruction::Finite:
	{
		ir::LLVMFcmp isFin;
	
		isFin.d = _destination( i );
	
		isFin.a.type.type     = _translate( i.type );
		isFin.a.type.category = ir::LLVMInstruction::Type::Element;
		isFin.a.constant      = true;
		
		if( i.type == ir::PTXOperand::f32 )
		{
			isFin.a.i32 = hydrazine::bit_cast< ir::LLVMI32 >(
				 std::numeric_limits<float>::infinity() );
		}
		else
		{
			isFin.a.i64 = hydrazine::bit_cast< ir::LLVMI64 >(
				 std::numeric_limits<double>::infinity() );
		}
		
		isFin.b = _translate( i.a );

		isFin.comparison = ir::LLVMInstruction::One;
		
		_add( isFin );
	}
	break;
	case ir::PTXInstruction::Infinite:
	{
		ir::LLVMFcmp isInf;
	
		isInf.d = _destination( i );
	
		isInf.a.type.type     = _translate( i.type );
		isInf.a.type.category = ir::LLVMInstruction::Type::Element;
		isInf.a.constant      = true;
		
		if( i.type == ir::PTXOperand::f32 )
		{
			isInf.a.i32 = hydrazine::bit_cast< ir::LLVMI32 >(
				 std::numeric_limits<float>::infinity() );
		}
		else
		{
			isInf.a.i64 = hydrazine::bit_cast< ir::LLVMI64 >(
				 std::numeric_limits<double>::infinity() );
		}
		
		isInf.b = _translate( i.a );

		isInf.comparison = ir::LLVMInstruction::Oeq;
	
		_add( isInf );
	}
	break;
	case ir::PTXInstruction::Number:
	{
		ir::LLVMFcmp isNum;
	
		isNum.d = _destination( i );
	
		isNum.a.type.type     = _translate( i.type );
		isNum.a.type.category = ir::LLVMInstruction::Type::Element;
		isNum.a.constant      = true;
		
		if( i.type == ir::PTXOperand::f32 )
		{
			isNum.a.f32 = 0.0f;
		}
		else
		{
			isNum.a.f64 = 0.0;
		}
	
		isNum.b = _translate( i.a );
	
		isNum.comparison = ir::LLVMInstruction::Ord;
	
		_add( isNum );
	}
	break;
	case ir::PTXInstruction::NotANumber:
	{
		ir::LLVMFcmp isNan;
	
		isNan.d = _destination( i );
	
		isNan.a.type.type     = _translate( i.type );
		isNan.a.type.category = ir::LLVMInstruction::Type::Element;
		isNan.a.constant      = true;
		
		if( i.type == ir::PTXOperand::f32 )
		{
			isNan.a.f32 = 0.0f;
		}
		else
		{
			isNan.a.f64 = 0.0;
		}
	
		isNan.b = _translate( i.a );
	
		isNan.comparison = ir::LLVMInstruction::Uno;
	
		_add( isNan );
	}
	break;
	case ir::PTXInstruction::Normal:
	{
		ir::LLVMFcmp equal;
		
		equal.comparison = ir::LLVMInstruction::One;
		
		equal.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
			ir::LLVMInstruction::Type::Element ) );
		equal.a = _translate( i.a );
		equal.b = ir::LLVMInstruction::Operand( (ir::LLVMI64) 0 );
		equal.b.type.type = _translate( i.a.type );
		
		_add( equal );
		
		ir::LLVMFcmp less;
		
		less.comparison = ir::LLVMInstruction::Olt;
		
		less.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
			ir::LLVMInstruction::Type::Element ) );
		less.a = equal.a;
		less.b = ir::LLVMInstruction::Operand( (ir::LLVMI64) 0 );
		less.b.type.type = equal.b.type.type;
		
		_add( less );

		ir::LLVMFsub subtract;
		
		subtract.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( equal.b.type.type, 
			ir::LLVMInstruction::Type::Element ) );
		subtract.a = ir::LLVMInstruction::Operand( (ir::LLVMI64) 0 );
		subtract.a.type.type = equal.b.type.type;
		subtract.b = equal.a;
		
		_add( subtract );
		
		ir::LLVMSelect select;
		
		select.condition = less.d;
		
		select.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( equal.b.type.type, 
			ir::LLVMInstruction::Type::Element ) );
		select.a = subtract.d;
		select.b = equal.a;
		
		_add( select );
		
		ir::LLVMFcmp greaterEqual;
		
		greaterEqual.comparison = ir::LLVMInstruction::Oge;
		
		greaterEqual.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
			ir::LLVMInstruction::Type::Element ) );
		greaterEqual.a = select.d;
		
		if( i.type == ir::PTXOperand::f32 )
		{
			greaterEqual.b = ir::LLVMInstruction::Operand( 
				(ir::LLVMI64) hydrazine::bit_cast< ir::LLVMI32 >(
				std::numeric_limits<float>::min() ) );
		}
		else
		{
			greaterEqual.b = ir::LLVMInstruction::Operand( 
				(ir::LLVMI64) hydrazine::bit_cast< ir::LLVMI64 >(
				std::numeric_limits<double>::min() ) );
		}
		
		greaterEqual.b.type.type = equal.b.type.type;
		
		_add( greaterEqual );
		
		ir::LLVMAnd land;
		
		land.a = greaterEqual.d;
		land.b = equal.d;
		land.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
			ir::LLVMInstruction::Type::Element ) );
		
		_add( land );

		ir::LLVMFcmp notInf;
		
		notInf.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
			ir::LLVMInstruction::Type::Element ) );
		notInf.comparison = ir::LLVMInstruction::One;
		notInf.a = select.d;

		if( i.type == ir::PTXOperand::f32 )
		{
			notInf.b = ir::LLVMInstruction::Operand( 
				(ir::LLVMI64) hydrazine::bit_cast< ir::LLVMI32 >(
				std::numeric_limits<float>::infinity() ) );
		}
		else
		{
			notInf.b = ir::LLVMInstruction::Operand( 
				(ir::LLVMI64) hydrazine::bit_cast< ir::LLVMI64 >(
				std::numeric_limits<double>::infinity() ) );
		}

		notInf.b.type.type = equal.b.type.type;
		
		_add( notInf );
		
		land.b = notInf.d;
		land.a = land.d;
		land.d = _destination( i );

		_add( land );
	}
	break;
	case ir::PTXInstruction::SubNormal:
	{
		ir::LLVMFcmp less;
		
		less.comparison = ir::LLVMInstruction::Olt;
		
		less.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
			ir::LLVMInstruction::Type::Element ) );
		less.a = _translate( i.a );
		less.b = ir::LLVMInstruction::Operand( (ir::LLVMI64) 0 );
		less.b.type.type = less.a.type.type;
		
		_add( less );

		ir::LLVMFsub subtract;
		
		subtract.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( less.a.type.type, 
			ir::LLVMInstruction::Type::Element ) );
		subtract.a = ir::LLVMInstruction::Operand( (ir::LLVMI64) 0 );
		subtract.a.type.type = less.a.type.type;
		subtract.b = less.a;
		
		_add( subtract );
		
		ir::LLVMSelect select;
		
		select.condition = less.d;
		
		select.d = ir::LLVMInstruction::Operand( _tempRegister(),
			ir::LLVMInstruction::Type( less.a.type.type, 
			ir::LLVMInstruction::Type::Element ) );
		select.a = subtract.d;
		select.b = less.a;
		
		_add( select );
		
		ir::LLVMFcmp greaterEqual;
		
		greaterEqual.comparison = ir::LLVMInstruction::Olt;
		
		greaterEqual.d = _destination( i );
		greaterEqual.a = select.d;
		
		if( i.type == ir::PTXOperand::f32 )
		{
			greaterEqual.b = ir::LLVMInstruction::Operand( 
				(ir::LLVMI64) hydrazine::bit_cast< ir::LLVMI32 >(
				std::numeric_limits<float>::min() ) );
		}
		else
		{
			greaterEqual.b = ir::LLVMInstruction::Operand( 
				(ir::LLVMI64) hydrazine::bit_cast< ir::LLVMI64 >(
				std::numeric_limits<double>::min() ) );
		}
		
		greaterEqual.b.type.type = less.a.type.type;
		
		_add( greaterEqual );
	}
	break;
	default: assertM(false, "Invalid floating point mode.");
	}
	
}

void PTXToLLVMTranslator::_translateTex( const ir::PTXInstruction& i )
{
	_usesTextures = true;

	ir::LLVMCall call;
	
	ir::LLVMInstruction::Operand d1 = _translate( i.d.array[0] );
	ir::LLVMInstruction::Operand d2 = _translate( i.d.array[1] );
	ir::LLVMInstruction::Operand d3 = _translate( i.d.array[2] );
	ir::LLVMInstruction::Operand d4 = _translate( i.d.array[3] );
	
	ir::LLVMInstruction::Operand d;
	d.type.category = ir::LLVMInstruction::Type::Pointer;
	
	switch( i.d.type )
	{
		case ir::PTXOperand::u32: /* fall through */
		case ir::PTXOperand::s32:
		{
			d.name = "%integerTexture";
			d.type.type = ir::LLVMInstruction::I32;
			break;
		}
		case ir::PTXOperand::f32:
		{
			d.name = "%floatingPointTexture";
			d.type.type = ir::LLVMInstruction::F32;
			break;
		}
		default : assertM( false, "Invalid destination type " 
			<< ir::PTXOperand::toString( i.d.type ) << " for tex" );
	}
	
	switch( i.geometry )
	{
		case ir::PTXInstruction::_1d:
		{
			call.name = "@__ocelot_tex_1d";
			call.parameters.resize( 4 );
			call.parameters[0] = d;
			call.parameters[3] = _translate( i.c.array[0] );
			break;
		}
		case ir::PTXInstruction::_2d:
			call.name = "@__ocelot_tex_2d";
			call.parameters.resize( 5 );
			call.parameters[0] = d;
			call.parameters[3] = _translate( i.c.array[0] );
			call.parameters[4] = _translate( i.c.array[1] );
			break;
		case ir::PTXInstruction::_a2d:
			call.name = "@__ocelot_tex_a2d";
			call.parameters.resize( 6 );
			call.parameters[0] = d;
			call.parameters[3] = _translate( i.c.array[1] );
			call.parameters[4] = _translate( i.c.array[2] );
			call.parameters[5] = _translate( i.c.array[0] );
			break;
		case ir::PTXInstruction::_3d:
			call.name = "@__ocelot_tex_3d";
			call.parameters.resize( 7 );
			call.parameters[0] = d;
			call.parameters[3] = _translate( i.c.array[0] );
			call.parameters[4] = _translate( i.c.array[1] );
			call.parameters[5] = _translate( i.c.array[2] );
			call.parameters[6] = _translate( i.c.array[3] );
			break;
		default: assertM( false, "Invalid texture geometry" );
	}
	
	call.parameters[1] = _context();
	
	call.parameters[2].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[2].type.type = ir::LLVMInstruction::I32;
	call.parameters[2].constant = true;
	call.parameters[2].i32 = i.a.reg;
	
	switch( i.d.type )
	{
		case ir::PTXOperand::u32:
		{
			switch( i.type )
			{
				case ir::PTXOperand::u32:
				{
					call.name += "_uu";
					break;
				}
				case ir::PTXOperand::f32:
				{
					call.name += "_uf";
					break;
				}
				case ir::PTXOperand::s32:
				{
					call.name += "_us";
					break;
				}
				default : assertM( false, "Invalid source type " 
					<< ir::PTXOperand::toString( i.c.type ) << " for tex" );
			}
			break;
		}
		case ir::PTXOperand::s32:
		{
			switch( i.type )
			{
				case ir::PTXOperand::u32:
				{
					call.name += "_su";
					break;
				}
				case ir::PTXOperand::f32:
				{
					call.name += "_sf";
					break;
				}
				case ir::PTXOperand::s32:
				{
					call.name += "_ss";
					break;
				}
				default : assertM( false, "Invalid source type " 
					<< ir::PTXOperand::toString( i.c.type ) << " for tex" );
			}
			break;
		}
		case ir::PTXOperand::f32:
		{
			switch( i.type )
			{
				case ir::PTXOperand::u32:
				{
					call.name += "_fu";
					break;
				}
				case ir::PTXOperand::f32:
				{
					call.name += "_ff";
					break;
				}
				case ir::PTXOperand::s32:
				{
					call.name += "_fs";
					break;
				}
				default : assertM( false, "Invalid source type " 
					<< ir::PTXOperand::toString( i.c.type ) << " for tex" );
			}
			break;
		}
		default : assertM( false, "Invalid destination type " 
			<< ir::PTXOperand::toString( i.d.type ) << " for tex" );
	}
	
	_add( call );
	
	ir::LLVMLoad load;
		
	load.d = d1;
	load.a = d;

	_add( load );

	ir::LLVMGetelementptr get;
	
	get.d = d;
	get.d.name = _tempRegister();
	get.a = d;
	get.indices.push_back( 1 );
	
	_add( get );

	load.d = d2;
	load.a = get.d;

	_add( load );

	get.d = d;
	get.d.name = _tempRegister();
	get.a = d;
	get.indices.back() = 2;
	
	_add( get );

	load.d = d3;
	load.a = get.d;

	_add( load );		
	
	get.d = d;
	get.d.name = _tempRegister();
	get.a = d;
	get.indices.back() = 3;
	
	_add( get );

	load.d = d4;
	load.a = get.d;

	_add( load );
}

void PTXToLLVMTranslator::_translateTxq( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	call.name = "@__ocelot_txq";
	
	call.d = _destination( i );
	call.parameters.resize( 3 );
	call.parameters[0] = _context();
	
	call.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[1].type.type = ir::LLVMInstruction::I32;
	call.parameters[1].constant = true;
	call.parameters[1].i32 = i.a.reg;

	call.parameters[2].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[2].type.type = ir::LLVMInstruction::I32;
	call.parameters[2].constant = true;
	call.parameters[2].i32 = i.surfaceQuery;

	_add( call );
}

void PTXToLLVMTranslator::_translateTrap( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	call.name = "@trap";
	
	_add( call );
}

void PTXToLLVMTranslator::_translateVote( const ir::PTXInstruction& i )
{
	ir::LLVMCall call;
	
	call.name = "@__ocelot_vote";
	
	call.d = _destination( i );
	call.parameters.resize( 3 );
	call.parameters[0] = _translate( i.a );
	call.parameters[1].type.type = ir::LLVMInstruction::I32;
	call.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[1].i32 = i.vote;
	call.parameters[1].constant = true;
	call.parameters[2].type.type = ir::LLVMInstruction::I1;
	call.parameters[2].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[2].i1 = i.b.condition == ir::PTXOperand::InvPred;
	call.parameters[2].constant = true;
	
	_add( call );
}

void PTXToLLVMTranslator::_translateXor( const ir::PTXInstruction& i )
{
	ir::LLVMXor Xor;
	
	Xor.d = _destination( i );
	Xor.a = _translate( i.a );
	Xor.b = _translate( i.b );

	_add( Xor );
}

void PTXToLLVMTranslator::_bitcast( const ir::PTXInstruction& i )
{
	_bitcast( i.d, i.a );
}

void PTXToLLVMTranslator::_bitcast( const ir::PTXOperand& d, 
	const ir::PTXOperand& a )
{
	_bitcast( _translate( d ), _translate( a ), 
		ir::PTXOperand::isSigned( d.type ) );
}

void PTXToLLVMTranslator::_bitcast( const ir::LLVMInstruction::Operand& d, 
	const ir::LLVMInstruction::Operand& a, bool isSigned )
{
	if( ir::LLVMInstruction::bits( d.type.type ) 
		== ir::LLVMInstruction::bits( a.type.type ) )
	{
		ir::LLVMBitcast cast;
		cast.d = d;
		cast.a = a;
	
		_add( cast );
	}
	else
	{
		ir::LLVMInstruction::Operand temp;
		
		temp.name = _tempRegister();
		temp.type.category = ir::LLVMInstruction::Type::Element;
		temp.type.type = ir::LLVMInstruction::getIntOfSize( 
			ir::LLVMInstruction::bits( a.type.type ) );
		
		_bitcast( temp, a );
		
		if( ir::LLVMInstruction::bits( d.type.type ) 
			< ir::LLVMInstruction::bits( a.type.type ) )
		{
			ir::LLVMTrunc truncate;
			
			truncate.d.name = _tempRegister();
			truncate.d.type.category = ir::LLVMInstruction::Type::Element;
			truncate.d.type.type = ir::LLVMInstruction::getIntOfSize( 
				ir::LLVMInstruction::bits( d.type.type ) );
			
			truncate.a = temp;
			
			_add( truncate );
			
			temp = truncate.d;
		}
		else
		{
			if( isSigned )
			{
				ir::LLVMSext extend;
			
				extend.d.name = _tempRegister();
				extend.d.type.category = ir::LLVMInstruction::Type::Element;
				extend.d.type.type = ir::LLVMInstruction::getIntOfSize( 
					ir::LLVMInstruction::bits( d.type.type ) );
			
				extend.a = temp;
			
				_add( extend );
			
				temp = extend.d;				
			}
			else
			{
				ir::LLVMZext extend;
			
				extend.d.name = _tempRegister();
				extend.d.type.category = ir::LLVMInstruction::Type::Element;
				extend.d.type.type = ir::LLVMInstruction::getIntOfSize( 
					ir::LLVMInstruction::bits( d.type.type ) );
			
				extend.a = temp;
			
				_add( extend );
			
				temp = extend.d;				
			}

		}
		
		_bitcast( d, temp );
	}
}
	
void PTXToLLVMTranslator::_convert( const ir::LLVMInstruction::Operand& d, 
	ir::PTXOperand::DataType dType, const ir::LLVMInstruction::Operand& a, 
	ir::PTXOperand::DataType aType, int modifier )
{
	switch( aType )
	{
		case ir::PTXOperand::s8:
		{
			switch( dType )
			{
				case ir::PTXOperand::pred:
				case ir::PTXOperand::s16:
				case ir::PTXOperand::b16:
				case ir::PTXOperand::u16:
				case ir::PTXOperand::s32:
				case ir::PTXOperand::b32:
				case ir::PTXOperand::u32:
				case ir::PTXOperand::b64:
				case ir::PTXOperand::s64:
				case ir::PTXOperand::u64:
				{
					ir::LLVMSext sext;
					sext.d = d;
					sext.a = a;

					_add( sext );
					break;
				}
				case ir::PTXOperand::s8:
				case ir::PTXOperand::u8:
				case ir::PTXOperand::b8:
				{
					_bitcast( d, a );
					break;
				}
				case ir::PTXOperand::f16:
				case ir::PTXOperand::f64:
				case ir::PTXOperand::f32:
				{
					ir::LLVMSitofp sitofp;
					sitofp.d = d;
					sitofp.a = a;
					
					_add( sitofp );
					break;
				}
				case ir::PTXOperand::TypeSpecifier_invalid:
				{
					assertM( false, "Invalid convert." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::s16:
		{
			switch( dType )
			{
				case ir::PTXOperand::s8:
				case ir::PTXOperand::pred:
				case ir::PTXOperand::b8:
				case ir::PTXOperand::u8:
				{
					ir::LLVMTrunc trunc;
					trunc.d = d;
					trunc.a = a;

					_add( trunc );
					break;
				}
				case ir::PTXOperand::b32:
				case ir::PTXOperand::u32:
				case ir::PTXOperand::s32:
				case ir::PTXOperand::b64:
				case ir::PTXOperand::u64:
				case ir::PTXOperand::s64:
				{
					ir::LLVMSext sext;
					sext.d = d;
					sext.a = a;
					
					_add( sext );
					break;
				}
				case ir::PTXOperand::s16:
				case ir::PTXOperand::b16:
				case ir::PTXOperand::u16:
				{
					_bitcast( d, a );
					break;
				}
				case ir::PTXOperand::f16:
				case ir::PTXOperand::f64:
				case ir::PTXOperand::f32:
				{
					ir::LLVMSitofp sitofp;
					sitofp.d = d;
					sitofp.a = a;
					
					_add( sitofp );
					break;
				}
				case ir::PTXOperand::TypeSpecifier_invalid:
				{
					assertM( false, "Invalid convert." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::s32:
		{
			switch( dType )
			{
				case ir::PTXOperand::pred:
				case ir::PTXOperand::b8:
				case ir::PTXOperand::u8:
				case ir::PTXOperand::s8:
				case ir::PTXOperand::b16:
				case ir::PTXOperand::u16:
				case ir::PTXOperand::s16:
				{
					ir::LLVMTrunc trunc;
					trunc.d = d;
					trunc.a = a;

					_add( trunc );
					break;
				}
				case ir::PTXOperand::s32:
				case ir::PTXOperand::b32:
				case ir::PTXOperand::u32:
				{
					_bitcast( d, a );
					break;
				}
				case ir::PTXOperand::s64:
				case ir::PTXOperand::b64:
				case ir::PTXOperand::u64:
				{
					ir::LLVMSext sext;
					sext.d = d;
					sext.a = a;
					
					_add( sext );
					break;
				}
				case ir::PTXOperand::f16:
				case ir::PTXOperand::f64:
				case ir::PTXOperand::f32:
				{
					ir::LLVMSitofp sitofp;
					sitofp.d = d;
					sitofp.a = a;
					
					_add( sitofp );
					break;
				}
				case ir::PTXOperand::TypeSpecifier_invalid:
				{
					assertM( false, "Invalid convert." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::s64:
		{
			switch( dType )
			{
				case ir::PTXOperand::pred:
				case ir::PTXOperand::s8:
				case ir::PTXOperand::b8:
				case ir::PTXOperand::u8:
				case ir::PTXOperand::s16:
				case ir::PTXOperand::b16:
				case ir::PTXOperand::u16:
				case ir::PTXOperand::s32:
				case ir::PTXOperand::b32:
				case ir::PTXOperand::u32:
				{
					ir::LLVMTrunc trunc;
					trunc.d = d;
					trunc.a = a;

					_add( trunc );
					break;
				}
				case ir::PTXOperand::s64:
				case ir::PTXOperand::b64:
				case ir::PTXOperand::u64:
				{
					_bitcast( d, a );
					break;
				}
				case ir::PTXOperand::f16:
				case ir::PTXOperand::f64:
				case ir::PTXOperand::f32:
				{
					ir::LLVMSitofp sitofp;
					sitofp.d = d;
					sitofp.a = a;
					
					_add( sitofp );
					break;
				}
				case ir::PTXOperand::TypeSpecifier_invalid:
				{
					assertM( false, "Invalid convert." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::pred:
		case ir::PTXOperand::b8:
		case ir::PTXOperand::u8:
		{
			switch( dType )
			{
				case ir::PTXOperand::s16:
				case ir::PTXOperand::s32:
				case ir::PTXOperand::s64:
				case ir::PTXOperand::b16:
				case ir::PTXOperand::u16:
				case ir::PTXOperand::b32:
				case ir::PTXOperand::u32:
				case ir::PTXOperand::b64:
				case ir::PTXOperand::u64:
				{
					ir::LLVMZext zext;
					zext.d = d;
					zext.a = a;
					
					_add( zext );
					break;
				}
				case ir::PTXOperand::pred:
				case ir::PTXOperand::s8:
				case ir::PTXOperand::b8:
				case ir::PTXOperand::u8:
				{
					_bitcast( d, a );
					break;
				}
				case ir::PTXOperand::f16:
				case ir::PTXOperand::f64:
				case ir::PTXOperand::f32:
				{
					ir::LLVMUitofp uitofp;
					uitofp.d = d;
					uitofp.a = a;
					
					_add( uitofp );
					break;
				}
				case ir::PTXOperand::TypeSpecifier_invalid:
				{
					assertM( false, "Invalid convert." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::b16:
		case ir::PTXOperand::u16:
		{
			switch( dType )
			{
				case ir::PTXOperand::b8:
				case ir::PTXOperand::u8:
				case ir::PTXOperand::s8:
				case ir::PTXOperand::pred:
				{
					ir::LLVMTrunc trunc;
					trunc.d = d;
					trunc.a = a;

					_add( trunc );
					break;
				}
				case ir::PTXOperand::s32:
				case ir::PTXOperand::s64:
				case ir::PTXOperand::b32:
				case ir::PTXOperand::u32:
				case ir::PTXOperand::b64:
				case ir::PTXOperand::u64:
				{
					ir::LLVMZext zext;
					zext.d = d;
					zext.a = a;
					
					_add( zext );
					break;
				}
				case ir::PTXOperand::s16:
				case ir::PTXOperand::b16:
				case ir::PTXOperand::u16:
				{
					_bitcast( d, a );
					break;
				}
				case ir::PTXOperand::f16:
				{
					assertM( false, "F16 type not supported" );
					break;
				}
				case ir::PTXOperand::f64:
				case ir::PTXOperand::f32:
				{
					ir::LLVMUitofp uitofp;
					uitofp.d = d;
					uitofp.a = a;
					
					_add( uitofp );
					break;
				}
				case ir::PTXOperand::TypeSpecifier_invalid:
				{
					assertM( false, "Invalid convert." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::b32:
		case ir::PTXOperand::u32:
		{
			switch( dType )
			{
				case ir::PTXOperand::pred:
				case ir::PTXOperand::b8:
				case ir::PTXOperand::u8:
				case ir::PTXOperand::s8:
				case ir::PTXOperand::b16:
				case ir::PTXOperand::u16:
				case ir::PTXOperand::s16:
				{
					ir::LLVMTrunc trunc;
					trunc.d = d;
					trunc.a = a;

					_add( trunc );
					break;
				}
				case ir::PTXOperand::b64:
				case ir::PTXOperand::u64:
				case ir::PTXOperand::s64:
				{
					ir::LLVMZext zext;
					zext.d = d;
					zext.a = a;
					
					_add( zext );
					break;
				}
				case ir::PTXOperand::b32:
				case ir::PTXOperand::s32:
				case ir::PTXOperand::u32:
				{
					_bitcast( d, a );
					break;
				}
				case ir::PTXOperand::f16:
				{
					assertM( false, "F16 type not supported" );
					break;
				}
				case ir::PTXOperand::f64:
				case ir::PTXOperand::f32:
				{
					ir::LLVMUitofp uitofp;
					uitofp.d = d;
					uitofp.a = a;
					
					_add( uitofp );
					break;
				}
				case ir::PTXOperand::TypeSpecifier_invalid:
				{
					assertM( false, "Invalid convert." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::b64:
		case ir::PTXOperand::u64:
		{
			switch( dType )
			{
				case ir::PTXOperand::pred:
				case ir::PTXOperand::s8:
				case ir::PTXOperand::b8:
				case ir::PTXOperand::u8:
				case ir::PTXOperand::b16:
				case ir::PTXOperand::u16:
				case ir::PTXOperand::s16:
				case ir::PTXOperand::b32:
				case ir::PTXOperand::u32:
				case ir::PTXOperand::s32:
				{
					ir::LLVMTrunc trunc;
					trunc.d = d;
					trunc.a = a;

					_add( trunc );
					break;
				}
				case ir::PTXOperand::b64:
				case ir::PTXOperand::s64:
				case ir::PTXOperand::u64:
				{
					_bitcast( d, a );
					break;
				}
				case ir::PTXOperand::f16:
				{
					assertM( false, "F16 type not supported" );
					break;
				}
				case ir::PTXOperand::f64:
				case ir::PTXOperand::f32:
				{
					ir::LLVMUitofp uitofp;
					uitofp.d = d;
					uitofp.a = a;
					
					_add( uitofp );
					break;
				}
				case ir::PTXOperand::TypeSpecifier_invalid:
				{
					assertM( false, "Invalid convert." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::f16:
		{
			switch( dType )
			{
				case ir::PTXOperand::s8:
				case ir::PTXOperand::s16:
				case ir::PTXOperand::s32:
				case ir::PTXOperand::s64:
				{
					ir::LLVMFptosi fptosi;
					fptosi.d = d;
					fptosi.a = a;
					
					_add( fptosi );
					break;
				}
				case ir::PTXOperand::pred:
				case ir::PTXOperand::b8:
				case ir::PTXOperand::u8:
				case ir::PTXOperand::b16:
				case ir::PTXOperand::u16:
				case ir::PTXOperand::b32:
				case ir::PTXOperand::u32:
				case ir::PTXOperand::b64:
				case ir::PTXOperand::u64:
				{
					ir::LLVMFptoui fptoui;
					fptoui.d = d;
					fptoui.a = a;
					
					_add( fptoui );
					break;
				}
				case ir::PTXOperand::f16:
				{
					_bitcast( d, a );
					break;
				}
				case ir::PTXOperand::f32:
				case ir::PTXOperand::f64:
				{
					ir::LLVMFpext fpext;
					fpext.d = d;
					fpext.a = a;
					
					_add( fpext );
					break;
				}
				case ir::PTXOperand::TypeSpecifier_invalid:
				{
					assertM( false, "Invalid convert." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::f32:
		{
			ir::LLVMInstruction::Operand tempA = a;
			
			if( !ir::PTXOperand::isFloat( dType ) )
			{
				if( modifier & ir::PTXInstruction::rni )
				{
					ir::LLVMFcmp compare;
				
					compare.d.type.category 
						= ir::LLVMInstruction::Type::Element;
					compare.d.type.type = ir::LLVMInstruction::I1;
					compare.d.name = _tempRegister();
					compare.comparison = ir::LLVMInstruction::Olt;

					compare.a = tempA;
					compare.b.constant = true;
					compare.b.f32 = 0.0;
					compare.b.type.category 
						= ir::LLVMInstruction::Type::Element;
					compare.b.type.type = ir::LLVMInstruction::F32;
				
					_add( compare );
				
					ir::LLVMSelect select;
				
					select.d = tempA;
					select.d.name = _tempRegister();
				
					select.condition = compare.d;
					select.a = compare.b;
					select.a.f32 = -0.5;
					select.b = compare.b;
					select.b.f32 = 0.5;
				
					_add( select );
				
					ir::LLVMFadd add;
				
					add.d = tempA;
					add.d.name = _tempRegister();
					add.a = tempA;
				
					add.b = select.d;
				
					_add( add );

					ir::LLVMFptosi fptosi;
				
					fptosi.d.name = _tempRegister();
					fptosi.d.type.category = ir::LLVMInstruction::Type::Element;
					fptosi.d.type.type = ir::LLVMInstruction::I32;

					fptosi.a = add.d;
				
					_add( fptosi );

					ir::LLVMSitofp sitofp;
				
					sitofp.d.name = _tempRegister();
					sitofp.d.type.category = ir::LLVMInstruction::Type::Element;
					sitofp.d.type.type = ir::LLVMInstruction::F32;

					sitofp.a = fptosi.d;
				
					_add( sitofp );
				
					tempA = sitofp.d;
				}
				else if( modifier & ir::PTXInstruction::rpi )
				{
					ir::LLVMFadd add;
				
					add.d.name = _tempRegister();
					add.d.type.category = ir::LLVMInstruction::Type::Element;
					add.d.type.type = ir::LLVMInstruction::F32;

					add.a = tempA;
				
					add.b.constant = true;
					add.b.f32 = 1.0f;
					add.b.type.category 
						= ir::LLVMInstruction::Type::Element;
					add.b.type.type = ir::LLVMInstruction::F32;
				
					_add( add );
				
					tempA = add.d;
				}
			}
			
			if( (ir::PTXInstruction::sat & modifier) &&
				ir::PTXOperand::isFloat( dType ) )
			{
				ir::LLVMFcmp compare;
				
				compare.d.type.category 
					= ir::LLVMInstruction::Type::Element;
				compare.d.type.type = ir::LLVMInstruction::I1;
				compare.d.name = _tempRegister();
				compare.comparison = ir::LLVMInstruction::Olt;

				compare.a = tempA;
				compare.b.constant = true;
				compare.b.f32 = 1.0;
				compare.b.type.category 
					= ir::LLVMInstruction::Type::Element;
				compare.b.type.type = ir::LLVMInstruction::F32;
				
				_add( compare );
				
				ir::LLVMSelect select;
				
				select.d = tempA;
				select.d.name = _tempRegister();
				
				select.condition = compare.d;
				select.a = tempA;
				select.b = compare.b;
				
				_add( select );
				
				compare.d.name = _tempRegister();
				compare.a = select.d;
				compare.b.f32 = 0.0;
				
				_add( compare );
				
				select.d.name = _tempRegister();
				
				select.condition = compare.d;
				select.a = compare.b;
				select.b = compare.a;
				
				_add( select );
				
				tempA = select.d;			
			}
			
			switch( dType )
			{
				case ir::PTXOperand::s8:
				case ir::PTXOperand::s16:
				case ir::PTXOperand::s32:
				case ir::PTXOperand::s64:
				{
					ir::LLVMFptosi fptosi;
					fptosi.d = ir::LLVMInstruction::Operand( 
						_tempRegister(), d.type );
					fptosi.a = tempA;
					
					_add( fptosi );

					_floatToIntSaturate( d, fptosi.d, tempA, true );
					break;
				}
				case ir::PTXOperand::pred:
				case ir::PTXOperand::b8:
				case ir::PTXOperand::u8:
				case ir::PTXOperand::b16:
				case ir::PTXOperand::u16:
				case ir::PTXOperand::b32:
				case ir::PTXOperand::u32:
				case ir::PTXOperand::b64:
				case ir::PTXOperand::u64:
				{
					ir::LLVMFptoui fptoui;
					fptoui.d = ir::LLVMInstruction::Operand( 
						_tempRegister(), d.type );
					fptoui.a = tempA;
					
					_add( fptoui );

					_floatToIntSaturate( d, fptoui.d, tempA, false );
					break;
				}
				case ir::PTXOperand::f16:
				{
					ir::LLVMFptrunc fptrunc;
					fptrunc.d = d;
					fptrunc.a = tempA;
					
					_add( fptrunc );
					break;
				}
				case ir::PTXOperand::f32:
				{
					if( ir::PTXInstruction::rzi & modifier )
					{
						_trunc( d, tempA );
					}
					else if( ir::PTXInstruction::rni & modifier )
					{
						_nearbyint( d, tempA );
					}
					else if( ir::PTXInstruction::rmi & modifier )
					{
						_floor( d, tempA );
					}
					else if( ir::PTXInstruction::rpi & modifier )
					{
						_ceil( d, tempA );
					}
					else
					{
						_bitcast( d, tempA );
					}
					break;
				}
				case ir::PTXOperand::f64:
				{
					ir::LLVMFpext fpext;
					fpext.d = d;
					fpext.a = tempA;
					
					_add( fpext );
					break;
				}
				case ir::PTXOperand::TypeSpecifier_invalid:
				{
					assertM( false, "Invalid convert." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::f64:
		{
			switch( dType )
			{
				case ir::PTXOperand::s8:
				case ir::PTXOperand::s16:
				case ir::PTXOperand::s32:
				case ir::PTXOperand::s64:
				{
					ir::LLVMFptosi fptosi;
					fptosi.d = ir::LLVMInstruction::Operand( 
						_tempRegister(), d.type );
					fptosi.a = a;
					
					_add( fptosi );

					_floatToIntSaturate( d, fptosi.d, a, true );
					break;
				}
				case ir::PTXOperand::pred:
				case ir::PTXOperand::b8:
				case ir::PTXOperand::u8:
				case ir::PTXOperand::b16:
				case ir::PTXOperand::u16:
				case ir::PTXOperand::b32:
				case ir::PTXOperand::u32:
				case ir::PTXOperand::b64:
				case ir::PTXOperand::u64:
				{
					ir::LLVMFptoui fptoui;
					fptoui.d = ir::LLVMInstruction::Operand( 
						_tempRegister(), d.type );
					fptoui.a = a;
					
					_add( fptoui );

					_floatToIntSaturate( d, fptoui.d, a, false );
					break;
				}
				case ir::PTXOperand::f16:
				{
					ir::LLVMFptrunc fptrunc;
					fptrunc.d = d;
					fptrunc.a = a;
					
					_add( fptrunc );
					break;
				}
				case ir::PTXOperand::f32:
				{
					ir::LLVMFptrunc fptrunc;
					fptrunc.d = d;
					fptrunc.a = a;
					
					_add( fptrunc );
					break;
				}
				case ir::PTXOperand::f64:
				{
					_bitcast( d, a );
					break;
				}
				case ir::PTXOperand::TypeSpecifier_invalid:
				{
					assertM( false,"Invalid convert." );
					break;
				}
			}
			break;
		}
		case ir::PTXOperand::TypeSpecifier_invalid:
		{
			assertM( false, "Invalid convert." );
			break;
		}
	}
}

void PTXToLLVMTranslator::_flushToZero(
	const ir::LLVMInstruction::Operand& d,
	const ir::LLVMInstruction::Operand& a )
{
	ir::LLVMFcmp less;
	
	less.comparison = ir::LLVMInstruction::Olt;
	
	less.d = ir::LLVMInstruction::Operand( _tempRegister(),
		ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
		ir::LLVMInstruction::Type::Element ) );
	less.a = a;
	less.b = ir::LLVMInstruction::Operand( (ir::LLVMI64) 0 );
	less.b.type.type = less.a.type.type;
	
	_add( less );

	ir::LLVMFsub subtract;
	
	subtract.d = ir::LLVMInstruction::Operand( _tempRegister(),
		ir::LLVMInstruction::Type( less.a.type.type, 
		ir::LLVMInstruction::Type::Element ) );
	subtract.a = ir::LLVMInstruction::Operand( (ir::LLVMI64) 0 );
	subtract.a.type.type = less.a.type.type;
	subtract.b = less.a;
	
	_add( subtract );
	
	ir::LLVMSelect select;
	
	select.condition = less.d;
	
	select.d = ir::LLVMInstruction::Operand( _tempRegister(),
		ir::LLVMInstruction::Type( less.a.type.type, 
		ir::LLVMInstruction::Type::Element ) );
	select.a = subtract.d;
	select.b = less.a;
	
	_add( select );
	
	ir::LLVMFcmp greaterEqual;
	
	greaterEqual.comparison = ir::LLVMInstruction::Olt;
	
	greaterEqual.d = ir::LLVMInstruction::Operand( _tempRegister(),
		ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
		ir::LLVMInstruction::Type::Element ) );
	greaterEqual.a = select.d;
	
	greaterEqual.b = ir::LLVMInstruction::Operand( 
		(ir::LLVMI32) hydrazine::bit_cast< ir::LLVMI32 >(
		std::numeric_limits<float>::min() ) );
	
	greaterEqual.b.type.type = less.a.type.type;
	
	_add( greaterEqual );

	ir::LLVMSelect flush;
	
	flush.d = d;
	flush.condition = greaterEqual.d;
	flush.a = ir::LLVMInstruction::Operand( (ir::LLVMF32) 0.0f );
	flush.b = a;
	
	_add( flush );
}

void PTXToLLVMTranslator::_saturate( const ir::LLVMInstruction::Operand& d,
	const ir::LLVMInstruction::Operand& a )
{
	ir::LLVMFcmp compare;
	
	compare.d.name = _tempRegister();
	compare.d.type.type = ir::LLVMInstruction::I1;
	compare.d.type.category = ir::LLVMInstruction::Type::Element;
	compare.comparison = ir::LLVMInstruction::Ult;
	compare.a = a;
	compare.b.type.type = compare.a.type.type;
	compare.b.type.category = ir::LLVMInstruction::Type::Element;
	compare.b.constant = true;
	compare.b.f32 = 0.0f;
	
	ir::LLVMSelect select;
	
	select.d = ir::LLVMInstruction::Operand( _tempRegister(), 
		ir::LLVMInstruction::Type( ir::LLVMInstruction::F32,
		ir::LLVMInstruction::Type::Element ) );
	select.condition = compare.d;
	select.a = compare.b;
	select.b = a;
	
	_add( compare );
	_add( select );
	
	compare.d.name = _tempRegister();
	compare.comparison = ir::LLVMInstruction::Ogt;
	compare.b.f32  = 1.0f;
	compare.a = a;
	
	select.b = select.d;
	
	select.d = d;		
	select.condition = compare.d;
	select.a = compare.b;
	
	_add( compare );
	_add( select );	
}

void PTXToLLVMTranslator::_floatToIntSaturate(
	const ir::LLVMInstruction::Operand& d, 
	const ir::LLVMInstruction::Operand& ftoint,
	const ir::LLVMInstruction::Operand& f, bool isSigned)
{
	ir::LLVMInstruction::Operand min( _tempRegister(), f.type );
	ir::LLVMInstruction::Operand max( _tempRegister(), f.type );
	ir::LLVMInstruction::Operand minInt( _tempRegister(), d.type );
	ir::LLVMInstruction::Operand maxInt( _tempRegister(), d.type );
	ir::LLVMInstruction::Operand lessThanMin( _tempRegister(),
		ir::LLVMInstruction::Type( ir::LLVMInstruction::I1, 
		ir::LLVMInstruction::Type::Element ) );
	ir::LLVMInstruction::Operand greaterThanMax( _tempRegister(),
		lessThanMin.type );
	ir::LLVMInstruction::Operand nan( _tempRegister(), lessThanMin.type );

	if( isSigned )
	{
		ir::LLVMSitofp sitofp;
		
		sitofp.d          = min;
		sitofp.a.constant = true;
		sitofp.a.type     = d.type;

		switch( d.type.type )
		{
		case ir::LLVMInstruction::I1:
		{
			sitofp.a.i64 = 0;
			break;
		}
		case ir::LLVMInstruction::I8:
		{
			sitofp.a.i64 = std::numeric_limits<char>::min();
			break;
		}
		case ir::LLVMInstruction::I16:
		{
			sitofp.a.i64 = std::numeric_limits<short>::min();
			break;
		}
		case ir::LLVMInstruction::I32:
		{
			sitofp.a.i64 = std::numeric_limits<int>::min();
			break;
		}
		case ir::LLVMInstruction::I64:
		{
			sitofp.a.i64 = std::numeric_limits<long long int>::min();
			break;
		}
		default: break;
		}

		minInt = sitofp.a;

		_add( sitofp );
		
		sitofp.d = max;

		switch( d.type.type )
		{
		case ir::LLVMInstruction::I1:
		{
			sitofp.a.i64 = 0;
			break;
		}
		case ir::LLVMInstruction::I8:
		{
			sitofp.a.i64 = std::numeric_limits<char>::max();
			break;
		}
		case ir::LLVMInstruction::I16:
		{
			sitofp.a.i64 = std::numeric_limits<short>::max();
			break;
		}
		case ir::LLVMInstruction::I32:
		{
			sitofp.a.i64 = std::numeric_limits<int>::max();
			break;
		}
		case ir::LLVMInstruction::I64:
		{
			sitofp.a.i64 = std::numeric_limits<long long int>::max();
			break;
		}
		default: break;
		}

		maxInt = sitofp.a;

		_add( sitofp );
	}
	else
	{
		ir::LLVMUitofp uitofp;
		
		uitofp.d          = min;
		uitofp.a.constant = true;
		uitofp.a.type     = d.type;

		switch( d.type.type )
		{
		case ir::LLVMInstruction::I1:
		{
			uitofp.a.i64 = 1;
			break;
		}
		case ir::LLVMInstruction::I8:
		{
			uitofp.a.i64 = std::numeric_limits<unsigned char>::min();
			break;
		}
		case ir::LLVMInstruction::I16:
		{
			uitofp.a.i64 = std::numeric_limits<unsigned short>::min();
			break;
		}
		case ir::LLVMInstruction::I32:
		{
			uitofp.a.i64 = std::numeric_limits<unsigned int>::min();
			break;
		}
		case ir::LLVMInstruction::I64:
		{
			uitofp.a.i64 = std::numeric_limits<
				long long unsigned int>::min();
			break;
		}
		default: break;
		}

		minInt = uitofp.a;

		_add( uitofp );
		
		uitofp.d = max;

		switch( d.type.type )
		{
		case ir::LLVMInstruction::I1:
		{
			uitofp.a.i64 = 0;
			break;
		}
		case ir::LLVMInstruction::I8:
		{
			uitofp.a.i64 = std::numeric_limits<unsigned char>::max();
			break;
		}
		case ir::LLVMInstruction::I16:
		{
			uitofp.a.i64 = std::numeric_limits<unsigned short>::max();
			break;
		}
		case ir::LLVMInstruction::I32:
		{
			uitofp.a.i64 = std::numeric_limits<unsigned int>::max();
			break;
		}
		case ir::LLVMInstruction::I64:
		{
			uitofp.a.i64 = std::numeric_limits<
				long long unsigned int>::max();
			break;
		}
		default: break;
		}
		
		maxInt = uitofp.a;
		
		_add( uitofp );
	}
	
	ir::LLVMFcmp compare;
	
	compare.comparison = ir::LLVMInstruction::Olt;
	
	compare.d = lessThanMin;
	compare.a = f;
	compare.b = min;
	
	_add( compare );
	
	compare.comparison = ir::LLVMInstruction::Ogt;
	
	compare.d = greaterThanMax;
	compare.a = f;
	compare.b = max;
	
	_add( compare );
	
	compare.comparison = ir::LLVMInstruction::Uno;
	compare.d = nan;
	
	_add( compare );
	
	ir::LLVMSelect select;
	
	select.condition = lessThanMin;
	select.a = minInt;
	select.b = ftoint;
	select.d = ir::LLVMInstruction::Operand( _tempRegister(), d.type );

	_add( select );
	
	select.condition = greaterThanMax;
	select.a = maxInt;
	select.b = select.d;
	select.d = ir::LLVMInstruction::Operand( _tempRegister(), d.type );
	
	_add( select );
	
	select.condition = nan;
	select.a.i64     = 0;
	select.b         = select.d;
	select.d         = d;
	
	_add( select );
}

void PTXToLLVMTranslator::_trunc(const ir::LLVMInstruction::Operand& d, 
	const ir::LLVMInstruction::Operand& a)
{
	ir::LLVMCall call;
	
	if( d.type.type == ir::LLVMInstruction::F32 )
	{
		call.name = "@truncf";
	}
	else
	{
		call.name = "@trunc";
	}
	
	call.d = d;
	call.parameters.push_back( a );
	
	_add( call );
}

void PTXToLLVMTranslator::_nearbyint(const ir::LLVMInstruction::Operand& d, 
	const ir::LLVMInstruction::Operand& a)
{
	ir::LLVMCall call;
	
	if( d.type.type == ir::LLVMInstruction::F32 )
	{
		call.name = "@nearbyintf";
	}
	else
	{
		call.name = "@nearbyint";
	}
	
	call.d = d;
	call.parameters.push_back( a );
	
	_add( call );

}

void PTXToLLVMTranslator::_floor(const ir::LLVMInstruction::Operand& d, 
	const ir::LLVMInstruction::Operand& a)
{
	ir::LLVMCall call;
	
	if( d.type.type == ir::LLVMInstruction::F32 )
	{
		call.name = "@floorf";
	}
	else
	{
		call.name = "@floor";
	}
	
	call.d = d;
	call.parameters.push_back( a );
	
	_add( call );
}

void PTXToLLVMTranslator::_ceil(const ir::LLVMInstruction::Operand& d, 
	const ir::LLVMInstruction::Operand& a)
{
	ir::LLVMCall call;
	
	if( d.type.type == ir::LLVMInstruction::F32 )
	{
		call.name = "@ceilf";
	}
	else
	{
		call.name = "@ceil";
	}
	
	call.d = d;
	call.parameters.push_back( a );
	
	_add( call );
}

std::string PTXToLLVMTranslator::_tempRegister()
{
	std::stringstream stream;
	stream << "%rt" << _tempRegisterCount++;
	return stream.str();
}

std::string PTXToLLVMTranslator::_loadSpecialRegister( 
	ir::PTXOperand::SpecialRegister s, ir::PTXOperand::VectorIndex index )
{
	std::string reg;

	ir::LLVMGetelementptr get;
	
	get.d.type.category = ir::LLVMInstruction::Type::Pointer;
	get.d.type.type = ir::LLVMInstruction::I32;
	get.a = _context();
	get.indices.push_back( 0 );
	
	switch( s )
	{
		case ir::PTXOperand::tid:
		{
			switch( index )
			{
				case ir::PTXOperand::ix:
				{
					get.indices.push_back( 0 );
					get.indices.push_back( 0 );
					break;
				}
				case ir::PTXOperand::iy:
				{
					get.indices.push_back( 0 );
					get.indices.push_back( 1 );
					break;
				}
				case ir::PTXOperand::iz:
				{
					get.indices.push_back( 0 );
					get.indices.push_back( 2 );
					break;
				}
				default: assertM( false, "Invalid Special register " 
					<< ir::PTXOperand::toString( s ) << "." );
			}
			break;
		}
		case ir::PTXOperand::ntid:
		{
			switch( index )
			{
				case ir::PTXOperand::ix:
				{
					get.indices.push_back( 1 );
					get.indices.push_back( 0 );
					break;
				}
				case ir::PTXOperand::iy:
				{
					get.indices.push_back( 1 );
					get.indices.push_back( 1 );
					break;
				}
				case ir::PTXOperand::iz:
				{
					get.indices.push_back( 1 );
					get.indices.push_back( 2 );
					break;
				}
				default: assertM( false, "Invalid Special register " 
					<< ir::PTXOperand::toString( s ) << "." );
			}
			break;
		}
		case ir::PTXOperand::laneId:
		{
			get.indices.push_back( 10 );
			break;
//			ir::LLVMBitcast bitcast;
//			
//			bitcast.d.type.category = ir::LLVMInstruction::Type::Element;
//			bitcast.d.type.type = ir::LLVMInstruction::I32;
//			bitcast.d.name = _tempRegister();
//			
//			bitcast.a = ir::LLVMInstruction::Operand((ir::LLVMI32) 0);
//			
//			_add( bitcast );
//			
//			return bitcast.d.name;
		}
		case ir::PTXOperand::warpId:
		{
			ir::LLVMBitcast bitcast;
			
			bitcast.d.type.category = ir::LLVMInstruction::Type::Element;
			bitcast.d.type.type = ir::LLVMInstruction::I32;
			bitcast.d.name = _tempRegister();
			
			bitcast.a = ir::LLVMInstruction::Operand((ir::LLVMI32) 0);
			
			_add( bitcast );
			
			return bitcast.d.name;
		}
		case ir::PTXOperand::warpSize:
		{
			ir::LLVMBitcast bitcast;
			
			bitcast.d.type.category = ir::LLVMInstruction::Type::Element;
			bitcast.d.type.type = ir::LLVMInstruction::I32;
			bitcast.d.name = _tempRegister();
			
			bitcast.a = ir::LLVMInstruction::Operand((ir::LLVMI32) 1);
			
			_add( bitcast );
			
			return bitcast.d.name;
		}
		case ir::PTXOperand::ctaId:
		{
			switch( index )
			{
				case ir::PTXOperand::ix:
				{
					get.indices.push_back( 2 );
					get.indices.push_back( 0 );
					break;
				}
				case ir::PTXOperand::iy:
				{
					get.indices.push_back( 2 );
					get.indices.push_back( 1 );
					break;
				}
				case ir::PTXOperand::iz:
				{
					get.indices.push_back( 2 );
					get.indices.push_back( 2 );
					break;
				}
				default: assertM( false, "Invalid Special register " 
					<< ir::PTXOperand::toString( s ) << "." );
			}
			break;
		}
		case ir::PTXOperand::nctaId:
		{
			switch( index )
			{
				case ir::PTXOperand::ix:
				{
					get.indices.push_back( 3 );
					get.indices.push_back( 0 );
					break;
				}
				case ir::PTXOperand::iy:
				{
					get.indices.push_back( 3 );
					get.indices.push_back( 1 );
					break;
				}
				case ir::PTXOperand::iz:
				{
					get.indices.push_back( 3 );
					get.indices.push_back( 2 );
					break;
				}
				default: assertM( false, "Invalid Special register " 
					<< ir::PTXOperand::toString( s ) << "." );
			}
			break;
		}
		case ir::PTXOperand::smId:
		{
			ir::LLVMBitcast bitcast;
			
			bitcast.d.type.category = ir::LLVMInstruction::Type::Element;
			bitcast.d.type.type = ir::LLVMInstruction::I32;
			bitcast.d.name = _tempRegister();
			
			bitcast.a = ir::LLVMInstruction::Operand((ir::LLVMI32) 0);
			
			_add( bitcast );
			
			return bitcast.d.name;
		}
		case ir::PTXOperand::nsmId:
		{
			ir::LLVMBitcast bitcast;
			
			bitcast.d.type.category = ir::LLVMInstruction::Type::Element;
			bitcast.d.type.type = ir::LLVMInstruction::I32;
			bitcast.d.name = _tempRegister();
			
			bitcast.a = ir::LLVMInstruction::Operand((ir::LLVMI32) 1);
			
			_add( bitcast );
			
			return bitcast.d.name;
		}
		case ir::PTXOperand::gridId:
		{
			assertM( false, "Special register " 
				<< ir::PTXOperand::toString( s ) << " not supported." );
			break;
		}
		case ir::PTXOperand::clock:
		{
			ir::LLVMCall call;
			
			call.name = "@llvm.readcyclecounter";
			call.d.type.category = ir::LLVMInstruction::Type::Element;
			call.d.type.type = ir::LLVMInstruction::I64;
			call.d.name = _tempRegister();

			_add( call );
							
			ir::LLVMTrunc cast;
			
			cast.d.type.category = ir::LLVMInstruction::Type::Element;
			cast.d.type.type = ir::LLVMInstruction::I32;
			cast.d.name = _tempRegister();
			
			cast.a = call.d;
			
			_add( cast );
			
			return cast.d.name;
			break;
		}
		case ir::PTXOperand::pm0:
		{
			assertM( false, "Special register " 
				<< ir::PTXOperand::toString( s ) << " not supported." );
			break;
		}
		case ir::PTXOperand::pm1:
		{
			assertM( false, "Special register " 
				<< ir::PTXOperand::toString( s ) << " not supported." );
			break;
		}
		case ir::PTXOperand::pm2:
		{
			assertM( false, "Special register " 
				<< ir::PTXOperand::toString( s ) << " not supported." );
			break;
		}
		case ir::PTXOperand::pm3:
		{
			assertM( false, "Special register " 
				<< ir::PTXOperand::toString( s ) << " not supported." );
			break;
		}
		default: break;
	}
	
	get.d.name = _tempRegister();
	
	_add( get );
	
	ir::LLVMLoad load;
	
	load.d.name = _tempRegister();;
	load.d.type.category = ir::LLVMInstruction::Type::Element;
	load.d.type.type = ir::LLVMInstruction::I32;
	load.a = get.d;
	
	_add( load );
	
	return load.d.name;
}

ir::LLVMInstruction::Operand PTXToLLVMTranslator::_getMemoryExtent( 
 	ir::PTXInstruction::AddressSpace space )
{
	ir::LLVMCall call;
	
	call.name = "@__ocelot_get_extent";

	call.d = ir::LLVMInstruction::Operand( _tempRegister(), 
		ir::LLVMInstruction::Type( ir::LLVMInstruction::I32, 
		ir::LLVMInstruction::Type::Element ) );
	
	call.parameters.resize( 2 );

	call.parameters[0] = _context();

	call.parameters[1].type.type = ir::LLVMInstruction::I32;
	call.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	call.parameters[1].constant = true;
	call.parameters[1].i32 = space;
	
	_add( call );
	
	return call.d;
}			

ir::LLVMInstruction::Operand PTXToLLVMTranslator::_getMemoryBasePointer( 
	ir::PTXInstruction::AddressSpace space, bool isArgument,
	bool isGlobalLocal )
{
	ir::LLVMGetelementptr get;
	
	get.d.name = _tempRegister();
	get.d.type.category = ir::LLVMInstruction::Type::Pointer;
	get.d.type.members.resize( 1 );
	get.d.type.members[0].category = ir::LLVMInstruction::Type::Pointer;
	get.d.type.members[0].type = ir::LLVMInstruction::I8;
	
	get.a = _context();
	get.indices.push_back( 0 );
	
	switch( space )
	{
		case ir::PTXInstruction::Const:
		{
			get.indices.push_back( 6 );
			break;
		}
		case ir::PTXInstruction::Shared:
		{
			get.indices.push_back( 5 );
			break;
		}
		case ir::PTXInstruction::Local:
		{
			if( isGlobalLocal )
			{
				get.indices.push_back( 9 );
			}
			else
			{
				get.indices.push_back( 4 );
			}
			break;
		}
		case ir::PTXInstruction::Param:
		{
			if( isArgument )
			{
				get.indices.push_back( 8 );
			}
			else
			{
				get.indices.push_back( 7 );
			}
			break;
		}
		default:
		{
			assertM( false, "Invalid memory space." );
		}			
	}
			
	_add( get );
	
	ir::LLVMLoad load;
	
	load.d.name = _tempRegister();
	load.d.type.category = ir::LLVMInstruction::Type::Pointer;
	load.d.type.type = ir::LLVMInstruction::I8;
	
	load.a = get.d;
	
	_add( load );
	
	return load.d;
}
		
ir::LLVMInstruction::Operand 
	PTXToLLVMTranslator::_getAddressableVariablePointer( 
	ir::PTXInstruction::AddressSpace space, const ir::PTXOperand& o )
{		
	ir::LLVMInstruction::Operand index;
	
	if( o.offset != 0 )
	{
		ir::LLVMGetelementptr getIndex;
	
		getIndex.d.name = _tempRegister();
		getIndex.d.type.category = ir::LLVMInstruction::Type::Pointer;
		getIndex.d.type.type = ir::LLVMInstruction::I8;
	
		getIndex.a = _getMemoryBasePointer( space, o.isArgument,
			o.isGlobalLocal );
		getIndex.indices.push_back( o.offset );
	
		_add( getIndex );
		
		index = getIndex.d;
	}
	else
	{
		index = _getMemoryBasePointer( space, o.isArgument, o.isGlobalLocal );
	}
	
	return index;
}

ir::LLVMInstruction::Operand 
	PTXToLLVMTranslator::_getAddressableGlobalPointer( 
	const ir::PTXOperand& o )
{
	ir::Module::GlobalMap::const_iterator 
		global = _llvmKernel->module->globals().find( o.identifier );
	assert( global != _llvmKernel->module->globals().end() );
	
	if( global->second.statement.elements() == 1 )
	{
		ir::LLVMInstruction::Operand result;
		
		result.type.category = ir::LLVMInstruction::Type::Pointer;
		result.type.type = _translate( global->second.statement.type );
		result.name = "@" + o.identifier;
		
		return result;
	}
	
	ir::LLVMGetelementptr get;
	get.a.type.category = ir::LLVMInstruction::Type::Pointer;
	get.a.name = "@" + o.identifier;
	
	get.a.type.members.resize( 1 );
	get.a.type.members[0].category 
		= ir::LLVMInstruction::Type::Array;
	get.a.type.members[0].vector 
		= global->second.statement.elements();
	
	get.a.type.members[0].type = _translate( 
		global->second.statement.type );
	
	get.d.type.category = ir::LLVMInstruction::Type::Pointer;
	get.d.type.type = get.a.type.members[0].type;
	get.d.name = _tempRegister();
	get.indices.push_back( 0 );
	get.indices.push_back( o.offset );
	
	_add( get );
	
	return get.d;
}

ir::LLVMInstruction::Operand PTXToLLVMTranslator::_getLoadOrStorePointer( 
	const ir::PTXOperand& o, ir::PTXInstruction::AddressSpace space, 
	ir::LLVMInstruction::DataType type, unsigned int vector )
{
	ir::LLVMInstruction::Operand pointer;
	
	if( o.addressMode == ir::PTXOperand::Address )
	{
		if( space == ir::PTXInstruction::Global )
		{
			pointer = _getAddressableGlobalPointer( o );
		}
		else
		{
			pointer = _getAddressableVariablePointer( space, o );
		}
		
		if( type == ir::LLVMInstruction::I8 )
		{
			return pointer;
		}
		
		ir::LLVMBitcast cast;
		
		cast.d.name = _tempRegister();
		cast.d.type.category = ir::LLVMInstruction::Type::Pointer;
		cast.a = pointer;
	
		if( vector == ir::PTXOperand::v1 )
		{
			cast.d.type.type = type;
		}
		else
		{
			cast.d.type.members.resize( 1 );
			cast.d.type.members[ 0 ].category 
				= ir::LLVMInstruction::Type::Vector;
			cast.d.type.members[ 0 ].type = type;
			cast.d.type.members[ 0 ].vector = vector;		
		}
	
		_add( cast );
	
		return cast.d;
	}
	else
	{
		assert( o.addressMode == ir::PTXOperand::Register 
			|| o.addressMode == ir::PTXOperand::Indirect
			|| o.addressMode == ir::PTXOperand::Immediate );

		ir::LLVMInstruction::Operand reg = _translate( o );
		
		// Possibly cast this to a 64-bit address
		#ifndef __i386__
		if( reg.type.type != ir::LLVMInstruction::I64 )
		{
			ir::LLVMZext extend;
			
			extend.a = reg;
			extend.d = ir::LLVMInstruction::Operand( _tempRegister(), 
				ir::LLVMInstruction::Type( ir::LLVMInstruction::I64,
				ir::LLVMInstruction::Type::Element ) );
		
			_add( extend );
			
			reg = extend.d;
		}
		#endif
		
		if( o.offset != 0 )
		{
			ir::LLVMAdd add;
		
			add.d.name = _tempRegister();
			add.d.type.category = ir::LLVMInstruction::Type::Element;
			add.d.type.type = reg.type.type;
			
			add.a = reg;
			
			add.b.type.category = ir::LLVMInstruction::Type::Element;
			add.b.type.type = reg.type.type;
			add.b.constant = true;
			add.b.i64 = o.offset;
	
			_add( add );
			
			reg = add.d;
		}
		
		if( space == ir::PTXInstruction::Global
			|| space == ir::PTXInstruction::Generic )
		{
			pointer = reg;
		}
		else
		{
			ir::LLVMPtrtoint toint;
		
			toint.a = _getMemoryBasePointer( space, false, false );
			toint.d.name = _tempRegister();
			toint.d.type.category = ir::LLVMInstruction::Type::Element;
			toint.d.type.type = reg.type.type;
		
			_add( toint );
		
			ir::LLVMAdd add;
		
			add.d.name = _tempRegister();
			add.d.type.category = ir::LLVMInstruction::Type::Element;
			add.d.type.type = reg.type.type;
		
			add.a = reg;	
			add.b = toint.d;

			_add( add );
		
			pointer = add.d;
		}
		
		ir::LLVMInttoptr toptr;
		
		toptr.d.name = _tempRegister();
		toptr.d.type.category = ir::LLVMInstruction::Type::Pointer;
		toptr.a = pointer;
	
		if( vector == ir::PTXOperand::v1 )
		{
			toptr.d.type.type = type;
		}
		else
		{
			toptr.d.type.members.resize( 1 );
			toptr.d.type.members[ 0 ].category 
				= ir::LLVMInstruction::Type::Vector;
			toptr.d.type.members[ 0 ].type = type;
			toptr.d.type.members[ 0 ].vector = vector;		
		}
	
		_add( toptr );
	
		return toptr.d;
	}
}

ir::LLVMInstruction::Operand PTXToLLVMTranslator::_destination( 
	const ir::PTXInstruction& i, bool pq )
{
	ir::LLVMInstruction::Operand destination;
	if( pq )
	{
		destination = _translate( i.pq );
	}
	else
	{
		destination = _translate( i.d );
	}

	assertM( i.pg.condition == ir::PTXOperand::PT, 
		"Predication not supported." )

	return destination;
}

ir::LLVMInstruction::Operand PTXToLLVMTranslator::_destinationCC( 
	const ir::PTXInstruction& i )
{
	ir::LLVMInstruction::Operand destination;
	destination.type.category = ir::LLVMInstruction::Type::Element;
	destination.type.type = ir::LLVMInstruction::I64;
	
	assertM( i.pg.condition == ir::PTXOperand::PT, 
		"Predication not supported." );
	std::stringstream stream;
	stream << "rcc" << _tempCCRegisterCount++;
	destination.name = stream.str();
	
	return destination;
}

ir::LLVMInstruction::Operand PTXToLLVMTranslator::_conditionCodeRegister( 
	const ir::PTXOperand& op )
{
	ir::LLVMInstruction::Operand cc;

	std::stringstream stream;
	stream << "rcc" << op.reg;
	
	cc.name = stream.str();
	cc.type.category = ir::LLVMInstruction::Type::Element;
	cc.type.type = ir::LLVMInstruction::I64;

	return cc;
}

void PTXToLLVMTranslator::_add( const ir::LLVMInstruction& i )
{
	assertM( i.valid() == "", "Instruction " << i.toString() 
		<< " is not valid: " << i.valid() );
	report( "    Added instruction '" << i.toString() << "'" );
	_llvmKernel->push_back( ir::LLVMStatement( i ) );	
}

void PTXToLLVMTranslator::_initializeRegisters()
{
	report( " Adding initialization instructions." );

	if( !_uninitialized.empty() )
	{
		ir::LLVMBr branch;
		branch.iftrue = "%" + (++_dfg->begin())->label();
	
		_llvmKernel->push_front( 
			ir::LLVMStatement( branch ) );
	}

	for( RegisterVector::const_iterator reg = _uninitialized.begin(); 
		reg != _uninitialized.end(); ++reg )
	{
		ir::LLVMBitcast move;
		
		std::stringstream stream;
		
		stream << "%ri" << reg->id;
		
		move.d.name = stream.str();
		move.d.type.category = ir::LLVMInstruction::Type::Element;
		move.d.type.type = _translate( reg->type );
		
		move.a = move.d;
		move.a.constant = true;
		move.a.i64 = 0;
		
		report( "  Adding instruction '" << move.toString() << "'" );		
		
		_llvmKernel->push_front( ir::LLVMStatement( move ) );
	}
	
	if( !_uninitialized.empty() )
	{
		_llvmKernel->push_front( 
			ir::LLVMStatement( "$OcelotRegisterInitializerBlock" ) );
	}
}

void PTXToLLVMTranslator::_addGlobalDeclarations()
{
	report("Translating Globals...");
	for( ir::Module::GlobalMap::const_iterator 
		global = _llvmKernel->module->globals().begin(); 
		global != _llvmKernel->module->globals().end(); ++global )
	{
		if( global->second.statement.directive 
			!= ir::PTXStatement::Global ) continue;
	
		report(" translating '" << global->second.statement.toString() << "'");
		ir::LLVMStatement statement( 
			ir::LLVMStatement::VariableDeclaration );

		statement.label = global->second.statement.name;
		statement.linkage = ir::LLVMStatement::External;
		statement.visibility = ir::LLVMStatement::Default;
	
		if( global->second.statement.elements() == 1 )
		{
			statement.operand.type.category 
				= ir::LLVMInstruction::Type::Element;
		}
		else
		{
			statement.operand.type.category 
				= ir::LLVMInstruction::Type::Array;
			if( global->second.statement.attribute 
				== ir::PTXStatement::Extern )
			{
				statement.operand.type.vector = 0;
			}
			else
			{
				assert( global->second.statement.elements() > 0 );
				statement.operand.type.vector 
					= global->second.statement.elements();
			}
		}
		
		statement.operand.type.type = _translate( 
			global->second.statement.type );
		statement.alignment = ir::PTXOperand::bytes( 
			global->second.statement.type );
	
		report("  adding LLVM statement '" << statement.toString() << "'");
	
		_llvmKernel->push_front( statement );
	}
}

void PTXToLLVMTranslator::_addExternalFunctionDeclarations()
{
	if( _externals == 0 ) return;
	
	for( StringSet::const_iterator name = _usedExternalCalls.begin();
		name != _usedExternalCalls.end(); ++name )
	{
		ir::ExternalFunctionSet::ExternalFunction*
			external = _externals->find( *name );
		assert( external != 0 );

		ir::LLVMStatement function(
			ir::LLVMStatement::FunctionDeclaration );

		function.label      = external->name();
		function.linkage    = ir::LLVMStatement::InvalidLinkage;
		function.convention = ir::LLVMInstruction::DefaultCallingConvention;
		function.visibility = ir::LLVMStatement::Default;
		
		ir::Module::FunctionPrototypeMap::const_iterator prototype =
			_ptx->module->prototypes().find( *name );
		assert( prototype != _ptx->module->prototypes().end() );
		
		if( prototype->second.returnArguments.size() > 0 )
		{
			function.operand.type.type = _translate(
				prototype->second.returnArguments[ 0 ].type );
			function.operand.type.category = 
				ir::LLVMInstruction::Type::Element;
		}
		
		for( ir::PTXKernel::Prototype::ArgumentVector::const_iterator
			argument = prototype->second.arguments.begin();
			argument != prototype->second.arguments.end(); ++argument )
		{
			function.parameters.push_back( ir::LLVMInstruction::Operand( "", 
				ir::LLVMInstruction::Type( _translate( argument->type ), 
				ir::LLVMInstruction::Type::Element ) ) );
		}
		
		_llvmKernel->push_front( function );
	}
}

void PTXToLLVMTranslator::_addStackAllocations()
{
	if( !_usesTextures ) return;
	
	ir::LLVMBr branch;
	
	if( _uninitialized.empty() )
	{
		branch.iftrue = "%" + (++_dfg->begin())->label();
	}
	else
	{
		branch.iftrue = "%$OcelotRegisterInitializerBlock";
	}
	
	_llvmKernel->push_front( ir::LLVMStatement( branch ) );
	
	ir::LLVMAlloca allocate( 4, 16 );
	
	allocate.d.name = "%floatingPointTexture";
	allocate.d.type.category = ir::LLVMInstruction::Type::Pointer;
	allocate.d.type.type = ir::LLVMInstruction::F32;
	
	_llvmKernel->push_front( 
		ir::LLVMStatement( allocate ) );

	allocate.d.name = "%integerTexture";
	allocate.d.type.type = ir::LLVMInstruction::I32;

	_llvmKernel->push_front( ir::LLVMStatement( allocate ) );

	_llvmKernel->push_front( 
		ir::LLVMStatement( "$OcelotTextureAllocateBlock" ) );
}

void PTXToLLVMTranslator::_addTextureCalls()
{
	ir::LLVMStatement tex( ir::LLVMStatement::FunctionDeclaration );

	tex.label = "__ocelot_tex_1d_uu";
	tex.linkage = ir::LLVMStatement::InvalidLinkage;
	tex.convention = ir::LLVMInstruction::DefaultCallingConvention;
	tex.visibility = ir::LLVMStatement::Default;
	
	tex.parameters.resize( 4 );
	tex.parameters[0].type.category = ir::LLVMInstruction::Type::Pointer;
	tex.parameters[0].type.type = ir::LLVMInstruction::I32;
	
	tex.parameters[1].type.category = ir::LLVMInstruction::Type::Pointer;
	tex.parameters[1].type.members.resize(1);
	tex.parameters[1].type.members[0].category 
		= ir::LLVMInstruction::Type::Structure;
	tex.parameters[1].type.members[0].label = "%LLVMContext";
	
	tex.parameters[2].type.category = ir::LLVMInstruction::Type::Element;
	tex.parameters[2].type.type = ir::LLVMInstruction::I32;

	tex.parameters[3].type.category = ir::LLVMInstruction::Type::Element;
	tex.parameters[3].type.type = ir::LLVMInstruction::I32;
	
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_1d_us";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_1d_su";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_1d_ss";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_1d_uf";
	tex.parameters[3].type.type = ir::LLVMInstruction::F32;
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_1d_sf";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_1d_ff";
	tex.parameters[0].type.type = ir::LLVMInstruction::F32;
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_1d_fu";
	tex.parameters[3].type.type = ir::LLVMInstruction::I32;
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_1d_fs";
	_llvmKernel->push_front( tex );
	

	tex.label = "__ocelot_tex_2d_uu";
	tex.parameters[0].type.type = ir::LLVMInstruction::I32;
	
	tex.parameters.resize( 5 );
	
	tex.parameters[3].type.type = ir::LLVMInstruction::I32;

	tex.parameters[4].type.category = ir::LLVMInstruction::Type::Element;
	tex.parameters[4].type.type = ir::LLVMInstruction::I32;
	
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_2d_us";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_2d_su";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_2d_ss";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_2d_uf";
	tex.parameters[3].type.type = ir::LLVMInstruction::F32;
	tex.parameters[4].type.type = ir::LLVMInstruction::F32;
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_2d_sf";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_2d_ff";
	tex.parameters[0].type.type = ir::LLVMInstruction::F32;
	_llvmKernel->push_front( tex );

	tex.parameters.resize( 6 );
	
	tex.label = "__ocelot_tex_a2d_ff";
	tex.parameters[5].type.category = ir::LLVMInstruction::Type::Element;
	tex.parameters[5].type.type = ir::LLVMInstruction::I32;
	_llvmKernel->push_front( tex );

	tex.parameters.resize( 5 );
	tex.label = "__ocelot_tex_2d_fu";
	tex.parameters[3].type.type = ir::LLVMInstruction::I32;
	tex.parameters[4].type.type = ir::LLVMInstruction::I32;
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_2d_fs";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_3d_uu";
	tex.parameters[0].type.type = ir::LLVMInstruction::I32;
	
	tex.parameters.resize( 7 );
	
	tex.parameters[3].type.type = ir::LLVMInstruction::I32;

	tex.parameters[4].type.type = ir::LLVMInstruction::I32;

	tex.parameters[5].type.category = ir::LLVMInstruction::Type::Element;
	tex.parameters[5].type.type = ir::LLVMInstruction::I32;

	tex.parameters[6].type.category = ir::LLVMInstruction::Type::Element;
	tex.parameters[6].type.type = ir::LLVMInstruction::I32;
	
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_3d_us";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_3d_su";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_3d_ss";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_3d_uf";
	tex.parameters[3].type.type = ir::LLVMInstruction::F32;
	tex.parameters[4].type.type = ir::LLVMInstruction::F32;
	tex.parameters[5].type.type = ir::LLVMInstruction::F32;
	tex.parameters[6].type.type = ir::LLVMInstruction::F32;
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_3d_sf";
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_3d_ff";
	tex.parameters[0].type.type = ir::LLVMInstruction::F32;
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_3d_fu";
	tex.parameters[3].type.type = ir::LLVMInstruction::I32;
	tex.parameters[4].type.type = ir::LLVMInstruction::I32;
	tex.parameters[5].type.type = ir::LLVMInstruction::I32;
	tex.parameters[6].type.type = ir::LLVMInstruction::I32;
	_llvmKernel->push_front( tex );

	tex.label = "__ocelot_tex_3d_fs";
	_llvmKernel->push_front( tex );
}

void PTXToLLVMTranslator::_addSurfaceCalls()
{

}

void PTXToLLVMTranslator::_addQueryCalls()
{
	ir::LLVMStatement query( ir::LLVMStatement::FunctionDeclaration );
	query.linkage    = ir::LLVMStatement::InvalidLinkage;
	query.convention = ir::LLVMInstruction::DefaultCallingConvention;
	query.visibility = ir::LLVMStatement::Default;

	query.operand.type.category = ir::LLVMInstruction::Type::Pointer;
	query.operand.type.type = ir::LLVMInstruction::I32;
	
	query.parameters.resize( 3 );
	
	query.parameters[0].type.category = ir::LLVMInstruction::Type::Pointer;
	query.parameters[0].type.members.resize(1);
	query.parameters[0].type.members[0].category 
		= ir::LLVMInstruction::Type::Structure;
	query.parameters[0].type.members[0].label = "%LLVMContext";
	
	query.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	query.parameters[1].type.type = ir::LLVMInstruction::I32;
	
	query.parameters[2].type.category = ir::LLVMInstruction::Type::Element;
	query.parameters[2].type.type = ir::LLVMInstruction::I32;
	
	query.label = "__ocelot_txq";
	_llvmKernel->push_front( query );
}

void PTXToLLVMTranslator::_addAtomicCalls()
{
	ir::LLVMStatement atom( ir::LLVMStatement::FunctionDeclaration );

	atom.label = "__ocelot_atomic_inc_32";
	atom.linkage = ir::LLVMStatement::InvalidLinkage;
	atom.convention = ir::LLVMInstruction::DefaultCallingConvention;
	atom.visibility = ir::LLVMStatement::Default;
	
	atom.operand.type.category = ir::LLVMInstruction::Type::Element;
	atom.operand.type.type = ir::LLVMInstruction::I32;
	
	atom.parameters.resize( 2 );

	atom.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	atom.parameters[0].type.type = ir::LLVMInstruction::I64;

	atom.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	atom.parameters[1].type.type = ir::LLVMInstruction::I32;

	_llvmKernel->push_front( atom );		

	atom.label = "__ocelot_atomic_dec_32";
	
	_llvmKernel->push_front( atom );
}

void PTXToLLVMTranslator::_addMathCalls()
{
	ir::LLVMStatement mul( ir::LLVMStatement::FunctionDeclaration );

	mul.label = "__ocelot_mul_hi_u64";
	mul.linkage = ir::LLVMStatement::InvalidLinkage;
	mul.convention = ir::LLVMInstruction::DefaultCallingConvention;
	mul.visibility = ir::LLVMStatement::Default;
	
	mul.operand.type.category = ir::LLVMInstruction::Type::Element;
	mul.operand.type.type = ir::LLVMInstruction::I64;
	
	mul.parameters.resize( 2 );

	mul.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	mul.parameters[0].type.type = ir::LLVMInstruction::I64;

	mul.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	mul.parameters[1].type.type = ir::LLVMInstruction::I64;

	_llvmKernel->push_front( mul );		

	mul.label = "__ocelot_mul_hi_s64";
	
	_llvmKernel->push_front( mul );

	ir::LLVMStatement math( ir::LLVMStatement::FunctionDeclaration );

	math.label = "floor";
	math.linkage = ir::LLVMStatement::InvalidLinkage;
	math.convention = ir::LLVMInstruction::DefaultCallingConvention;
	math.visibility = ir::LLVMStatement::Default;
	
	math.operand.type.category = ir::LLVMInstruction::Type::Element;
	math.operand.type.type = ir::LLVMInstruction::F64;
	
	math.parameters.resize( 1 );

	math.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	math.parameters[0].type.type = ir::LLVMInstruction::F64;

	_llvmKernel->push_front( math );

	math.label = "ceil";
	_llvmKernel->push_front( math );

	math.label = "trunc";
	_llvmKernel->push_front( math );

	math.label = "nearbyint";
	_llvmKernel->push_front( math );

	math.operand.type.type = ir::LLVMInstruction::F32;
	math.parameters[0].type.type = ir::LLVMInstruction::F32;

	math.label = "floorf";
	_llvmKernel->push_front( math );

	math.label = "ceilf";
	_llvmKernel->push_front( math );

	math.label = "truncf";
	_llvmKernel->push_front( math );

	math.label = "nearbyintf";
	_llvmKernel->push_front( math );
}

void PTXToLLVMTranslator::_addLLVMIntrinsics()
{
	// @llvm.ctpop
	ir::LLVMStatement ctpop( ir::LLVMStatement::FunctionDeclaration );

	ctpop.label = "llvm.ctpop.i8";
	ctpop.linkage = ir::LLVMStatement::InvalidLinkage;
	ctpop.convention = ir::LLVMInstruction::DefaultCallingConvention;
	ctpop.visibility = ir::LLVMStatement::Default;
	
	ctpop.operand.type.category = ir::LLVMInstruction::Type::Element;
	ctpop.operand.type.type = ir::LLVMInstruction::I8;
	
	ctpop.parameters.resize( 1 );

	ctpop.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	ctpop.parameters[0].type.type = ir::LLVMInstruction::I8;

	_llvmKernel->push_front( ctpop );

	ctpop.label = "llvm.ctpop.i16";
	ctpop.operand.type.type = ir::LLVMInstruction::I16;
	ctpop.parameters[0].type.type = ir::LLVMInstruction::I16;
	_llvmKernel->push_front( ctpop );

	ctpop.label = "llvm.ctpop.i32";
	ctpop.operand.type.type = ir::LLVMInstruction::I32;
	ctpop.parameters[0].type.type = ir::LLVMInstruction::I32;
	_llvmKernel->push_front( ctpop );

	ctpop.label = "llvm.ctpop.i64";
	ctpop.operand.type.type = ir::LLVMInstruction::I64;
	ctpop.parameters[0].type.type = ir::LLVMInstruction::I64;
	_llvmKernel->push_front( ctpop );

	// @llvm.readcyclecounter
	ir::LLVMStatement rdtsc( ir::LLVMStatement::FunctionDeclaration );

	rdtsc.label = "llvm.readcyclecounter";
	rdtsc.linkage    = ir::LLVMStatement::InvalidLinkage;
	rdtsc.convention = ir::LLVMInstruction::DefaultCallingConvention;
	rdtsc.visibility = ir::LLVMStatement::Default;

	rdtsc.operand.type.category = ir::LLVMInstruction::Type::Element;
	rdtsc.operand.type.type = ir::LLVMInstruction::I64;

	_llvmKernel->push_front( rdtsc );

	// @llvm.ctlz
	ir::LLVMStatement ctlz( ir::LLVMStatement::FunctionDeclaration );

	ctlz.label      = "llvm.ctlz.i8";
	ctlz.linkage    = ir::LLVMStatement::InvalidLinkage;
	ctlz.convention = ir::LLVMInstruction::DefaultCallingConvention;
	ctlz.visibility = ir::LLVMStatement::Default;
	
	ctlz.operand.type.category = ir::LLVMInstruction::Type::Element;
	ctlz.operand.type.type     = ir::LLVMInstruction::I8;
	
	ctlz.parameters.resize( 1 );

	ctlz.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	ctlz.parameters[0].type.type     = ir::LLVMInstruction::I8;

	_llvmKernel->push_front( ctlz );

	ctlz.label = "llvm.ctlz.i16";
	ctlz.operand.type.type = ir::LLVMInstruction::I16;
	ctlz.parameters[0].type.type = ir::LLVMInstruction::I16;
	_llvmKernel->push_front( ctlz );

	ctlz.label = "llvm.ctlz.i32";
	ctlz.operand.type.type = ir::LLVMInstruction::I32;
	ctlz.parameters[0].type.type = ir::LLVMInstruction::I32;
	_llvmKernel->push_front( ctlz );

	ctlz.label = "llvm.ctlz.i64";
	ctlz.operand.type.type = ir::LLVMInstruction::I64;
	ctlz.parameters[0].type.type = ir::LLVMInstruction::I64;
	_llvmKernel->push_front( ctlz );
	
	// @llvm.sqrt
	ir::LLVMStatement sqrt( ir::LLVMStatement::FunctionDeclaration );

	sqrt.label      = "llvm.sqrt.f32";
	sqrt.linkage    = ir::LLVMStatement::InvalidLinkage;
	sqrt.convention = ir::LLVMInstruction::DefaultCallingConvention;
	sqrt.visibility = ir::LLVMStatement::Default;
	
	sqrt.operand.type.category = ir::LLVMInstruction::Type::Element;
	sqrt.operand.type.type     = ir::LLVMInstruction::F32;
	
	sqrt.parameters.resize( 1 );

	sqrt.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	sqrt.parameters[0].type.type     = ir::LLVMInstruction::F32;

	_llvmKernel->push_front( sqrt );

	sqrt.label                   = "llvm.sqrt.f64";
	sqrt.operand.type.type       = ir::LLVMInstruction::F64;
	sqrt.parameters[0].type.type = ir::LLVMInstruction::F64;
	
	_llvmKernel->push_front( sqrt );

	// @llvm.cos
	ir::LLVMStatement cos( ir::LLVMStatement::FunctionDeclaration );

	cos.label      = "llvm.cos.f32";
	cos.linkage    = ir::LLVMStatement::InvalidLinkage;
	cos.convention = ir::LLVMInstruction::DefaultCallingConvention;
	cos.visibility = ir::LLVMStatement::Default;
	
	cos.operand.type.category = ir::LLVMInstruction::Type::Element;
	cos.operand.type.type     = ir::LLVMInstruction::F32;
	
	cos.parameters.resize( 1 );

	cos.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	cos.parameters[0].type.type     = ir::LLVMInstruction::F32;

	_llvmKernel->push_front( cos );

	// @llvm.sin
	cos.label      = "llvm.sin.f32";
	_llvmKernel->push_front( cos );

	// @llvm.log
	cos.label      = "llvm.log.f32";
	_llvmKernel->push_front( cos );

	// @llvm.log2
	cos.label      = "llvm.log2.f32";
	_llvmKernel->push_front( cos );

	// @llvm.exp2
	cos.label      = "llvm.exp2.f32";
	_llvmKernel->push_front( cos );

	// @llvm.pow
	ir::LLVMStatement pow( ir::LLVMStatement::FunctionDeclaration );

	pow.label      = "llvm.pow.f32";
	pow.linkage    = ir::LLVMStatement::InvalidLinkage;
	pow.convention = ir::LLVMInstruction::DefaultCallingConvention;
	pow.visibility = ir::LLVMStatement::Default;
	
	pow.operand.type.category = ir::LLVMInstruction::Type::Element;
	pow.operand.type.type     = ir::LLVMInstruction::F32;
	
	pow.parameters.resize( 2 );

	pow.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	pow.parameters[0].type.type     = ir::LLVMInstruction::F32;
	pow.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	pow.parameters[1].type.type     = ir::LLVMInstruction::F32;
 
 	_llvmKernel->push_front( pow );	

}

void PTXToLLVMTranslator::_addUtilityCalls()
{
	ir::LLVMStatement extent( ir::LLVMStatement::FunctionDeclaration );

	extent.operand.type.category = ir::LLVMInstruction::Type::Element;
	extent.operand.type.type     = ir::LLVMInstruction::I32;

	extent.label = "__ocelot_get_extent";
	extent.linkage = ir::LLVMStatement::InvalidLinkage;
	extent.convention = ir::LLVMInstruction::DefaultCallingConvention;
	extent.visibility = ir::LLVMStatement::Default;
	
	extent.parameters.resize( 2 );
	
	extent.parameters[ 0 ].type.category = ir::LLVMInstruction::Type::Pointer;
	extent.parameters[ 0 ].type.members.resize( 1 );
	extent.parameters[ 0 ].type.members[ 0 ].category 
		= ir::LLVMInstruction::Type::Structure;
	extent.parameters[ 0 ].type.members[ 0 ].label = "%LLVMContext";
	
	extent.parameters[ 1 ].type.category = ir::LLVMInstruction::Type::Element;
	extent.parameters[ 1 ].type.type = ir::LLVMInstruction::I32;
	
	_llvmKernel->push_front( extent );
}

void PTXToLLVMTranslator::_addKernelPrefix()
{
	_llvmKernel->push_front( 
		ir::LLVMStatement( ir::LLVMStatement::BeginFunctionBody ) );

	ir::LLVMStatement kernel( ir::LLVMStatement::FunctionDefinition );

	kernel.label = "_Z_ocelotTranslated_" + _llvmKernel->name;
	kernel.linkage = ir::LLVMStatement::InvalidLinkage;
	kernel.convention = ir::LLVMInstruction::DefaultCallingConvention;
	kernel.visibility = ir::LLVMStatement::Default;
	kernel.functionAttributes = ir::LLVMInstruction::NoUnwind;
	
	kernel.parameters.resize( 1 );
	kernel.parameters[ 0 ].attribute = ir::LLVMInstruction::NoAlias;
	kernel.parameters[ 0 ].type.label = "%LLVMContext";
	kernel.parameters[ 0 ].type.category 
		= ir::LLVMInstruction::Type::Pointer;
	kernel.parameters[ 0 ].name = "%__ctaContext";
	
	_llvmKernel->push_front( kernel );

	ir::LLVMStatement dim3( ir::LLVMStatement::TypeDeclaration );
	
	dim3.label = "%Dimension";
	dim3.operand.type.category = ir::LLVMInstruction::Type::Structure;
	dim3.operand.type.members.resize( 3 );
	dim3.operand.type.members[ 0 ].category 
		= ir::LLVMInstruction::Type::Element;
	dim3.operand.type.members[ 0 ].type = ir::LLVMInstruction::I32;
	dim3.operand.type.members[ 1 ] = dim3.operand.type.members[ 0 ];
	dim3.operand.type.members[ 2 ] = dim3.operand.type.members[ 0 ];
	
	_llvmKernel->push_front( dim3 );

	_llvmKernel->push_front( 
		ir::LLVMStatement( ir::LLVMStatement::NewLine ) );		

	ir::LLVMStatement brev( ir::LLVMStatement::FunctionDeclaration );

	brev.label = "__ocelot_brev_b32";
	brev.linkage = ir::LLVMStatement::InvalidLinkage;
	brev.convention = ir::LLVMInstruction::DefaultCallingConvention;
	brev.visibility = ir::LLVMStatement::Default;
	
	brev.operand.type.category = ir::LLVMInstruction::Type::Element;
	brev.operand.type.type = ir::LLVMInstruction::I32;
	
	brev.parameters.resize( 1 );
	brev.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	brev.parameters[0].type.type = ir::LLVMInstruction::I32;

	_llvmKernel->push_front( brev );		
	brev.operand.type.type = ir::LLVMInstruction::I64;
	brev.parameters[0].type.type = ir::LLVMInstruction::I64;
	brev.label = "__ocelot_brev_b64";
	_llvmKernel->push_front( brev );		

    ir::LLVMStatement bfe( ir::LLVMStatement::FunctionDeclaration );

    bfe.label = "__ocelot_bfe_b32";
    bfe.linkage = ir::LLVMStatement::InvalidLinkage;
    bfe.convention = ir::LLVMInstruction::DefaultCallingConvention;
    bfe.visibility = ir::LLVMStatement::Default;
	
    bfe.operand.type.category = ir::LLVMInstruction::Type::Element;
    bfe.operand.type.type = ir::LLVMInstruction::I32;
	
    bfe.parameters.resize( 4 );
    bfe.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
    bfe.parameters[0].type.type = ir::LLVMInstruction::I32;
    bfe.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
    bfe.parameters[1].type.type = ir::LLVMInstruction::I32;
    bfe.parameters[2].type.category = ir::LLVMInstruction::Type::Element;
    bfe.parameters[2].type.type = ir::LLVMInstruction::I32;
    bfe.parameters[3].type.category = ir::LLVMInstruction::Type::Element;
    bfe.parameters[3].type.type = ir::LLVMInstruction::I1;

    _llvmKernel->push_front( bfe );		

    bfe.label = "__ocelot_bfe_b64";
    bfe.operand.type.type = ir::LLVMInstruction::I64;
    bfe.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
    bfe.parameters[0].type.type = ir::LLVMInstruction::I64;
	
    _llvmKernel->push_front( bfe );		
    
    ir::LLVMStatement bfi( ir::LLVMStatement::FunctionDeclaration );

	bfi.label = "__ocelot_bfi_b32";
	bfi.linkage = ir::LLVMStatement::InvalidLinkage;
	bfi.convention = ir::LLVMInstruction::DefaultCallingConvention;
	bfi.visibility = ir::LLVMStatement::Default;
	
	bfi.operand.type.category = ir::LLVMInstruction::Type::Element;
	bfi.operand.type.type = ir::LLVMInstruction::I32;
	
	bfi.parameters.resize( 4 );
	bfi.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	bfi.parameters[0].type.type = ir::LLVMInstruction::I32;
	bfi.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	bfi.parameters[1].type.type = ir::LLVMInstruction::I32;
	bfi.parameters[2].type.category = ir::LLVMInstruction::Type::Element;
	bfi.parameters[2].type.type = ir::LLVMInstruction::I32;
	bfi.parameters[3].type.category = ir::LLVMInstruction::Type::Element;
	bfi.parameters[3].type.type = ir::LLVMInstruction::I32;

	_llvmKernel->push_front( bfi );		
	
	bfi.label = "__ocelot_bfi_b64";
	bfi.operand.type.type = ir::LLVMInstruction::I64;
	bfi.parameters[0].type.type = ir::LLVMInstruction::I64;
	bfi.parameters[1].type.type = ir::LLVMInstruction::I64;
	_llvmKernel->push_front( bfi );
	
	ir::LLVMStatement bfind( ir::LLVMStatement::FunctionDeclaration );

	bfind.label = "__ocelot_bfind_b32";
	bfind.linkage = ir::LLVMStatement::InvalidLinkage;
	bfind.convention = ir::LLVMInstruction::DefaultCallingConvention;
	bfind.visibility = ir::LLVMStatement::Default;
	
	bfind.operand.type.category = ir::LLVMInstruction::Type::Element;
	bfind.operand.type.type = ir::LLVMInstruction::I32;
	
	bfind.parameters.resize( 2 );
	bfind.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	bfind.parameters[0].type.type = ir::LLVMInstruction::I32;
	bfind.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	bfind.parameters[1].type.type = ir::LLVMInstruction::I1;

	_llvmKernel->push_front( bfind );
	bfind.parameters[0].type.type = ir::LLVMInstruction::I64;
	bfind.label = "__ocelot_bfind_b64";
	_llvmKernel->push_front( bfind );		

	ir::LLVMStatement prmt( ir::LLVMStatement::FunctionDeclaration );

	prmt.label = "__ocelot_prmt";
	prmt.linkage = ir::LLVMStatement::InvalidLinkage;
	prmt.convention = ir::LLVMInstruction::DefaultCallingConvention;
	prmt.visibility = ir::LLVMStatement::Default;
	
	prmt.operand.type.category = ir::LLVMInstruction::Type::Element;
	prmt.operand.type.type = ir::LLVMInstruction::I32;
	
	prmt.parameters.resize( 3 );
	prmt.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	prmt.parameters[0].type.type = ir::LLVMInstruction::I32;
	prmt.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	prmt.parameters[1].type.type = ir::LLVMInstruction::I32;
	prmt.parameters[2].type.category = ir::LLVMInstruction::Type::Element;
	prmt.parameters[2].type.type = ir::LLVMInstruction::I32;

	_llvmKernel->push_front( prmt );		
	prmt.label = "__ocelot_prmt_f4e";
	_llvmKernel->push_front( prmt );		
	prmt.label = "__ocelot_prmt_b4e";
	_llvmKernel->push_front( prmt );		
	prmt.label = "__ocelot_prmt_rc8";
	_llvmKernel->push_front( prmt );		
	prmt.label = "__ocelot_prmt_ecl";
	_llvmKernel->push_front( prmt );		
	prmt.label = "__ocelot_prmt_ecr";
	_llvmKernel->push_front( prmt );		
	prmt.label = "__ocelot_prmt_rc16";
	_llvmKernel->push_front( prmt );

	ir::LLVMStatement vote( ir::LLVMStatement::FunctionDeclaration );

	vote.label = "__ocelot_vote";
	vote.linkage = ir::LLVMStatement::InvalidLinkage;
	vote.convention = ir::LLVMInstruction::DefaultCallingConvention;
	vote.visibility = ir::LLVMStatement::Default;
	
	vote.operand.type.category = ir::LLVMInstruction::Type::Element;
	vote.operand.type.type = ir::LLVMInstruction::I1;
	
	vote.parameters.resize( 3 );
	vote.parameters[0].type.category = ir::LLVMInstruction::Type::Element;
	vote.parameters[0].type.type = ir::LLVMInstruction::I1;
	vote.parameters[1].type.category = ir::LLVMInstruction::Type::Element;
	vote.parameters[1].type.type = ir::LLVMInstruction::I32;
	vote.parameters[2].type.category = ir::LLVMInstruction::Type::Element;
	vote.parameters[2].type.type = ir::LLVMInstruction::I1;

	_llvmKernel->push_front( vote );		

	_addMemoryCheckingDeclarations();
	_insertDebugSymbols();
	_addTextureCalls();
	_addSurfaceCalls();
	_addAtomicCalls();
	_addQueryCalls();
	_addMathCalls();
	_addLLVMIntrinsics();
	_addUtilityCalls();

	_llvmKernel->push_back( 
		ir::LLVMStatement( ir::LLVMStatement::NewLine ) );
	
	ir::LLVMStatement contextType( ir::LLVMStatement::TypeDeclaration );
	
	contextType.label = "%LLVMContext";
	contextType.operand.type = _getCtaContextType();
	
	_llvmKernel->push_front( contextType );

	_llvmKernel->push_front( 
		ir::LLVMStatement( ir::LLVMStatement::NewLine ) );
}

void PTXToLLVMTranslator::_addKernelSuffix()
{
	_llvmKernel->push_back( 
		ir::LLVMStatement( ir::LLVMStatement::EndFunctionBody ) );	
}
	
}

#endif

