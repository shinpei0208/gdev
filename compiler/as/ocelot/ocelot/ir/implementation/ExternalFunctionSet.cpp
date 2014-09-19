/*! \file   ExternalFunctionSet.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Wednesday April 6, 2011
	\brief  The source file for the ExternalFunctionSet class.
*/

#ifndef EXTERNAL_FUCNTION_SET_CPP_INCLUDED
#define EXTERNAL_FUNCTION_SET_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/ir/interface/ExternalFunctionSet.h>
#include <ocelot/executive/interface/LLVMState.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/LLVMKernel.h>

#include <configure.h>

// Hydrazine Includes
#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Exception.h>

// LLVM Includes
#if HAVE_LLVM
#include <llvm/Transforms/Scalar.h>
#include <llvm/PassManager.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Assembly/Parser.h>
#include <llvm/Analysis/Verifier.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#endif

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0
#define REPORT_LLVM 0

namespace ir
{

#if HAVE_LLVM
static unsigned int align(unsigned int address, unsigned int alignment)
{
	unsigned int remainder = address % alignment;
	return remainder == 0 ? address : (address + alignment - remainder);
}

static LLVMInstruction::DataType translateType(PTXOperand::DataType t)
{
	switch(t)
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
		break;
	}
	}
	return ir::LLVMInstruction::InvalidDataType;
}

static std::string getValueString(unsigned int value)
{
	std::stringstream stream;
	
	stream << "%r" << value;
	
	return stream.str();
}

static llvm::Function* jitFunction(
	const ExternalFunctionSet::ExternalFunction& f,
	const PTXKernel::Prototype& prototype, llvm::Module* m)
{
	LLVMKernel kernel;

	// Add a prototype for the external function
	LLVMStatement proto(LLVMStatement::FunctionDeclaration);

	proto.label      = f.name();
	proto.linkage    = ir::LLVMStatement::InvalidLinkage;
	proto.convention = ir::LLVMInstruction::DefaultCallingConvention;
	proto.visibility = ir::LLVMStatement::Default;
	
	if(!prototype.returnArguments.empty())
	{
		const Parameter& parameter = prototype.returnArguments[0];
		
		proto.operand.type = LLVMInstruction::Type(
			translateType(parameter.type), LLVMInstruction::Type::Element);
	}
	
	for(PTXKernel::Prototype::ArgumentVector::const_iterator
		argument = prototype.arguments.begin();
		argument != prototype.arguments.end(); ++argument)
	{
		const Parameter& parameter = *argument;

		proto.parameters.push_back(LLVMInstruction::Parameter());

		proto.parameters.back().type = LLVMInstruction::Type(
			translateType(parameter.type),
			LLVMInstruction::Type::Element);
	}
	
	kernel.push_back(proto);
	
	// Add the function
	LLVMStatement func(LLVMStatement::FunctionDefinition);

	func.label              = f.mangledName();
	func.linkage            = LLVMStatement::InvalidLinkage;
	func.convention         = LLVMInstruction::DefaultCallingConvention;
	func.visibility         = LLVMStatement::Default;
	func.functionAttributes = LLVMInstruction::NoUnwind;

	func.parameters.resize(1);

	func.parameters[0].attribute     = LLVMInstruction::NoAlias;
	func.parameters[0].type.type     = LLVMInstruction::I8;
	func.parameters[0].type.category = LLVMInstruction::Type::Pointer;
	func.parameters[0].name          = "%parameters";

	kernel.push_back(func);

	kernel.push_back(LLVMStatement(LLVMStatement::BeginFunctionBody));

	kernel.push_back(ir::LLVMStatement("entry"));

	unsigned int offset = 0;
	unsigned int value  = 0;

	ir::LLVMCall call;

	call.name = "@" + f.name();
	
	// translate the return parameters
	assert(prototype.returnArguments.size() <= 1);
	
	if(!prototype.returnArguments.empty())
	{
		const Parameter& parameter = prototype.returnArguments[0];

		LLVMInstruction::Operand operand = LLVMInstruction::Operand(
			getValueString(value++), 
			LLVMInstruction::Type(translateType(parameter.type),
			LLVMInstruction::Type::Element));
	
		call.d = operand;
		
		offset += parameter.getSize();
	}
	
	// load and translate the parameters
	for(PTXKernel::Prototype::ArgumentVector::const_iterator
		argument = prototype.arguments.begin();
		argument != prototype.arguments.end(); ++argument)
	{
		const Parameter& parameter = *argument;

		offset = align(offset, parameter.getAlignment());
		
		LLVMInstruction::Operand operand = LLVMInstruction::Operand(
			getValueString(value++), 
			LLVMInstruction::Type(translateType(parameter.type),
			LLVMInstruction::Type::Element));

		LLVMInstruction::Operand pointer = LLVMInstruction::Operand(
			operand.name + "p",
			LLVMInstruction::Type(translateType(parameter.type),
			LLVMInstruction::Type::Pointer));
	
		call.parameters.push_back(operand);
		
		LLVMGetelementptr get;
		
		get.d = LLVMInstruction::Operand(
			getValueString(value++), 
			LLVMInstruction::Type(LLVMInstruction::I8,
			LLVMInstruction::Type::Pointer));
		get.a = LLVMInstruction::Operand("%parameters",
			LLVMInstruction::Type(LLVMInstruction::I8,
			LLVMInstruction::Type::Pointer));
		
		get.indices.push_back(offset);
		
		kernel.push_back(LLVMStatement(get));
		
		LLVMBitcast bitcast;
		
		bitcast.d = pointer;
		bitcast.a = get.d;
		
		kernel.push_back(LLVMStatement(bitcast));
		
		LLVMLoad load;
		
		load.d = operand;
		load.a = pointer;
		
		kernel.push_back(LLVMStatement(load));
		
		offset += parameter.getSize();		
	}
	
	kernel.push_back(ir::LLVMStatement(call));

	// save the return operand
	if(!prototype.returnArguments.empty())
	{
		// store the operand in the parameter list
		const Parameter& parameter = prototype.returnArguments[0];

		LLVMInstruction::Operand pointer = LLVMInstruction::Operand(
			call.d.name + "p",
			LLVMInstruction::Type(translateType(parameter.type),
			LLVMInstruction::Type::Pointer));
		
		LLVMGetelementptr get;
		
		get.d = LLVMInstruction::Operand(
			getValueString(value++), 
			LLVMInstruction::Type(LLVMInstruction::I8,
			LLVMInstruction::Type::Pointer));
		get.a = LLVMInstruction::Operand("%parameters",
			LLVMInstruction::Type(LLVMInstruction::I8,
			LLVMInstruction::Type::Pointer));
		
		get.indices.push_back(0);
		
		kernel.push_back(LLVMStatement(get));
		
		LLVMBitcast bitcast;
		
		bitcast.d = pointer;
		bitcast.a = get.d;
		
		kernel.push_back(LLVMStatement(bitcast));
		
		LLVMStore store;
		
		store.a = call.d;
		store.d = pointer;
		
		kernel.push_back(LLVMStatement(store));
	}

	// return void
	kernel.push_back(LLVMStatement(LLVMRet()));

	kernel.push_back(LLVMStatement(LLVMStatement::EndFunctionBody));

	// assemble the function
	kernel.assemble();

	// parse the function
	llvm::SMDiagnostic error;
	
	m = llvm::ParseAssemblyString(kernel.code().c_str(), 
		m, error, llvm::getGlobalContext());

	reportE(REPORT_LLVM,
		"Generated the following LLVM:\n" << kernel.numberedCode());

	if(m == 0)
	{
		report("   Generating interface to external function failed, "
			"dumping code:\n" << kernel.numberedCode());
		std::string string;
		llvm::raw_string_ostream message(string);
		message << "LLVM Parser failed: ";
		error.print(kernel.name.c_str(), message);

		throw hydrazine::Exception(message.str());
	}
	
	// verify the function
	std::string verifyError;
	
	if(llvm::verifyModule(*m, llvm::ReturnStatusAction, &verifyError))
	{
		report("   Checking kernel failed, dumping code:\n" 
			<< kernel.numberedCode());
		throw hydrazine::Exception("LLVM Verifier failed for kernel: " 
			+ kernel.name + " : \"" + verifyError + "\"");
	}
	
	llvm::GlobalValue* global = m->getNamedValue(f.name());
	assertM(global != 0, "Global function " << f.name()
		<< " not found in llvm module.");
	executive::LLVMState::jit()->addGlobalMapping(global, f.functionPointer());
	
	// done, the function is now in the module
	return m->getFunction(f.mangledName());
}

#endif

ExternalFunctionSet::ExternalFunction::ExternalFunction(const std::string& i,
	void* f, llvm::Module* m)
: _name(i), _functionPointer(f), _module(m), _externalFunctionPointer(0)
{
	
}

void ExternalFunctionSet::ExternalFunction::call(void* parameters,
	const ir::PTXKernel::Prototype& p)
{
	#if HAVE_LLVM
	if(!_externalFunctionPointer)
	{	
		assert(_module);
		
		llvm::Function* function = _module->getFunction(mangledName());

		if(function == 0) function = jitFunction(*this, p, _module);

		// This invokes the jit
		_externalFunctionPointer = hydrazine::bit_cast<ExternalCallType>(
			executive::LLVMState::jit()->getPointerToFunction(function));
	}
	
	// call through the interface to the external function
	_externalFunctionPointer(parameters);
	
	#else
	assertM(false, "LLVM required to call external host functions from PTX.");
	#endif
}

const std::string& ExternalFunctionSet::ExternalFunction::name() const
{
	return _name;
}

void* ExternalFunctionSet::ExternalFunction::functionPointer() const
{
	return _functionPointer;
}

std::string ExternalFunctionSet::ExternalFunction::mangledName() const
{
	return "_Z_OcelotUnpackParametersAndCall_" + _name;
}

ExternalFunctionSet::ExternalFunctionSet()
: _module(0)
{
	#if HAVE_LLVM
	_module = new llvm::Module("_ZOcelotExternalFunctionModule",
		llvm::getGlobalContext());
	#endif
}

ExternalFunctionSet::~ExternalFunctionSet()
{
	#if HAVE_LLVM
	for(FunctionSet::const_iterator external = _functions.begin();
		external != _functions.end(); ++external)
	{
		llvm::Function* function = _module->getFunction(
			external->second.mangledName());
		if(function != 0)
		{
			executive::LLVMState::jit()->freeMachineCodeForFunction(function);
		}
	}

	executive::LLVMState::jit()->removeModule(_module);

	delete _module;
	#endif
}

void ExternalFunctionSet::add(const std::string& name, void* pointer)
{
	FunctionSet::iterator function = _functions.find(name);
	if(function != _functions.end())
	{
		ExternalFunction* ef = const_cast<ExternalFunction*>(&function->second);
		assert(ef->functionPointer() == pointer);
		return;
	}

	report("Adding function " << name << " with CPU function pointer "
		<< pointer);

	_functions.insert(std::make_pair(name, 
		ExternalFunction(name, pointer, _module)));
}

void ExternalFunctionSet::remove(const std::string& name)
{
	FunctionSet::iterator function = _functions.find(name);
	assert(function != _functions.end());
	
	report("Removing function " << name);

	#if HAVE_LLVM
	llvm::Function* llvmFunction = _module->getFunction(
		function->second.mangledName());
	if(llvmFunction != 0)
	{
		executive::LLVMState::jit()->freeMachineCodeForFunction(llvmFunction);
		llvmFunction->eraseFromParent();
		assert(_module->getFunction(function->second.mangledName()) == 0);
	
		llvm::GlobalValue* global = _module->getNamedValue(
			function->second.name());
		assertM(global != 0, "Could not find global "
			<< function->second.name() << " in module.");
		global->eraseFromParent();
	}
	#endif

	_functions.erase(function);
}

ExternalFunctionSet::ExternalFunction* ExternalFunctionSet::find(
	const std::string& name) const
{
	FunctionSet::const_iterator function = _functions.find(name);

	if(function == _functions.end()) return 0;
	
	return const_cast<ExternalFunction*>(&function->second);
}


}

#endif

