/*!	\file LLVMModuleManager.cpp
	\date Thursday September 23, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the LLVMModuleManager class
*/

#ifndef LLVM_MODULE_MANAGER_CPP_INCLUDED
#define LLVM_MODULE_MANAGER_CPP_INCLUDED

// Ocelot Includes
#include <ocelot/executive/interface/LLVMModuleManager.h>
#include <ocelot/executive/interface/LLVMState.h>
#include <ocelot/executive/interface/Device.h>

#include <ocelot/translator/interface/PTXToLLVMTranslator.h>

#include <ocelot/transforms/interface/SubkernelFormationPass.h>
#include <ocelot/transforms/interface/ConvertPredicationToSelectPass.h>
#include <ocelot/transforms/interface/RemoveBarrierPass.h>
#include <ocelot/transforms/interface/SimplifyExternalCallsPass.h>
#include <ocelot/transforms/interface/PassManager.h>

#include <ocelot/ir/interface/LLVMKernel.h>
#include <ocelot/ir/interface/Module.h>

#include <ocelot/api/interface/OcelotConfiguration.h>

#include <configure.h>

// Hydrazine Includes
#include <hydrazine/interface/Casts.h>

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

#define REPORT_ALL_LLVM_ASSEMBLY 0

namespace executive
{

////////////////////////////////////////////////////////////////////////////////
// LLVMModuleManager
void LLVMModuleManager::loadModule(const ir::Module* m, 
	translator::Translator::OptimizationLevel l, Device* d)
{
	_database.loadModule(m, l, d);
}

bool LLVMModuleManager::isModuleLoaded(const std::string& moduleName)
{
	return _database.isModuleLoaded(moduleName);
}

void LLVMModuleManager::unloadModule(const std::string& moduleName)
{
	_database.unloadModule(moduleName);
}

unsigned int LLVMModuleManager::totalFunctionCount()
{
	return _database.totalFunctionCount();
}

void LLVMModuleManager::associate(hydrazine::Thread* thread)
{
	_database.associate(thread);
}

hydrazine::Thread::Id LLVMModuleManager::id()
{
	return _database.id();
}

void LLVMModuleManager::setExternalFunctionSet(
	const ir::ExternalFunctionSet& s)
{
	_database.setExternalFunctionSet(s);
}

void LLVMModuleManager::clearExternalFunctionSet()
{
	_database.clearExternalFunctionSet();
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Helper Functions
#if HAVE_LLVM
static unsigned int pad(unsigned int& size, unsigned int alignment)
{
	unsigned int padding = alignment - (size % alignment);
	padding = (alignment == padding) ? 0 : padding;
	size += padding;
	return padding;
}

static void setupGlobalMemoryReferences(ir::PTXKernel& kernel,
	const ir::PTXKernel& parent)
{
	for(ir::ControlFlowGraph::iterator block = kernel.cfg()->begin(); 
		block != kernel.cfg()->end(); ++block)
	{
		for( ir::ControlFlowGraph::InstructionList::iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction )
		{
			ir::PTXInstruction& ptx = static_cast<
				ir::PTXInstruction&>(**instruction);

			if(ptx.mayHaveAddressableOperand()
				&& (ptx.a.addressMode == ir::PTXOperand::Address
				|| ptx.a.addressMode == ir::PTXOperand::Indirect))
			{
				ir::Module::GlobalMap::const_iterator 
					global = parent.module->globals().find(ptx.a.identifier);
				if(global == parent.module->globals().end()) continue;
				if(global->second.statement.directive
					!= ir::PTXStatement::Global) continue;
					
				ptx.addressSpace = ir::PTXInstruction::Global;				
				report("   For instruction \"" << ptx.toString() 
					<< "\" setting address space to global.");
			}
		}
	}
}

static void setupArgumentMemoryReferences(ir::PTXKernel& kernel,
	LLVMModuleManager::KernelAndTranslation::MetaData* metadata,
	const ir::PTXKernel& parent)
{
	typedef std::unordered_map<std::string, unsigned int> OffsetMap;
	report("  Setting up argument memory references.");

	unsigned int offset = 0;
	
	OffsetMap offsets;
	
	for(ir::Kernel::ParameterVector::const_iterator
		argument = parent.arguments.begin();
		argument != parent.arguments.end(); ++argument)
	{
		pad(offset, argument->getAlignment());
		offsets.insert(std::make_pair(argument->name, offset));
		report("   Argument " << argument->name << ", offset " << offset);
		offset += argument->getSize();
	}

	for(ir::ControlFlowGraph::iterator block = kernel.cfg()->begin(); 
		block != kernel.cfg()->end(); ++block)
	{
		for( ir::ControlFlowGraph::InstructionList::iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction )
		{
			ir::PTXInstruction& ptx = static_cast<
				ir::PTXInstruction&>(**instruction);

			ir::PTXOperand* operands[] = {&ptx.d, &ptx.a, &ptx.b, &ptx.c};

			if(ptx.mayHaveAddressableOperand())
			{
				for(unsigned int i = 0; i != 4; ++i)
				{
					if(operands[i]->addressMode == ir::PTXOperand::Address)
					{
						OffsetMap::iterator argument = offsets.find( 
							operands[i]->identifier);
						if(argument != offsets.end())
						{
							report("   For instruction \"" << ptx.toString() 
								<< "\" mapping \"" << argument->first 
								<< "\" to "
								<< (operands[i]->offset + argument->second));
							operands[i]->offset     += argument->second;
							operands[i]->isArgument = true;
						}
					}
				}
			}
		}
	}
	
	metadata->argumentSize = offset;
	
	report("   total argument memory size is " << metadata->argumentSize);
}

static void setupParameterMemoryReferences(ir::PTXKernel& kernel,
	LLVMModuleManager::KernelAndTranslation::MetaData* metadata,
	const ir::PTXKernel& parent, const ir::ExternalFunctionSet& externals)
{
	typedef std::unordered_map<std::string, unsigned int> OffsetMap;
	report("  Setting up parameter memory references.");

	metadata->parameterSize	= 0;
	
	OffsetMap offsets;
	
	// Determine the order that parameters are passed as arguments to calls
	for(ir::ControlFlowGraph::iterator block = kernel.cfg()->begin(); 
		block != kernel.cfg()->end(); ++block)
	{
		for(ir::ControlFlowGraph::InstructionList::iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<
				ir::PTXInstruction&>(**instruction);
			if(ptx.opcode != ir::PTXInstruction::Call) continue;
			if(&externals != 0)
			{
				if(externals.find(ptx.a.identifier) != 0)  continue;
			}
					
			unsigned int offset = 0;
			
			report("   For arguments of call instruction '"
				<< ptx.toString() << "'");
			for (ir::PTXOperand::Array::const_iterator 
				argument = ptx.d.array.begin();
				argument != ptx.d.array.end(); ++argument) 
			{
				pad(offset, ir::PTXOperand::bytes(argument->type));
				assert(offsets.count(argument->identifier) == 0);
				offsets.insert(std::make_pair(argument->identifier, offset));
				report("    mapping '" << argument->identifier
					<< "' to " << offset);
				offset += ir::PTXOperand::bytes(argument->type);
			}
			
			for (ir::PTXOperand::Array::const_iterator 
				argument = ptx.b.array.begin();
				argument != ptx.b.array.end(); ++argument) 
			{
				pad(offset, ir::PTXOperand::bytes(argument->type));
				assert(offsets.count(argument->identifier) == 0);
				offsets.insert(std::make_pair(argument->identifier, offset));
				report("    mapping '" << argument->identifier
					<< "' to " << offset);
				offset += ir::PTXOperand::bytes(argument->type);
			}
			
			metadata->parameterSize = std::max(offset, metadata->parameterSize);
		}
	}
	
	for(ir::ControlFlowGraph::iterator block = kernel.cfg()->begin(); 
		block != kernel.cfg()->end(); ++block)
	{
		for( ir::ControlFlowGraph::InstructionList::iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction )
		{
			ir::PTXInstruction& ptx = static_cast<
				ir::PTXInstruction&>(**instruction);

			ir::PTXOperand* operands[] = {&ptx.d, &ptx.a, &ptx.b, &ptx.c};

			if(ptx.mayHaveAddressableOperand())
			{
				for(unsigned int i = 0; i != 4; ++i)
				{
					if(operands[i]->addressMode == ir::PTXOperand::Address)
					{
						OffsetMap::iterator parameter = offsets.find( 
							operands[i]->identifier);
						if(parameter != offsets.end())
						{
							report("   For instruction \"" 
							<< ptx.toString() << "\" mapping \"" 
							<< parameter->first << "\" to " 
							<< (parameter->second + operands[i]->offset));
							operands[i]->offset += parameter->second;
							operands[i]->isArgument = false;
						}
					}
				}
			}
		}
	}
	
	// In order to handle tail calls resuing the current stack frame, allocate 
	//  enough space for the max number of parameters in the module
	for(ir::Module::KernelMap::const_iterator
		function = kernel.module->kernels().begin();
		function != kernel.module->kernels().end(); ++function)
	{
		if(!function->second->function()) continue;
		
		unsigned int bytes = 0;
		
		for(ir::Kernel::ParameterVector::const_iterator
			argument = function->second->arguments.begin();
			argument != function->second->arguments.end(); ++argument)
		{
			pad(bytes, argument->getSize());
			bytes += argument->getSize();
		}
		
		metadata->parameterSize = std::max(bytes, metadata->parameterSize);
	}
	
	report("   total parameter memory size is " << metadata->parameterSize);
}

static void setupSharedMemoryReferences(ir::PTXKernel& kernel,
	LLVMModuleManager::KernelAndTranslation::MetaData* metadata,
	const ir::PTXKernel& parent)
{
	typedef std::unordered_map<std::string, unsigned int> OffsetMap;
	typedef std::unordered_set<std::string> StringSet;
	typedef std::deque<ir::PTXOperand*> OperandVector;
	typedef std::unordered_map<std::string, 
		ir::Module::GlobalMap::const_iterator> GlobalMap;

	report( "  Setting up shared memory references." );

	OffsetMap offsets;
	StringSet external;
	OperandVector externalOperands;

	unsigned int externalAlignment = 1;             
	metadata->sharedSize = 0;

	for(ir::Module::GlobalMap::const_iterator 
		global = kernel.module->globals().begin(); 
		global != kernel.module->globals().end(); ++global) 
	{
		if(global->second.statement.directive == ir::PTXStatement::Shared) 
		{
			if(global->second.statement.attribute == ir::PTXStatement::Extern)
			{
				report("   Allocating global external shared variable " 
					<< global->second.statement.name);
				assertM(external.count(global->second.statement.name) == 0, 
					"External global " << global->second.statement.name 
					<< " more than once.");
				external.insert(global->second.statement.name);
				externalAlignment = std::max(externalAlignment, 
					(unsigned) global->second.statement.alignment);
				externalAlignment = std::max(externalAlignment, 
					ir::PTXOperand::bytes(global->second.statement.type));
			}
			else 
			{
				report("   Allocating global shared variable " 
					<< global->second.statement.name);
				pad(metadata->sharedSize, global->second.statement.alignment);
				offsets.insert(std::make_pair(global->second.statement.name, 
					metadata->sharedSize));
				metadata->sharedSize += global->second.statement.bytes();
			}
		}
	}

	ir::Kernel::LocalMap::const_iterator local = parent.locals.begin();
	for( ; local != parent.locals.end(); ++local)
	{
		if(local->second.space == ir::PTXInstruction::Shared)
		{
			if(local->second.attribute == ir::PTXStatement::Extern)
			{
				report("    Found local external shared variable " 
					<< local->second.name);
				assert( external.count(local->second.name) == 0 );
				external.insert( local->second.name);
				externalAlignment = std::max(externalAlignment, 
					(unsigned) local->second.alignment);
				externalAlignment = std::max(externalAlignment, 
				ir::PTXOperand::bytes(local->second.type));
			}
			else
			{
				report("   Found local shared variable " 
					<< local->second.name << " of size " 
					<< local->second.getSize());
				pad(metadata->sharedSize, local->second.alignment);
				offsets.insert(std::make_pair(local->second.name, 
					metadata->sharedSize));
				metadata->sharedSize += local->second.getSize();
			}
		}
	}
                
	for(ir::ControlFlowGraph::iterator block = kernel.cfg()->begin(); 
		block != kernel.cfg()->end(); ++block)
	{
		for(ir::ControlFlowGraph::InstructionList::iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<
				ir::PTXInstruction&>(**instruction);

			ir::PTXOperand* operands[] = {&ptx.d, &ptx.a, &ptx.b, &ptx.c};

			if(ptx.mayHaveAddressableOperand())
			{
				for(unsigned int i = 0; i != 4; ++i)
				{
					if(operands[i]->addressMode == ir::PTXOperand::Address)
					{
						StringSet::iterator si = external.find(
							operands[i]->identifier);
						if(si != external.end()) 
						{
							report("   For instruction \"" 
								<< ptx.toString() 
								<< "\", mapping shared label \"" << *si 
								<< "\" to external shared memory.");
							externalOperands.push_back(operands[i]);
							continue;
						}
	
						OffsetMap::iterator offset = offsets.find(
							operands[i]->identifier);
						if(offsets.end() != offset) 
						{
							ptx.addressSpace = ir::PTXInstruction::Shared;
							operands[i]->offset += offset->second;
							report("   For instruction " 
								<< ptx.toString() << ", mapping shared label " 
								<< offset->first << " to " << offset->second);
						}
					}
				}
			}
		}
	}

	pad(metadata->sharedSize, externalAlignment);

	report("   Mapping external shared variables.");
	for( OperandVector::iterator operand = externalOperands.begin(); 
		operand != externalOperands.end(); ++operand) 
	{
		report("    Mapping external shared label " 
			<< (*operand)->identifier << " to " << metadata->sharedSize);
		(*operand)->offset += metadata->sharedSize;
	}

	report( "   Total shared memory size is " << metadata->sharedSize << "." );
}

static void setupConstantMemoryReferences(ir::PTXKernel& kernel,
	LLVMModuleManager::KernelAndTranslation::MetaData* metadata,
	const ir::PTXKernel& parent)
{
	report( "  Setting up constant memory references." );
	typedef std::unordered_map<std::string, unsigned int> OffsetMap;

	metadata->constantSize = 0;
	OffsetMap constants;
	
	for(ir::Module::GlobalMap::const_iterator 
		global = parent.module->globals().begin(); 
		global != parent.module->globals().end(); ++global) 
	{
		if(global->second.statement.directive == ir::PTXStatement::Const) 
		{
			report( "   Found global constant variable " 
				<< global->second.statement.name << " of size " 
				<< global->second.statement.bytes() );
			pad(metadata->constantSize, global->second.statement.alignment);
			constants.insert(std::make_pair(global->second.statement.name,
				metadata->constantSize));
			metadata->constantSize += global->second.statement.bytes();
		}
	}

	report("   Total constant memory size is " << metadata->constantSize);

	for(ir::ControlFlowGraph::iterator block = kernel.cfg()->begin(); 
		block != kernel.cfg()->end(); ++block)
	{
		for(ir::ControlFlowGraph::InstructionList::iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<
				ir::PTXInstruction&>(**instruction);
			ir::PTXOperand* operands[] = {&ptx.d, &ptx.a, &ptx.b, &ptx.c};

			if(ptx.mayHaveAddressableOperand())
			{
				for(unsigned int i = 0; i != 4; ++i)
				{
					if(operands[i]->addressMode == ir::PTXOperand::Address)
					{
						OffsetMap::iterator mapping = constants.find( 
							operands[i]->identifier);
						if(constants.end() != mapping) 
						{
							ptx.addressSpace = ir::PTXInstruction::Const;
							operands[i]->offset += mapping->second;
							report("   For instruction " 
								<< ptx.toString() 
								<< ", mapping constant label " << mapping->first 
								<< " to " << mapping->second);
						}
					}
				}
			}
		}
	}
}

static void setupTextureMemoryReferences(ir::PTXKernel& kernel,
	LLVMModuleManager::KernelAndTranslation::MetaData* metadata,
	const ir::PTXKernel& parent, executive::Device* device)
{
	typedef std::unordered_map<std::string, unsigned int> TextureMap;
	report(" Setting up texture memory references.");
	
	TextureMap textures;
	
	for(ir::ControlFlowGraph::iterator block = kernel.cfg()->begin(); 
		block != kernel.cfg()->end(); ++block)
	{
		for(ir::ControlFlowGraph::InstructionList::iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<
				ir::PTXInstruction&>(**instruction);
			if(ptx.opcode == ir::PTXInstruction::Tex)
			{
				report("  found texture instruction: " << ptx.toString());

				TextureMap::iterator reference =
					textures.find(ptx.a.identifier);
				if(reference != textures.end())
				{
					ptx.a.reg = reference->second;
				}
				else
				{
					ptx.a.reg = textures.size();
					textures.insert(std::make_pair(
						ptx.a.identifier, textures.size()));
						
					ir::Texture* texture = (ir::Texture*)
						device->getTextureReference(
						kernel.module->path(), ptx.a.identifier);
					assert(texture != 0);
					
					metadata->textures.push_back(texture);
				}
			}
		}
	}
}

static void setupLocalMemoryReferences(ir::PTXKernel& kernel,
	LLVMModuleManager::KernelAndTranslation::MetaData* metadata,
	const ir::PTXKernel& parent)
{
	report( "  Setting up local memory references." );
	typedef std::unordered_map<std::string, unsigned int> OffsetMap;

	OffsetMap offsets;
	
	// Reserve the first few 32-bit words
	// [0] == subkernel-id
	// [1] == call type
	// [2] == barrier resume point if it exists
	// [3] == resume point if it exists
	metadata->localSize = 16;
	
	// give preference to barrier resume point
	ir::Kernel::LocalMap::const_iterator local = kernel.locals.find(
		"_Zocelot_barrier_next_kernel");
	if(local != kernel.locals.end())
	{
		if(local->second.space == ir::PTXInstruction::Local)
		{
			report("   Found local local variable " 
				<< local->second.name << " of size " 
				<< local->second.getSize());
			
			offsets.insert(std::make_pair(local->second.name, 8));
		}
	}

	// give preference to resume point
	local = kernel.locals.find("_Zocelot_resume_point");
	if(local != kernel.locals.end())
	{
		if(local->second.space == ir::PTXInstruction::Local)
		{
			report("   Found local local variable " 
				<< local->second.name << " of size " 
				<< local->second.getSize());
			
			offsets.insert(std::make_pair(local->second.name, 12));
		}
	}

	for(ir::Kernel::LocalMap::const_iterator local = kernel.locals.begin(); 
		local != kernel.locals.end(); ++local)
	{
		if(local->first == "_Zocelot_barrier_next_kernel") continue;
		if(local->first == "_Zocelot_spill_area")          continue;
		if(local->first == "_Zocelot_resume_point")        continue;
		
		if(local->second.space == ir::PTXInstruction::Local)
		{
			report("   Found local local variable " 
				<< local->second.name << " of size " 
				<< local->second.getSize());
			
			pad(metadata->localSize, local->second.alignment);
			offsets.insert(std::make_pair(local->second.name,
				metadata->localSize));
			metadata->localSize += local->second.getSize();
		}
	}

	// defer the spill area
	local = kernel.locals.find("_Zocelot_spill_area");
	if(local != kernel.locals.end())
	{
		if(local->second.space == ir::PTXInstruction::Local)
		{
			report("   Found local local variable " 
				<< local->second.name << " of size " 
				<< local->second.getSize());
			
			pad(metadata->localSize, local->second.alignment);
			offsets.insert(std::make_pair(local->second.name,
				metadata->localSize));
			metadata->localSize += local->second.getSize();
		}
	}
    
	for(ir::ControlFlowGraph::iterator block = kernel.cfg()->begin(); 
		block != kernel.cfg()->end(); ++block)
	{
		for(ir::ControlFlowGraph::InstructionList::iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<
				ir::PTXInstruction&>(**instruction);
			ir::PTXOperand* operands[] = {&ptx.d, &ptx.a, &ptx.b, &ptx.c};
	
			if(ptx.mayHaveAddressableOperand())
			{
				for(unsigned int i = 0; i != 4; ++i)
				{
					if(operands[i]->addressMode == ir::PTXOperand::Address)
					{
						OffsetMap::iterator offset = offsets.find( 
							operands[i]->identifier);
						if(offsets.end() != offset) 
						{
							ptx.addressSpace = ir::PTXInstruction::Local;
							operands[i]->isGlobalLocal = false;
							operands[i]->offset += offset->second;
							report("   For instruction " 
								<< ptx.toString() << ", mapping local label " 
								<< offset->first << " to " << offset->second);
						}
					}
				}
			}
		}
	}

    report("   Total local memory size is " << metadata->localSize << ".");
}

static void setupGlobalLocalMemoryReferences(ir::PTXKernel& kernel,
	LLVMModuleManager::KernelAndTranslation::MetaData* metadata,
	const ir::PTXKernel& parent)
{
	report( "  Setting up globally scoped local memory references." );
	typedef std::unordered_map<std::string, unsigned int> OffsetMap;

	OffsetMap offsets;
	metadata->globalLocalSize = 0;
	
	for(ir::Module::GlobalMap::const_iterator global =
		kernel.module->globals().begin();
		global != kernel.module->globals().end(); ++global)
	{
		if(global->second.statement.directive == ir::PTXStatement::Local)
		{
			report("   Found globally scoped local variable " 
				<< global->second.statement.name << " of size " 
				<< global->second.statement.bytes());
			
			pad(metadata->globalLocalSize,
				global->second.statement.accessAlignment());
			offsets.insert(std::make_pair(global->second.statement.name,
				metadata->globalLocalSize));
			metadata->globalLocalSize += global->second.statement.bytes();
		}
	}
    
	for(ir::ControlFlowGraph::iterator block = kernel.cfg()->begin(); 
		block != kernel.cfg()->end(); ++block)
	{
		for(ir::ControlFlowGraph::InstructionList::iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<
				ir::PTXInstruction&>(**instruction);
			ir::PTXOperand* operands[] = {&ptx.d, &ptx.a, &ptx.b, &ptx.c};
	
			if(ptx.mayHaveAddressableOperand())
			{
				for(unsigned int i = 0; i != 4; ++i)
				{
					if(operands[i]->addressMode == ir::PTXOperand::Address)
					{
						OffsetMap::iterator offset = offsets.find( 
							operands[i]->identifier);
						if(offsets.end() != offset) 
						{
							ptx.addressSpace = ir::PTXInstruction::Local;
							operands[i]->isGlobalLocal = true;
							operands[i]->offset += offset->second;
							report("   For instruction " 
								<< ptx.toString()
								<< ", mapping globally scoped local label " 
								<< offset->first << " to " << offset->second);
						}
					}
				}
			}
		}
	}

    report("   Total globally scoped local memory size is "
    	<< metadata->globalLocalSize << ".");
}

static void setupPTXMemoryReferences(ir::PTXKernel& kernel,
	LLVMModuleManager::KernelAndTranslation::MetaData* metadata,
	const ir::PTXKernel& parent, executive::Device* device,
	const ir::ExternalFunctionSet& externals)
{
	report(" Setting up memory references for kernel variables.");
	
	setupGlobalMemoryReferences(kernel, parent);
	setupArgumentMemoryReferences(kernel, metadata, parent);
	setupParameterMemoryReferences(kernel, metadata, parent, externals);
	setupSharedMemoryReferences(kernel, metadata, parent);
	setupConstantMemoryReferences(kernel, metadata, parent);
	setupTextureMemoryReferences(kernel, metadata, parent, device);
	setupLocalMemoryReferences(kernel, metadata, parent);
	setupGlobalLocalMemoryReferences(kernel, metadata, parent);
}

static unsigned int optimizePTX(ir::PTXKernel& kernel,
	translator::Translator::OptimizationLevel optimization,
	LLVMModuleManager::FunctionId id, const ir::ExternalFunctionSet& externals)
{
	report(" Optimizing PTX");
	transforms::PassManager manager(const_cast<ir::Module*>(kernel.module));

	transforms::SimplifyExternalCallsPass simplifyExternals(externals);

	if(&externals != 0)
	{
		report("  Adding simplify externals pass");
		manager.addPass(&simplifyExternals);
	}
	
	transforms::ConvertPredicationToSelectPass convertPredicationToSelect;
	transforms::RemoveBarrierPass              removeBarriers(id, &externals);
	
	report("  Adding convert predication to select pass");
	manager.addPass(&convertPredicationToSelect);

	report("  Adding remove barriers pass");
	manager.addPass(&removeBarriers);
	
	manager.runOnKernel(kernel);

	manager.releasePasses();

	return removeBarriers.usesBarriers;
}

static void setupCallTargets(ir::PTXKernel& kernel,
	const LLVMModuleManager::ModuleDatabase& database)
{
	// replace all call instruction operands with kernel id
	report("  Setting up targets of call instructions.");
	for(ir::ControlFlowGraph::iterator block = kernel.cfg()->begin(); 
		block != kernel.cfg()->end(); ++block)
	{
		for(ir::ControlFlowGraph::InstructionList::iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<
				ir::PTXInstruction&>(**instruction);
			if(ptx.opcode != ir::PTXInstruction::Call 
				&& ptx.opcode != ir::PTXInstruction::Mov) continue;

			if(ptx.opcode == ir::PTXInstruction::Call)
			{
				if(ptx.tailCall) continue;
				
				const ir::ExternalFunctionSet& externals =
					database.getExternalFunctionSet();

				if(&externals != 0)
				{
					if(externals.find(ptx.a.identifier) != 0) continue;
				}
			}
			
			if(ptx.a.addressMode == ir::PTXOperand::FunctionName)
			{
				LLVMModuleManager::FunctionId id = database.getFunctionId(
					kernel.module->path(), ptx.a.identifier);
				report("   setting target '" << ptx.a.identifier 
					<< "' of instruction '" << ptx.toString()
					<< "' to id " << id);
				ptx.reentryPoint = id;
			}
		}
	}
}

static void translate(llvm::Module*& module, ir::PTXKernel& kernel,
	translator::Translator::OptimizationLevel optimization,
	const ir::ExternalFunctionSet& externals)
{
	assert(module == 0);

	report(" Translating kernel.");
	
	report("  Converting from PTX IR to LLVM IR.");
	translator::PTXToLLVMTranslator translator(optimization, &externals);

	transforms::PassManager manager(const_cast<ir::Module*>(kernel.module));
	
	manager.addPass(&translator);
	manager.runOnKernel(kernel);
	manager.releasePasses();

	ir::LLVMKernel* llvmKernel = static_cast<ir::LLVMKernel*>(
		translator.translatedKernel());
	
	report("  Assembling LLVM kernel.");
	llvmKernel->assemble();
	llvm::SMDiagnostic error;

	module = new llvm::Module(kernel.name.c_str(), llvm::getGlobalContext());

	reportE(REPORT_ALL_LLVM_ASSEMBLY, llvmKernel->code());

	report("  Parsing LLVM assembly.");
	module = llvm::ParseAssemblyString(llvmKernel->code().c_str(), 
		module, error, llvm::getGlobalContext());

	if(module == 0)
	{
		report("   Parsing kernel failed, dumping code:\n" 
			<< llvmKernel->numberedCode());
		std::string m;
		llvm::raw_string_ostream message(m);
		message << "LLVM Parser failed: ";
		error.print(kernel.name.c_str(), message);

		throw hydrazine::Exception(message.str());
	}

	report("  Checking llvm module for errors.");
	std::string verifyError;
	
	if(llvm::verifyModule(*module, llvm::ReturnStatusAction, &verifyError))
	{
		report("   Checking kernel failed, dumping code:\n" 
			<< llvmKernel->numberedCode());
		delete llvmKernel;
		delete module;
		module = 0;

		throw hydrazine::Exception("LLVM Verifier failed for kernel: " 
			+ kernel.name + " : \"" + verifyError + "\"");
	}

	delete llvmKernel;
}

static LLVMModuleManager::KernelAndTranslation::MetaData* generateMetadata(
	ir::PTXKernel& kernel, translator::Translator::OptimizationLevel level)
{
	LLVMModuleManager::KernelAndTranslation::MetaData* 
		metadata = new LLVMModuleManager::KernelAndTranslation::MetaData;
	report(" Building metadata.");
	
	if(level == translator::Translator::DebugOptimization
		|| level == translator::Translator::ReportOptimization)
	{
		report("  Adding debugging symbols");
		ir::ControlFlowGraph::BasicBlock::Id id = 0;
		
		for(ir::ControlFlowGraph::iterator block = kernel.cfg()->begin();
			block != kernel.cfg()->end(); ++block)
		{
			block->id = id++;
			metadata->blocks.insert(std::make_pair(block->id, block));
		}
	}
	
	metadata->kernel = &kernel;
	metadata->warpSize = 1;
	
	return metadata;
}

static void optimize(llvm::Module& module,
	translator::Translator::OptimizationLevel optimization)
{
	report(" Optimizing kernel at level " 
		<< translator::Translator::toString(optimization));

    unsigned int level = 0;
    bool space         = false;

	if(optimization == translator::Translator::BasicOptimization)
	{
		level = 1;
	}
	else if(optimization == translator::Translator::AggressiveOptimization)
	{
		level = 2;
	}
	else if(optimization == translator::Translator::SpaceOptimization)
	{
		level = 2;
		space = true;
	}
	else if(optimization == translator::Translator::FullOptimization)
	{
		level = 3;
	}

	if(level == 0) return;

	llvm::PassManager manager;

	if(level < 2)
	{
		manager.add(llvm::createInstructionCombiningPass());
		manager.add(llvm::createReassociatePass());
		manager.add(llvm::createGVNPass());
		manager.add(llvm::createCFGSimplificationPass());
	}
	else
	{
//		manager.add(llvm::createSimplifyLibCallsPass());
		manager.add(llvm::createInstructionCombiningPass());
		manager.add(llvm::createJumpThreadingPass());
		manager.add(llvm::createCFGSimplificationPass());
		manager.add(llvm::createScalarReplAggregatesPass());
		manager.add(llvm::createInstructionCombiningPass());
		manager.add(llvm::createTailCallEliminationPass());
		manager.add(llvm::createCFGSimplificationPass());
		manager.add(llvm::createReassociatePass());
		manager.add(llvm::createLoopRotatePass());
		manager.add(llvm::createLICMPass());
		manager.add(llvm::createLoopUnswitchPass(space || level < 3));
		manager.add(llvm::createInstructionCombiningPass());
		manager.add(llvm::createIndVarSimplifyPass());
		manager.add(llvm::createLoopDeletionPass());
		if( level > 2 )
		{
			manager.add(llvm::createLoopUnrollPass());
		}
		manager.add(llvm::createInstructionCombiningPass());
		manager.add(llvm::createGVNPass());
		manager.add(llvm::createMemCpyOptPass());
		manager.add(llvm::createSCCPPass());

		// Run instcombine after redundancy elimination to exploit opportunities
		// opened up by them.
		manager.add(llvm::createInstructionCombiningPass());
		manager.add(llvm::createDeadStoreEliminationPass());
		manager.add(llvm::createAggressiveDCEPass());
		manager.add(llvm::createCFGSimplificationPass());
	}
	manager.run(module);
}


static void link(llvm::Module& module, const ir::PTXKernel& kernel, 
	Device* device, const ir::ExternalFunctionSet& externals,
	const LLVMModuleManager::ModuleDatabase& database)
{
	// Add global variables
	report("  Linking global variables.");
	
	for(ir::Module::GlobalMap::const_iterator 
		global = kernel.module->globals().begin(); 
		global != kernel.module->globals().end(); ++global) 
	{
		if(global->second.statement.directive == ir::PTXStatement::Global) 
		{
			assert(device != 0);

			llvm::GlobalValue* value = module.getNamedValue(global->first);
			assertM(value != 0, "Global variable " << global->first 
				<< " not found in llvm module.");
			Device::MemoryAllocation* allocation = device->getGlobalAllocation( 
				kernel.module->path(), global->first);
			assert(allocation != 0);
			report("   Binding global variable " << global->first 
				<< " to " << allocation->pointer());
			LLVMState::jit()->addGlobalMapping(value, allocation->pointer());
		}
	}
	
	// Add global references to function entry points
	report("  Linking global references to function entry points.");
	for(ir::Module::GlobalMap::const_iterator 
		global = kernel.module->globals().begin(); 
		global != kernel.module->globals().end(); ++global) 
	{
		for(ir::PTXStatement::SymbolVector::const_iterator symbol =
			global->second.statement.array.symbols.begin(); symbol !=
			global->second.statement.array.symbols.end(); ++symbol)
		{
			assert(device != 0);
			
			size_t size = ir::PTXOperand::bytes(global->second.statement.type);
			size_t offset = symbol->offset * size;
			
			Device::MemoryAllocation* allocation = device->getGlobalAllocation( 
				kernel.module->path(), global->first);
			assert(allocation != 0);
			report("   Adding symbol " << symbol->name 
				<< " to global " << global->first << " at byte-offset "
				<< offset);
			
			LLVMModuleManager::FunctionId id = database.getFunctionId(
				kernel.module->path(), symbol->name);
			
			std::memcpy((char*)allocation->pointer() + offset, &id, size);
		}
	}
	
	// Add externals
	report("  Linking global pointers to external (host) functions.");
	if(&externals == 0) return;
	
	for(ir::Module::FunctionPrototypeMap::const_iterator
		prototype = kernel.module->prototypes().begin();
		prototype != kernel.module->prototypes().end(); ++prototype)
	{
		ir::ExternalFunctionSet::ExternalFunction* external = externals.find(
			prototype->second.identifier);
	
		if(external != 0)
		{
			// Would you ever want to call into address 0?
			assert(external->functionPointer() != 0);
			
			llvm::GlobalValue* value = module.getNamedValue(external->name());
			assertM(value != 0, "Global function " << external->name() 
				<< " not found in llvm module.");
			report("   Binding global variable " << external->name() 
				<< " to " << external->functionPointer());
			LLVMState::jit()->addGlobalMapping(value,
				external->functionPointer());
		}
	}
}

static void codegen(LLVMModuleManager::Function& function, llvm::Module& module,
	const ir::PTXKernel& kernel, Device* device,
	const ir::ExternalFunctionSet& externals,
	const LLVMModuleManager::ModuleDatabase& database)
{
	report(" Generating native code.");
	
	LLVMState::jit()->addModule(&module);

	link(module, kernel, device, externals, database);

	report("  Invoking LLVM to Native JIT");

	std::string name = "_Z_ocelotTranslated_" + kernel.name;
	
	llvm::Function* llvmFunction = module.getFunction(name);
	
	assertM(llvmFunction != 0, "Could not find function " + name);
	function = hydrazine::bit_cast<LLVMModuleManager::Function>(
		LLVMState::jit()->getPointerToFunction(llvmFunction));
}
#endif

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// KernelAndTranslation
LLVMModuleManager::KernelAndTranslation::KernelAndTranslation(ir::PTXKernel* k, 
	translator::Translator::OptimizationLevel l, const ir::PTXKernel* p, 
	FunctionId o, unsigned int s, Device* d, const ModuleDatabase* m)
	: _kernel(k), _module(0), _optimizationLevel(l), _metadata(0), _parent(p),
	_offsetId(o), _subkernels(s), _device(d), _database(m)
{

}

void LLVMModuleManager::KernelAndTranslation::unload()
{
	#if HAVE_LLVM
	if(_metadata == 0)
	{
		delete _kernel;
		return;
	}
	assert(_module != 0);

	llvm::Function* function = _module->getFunction(_kernel->name);

	LLVMState::jit()->freeMachineCodeForFunction(function);

	LLVMState::jit()->removeModule(_module);
	delete _kernel;
	delete _module;
	delete _metadata;
	#else
	// Is it possible this is called when LLVMModuleManager 
	// is being destructed even with no LLVM device present?
	// assertM(false, "LLVM support not compiled into ocelot. 
	//         You should use a different device.");
	#endif
}

LLVMModuleManager::KernelAndTranslation::MetaData*
	LLVMModuleManager::KernelAndTranslation::metadata()
{
	#if HAVE_LLVM
	report("Getting metadata for kernel '" << _kernel->name << "'");

	if(_metadata != 0) return _metadata;
	
	report("Translating PTX");
	
	unsigned int barriers = optimizePTX(*_kernel,
		_optimizationLevel, _offsetId, _database->getExternalFunctionSet());
	
	try
	{
		_metadata = generateMetadata(*_kernel, _optimizationLevel);
		
		_metadata->subkernels = barriers + _subkernels;
		
		setupPTXMemoryReferences(*_kernel, _metadata, *_parent, _device,
			_database->getExternalFunctionSet());
		setupCallTargets(*_kernel, *_database);
		translate(_module, *_kernel, _optimizationLevel,
			_database->getExternalFunctionSet());
	}
	catch(...)
	{
		delete _metadata;
		_metadata = 0;
		throw;
	}
	
	try
	{
		optimize(*_module, _optimizationLevel);
		codegen(_metadata->function, *_module, *_kernel, _device,
			_database->getExternalFunctionSet(), *_database);
	}
	catch(...)
	{
		llvm::Function* function = _module->getFunction(_kernel->name);

		LLVMState::jit()->freeMachineCodeForFunction(function);

		LLVMState::jit()->removeModule(_module);
		delete _module;
		delete _metadata;
		_metadata = 0;
		
		throw;
	}

	return _metadata;
	#else
	assertM(false, "LLVM support not compiled into ocelot.");
	return 0;
	#endif
}

const std::string& LLVMModuleManager::KernelAndTranslation::name() const
{
	return _kernel->name;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Module
LLVMModuleManager::Module::Module(const KernelVector& kernels,
	FunctionId nextFunctionId, ir::Module* m)
: _originalModule(m)
{	
	for(KernelVector::const_iterator kernel = kernels.begin();
		kernel != kernels.end(); ++kernel)
	{
		_ids.insert(std::make_pair(kernel->name(), nextFunctionId++));
	}
}

void LLVMModuleManager::Module::destroy()
{
	delete _originalModule;
}

LLVMModuleManager::FunctionId LLVMModuleManager::Module::getFunctionId(
	const std::string& kernelName) const
{
	FunctionIdMap::const_iterator id = _ids.find(kernelName);
	
	assert(id != _ids.end());
	
	return id->second;
}

LLVMModuleManager::FunctionId LLVMModuleManager::Module::lowId() const
{
	LLVMModuleManager::FunctionId min = std::numeric_limits<FunctionId>::max();
	
	for(FunctionIdMap::const_iterator id = _ids.begin(); id != _ids.end(); ++id)
	{
		min = std::min(min, id->second);
	}
	
	return min;
}

LLVMModuleManager::FunctionId LLVMModuleManager::Module::highId() const
{
	LLVMModuleManager::FunctionId max = std::numeric_limits<FunctionId>::min();
	
	for(FunctionIdMap::const_iterator id = _ids.begin(); id != _ids.end(); ++id)
	{
		max = std::max(max, id->second);
	}
	
	return max;
}

bool LLVMModuleManager::Module::empty() const
{
	return _ids.empty();
}

void LLVMModuleManager::Module::shiftId(FunctionId nextId)
{
	FunctionIdMap newIds;
	
	for(FunctionIdMap::const_iterator id = _ids.begin(); id != _ids.end(); ++id)
	{
		newIds.insert(std::make_pair(id->first, id->second - nextId));
	}
	
	_ids = std::move(newIds);
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// ModuleDatabase

LLVMModuleManager::ModuleDatabase::ModuleDatabase()
{
	#if HAVE_LLVM
	start();
	#endif
}

LLVMModuleManager::ModuleDatabase::~ModuleDatabase()
{
	#if HAVE_LLVM
	DatabaseMessage message;

	message.type = DatabaseMessage::KillThread;

	send(&message);

	DatabaseMessage* reply;	
		
	while(true)
	{
		if(killed()) break;
		
		if(test())
		{
			receive(reply);
			break;
		}
	}
	#endif
	
	for(KernelVector::iterator kernel = _kernels.begin();
		kernel != _kernels.end(); ++kernel)
	{
		kernel->unload();
	}
	
	for(ModuleMap::iterator module = _modules.begin();
		module != _modules.end(); ++module)
	{
		module->second.destroy();
	}
}

void LLVMModuleManager::ModuleDatabase::loadModule(const ir::Module* module, 
	translator::Translator::OptimizationLevel level, Device* device)
{
	if(!_barrierModule.loaded())
	{
		std::stringstream ptx;
	
		ptx << 
			".entry _ZOcelotBarrierKernel()\n"
			"{\t\n"
			"\t.reg .u32 %r<2>;\n"
			"\t.local .u32 _Zocelot_barrier_next_kernel;\n"
			"\tentry:\n"
			"\tmov.u32 %r0, _Zocelot_barrier_next_kernel;\n"
			"\tld.local.u32 %r1, [%r0];\n"
			"BarrierPrototype: .callprototype _ ();\n"
			"\tcall.tail %r1, BarrierPrototype;\n"
			"\texit;\n"
			"}\n";
	
		_barrierModule.load(ptx, "_ZOcelotBarrierModule");

		loadModule(&_barrierModule, translator::Translator::NoOptimization, 0);
	}
	
	typedef api::OcelotConfiguration config;

	assert(!isModuleLoaded(module->path()));

	report("Loading module '" << module->path() << "'");

	ir::Module* newModule = new ir::Module(*module);

	typedef transforms::SubkernelFormationPass::ExtractKernelsPass Pass;
	Pass pass(config::get().optimizations.subkernelSize);
	transforms::PassManager manager(newModule);

	manager.addPass(&pass);
	manager.runOnModule();
	manager.releasePasses();

	KernelVector subkernels;

	for(Pass::KernelVectorMap::const_iterator 
		kernel = pass.kernels.begin(); 
		kernel != pass.kernels.end(); ++kernel)
	{
		for(transforms::SubkernelFormationPass::KernelVector::const_iterator 
			subkernel = kernel->second.begin();
			subkernel != kernel->second.end(); ++subkernel)
		{
			report(" adding subkernel '" << (*subkernel)->name 
				<< "' at index " << (subkernels.size() + _kernels.size()));
			subkernels.push_back(KernelAndTranslation(*subkernel,
				level, kernel->first, std::distance(
				kernel->second.begin(), subkernel), kernel->second.size() - 1,
				device, this));
		}
	}

	_modules.insert(std::make_pair(module->path(),
		Module(subkernels, _kernels.size(), newModule)));
	_kernels.insert(_kernels.end(), subkernels.begin(), subkernels.end());
}

bool LLVMModuleManager::ModuleDatabase::isModuleLoaded(
	const std::string& moduleName)
{
	return _modules.count(moduleName) != 0;
}

void LLVMModuleManager::ModuleDatabase::unloadModule(
	const std::string& moduleName)
{
	report("Unloading module '" << moduleName << "'");
	
	ModuleMap::iterator module = _modules.find(moduleName);
	assert(module != _modules.end());

	if(module->second.empty())
	{
		_modules.erase(module);
		return;
	}

	FunctionId lowId  = module->second.lowId();
	FunctionId highId = module->second.highId();

	report(" Removing kernels between " << lowId << " and " << highId);
	
	KernelVector::iterator kernelStart = _kernels.begin();
	KernelVector::iterator kernelEnd   = _kernels.begin();
	std::advance(kernelEnd, lowId);
	
	KernelVector newKernels(kernelStart, kernelEnd);
	
	kernelStart = _kernels.begin();
	std::advance(kernelStart, highId + 1);
	
	for(KernelVector::iterator unloaded = kernelEnd;
		unloaded != kernelStart; ++unloaded)
	{
		unloaded->unload();
	}
	
	newKernels.insert(newKernels.end(), kernelStart, _kernels.end());

	report(" Removed " << (_kernels.size() - newKernels.size()) << " kernels.");

	_kernels = std::move(newKernels);
	
	module->second.destroy();
	
	_modules.erase(module);
	
	for(ModuleMap::iterator module = _modules.begin();
		module != _modules.end(); ++module)
	{
		if(module->second.lowId() > lowId)
		{
			module->second.shiftId(highId - lowId + 1);
		}
	}
}

unsigned int LLVMModuleManager::ModuleDatabase::totalFunctionCount() const
{
	return _kernels.size();
}

LLVMModuleManager::FunctionId LLVMModuleManager::ModuleDatabase::getFunctionId(
	const std::string& moduleName, const std::string& kernelName) const
{
	ModuleMap::const_iterator module = _modules.find(moduleName);

	assert(module != _modules.end());
	return module->second.getFunctionId(kernelName);
}

void LLVMModuleManager::ModuleDatabase::setExternalFunctionSet(
	const ir::ExternalFunctionSet& s)
{
	_externals = &s;
}

void LLVMModuleManager::ModuleDatabase::clearExternalFunctionSet()
{
	_externals = 0;
}

const ir::ExternalFunctionSet& 
	LLVMModuleManager::ModuleDatabase::getExternalFunctionSet() const
{
	return *_externals;
}

void LLVMModuleManager::ModuleDatabase::execute()
{
	DatabaseMessage* m;
	
	Id id = threadReceive(m);
	
	while(m->type != DatabaseMessage::KillThread)
	{
		GetFunctionMessage* message = static_cast<GetFunctionMessage*>(m);
		
		try
		{
			if(message->type == DatabaseMessage::GetId)
			{
				ModuleMap::iterator module = _modules.find(message->moduleName);

				if(module != _modules.end())
				{
					message->id = module->second.getFunctionId(
						message->kernelName);
				}
			}
			else
			{
				assert(message->id < _kernels.size());
				message->metadata = _kernels[message->id].metadata();
			}
		}
		catch(const hydrazine::Exception& e)
		{
			report("Operation failed, replying with exception.");
			message->type = DatabaseMessage::Exception;
			message->errorMessage = e.what();
		}
		
		threadSend(message, id);
		id = threadReceive(m);
	}

	threadSend(m, id);
}
////////////////////////////////////////////////////////////////////////////////

LLVMModuleManager::ModuleDatabase LLVMModuleManager::_database;

}

#endif

