/*! \file   FunctionInliningPass.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Wednesday June 20, 2012
	\brief  The source file for the FunctionInliningPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/FunctionInliningPass.h>

#include <ocelot/ir/interface/IRKernel.h>
#include <ocelot/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE    0
#define REPORT_DETAILS 1

namespace transforms
{

FunctionInliningPass::FunctionInliningPass(unsigned int threshold)
: KernelPass({"DataflowGraphAnalysis"}, "FunctionInliningPass"),
	thresholdToInline(threshold), _nextRegister(0)
{

}

void FunctionInliningPass::initialize(const ir::Module& m)
{

}

void FunctionInliningPass::runOnKernel(ir::IRKernel& k)
{
	report("Running function inlining pass on kernel " << k.name);
	
	auto analysis = getAnalysis("DataflowGraphAnalysis");
	assert(analysis != 0);
	
	auto dfg = static_cast<analysis::DataflowGraph*>(analysis);

	_nextRegister = dfg->maxRegister() + 1;
	
	// Get the set of all function calls that satisfy the inlining criteria
	_getFunctionsToInline(k);

	// Inline all of the functions in this set
	_inlineSelectedFunctions(k);

	if(!_calls.empty())
	{
		invalidateAnalysis("DataflowGraphAnalysis");
	}
}

void FunctionInliningPass::finalize()
{
	// cleanup
	_calls.clear();
}

FunctionInliningPass::StringVector
	FunctionInliningPass::getDependentPasses() const
{
	return StringVector(1, "SimplifyExternalCallsPass");	
}

static bool isBuiltin(const std::string& functionName)
{
	if(functionName.find("_Z") == 0) return true;

	if(functionName == "cudaLaunchDevice") return true;

	return false;
}

void FunctionInliningPass::_getFunctionsToInline(ir::IRKernel& k)
{
	report(" Finding functions that are eligible for inlining...");

	for(auto block = k.cfg()->begin(); block != k.cfg()->end(); ++block)
	{
		bool linked = false;
		
		for(auto instruction = block->instructions.begin();
			instruction != block->instructions.end(); ++instruction)
		{
			auto ptx = static_cast<ir::PTXInstruction&>(**instruction);
			
			if(ptx.opcode != ir::PTXInstruction::Call) continue;
			
			report("  Examining " << ptx.toString());

			if(ptx.a.addressMode != ir::PTXOperand::FunctionName)
			{
				report("   skipping because it is an indirect call.");
				continue;
			}
			
			// Get the kernel being called if it is in this module
			auto calledKernel = k.module->getKernel(ptx.a.identifier);
			
			// Skip kernels in another module
			if(calledKernel == 0)
			{
				report("   skipping because it is in a different module.");
				continue;
			}
			
			// Skip kernels that are built-in functions
			if(isBuiltin(ptx.a.identifier))
			{
				report("   skipping because it is a reserved keyword.");
				continue;
			}
			
			// Skip kernels that are too large to inline
			if(calledKernel->cfg()->instructionCount() > thresholdToInline)
			{
				report("   skipping because it is too large ("
					<< calledKernel->cfg()->instructionCount()
					<< " > " << thresholdToInline << ").");
				continue;
			}
			
			report("   it is eligible for inlining!");
			
			if(linked)
			{
				_calls.back().linked = true;
			}
			
			_calls.push_back(FunctionCallDescriptor(
				instruction, block, calledKernel));
		
			linked = true;
		}
	}
}

typedef std::unordered_map<ir::ControlFlowGraph::const_iterator,
	ir::ControlFlowGraph::iterator> BasicBlockMap;
	
static void insertAndConnectBlocks(BasicBlockMap& newBlocks,
	ir::ControlFlowGraph::iterator& functionEntry,
	ir::ControlFlowGraph::iterator& functionExit,
	ir::IRKernel& kernel, unsigned int& nextRegister,
	const ir::IRKernel& inlinedKernel)
{
	typedef std::unordered_map<ir::PTXOperand::RegisterType,
		ir::PTXOperand::RegisterType> RegisterMap;
	
	ir::IRKernel copy;
	const ir::IRKernel* inlinedKernelPointer = &inlinedKernel;
	
	// create a copy if the call is recursive
	if(inlinedKernelPointer == &kernel)
	{
		copy = inlinedKernel;
		inlinedKernelPointer = &copy;
	}
	
	//  Insert new blocks
	for(auto block = inlinedKernelPointer->cfg()->begin();
		block != inlinedKernelPointer->cfg()->end(); ++block)
	{
		auto newBlock = kernel.cfg()->clone_block(block);
		
		newBlocks.insert(std::make_pair(block, newBlock));
	}
	
	//  Connect new blocks, rename branch labels
	for(auto block = newBlocks.begin(); block != newBlocks.end(); ++block)
	{
		for(auto edge = block->first->out_edges.begin();
			edge != block->first->out_edges.end(); ++edge)
		{
			auto headBlock = block->second;
			auto tail      = (*edge)->tail;
			
			auto tailBlock = newBlocks.find(tail);
			assert(tailBlock != newBlocks.end());
			
			kernel.cfg()->insert_edge(ir::Edge(headBlock,
				tailBlock->second, (*edge)->type));
			
			if((*edge)->type == ir::Edge::Branch)
			{
				assert(!headBlock->instructions.empty());
				auto instruction = headBlock->instructions.back();
				
				auto branch = static_cast<ir::PTXInstruction*>(instruction);

				if(branch->opcode == ir::PTXInstruction::Ret) continue;

				assertM(branch->opcode == ir::PTXInstruction::Bra, "Expecting "
					<< branch->toString() << " to be a branch");
				
				branch->d.identifier = tailBlock->second->label();
			}
		}
	}
	
	//  Assign copied blocks new registers
	RegisterMap newRegisters;
	
	for(auto block = newBlocks.begin(); block != newBlocks.end(); ++block)
	{
		for(auto instruction = block->second->instructions.begin();
			instruction != block->second->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<ir::PTXInstruction&>(
				**instruction);
		
			ir::PTXOperand* operands[] = {&ptx.pg, &ptx.pq, &ptx.d, &ptx.a,
				&ptx.b, &ptx.c};
				
			for(unsigned int i = 0; i < 6; ++i)
			{
				ir::PTXOperand& operand = *operands[i];
				
				if( operand.addressMode != ir::PTXOperand::Register &&
					operand.addressMode != ir::PTXOperand::Indirect &&
					operand.addressMode != ir::PTXOperand::ArgumentList)
				{
					continue;
				}
				
				if(operand.type != ir::PTXOperand::pred)
				{
					if(operand.array.empty() &&
						operand.addressMode != ir::PTXOperand::ArgumentList)
					{
						auto mapping = newRegisters.find(operand.reg);
						
						if(mapping == newRegisters.end())
						{
							mapping = newRegisters.insert(std::make_pair(
								operand.reg, nextRegister++)).first;
						}
						
						operand.reg = mapping->second;
					}
					else
					{
						for(auto subOperand = operand.array.begin(); 
							subOperand != operand.array.end(); ++subOperand )
						{
							if(!subOperand->isRegister()) continue;
							
							auto mapping = newRegisters.find(subOperand->reg);
						
							if(mapping == newRegisters.end())
							{
								mapping = newRegisters.insert(std::make_pair(
									subOperand->reg, nextRegister++)).first;
							}
						
							subOperand->reg = mapping->second;
						}
					}
				}
				else if(operand.addressMode != ir::PTXOperand::ArgumentList)
				{
					if(operand.condition == ir::PTXOperand::Pred
						|| operand.condition == ir::PTXOperand::InvPred)
					{
						auto mapping = newRegisters.find(operand.reg);
						
						if(mapping == newRegisters.end())
						{
							mapping = newRegisters.insert(std::make_pair(
								operand.reg, nextRegister++)).first;
						}
						
						operand.reg = mapping->second;
					}
				}
			}
		}
	}
	
	//  Assign copied blocks new local variables
	typedef std::unordered_map<std::string, std::string> LocalMap;
	
	LocalMap locals;
	
	for(auto local = inlinedKernel.locals.begin();
		local != inlinedKernel.locals.end(); ++local)
	{
		std::string newName = "_Zinlined_" + local->first;
	
		locals.insert(std::make_pair(local->first, newName));
		
		auto newLocal = kernel.locals.insert(
			std::make_pair(newName, local->second)).first;
		
		newLocal->second.name = newName;
	}
	
	for(auto block = newBlocks.begin(); block != newBlocks.end(); ++block)
	{
		for(auto instruction = block->second->instructions.begin();
			instruction != block->second->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<ir::PTXInstruction&>(
				**instruction);
		
			if(!ptx.mayHaveAddressableOperand()) continue;
		
			ir::PTXOperand* operands[] = {&ptx.pg, &ptx.pq, &ptx.d, &ptx.a,
				&ptx.b, &ptx.c};
				
			for(unsigned int i = 0; i < 6; ++i)
			{
				ir::PTXOperand& operand = *operands[i];
				
				if(operand.addressMode != ir::PTXOperand::Address) continue;
				
				auto local = locals.find(operand.identifier);
				
				if(local == locals.end()) continue;
				
				operand.identifier = local->second;
			}
		}
	}
	
	//  Get the entry and exit points
	auto entryMapping = newBlocks.find(
		inlinedKernelPointer->cfg()->get_entry_block());
	assert(entryMapping != newBlocks.end());
	
	functionEntry = entryMapping->second;
	
	auto exitMapping = newBlocks.find(
		inlinedKernelPointer->cfg()->get_exit_block());
	assert(exitMapping != newBlocks.end());
	
	functionExit = exitMapping->second;
}

static void convertParametersToRegisters(
	const BasicBlockMap& newBlocks, ir::IRKernel& kernel,
	ir::ControlFlowGraph::instruction_iterator callIterator,
	const ir::IRKernel& calledKernel)
{
	typedef std::unordered_map<std::string,	ir::PTXOperand> OperandMap;
	typedef std::unordered_set<std::string> StringSet;

	reportE(REPORT_DETAILS, "   Converting parameters to registers...");
	
	// Get a map from argument name to register in the calling function
	OperandMap  argumentMap;
	StringSet   bitBucketArguments;
	
	auto argument = calledKernel.arguments.begin();
	
	ir::PTXInstruction& call = static_cast<ir::PTXInstruction&>(**callIterator);
	
	for(auto parameter = call.d.array.begin();
		parameter != call.d.array.end(); ++parameter, ++argument)
	{
		if(parameter->addressMode == ir::PTXOperand::BitBucket)
		{
			bitBucketArguments.insert(argument->name);
			continue;
		}

		assert(argument != calledKernel.arguments.end());
		assert(parameter->addressMode == ir::PTXOperand::Register ||
			parameter->addressMode == ir::PTXOperand::Immediate);
		assert(argumentMap.count(argument->name) == 0);
		assert(argument->returnArgument);

		argumentMap.insert(std::make_pair(argument->name, *parameter));
	}

	for(auto parameter = call.b.array.begin();
		parameter != call.b.array.end(); ++parameter, ++argument)
	{
		if(parameter->addressMode == ir::PTXOperand::BitBucket)
		{
			bitBucketArguments.insert(argument->name);
			continue;
		}

		assert(argument != calledKernel.arguments.end());
		assert(parameter->addressMode == ir::PTXOperand::Register ||
			parameter->addressMode == ir::PTXOperand::Immediate);
		assert(argumentMap.count(argument->name) == 0);
		assert(!argument->returnArgument);

		argumentMap.insert(std::make_pair(argument->name, *parameter));
	}
	
	// Convert all stores to that parameter to moves to the associated register
	for(auto block = newBlocks.begin(); block != newBlocks.end(); ++block)
	{
		for(auto instruction = block->second->instructions.begin();
			instruction != block->second->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<ir::PTXInstruction&>(
				**instruction);
		
			if(ptx.opcode        != ir::PTXInstruction::St)    continue;
			if(ptx.addressSpace  != ir::PTXInstruction::Param) continue;
			if(ptx.d.addressMode != ir::PTXOperand::Address)   continue;
			
			if(bitBucketArguments.count(ptx.d.identifier))
			{
				delete *instruction;
				instruction = --block->second->instructions.erase(instruction);
				
				continue;
			}
			
			auto argument = argumentMap.find(ptx.d.identifier);
			
			if(argument == argumentMap.end()) continue;

			ptx.type = argument->second.type;
			ptx.pg   = call.pg;
			ptx.d    = argument->second;
				
			if(argument->second.addressMode == ir::PTXOperand::Register)
			{
				// If the types match, it is a move
				if(argument->second.type == ptx.d.type)
				{
					ptx.opcode = ir::PTXInstruction::Mov;
				}
				else
				{
					// otherwise, we need a cast
					ptx.opcode   = ir::PTXInstruction::Cvt;
					ptx.modifier = ir::PTXInstruction::Modifier_invalid;
				}
			}
			else
			{
				assert(argument->second.addressMode ==
					ir::PTXOperand::Immediate);
		
				ptx.opcode = ir::PTXInstruction::Mov;
			}
		}
	}
	
	// Convert all loads from that parameter to moves from the register
	for(auto block = newBlocks.begin(); block != newBlocks.end(); ++block)
	{
		for(auto instruction = block->second->instructions.begin();
			instruction != block->second->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<ir::PTXInstruction&>(
				**instruction);
	
			if(ptx.opcode        != ir::PTXInstruction::Ld)    continue;
			if(ptx.addressSpace  != ir::PTXInstruction::Param) continue;
			if(ptx.a.addressMode != ir::PTXOperand::Address)   continue;
			
			if(bitBucketArguments.count(ptx.a.identifier))
			{
				delete *instruction;
				instruction = --block->second->instructions.erase(instruction);
				
				continue;
			}
			
			auto argument = argumentMap.find(ptx.a.identifier);
			
			if(argument == argumentMap.end()) continue;
			
			assert(ptx.d.addressMode == ir::PTXOperand::Register);
			
			ptx.type = argument->second.type;			
			ptx.pg   = call.pg;
			ptx.a    = argument->second;
			
			// If the types match, it is a move
			if(ptx.type == ptx.a.type)
			{
				ptx.opcode = ir::PTXInstruction::Mov;
			}
			else
			{
				// otherwise, we need a cast
				ptx.opcode        = ir::PTXInstruction::Cvt;
				ptx.modifier      = ir::PTXInstruction::Modifier_invalid;
			}
		}
	}
}

static ir::ControlFlowGraph::iterator convertCallToJumps(
	const BasicBlockMap& newBlocks,
	ir::ControlFlowGraph::iterator functionEntry,
	ir::ControlFlowGraph::iterator functionExit, ir::IRKernel& kernel,
	ir::ControlFlowGraph::instruction_iterator call,
	ir::ControlFlowGraph::iterator block)
{
	// split the block
	auto firstInstructionOfSplitBlock = call;
	++firstInstructionOfSplitBlock;
	
	auto returnBlock = kernel.cfg()->split_block(block,	
		firstInstructionOfSplitBlock, ir::Edge::Invalid);
	kernel.cfg()->remove_edge(block->out_edges.front());	
	
	// add edges
	kernel.cfg()->insert_edge(ir::Edge(block, functionEntry, ir::Edge::Branch));
	kernel.cfg()->insert_edge(ir::Edge(functionExit,
		returnBlock, ir::Edge::Branch));
	
	ir::PTXInstruction& ptxCall = static_cast<ir::PTXInstruction&>(**call);
	
	if(ptxCall.pg.condition != ir::PTXOperand::PT)
	{
		kernel.cfg()->insert_edge(ir::Edge(block,
			returnBlock, ir::Edge::FallThrough));
	}
	else
	{
		ptxCall.uni = true;
	}

	// set branch to function instruction
	
	ptxCall = ir::PTXInstruction(ir::PTXInstruction::Bra);
	
	ptxCall.d.addressMode = ir::PTXOperand::Label;
	ptxCall.d.identifier  = functionEntry->label();
	
	// set all return instructions to branches to the exit node
	for(auto block = newBlocks.begin(); block != newBlocks.end(); ++block)
	{
		for(auto instruction = block->second->instructions.begin();
			instruction != block->second->instructions.end(); ++instruction)
		{
			ir::PTXInstruction& ptx = static_cast<ir::PTXInstruction&>(
				**instruction);
		
			if(ptx.opcode != ir::PTXInstruction::Ret) continue;
			
			ptx = ir::PTXInstruction(ir::PTXInstruction::Bra);

			ptx.d.addressMode = ir::PTXOperand::Label;
			ptx.d.identifier  = functionExit->label();
			
			if(block->second->has_fallthrough_edge() &&
				ptx.pg.condition == ir::PTXOperand::PT)
			{
				auto fallthrough = block->second->get_fallthrough_edge();
				
				ptx.uni = true;
				
				ir::Edge newEdge(fallthrough->head, fallthrough->tail,
					ir::Edge::Branch);
				
				kernel.cfg()->remove_edge(fallthrough);
				kernel.cfg()->insert_edge(newEdge);				
			}
			
			break;
		}
	}
	
	// set branch back after executing function
	auto ret = new ir::PTXInstruction(ir::PTXInstruction::Bra);
	
	ret->uni = true;
	
	ret->d.addressMode = ir::PTXOperand::Label;
	ret->d.identifier  = returnBlock->label();
	
	functionExit->instructions.push_back(ret);
	
	return returnBlock;
}

void FunctionInliningPass::_inlineSelectedFunctions(ir::IRKernel& k)
{
	report(" Inlining selected calls...");
	
	for(auto call = _calls.begin(); call != _calls.end(); ++call)
	{
		if(call->calledKernel->cfg()->instructionCount() > thresholdToInline)
		{
			report("  skipping " << (*call->call)->toString()
				<< " because to function being inlined has grown too large.");
			continue;
		}
		
		BasicBlockMap newBlocks;

		ir::ControlFlowGraph::iterator functionEntry;
		ir::ControlFlowGraph::iterator functionExit;

		report("  Inlining " << (*call->call)->toString());

		// Insert and connect the blocks from the inlined kernel
		insertAndConnectBlocks(newBlocks, functionEntry, functionExit,
			k, _nextRegister, *call->calledKernel);
		
		// Convert parameter accesses into register accesses
		convertParametersToRegisters(newBlocks, k, call->call,
			*call->calledKernel);
		
		// Split the original block, jump to the function at the call and back
		// to the second half of the split block at the return
		auto returnBlock = convertCallToJumps(newBlocks,
			functionEntry, functionExit, k,
			call->call, call->basicBlock);
		
		_updateLinks(returnBlock, call);
	}
}

FunctionInliningPass::FunctionCallDescriptor::FunctionCallDescriptor(
	ir::ControlFlowGraph::instruction_iterator c,
	ir::ControlFlowGraph::iterator b, const ir::IRKernel* k, bool l)
: call(c), basicBlock(b), calledKernel(k), linked(l)
{

}

void FunctionInliningPass::_updateLinks(
	ir::ControlFlowGraph::iterator splitBlock,
	FunctionDescriptorVector::iterator call)
{
	auto instruction = splitBlock->instructions.begin();

	while(call->linked)
	{
		++call;
		
		report("Updating linked call to '" << call->calledKernel->name << "'");
		
		call->basicBlock = splitBlock;
		
		while(true)
		{
			assert(instruction != splitBlock->instructions.end());
			
			ir::PTXInstruction& ptx =
				static_cast<ir::PTXInstruction&>(**instruction);
		
			if(ptx.opcode != ir::PTXInstruction::Call ||
				ptx.a.addressMode != ir::PTXOperand::FunctionName ||
				ptx.a.identifier != call->calledKernel->name)
			{
				++instruction;
				continue;
			}
			
			call->call = instruction;
			break;
		}
	}
}

}

