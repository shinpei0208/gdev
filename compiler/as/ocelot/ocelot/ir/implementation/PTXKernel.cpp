/*! \file PTXKernel.cpp
	\author Gregory Diamos <gregory.diamos@gatech>
	\date Thursday September 17, 2009
	\brief The header file for the PTXKernel class
*/

#ifndef PTX_KERNEL_H_INCLUDED
#define PTX_KERNEL_H_INCLUDED

// C++ includes
#include <cmath>

// Ocelot Includes
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>

#include <ocelot/transforms/interface/PassManager.h>
#include <ocelot/transforms/interface/ReadableLayoutPass.h>

// Hydrazine Includes
#include <hydrazine/interface/Version.h>
#include <hydrazine/interface/debug.h>

////////////////////////////////////////////////////////////////////////////////

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

////////////////////////////////////////////////////////////////////////////////

namespace ir
{

PTXKernel::Prototype::Prototype() {
	callType = Entry;
	linkingDirective = Visible;
}

std::string PTXKernel::Prototype::toString(const LinkingDirective ld) {
	switch (ld) {
		case Extern:  return ".extern";
		case Visible: return ".visible";
		case Weak:    return ".weak";
		default:      break;
	}
	return ""; // internal hidden is a valid state
}
std::string PTXKernel::Prototype::toString(const CallType ct) {
	switch (ct) {
		case Entry: return ".entry";
		case Func:  return ".func";
		default:    break;
	}
	return "invalid";
}

void PTXKernel::Prototype::clear() { 
	returnArguments.clear();
	arguments.clear();
}

/*!
*/
std::string PTXKernel::Prototype::toString(PTXEmitter::Target emitterTarget) const {
	std::stringstream ss;
	
	if (callType == Func) {
		ss << Prototype::toString(linkingDirective) << " ";
	}
	ss << Prototype::toString(callType) << " ";
	if (returnArguments.size()) {
		ss << "(";
		int n = 0;
		for (ParameterVector::const_iterator op_it
			= returnArguments.begin();
			op_it != returnArguments.end(); ++op_it) {
		
			ss << (n++ ? ", " : "") << op_it->toString();	
		}
		ss << ") ";
	}

	ss << identifier << " (";
	if (arguments.size()) {
		int n = 0;
		for (ParameterVector::const_iterator op_it = arguments.begin();
			op_it != arguments.end(); ++op_it) {
		
			ss << (n++ ? ", " : "") << op_it->toString();	
		}
	}
	ss << ")";

	return ss.str();
}
			
/*!
	\brief emits a mangled form of the function prototype that can be 
	used to identify the function
*/
std::string PTXKernel::Prototype::getMangledName() const {
	std::stringstream ss;

	ss << identifier << "(";
	if (arguments.size()) {
		int n = 0;
		for (ParameterVector::const_iterator op_it = arguments.begin();
			op_it != arguments.end(); ++op_it) {
		
			ss << (n++ ? "," : "") << op_it->toString();	
		}
	}
	ss << ")";

	return ss.str();
}

////////////////////////////////////////////////////////////////////////////

PTXKernel::PTXKernel( const std::string& name, bool isFunction,
	const ir::Module* module, Id id ) :
	IRKernel( Instruction::PTX, name, isFunction, module, id )
{
	_cfg = new ControlFlowGraph(this);
}

PTXKernel::PTXKernel( PTXStatementVector::const_iterator start,
	PTXStatementVector::const_iterator end, bool function, Id id) : 
	IRKernel( Instruction::PTX, "", function, 0, id )
{
	_cfg = new ControlFlowGraph(this);
	constructCFG( *_cfg, start, end );
	assignRegisters( *_cfg );
}

PTXKernel::PTXKernel( const PTXKernel& kernel ) : IRKernel( kernel )
{
	
}

const PTXKernel& PTXKernel::operator=(const PTXKernel &kernel) 
{
	if( &kernel == this ) return *this;
	
	IRKernel::operator=(kernel);
	_function = kernel.function();

	return *this;	
}

PTXKernel::RegisterVector PTXKernel::getReferencedRegisters() const
{
	report( "Getting list of all referenced registers" );				

	typedef std::unordered_set< analysis::DataflowGraph::RegisterId > 
		RegisterSet;

	RegisterSet encountered;
	RegisterSet predicates;
	RegisterSet addedRegisters;
	RegisterVector regs;
	
	for( ControlFlowGraph::const_iterator block = cfg()->begin(); 
		block != cfg()->end(); ++block )
	{
		report( " For block " << block->label());
					
		for( ControlFlowGraph::InstructionList::const_iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction )
		{
			report( "  For instruction " << (*instruction)->toString() );
			
			const ir::PTXInstruction& ptx = static_cast<
				const ir::PTXInstruction&>(**instruction);
			
			const ir::PTXOperand* operands[] = {&ptx.pq, &ptx.d, &ptx.a, &ptx.b,
				&ptx.c, &ptx.pg};

			for( unsigned int i = 0; i < 6; ++i )
			{
				const ir::PTXOperand& d = *operands[i];
				if( d.addressMode != ir::PTXOperand::Register &&
					d.addressMode != ir::PTXOperand::ArgumentList ) continue;
				
				if( d.type != ir::PTXOperand::pred )
				{
					if( d.array.empty() &&
						d.addressMode != ir::PTXOperand::ArgumentList )
					{
						if( encountered.insert( d.reg ).second )
						{
							report( "   Added %r" << d.reg );
							analysis::DataflowGraph::Register live_reg( 
								d.reg, d.type );
							if (addedRegisters.find(live_reg.id)
								== addedRegisters.end()) {
								regs.push_back( live_reg );
								addedRegisters.insert(live_reg.id);
							}
						}
					}
					else
					{
						for( PTXOperand::Array::const_iterator 
							operand = d.array.begin(); 
							operand != d.array.end(); ++operand )
						{
							if( !operand->isRegister() ) continue;
							if( operand->addressMode ==
								PTXOperand::BitBucket ) continue;
							report( "   Added %r" << operand->reg );
							analysis::DataflowGraph::Register live_reg( 
								operand->reg, operand->type );
							if (addedRegisters.find(live_reg.id)
								== addedRegisters.end()) {
								regs.push_back( live_reg );
								addedRegisters.insert(live_reg.id);
							}
						}
					}
				}
				else if( d.addressMode != ir::PTXOperand::ArgumentList )
				{
					if( d.condition == ir::PTXOperand::Pred
						|| d.condition == ir::PTXOperand::InvPred )
					{
						if( predicates.insert( d.reg ).second )
						{
							report( "   Added %p" << d.reg );
							analysis::DataflowGraph::Register live_reg( 
								d.reg, d.type );
							if (addedRegisters.find(live_reg.id)
								== addedRegisters.end()) {
								regs.push_back( live_reg );
								addedRegisters.insert(live_reg.id);
							}
						}
					}
				}
			}
		}
	}
	
	return regs;
}

PTXOperand::RegisterType PTXKernel::getUnusedRegister() const {
	RegisterVector regs = getReferencedRegisters();
	
	PTXOperand::RegisterType max = 0;

	for (RegisterVector::const_iterator reg = regs.begin();
		reg != regs.end(); ++reg) {
		max = std::max(max, reg->id);
	}
	
	return max + 1;
}

bool PTXKernel::executable() const {
	return false;
}

void PTXKernel::constructCFG( ControlFlowGraph &cfg,
	PTXStatementVector::const_iterator kernelStart,
	PTXStatementVector::const_iterator kernelEnd) {
	typedef std::unordered_map< std::string, 
		ControlFlowGraph::iterator > BlockToLabelMap;
	typedef std::vector< ControlFlowGraph::iterator > BlockPointerVector;

	BlockToLabelMap blocksByLabel;
	BlockPointerVector branchBlocks;

	ControlFlowGraph::iterator last_inserted_block = cfg.end();
	ControlFlowGraph::iterator block = cfg.insert_block(
		ControlFlowGraph::BasicBlock(cfg.newId()));
	ControlFlowGraph::Edge edge(cfg.get_entry_block(), block, 
		ControlFlowGraph::Edge::FallThrough);

	bool inParameterList = false;
	bool isReturnArgument = false;
	bool hasExit = false;
	unsigned int statementIndex = 0;
	for( ; kernelStart != kernelEnd; ++kernelStart, ++statementIndex ) 
	{
		const PTXStatement &statement = *kernelStart;
	
		if( statement.directive == PTXStatement::Label ) 
		{
			// a label indicates the termination of a previous block
			//
			// This implementation does not store any empty basic blocks.
			if( block->instructions.size() ) {
				//
				// insert old block
				//
				if (edge.type != ControlFlowGraph::Edge::Invalid) {
					cfg.insert_edge(edge);
				}
			
				edge.head = block;
				last_inserted_block = block;
				block = cfg.insert_block(
					ControlFlowGraph::BasicBlock(cfg.newId()));
				edge.tail = block;
				edge.type = ControlFlowGraph::Edge::FallThrough;
			}
			
			block->comment = statement.instruction.metadata;
			
			report( "Added block with label " << block->label() << ", comment "
				<< block->comment );
			
			assertM( blocksByLabel.count( statement.name ) == 0, 
				"Duplicate blocks with label " << statement.name << ", comment "
				<< block->comment )
			blocksByLabel.insert( std::make_pair( statement.name, block ) );
		}
		else if( statement.directive == PTXStatement::Instr ) 
		{
			block->instructions.push_back( statement.instruction.clone() );
			
			if (statement.instruction.opcode == PTXInstruction::Bra) 
			{
				last_inserted_block = block;
				// dont't add fall through edges for unconditional branches
				if (edge.type != ControlFlowGraph::Edge::Invalid) {
					cfg.insert_edge(edge);
				}
				edge.head = block;
				branchBlocks.push_back(block);
				block = cfg.insert_block(
					ControlFlowGraph::BasicBlock(cfg.newId()));
				if (statement.instruction.pg.condition 
					!= ir::PTXOperand::PT) {
					edge.tail = block;
					edge.type = ControlFlowGraph::Edge::FallThrough;
				}
				else {
					edge.type = ControlFlowGraph::Edge::Invalid;
				}
			}
			else if( statement.instruction.isExit() )
			{
				last_inserted_block = block;
				if (edge.type != ControlFlowGraph::Edge::Invalid) {
					cfg.insert_edge(edge);
				}
				edge.head = block;
				edge.tail = cfg.get_exit_block();
				if (hasExit)
				{
					edge.type = ControlFlowGraph::Edge::Branch;
				}
				else
				{
					edge.type = ControlFlowGraph::Edge::FallThrough;
					hasExit = true;
				}
				cfg.insert_edge(edge);
				
				block = cfg.insert_block(
					ControlFlowGraph::BasicBlock(cfg.newId()));
				edge.type = ControlFlowGraph::Edge::Invalid;
			}
		}
		else if( statement.directive == PTXStatement::Param )
		{
			if( inParameterList )
			{
				arguments.push_back( Parameter( statement,
					true, isReturnArgument) );
			}
			else
			{
				parameters.insert( std::make_pair( 
					statement.name, Parameter( statement, false ) ) );
			}
		}
		else if( statement.directive == PTXStatement::Local
			|| statement.directive == PTXStatement::Shared )
		{
			locals.insert( std::make_pair( 
				statement.name, Local( statement ) ) );
		}
		else if( statement.directive == PTXStatement::Entry )
		{
			assert( !function() );
			name = statement.name;
		}
		else if( statement.directive == PTXStatement::FunctionName )
		{
			assert( function() );
			name = statement.name;
		}
		else if( statement.directive == PTXStatement::StartParam )
		{
			assert( !inParameterList );
			inParameterList = true;
			isReturnArgument = statement.isReturnArgument;
		}
		else if( statement.directive == PTXStatement::EndParam )
		{
			assert( inParameterList );
			inParameterList = false;
			isReturnArgument = statement.isReturnArgument;
		}
	}

	if (!block->instructions.size()) 
	{
		cfg.remove_block(block);
	}

	// go back and add edges for basic blocks terminating in branches
	for( BlockPointerVector::iterator it = branchBlocks.begin();
		it != branchBlocks.end(); ++it ) 
	{
		PTXInstruction& bra = *static_cast<PTXInstruction*>(
			(*it)->instructions.back());
		// skip always false branches
		if( bra.pg.condition == ir::PTXOperand::nPT ) continue;
		
		BlockToLabelMap::iterator labeledBlockIt = 
			blocksByLabel.find( bra.d.identifier );
	
		assertM(labeledBlockIt != blocksByLabel.end(), 
			"undefined label " << bra.d.identifier);
	
		bra.d.identifier = labeledBlockIt->second->label();
		cfg.insert_edge(ControlFlowGraph::Edge(*it, 
			labeledBlockIt->second, ControlFlowGraph::Edge::Branch));
	}
}

PTXKernel::RegisterMap PTXKernel::assignRegisters( ControlFlowGraph& cfg ) 
{
	RegisterMap map;

	report( "Allocating registers " );

	for (ControlFlowGraph::iterator block = cfg.begin(); 
		block != cfg.end(); ++block) {
		for (ControlFlowGraph::InstructionList::iterator 
			instruction = block->instructions.begin(); 
			instruction != block->instructions.end(); ++instruction) {
			PTXInstruction& instr = *static_cast<PTXInstruction*>(
				*instruction);
			PTXOperand PTXInstruction:: * operands[] = 
			{ &PTXInstruction::a, &PTXInstruction::b, &PTXInstruction::c, 
				&PTXInstruction::d, &PTXInstruction::pg, 
				&PTXInstruction::pq };
	
			report( " For instruction '" << instr.toString() << "'" );
	
			for (int i = 0; i < 6; i++) {
				if ((instr.*operands[i]).addressMode 
					== PTXOperand::Invalid) {
					continue;
				}
				if ((instr.*operands[i]).type == PTXOperand::pred
					&& (instr.*operands[i]).condition == PTXOperand::PT) {
					continue;
				}
				if ((instr.*operands[i]).isRegister()
					|| (instr.*operands[i]).addressMode 
					== PTXOperand::ArgumentList) {
					if (!(instr.*operands[i]).array.empty()) {
						for (PTXOperand::Array::iterator a_it = 
							(instr.*operands[i]).array.begin(); 
							a_it != (instr.*operands[i]).array.end();
							++a_it) {
							
							if( !a_it->isRegister() ) continue;
							
							RegisterMap::iterator it =
								map.find(a_it->registerName());

							PTXOperand::RegisterType reg = 0;
							if (it == map.end()) {
								reg = (PTXOperand::RegisterType) map.size();
								map.insert(std::make_pair(
									a_it->registerName(), reg));
							}
							else {
								reg = it->second;
							}
							if (a_it->addressMode != PTXOperand::BitBucket
								&& a_it->identifier != "_") {
								report( "  [1] Assigning register " 
									<< a_it->registerName() 
									<< " to " << a_it->reg );
								a_it->identifier.clear();
							}
							else {
								report("  [1] " << a_it->registerName() 
									<< " is a bit bucket");
							}
							a_it->reg = reg;
						}
					}
					else if((instr.*operands[i]).addressMode 
					    != PTXOperand::ArgumentList) {
						RegisterMap::iterator it 
							= map.find((instr.*operands[i]).registerName());

						PTXOperand::RegisterType reg = 0;
						if (it == map.end()) {
							reg = (PTXOperand::RegisterType) map.size();
							map.insert(std::make_pair( 
								(instr.*operands[i]).registerName(), reg));
						}
						else {
							reg = it->second;
						}
						report("  [2] Assigning register " 
							<< (instr.*operands[i]).registerName() 
							<< " to " << reg);
						(instr.*operands[i]).identifier.clear();
						(instr.*operands[i]).reg = reg;
					}
				}
			}
		}
	}

	return map;
}


static unsigned int align(unsigned int offset, unsigned int _size) {
	unsigned int size = _size == 0 ? 1 : _size;
	unsigned int difference = offset % size;
	unsigned int alignedOffset = difference == 0 
		? offset : offset + size - difference;
	return alignedOffset;
}

void PTXKernel::computeOffset(
	const ir::PTXStatement& statement, unsigned int& offset, 
	unsigned int& totalOffset) {
	
	offset = align(totalOffset, statement.accessAlignment());

	totalOffset = offset;
	if(statement.array.stride.empty()) {
		totalOffset += statement.array.vec * ir::PTXOperand::bytes(statement.type);
	}
	else {
		for (int i = 0; i < (int)statement.array.stride.size(); i++) {
			totalOffset += statement.array.stride[i] * statement.array.vec * 
				ir::PTXOperand::bytes(statement.type);
		}
	}
}

unsigned int PTXKernel::getSharedMemoryLayout(
	std::map<std::string, unsigned int> &globalOffsets, 
	std::map<std::string, unsigned int> &localOffsets) const {
	
	typedef std::unordered_map<std::string, 
		ir::Module::GlobalMap::const_iterator> GlobalMap;
	typedef std::deque<ir::PTXOperand*> OperandVector;
	
	unsigned int sharedOffset = 0;

	report( "Initializing shared memory for kernel " << name );
	GlobalMap sharedGlobals;

	OperandVector externalOperands;
	
	if(module != 0) {
		for(ir::Module::GlobalMap::const_iterator it = module->globals().begin(); 
			it != module->globals().end(); ++it) {
			
			if (it->second.statement.directive == ir::PTXStatement::Shared) {
				if(it->second.statement.attribute == ir::PTXStatement::Extern) {
				
					report("Found global external shared variable " << it->second.statement.name);
				} 
				else {
					report("Found global shared variable " << it->second.statement.name);
					unsigned int offset;
					computeOffset(it->second.statement, offset, sharedOffset);
					globalOffsets[it->second.name()] = offset;
				}
			}
		}
	}
	
	LocalMap::const_iterator it = locals.begin();
	for (; it != locals.end(); ++it) {
		if (it->second.space == ir::PTXInstruction::Shared) {
			if(it->second.attribute == ir::PTXStatement::Extern) {
			
				report("Found local external shared variable " << it->second.name);
			}
			else {
				unsigned int offset;
				computeOffset(it->second.statement(), offset, sharedOffset);
				localOffsets[it->second.name] = offset;
				
				report("Found local shared variable " << it->second.name 
					<< " at offset " << offset << " with alignment " 
					<< it->second.getAlignment() << " of size " 
					<< (sharedOffset - offset ));
			}
		}
	}
	return sharedOffset;
}

unsigned int PTXKernel::sharedMemorySize() const
{
	std::map<std::string, unsigned int> globalOffsets;
	std::map<std::string, unsigned int> localOffsets;	

	return getSharedMemoryLayout(globalOffsets, localOffsets);
}

void PTXKernel::write(std::ostream& stream) const 
{
	writeWithEmitter(stream);
}

void PTXKernel::writeWithEmitter(std::ostream& stream,
	PTXEmitter::Target emitterTarget) const 
{
	std::stringstream strReturnArguments;
	std::stringstream strArguments;
	
	int returnArgCount = 0, argCount = 0;
	
	for( ParameterVector::const_iterator parameter = arguments.begin();
		parameter != arguments.end(); ++parameter) {
		if (parameter->returnArgument) {
			strReturnArguments << (returnArgCount++ ? ",\n\t\t" : "")
				<< parameter->toString(emitterTarget);
		}
		else {
			strArguments << (argCount++ ? ",\n\t\t" : "")
				<< parameter->toString(emitterTarget);
		}
	}
	
	
	if (_function) {
		stream << ".visible .func ";
		if (returnArgCount) {
			stream << "(" << strReturnArguments.str() << ") ";
		}
		stream << name;
	}
	else {
		stream << ".entry " << name;
	}
	if (argCount) {
		stream << "(" << strArguments.str() << ")\n";
	}
	else {
		stream << "()\n";
	}
	stream << "{\n";
	
	for (LocalMap::const_iterator local = locals.begin();
		local != locals.end(); ++local) {
		stream << "\t" << local->second.toString() << "\n";
	}
	
	stream << "\n";

	for (ParameterMap::const_iterator parameter = parameters.begin();
		parameter != parameters.end(); ++parameter ) {
		stream << "\t" << parameter->second.toString() << ";\n";
	}
	
	RegisterVector regs = getReferencedRegisters();

	for (RegisterVector::const_iterator reg = regs.begin();
		reg != regs.end(); ++reg) {
		if (reg->type == PTXOperand::pred) {
			stream << "\t.reg .pred %p" << reg->id << ";\n";
		}
		else {
			stream << "\t.reg ." 
				<< PTXOperand::toString( reg->type ) << " " 
				<< "%r" << reg->id << ";\n";
		}
	}
	
	typedef std::map<std::string, ir::PTXInstruction*> IndirectCallMap;
	
	// issue actual instructions
	if (_cfg != 0) {
		ControlFlowGraph::BlockPointerVector blocks; 
		
		if (emitterTarget == PTXEmitter::Target_OcelotIR
			|| emitterTarget == PTXEmitter::Target_NVIDIA_PTX30) {

			// TODO: implement a const version of the pass manager
			transforms::PassManager manager(const_cast<Module*>(module));
			
			transforms::ReadableLayoutPass pass;
			manager.addPass(&pass);
			
			manager.runOnKernel(const_cast<PTXKernel&>(*this));
			manager.releasePasses();
						
			blocks = pass.blocks;
		}
		else {
			blocks = _cfg->executable_sequence();
		}		
		
		IndirectCallMap indirectCalls;
	
		// look for and emit function prototypes
		for (ControlFlowGraph::BlockPointerVector::iterator 
			block = blocks.begin(); block != blocks.end(); ++block) {
			
			for( ControlFlowGraph::InstructionList::iterator 
				instruction = (*block)->instructions.begin(); 
				instruction != (*block)->instructions.end();
				++instruction ) {
				ir::PTXInstruction* inst =
					static_cast<ir::PTXInstruction *>(*instruction);
				
				if (inst->opcode == ir::PTXInstruction::Call) {
					bool needsPrototype = false;
					std::string name;
											
					if (inst->a.addressMode == PTXOperand::FunctionName) {
						needsPrototype = module->prototypes().count(
							inst->a.identifier) == 0;
						if (needsPrototype) {
							name = inst->a.identifier;
						}
					}
					else if (inst->a.addressMode == PTXOperand::Register) {
						name = inst->c.identifier;
						needsPrototype = true;
					}
					
					// indirect call
					if (needsPrototype) {
						indirectCalls[name] = inst;
					}
				}
			}
		}
		if (indirectCalls.size()) {
			stream << "\t\n";
			for (IndirectCallMap::const_iterator
				indCall = indirectCalls.begin();
				indCall != indirectCalls.end(); ++indCall) {

				stream << "\t" << indCall->first << ": .callprototype ";
				
				if (!indCall->second->d.array.empty()) {
					stream << "(";
				
					unsigned int n = 0;
					for (ir::PTXOperand::Array::const_iterator
						arg_it = indCall->second->d.array.begin();
						arg_it != indCall->second->d.array.end();
						++arg_it, ++n) {
				
						stream << (n ? ", " : "") << ".param ."
							<< ir::PTXOperand::toString(arg_it->type) << " _";
					}
			
					stream << ")";
				}
				
				stream << " " << indCall->first << " (";
				unsigned int n = 0;
				for (ir::PTXOperand::Array::const_iterator
					arg_it = indCall->second->b.array.begin();
					arg_it != indCall->second->b.array.end();
					++arg_it, ++n) {
				
					stream << (n ? ", " : "") << ".param ."
						<< ir::PTXOperand::toString(arg_it->type) << " _";
				}
				stream << ");\n";
			}
			stream << "\t\n";
		}

		//
	
		int blockIndex = 1;

		for (auto block = blocks.begin(); block != blocks.end(); 
			++block, ++blockIndex) {
			std::string label = (*block)->label();
			std::string comment = (*block)->comment;
			if ((*block)->instructions.size() 
				|| (label != "entry" && label != "exit")) {
				if (label == "") {
					std::stringstream ss;
					ss << "$__Block_" << (*block)->id;
					label = ss.str();
				}
				stream << "\t" << label << ":";
				if (comment != "") {
					stream << "\t\t" << comment << " ";
				}
				stream << "\n";
			}
			
			for (auto instruction = (*block)->instructions.begin(); 
				instruction != (*block)->instructions.end();
				++instruction ) {
				ir::PTXInstruction* inst =
					static_cast<ir::PTXInstruction *>(*instruction);
				
				stream << "\t\t" << inst->toString() << "; "
				    << inst->metadata << "\n";
			}
		}
	}
	stream << "}\n";
}

/*! \brief returns a prototype for this kernel */
const ir::PTXKernel::Prototype& ir::PTXKernel::getPrototype() const {
	auto it = module->prototypes().find(name);
	assert(it != module->prototypes().end());
	return it->second;
}

}

#endif

