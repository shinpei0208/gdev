/*! \file RemoveBarrierPass.cpp
	\date Tuesday September 15, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the RemoveBarrierPass class
*/

#ifndef REMOVE_BARRIER_PASS_CPP_INCLUDED
#define REMOVE_BARRIER_PASS_CPP_INCLUDED

#include <ocelot/transforms/interface/RemoveBarrierPass.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/ExternalFunctionSet.h>

#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace transforms
{

analysis::DataflowGraph& RemoveBarrierPass::_dfg()
{
	Analysis* dfg_structure = getAnalysis("DataflowGraphAnalysis");
	assert(dfg_structure != 0);

	return *static_cast<analysis::DataflowGraph*>(dfg_structure);
}

analysis::DataflowGraph::RegisterId RemoveBarrierPass::_tempRegister()
{
	return _dfg().newRegister();
}

void RemoveBarrierPass::_addSpillCode( analysis::DataflowGraph::iterator block, 
	analysis::DataflowGraph::iterator target, 
	const analysis::DataflowGraph::RegisterSet& alive, bool isBarrier )
{
	unsigned int bytes = 0;
	
	ir::PTXInstruction move ( ir::PTXInstruction::Mov );
	
	move.type = ir::PTXOperand::u32;
	move.addressSpace = ir::PTXInstruction::Local;
	move.a.identifier = "_Zocelot_spill_area";
	move.a.addressMode = ir::PTXOperand::Address;
	move.a.type = ir::PTXOperand::u32;
	
	move.d.reg = _tempRegister();
	move.d.addressMode = ir::PTXOperand::Register;
	move.d.type = ir::PTXOperand::u32;
	
	_dfg().insert( block, move, block->instructions().size() - 1 );

	report( "   Saving " << alive.size() << " Registers" );
	
	for( analysis::DataflowGraph::RegisterSet::const_iterator 
		reg = alive.begin(); reg != alive.end(); ++reg )
	{
		report( "    r" << reg->id << " (" 
			<< ir::PTXOperand::bytes( reg->type ) << " bytes)" );
		ir::PTXInstruction save( ir::PTXInstruction::St );
	
		save.type = reg->type;
		save.addressSpace = ir::PTXInstruction::Local;

		save.d.addressMode = ir::PTXOperand::Indirect;
		save.d.reg = move.d.reg;
		save.d.type = ir::PTXOperand::u32;
		save.d.offset = bytes;
		
		bytes += ir::PTXOperand::bytes( save.type );
	
		save.a.addressMode = ir::PTXOperand::Register;
		save.a.type = reg->type;
		save.a.reg = reg->id;
		
		_dfg().insert( block, save, 
			block->instructions().size() - 1 );
	}
	
	_spillBytes = std::max( bytes, _spillBytes );
	
	move.type = ir::PTXOperand::u32;
	move.addressSpace = ir::PTXInstruction::Local;
	move.a.identifier = "_Zocelot_resume_point";
	move.a.addressMode = ir::PTXOperand::Address;
	move.a.type = ir::PTXOperand::u32;
	
	move.d.reg = _tempRegister();
	move.d.addressMode = ir::PTXOperand::Register;
	move.d.type = ir::PTXOperand::u32;
	
	_dfg().insert( block, move, block->instructions().size() - 1 );
	
	ir::PTXInstruction save( ir::PTXInstruction::St );

	save.type = ir::PTXOperand::u32;
	save.addressSpace = ir::PTXInstruction::Local;

	save.d.addressMode = ir::PTXOperand::Indirect;
	save.d.reg = move.d.reg;
	save.d.type = ir::PTXOperand::u32;

	save.a.addressMode = ir::PTXOperand::Immediate;
	save.a.type = ir::PTXOperand::u32;
	save.a.imm_uint = target->id();
	
	_dfg().insert( block, save, block->instructions().size() - 1 );

	if( isBarrier )
	{
		move.d.reg = _tempRegister();
		move.a.identifier = "_Zocelot_barrier_next_kernel";
	
		_dfg().insert( block, move,
			block->instructions().size() - 1 );

		save.d.reg = move.d.reg;
		save.a.imm_uint = _kernelId;

		_dfg().insert( block, save,
			block->instructions().size() - 1 );
	}
}

void RemoveBarrierPass::_addRestoreCode(
	analysis::DataflowGraph::iterator block, 
	const analysis::DataflowGraph::RegisterSet& alive )
{
	unsigned int bytes = 0;

	ir::PTXInstruction move ( ir::PTXInstruction::Mov );
	
	move.type = ir::PTXOperand::u32;
	move.addressSpace = ir::PTXInstruction::Local;
	move.a.identifier = "_Zocelot_spill_area";
	move.a.addressMode = ir::PTXOperand::Address;
	move.a.type = ir::PTXOperand::u32;
	
	move.d.reg = _tempRegister();
	move.d.addressMode = ir::PTXOperand::Register;
	move.d.type = ir::PTXOperand::u32;

	for( analysis::DataflowGraph::RegisterSet::const_iterator 
		reg = alive.begin(); reg != alive.end(); ++reg )
	{
		ir::PTXInstruction load( 
			ir::PTXInstruction::Ld );
	
		load.type = reg->type;
		load.addressSpace = ir::PTXInstruction::Local;
		
		load.a.addressMode = ir::PTXOperand::Indirect;
		load.a.reg = move.d.reg;
		load.a.type = ir::PTXOperand::u32;
		load.a.offset = bytes;
		
		bytes += ir::PTXOperand::bytes( load.type );
	
		load.d.addressMode = ir::PTXOperand::Register;
		load.d.type = reg->type;
		load.d.reg = reg->id;
		
		_dfg().insert( block, load, 0 );
	}

	_dfg().insert( block, move, 0 );
}

void RemoveBarrierPass::_addEntryPoint(
	analysis::DataflowGraph::iterator block )
{
	analysis::DataflowGraph::iterator entry = _dfg().insert(_dfg().begin());
			
	ir::PTXInstruction move( ir::PTXInstruction::Mov );
	
	move.type = ir::PTXOperand::u32;
	move.addressSpace = ir::PTXInstruction::Local;
	move.a.identifier = "_Zocelot_resume_point";
	move.a.addressMode = ir::PTXOperand::Address;
	move.a.type = ir::PTXOperand::u32;
	
	move.d.reg = _tempRegister();
	move.d.addressMode = ir::PTXOperand::Register;
	move.d.type = ir::PTXOperand::u32;
	
	_dfg().insert( entry, move, 0 );

	ir::PTXInstruction load( ir::PTXInstruction::Ld );

	load.addressSpace = ir::PTXInstruction::Local;
	load.type = ir::PTXOperand::u32;
	load.a = move.d;

	load.d.reg = _tempRegister();
	load.d.addressMode = ir::PTXOperand::Register;
	load.d.type = ir::PTXOperand::u32;
	
	_dfg().insert( entry, load, 1 );

	ir::PTXInstruction setp( ir::PTXInstruction::SetP );
	
	setp.type = ir::PTXOperand::u32;
	setp.comparisonOperator = ir::PTXInstruction::Eq;

	setp.d.reg = _tempRegister();
	setp.d.addressMode = ir::PTXOperand::Register;
	setp.d.type = ir::PTXOperand::pred;
	
	setp.a = load.d;
	
	setp.b.addressMode = ir::PTXOperand::Immediate;
	setp.b.type = ir::PTXOperand::u32;
	setp.b.imm_uint = block->id();
	
	_dfg().insert( entry, setp, 2 );
	
	ir::PTXInstruction branch( ir::PTXInstruction::Bra );
	
	branch.d.addressMode = ir::PTXOperand::Label;
	branch.d.identifier = block->label();
	branch.pg = setp.d;

	_dfg().insert( entry, branch, 3 );
	
	_dfg().target( entry, block );
}

void RemoveBarrierPass::_removeBarrier( analysis::DataflowGraph::iterator block, 
	unsigned int id )
{
	typedef analysis::DataflowGraph::RegisterSet RegisterSet;
	
	analysis::DataflowGraph::InstructionVector::const_iterator 
		_instruction( block->instructions().begin() );
	std::advance( _instruction, id );
	analysis::DataflowGraph::iterator exitBlock( _dfg().end() );
	std::advance( exitBlock, -1 );

	ir::PTXInstruction& instruction = static_cast< ir::PTXInstruction& >( 
		*_instruction->i );

	bool isBarrier = instruction.opcode == ir::PTXInstruction::Bar;

	if( isBarrier )
	{
		report( "  Converting instruction " << instruction.toString() );
		instruction.opcode = ir::PTXInstruction::Call;
		instruction.tailCall = true;
		instruction.branchTargetInstruction = -1;
		instruction.a = ir::PTXOperand(
			ir::PTXOperand::FunctionName, "_ZOcelotBarrierKernel");
		instruction.d.addressMode = ir::PTXOperand::Invalid;

		report( "   Converted to " << instruction.toString() );		
	}
	
	RegisterSet alive = block->alive( _instruction );
	
	analysis::DataflowGraph::iterator bottom = _dfg().split( block, 
		id + 1, false );

	_addSpillCode( block, bottom, alive, isBarrier );
	_addRestoreCode( bottom, alive );
	
	_dfg().redirect( block, bottom, exitBlock );
	
	if( !isBarrier && instruction.pg.condition != ir::PTXOperand::PT )
	{
		_dfg().target( block, bottom, true );
	}
	
	_addEntryPoint( bottom );
}

void RemoveBarrierPass::_runOnBlock( analysis::DataflowGraph::iterator block )
{
	typedef analysis::DataflowGraph::InstructionVector::const_iterator
		const_iterator;
	for( const_iterator _instruction = block->instructions().begin(); 
		_instruction != block->instructions().end(); ++_instruction )
	{
		ir::PTXInstruction& instruction = static_cast< 
			ir::PTXInstruction& >( *_instruction->i );
		if( instruction.opcode == ir::PTXInstruction::Bar
			|| ( instruction.opcode == ir::PTXInstruction::Call 
				&& !instruction.tailCall ) )
		{
			if( _externals != 0
				&& instruction.opcode == ir::PTXInstruction::Call )
			{
				if( _externals->find( instruction.a.identifier ) != 0 )
				{
					report( "Skipping external call "
						<< instruction.toString() );
					continue;
				}
			}

			unsigned int bytes = _spillBytes;
			_spillBytes = 1;
			usesBarriers = true;
			_removeBarrier( block, std::distance( 
				const_iterator( block->instructions().begin() ),
				_instruction ) );
			_spillBytes = std::max( bytes, _spillBytes );
			++_reentryPoint;
			_dfg().compute();
			break;
		}
	}
}

void RemoveBarrierPass::_addLocalVariables()
{
	if( !usesBarriers ) return;
	
	if( _kernel->locals.count( "_Zocelot_resume_point" ) == 0 )
	{
		ir::PTXStatement syncVariable( ir::PTXStatement::Local );
	
		syncVariable.type = ir::PTXOperand::u32;
		syncVariable.name = "_Zocelot_resume_point";
	
		_kernel->locals.insert( std::make_pair( syncVariable.name, 
			ir::Local( syncVariable ) ) );
	}
	
	if( _kernel->locals.count( "_Zocelot_barrier_next_kernel" ) == 0 )
	{
		ir::PTXStatement nextVariable( ir::PTXStatement::Local );
	
		nextVariable.type = ir::PTXOperand::u32;
		nextVariable.name = "_Zocelot_barrier_next_kernel";
	
		_kernel->locals.insert( std::make_pair( nextVariable.name, 
			ir::Local( nextVariable ) ) );
	}
	
	ir::Kernel::LocalMap::iterator stackLocal = _kernel->locals.find( 
		"_Zocelot_spill_area" );
	
	report("Stack size is " << _spillBytes);
	
	if( stackLocal == _kernel->locals.end() )
	{
		ir::PTXStatement stack( ir::PTXStatement::Local );
	
		stack.type = ir::PTXOperand::b8;
		stack.name = "_Zocelot_spill_area";
		stack.array.stride.push_back( _spillBytes );
	
		_kernel->locals.insert( std::make_pair( stack.name, 
			ir::Local( stack ) ) );
		report(" creating new local memory stack entry.");
	
	}
	else
	{
		stackLocal->second.elements = std::max( _spillBytes, 
			stackLocal->second.elements );
		report(" updating local memory stack entry.");
	}
}

RemoveBarrierPass::RemoveBarrierPass( unsigned int i,
	const ir::ExternalFunctionSet* s )
: KernelPass( {"DataflowGraphAnalysis"}, "RemoveBarriersPass" ),
	_kernelId( i ), _externals( s )
{

}

void RemoveBarrierPass::initialize( const ir::Module& m )
{
	usesBarriers = false;
}

void RemoveBarrierPass::runOnKernel( ir::IRKernel& k )
{
	report( "Removing barriers from kernel " << k.name );
	assertM( k.ISA == ir::Instruction::PTX, 
		"This pass is valid for PTX kernels only." );
	_reentryPoint = 1;
	_spillBytes = 1;
	_kernel = static_cast< ir::PTXKernel* >( &k );
	
	for( analysis::DataflowGraph::iterator block = _dfg().begin(); 
		block != _dfg().end(); ++block )
	{
		_runOnBlock( block );
	}
	
	_addLocalVariables();
}

void RemoveBarrierPass::finalize( )
{

}

}

#endif

