/*! \file ConvertPredicationToSelectPass.cpp
	\date Friday September 25, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the ConvertPredicationToSelectPass class
*/

#ifndef REMOVE_BARRIER_PASS_CPP_INCLUDED
#define REMOVE_BARRIER_PASS_CPP_INCLUDED

#include <ocelot/transforms/interface/ConvertPredicationToSelectPass.h>
#include <ocelot/ir/interface/PTXKernel.h>

#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace transforms
{
	analysis::DataflowGraph& ConvertPredicationToSelectPass::dfg()
	{
		analysis::Analysis* graph = getAnalysis("DataflowGraphAnalysis");
		assert(graph != 0);
		
		return static_cast<analysis::DataflowGraph&>(*graph);
	}

	analysis::DataflowGraph::RegisterId
		ConvertPredicationToSelectPass::_tempRegister()
	{
		return dfg().newRegister();
	}

	void ConvertPredicationToSelectPass::_replacePredicate( 
		analysis::DataflowGraph::iterator block, unsigned int id )
	{
		typedef analysis::DataflowGraph::RegisterSet RegisterSet;

		analysis::DataflowGraph::InstructionVector::const_iterator 
			instruction( block->instructions().begin() );
		std::advance( instruction, id );

		report( "  Converting instruction " << instruction->i->toString() );
		
		ir::PTXInstruction select( ir::PTXInstruction::SelP );

		ir::PTXInstruction& ptx = static_cast< ir::PTXInstruction& >( 
			*instruction->i );

		select.d     = ptx.d;
		select.b     = select.d;
		select.a     = select.b;
		select.a.reg = _tempRegister();
		select.c     = ptx.pg;
		select.type  = ptx.type;
		
		ptx.pg.condition = ir::PTXOperand::PT;
		ptx.d.reg        = select.a.reg;
			
		dfg().insert( block, select, id + 1 );
	}
	
	void ConvertPredicationToSelectPass::_runOnBlock( 
		analysis::DataflowGraph::iterator block )
	{
		typedef analysis::DataflowGraph::InstructionVector::const_iterator 
			const_iterator;
	
		for( const_iterator	instruction = block->instructions().begin(); 
			instruction != block->instructions().end(); ++instruction )
		{
			ir::PTXInstruction& ptx = static_cast< ir::PTXInstruction& >( 
				*instruction->i );
		
			if( ptx.opcode != ir::PTXInstruction::Bra 
				&& ptx.opcode != ir::PTXInstruction::Call 
				&& ptx.opcode != ir::PTXInstruction::Ret )
			{
				if( ptx.pg.condition != ir::PTXOperand::PT )
				{
					_replacePredicate( block, std::distance( 
						const_iterator( block->instructions().begin() ),
						instruction ) );
				}
			}
		}
	}

	
	ConvertPredicationToSelectPass::ConvertPredicationToSelectPass()
		: KernelPass( {"DataflowGraphAnalysis"},
			"ConvertPredicationToSelectPass" )
	{
	}

	void ConvertPredicationToSelectPass::initialize( const ir::Module& m )
	{

	}
	
	void ConvertPredicationToSelectPass::runOnKernel( ir::IRKernel& k )
	{
		assertM( k.ISA == ir::Instruction::PTX, 
			"This pass is valid for PTX kernels only." );
		_kernel = static_cast< ir::PTXKernel* >( &k );
		
		for( analysis::DataflowGraph::iterator block = dfg().begin(); 
			block != dfg().end(); ++block )
		{
			_runOnBlock( block );
		}
	}
	
	void ConvertPredicationToSelectPass::finalize( )
	{
	
	}
	
}

#endif

