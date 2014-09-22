/*! \brief  SplitBasicBlockPass.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday August 12, 2011
	\brief  The source file for the SplitBasicBlockPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/SplitBasicBlockPass.h>

#include <ocelot/ir/interface/ControlFlowGraph.h>
#include <ocelot/ir/interface/IRKernel.h>

// Standard Library Includes
#include <cassert>

namespace transforms
{

SplitBasicBlockPass::SplitBasicBlockPass(unsigned int m)
: KernelPass({}, "SplitBasicBlockPass"), _maxSize(m)
{
	assert(_maxSize > 0);
}

void splitBlock(ir::ControlFlowGraph& cfg,
	ir::ControlFlowGraph::iterator block, unsigned int maxSize)
{
	while(block->instructions.size() > maxSize)
	{
		auto splitPoint = block->instructions.begin();
		bool split = true;
		
		// don't split up function call setup instructions
		unsigned int counter = 0;
		for(; splitPoint != block->instructions.end(); ++splitPoint, ++counter)
		{
			ir::PTXInstruction& ptx =
				*static_cast<ir::PTXInstruction*>(*splitPoint);
			
			bool isParamLoad = ptx.opcode == ir::PTXInstruction::Ld &&
				ptx.addressSpace == ir::PTXInstruction::Param &&
				ptx.a.addressMode == ir::PTXOperand::Address;
			
			if(ptx.opcode == ir::PTXInstruction::Call)
			{
				split = false;
			}
			else if(!isParamLoad)
			{
				split = true;
			}
			
			if(counter == maxSize - 1) break;
		}
		
		for(; splitPoint != block->instructions.end(); ++splitPoint, ++counter)
		{
			ir::PTXInstruction& ptx =
				*static_cast<ir::PTXInstruction*>(*splitPoint);
		
			bool isParamStore = ptx.opcode == ir::PTXInstruction::St &&
				ptx.addressSpace == ir::PTXInstruction::Param &&
				ptx.d.addressMode == ir::PTXOperand::Address;
			bool isParamLoad = ptx.opcode == ir::PTXInstruction::Ld &&
				ptx.addressSpace == ir::PTXInstruction::Param &&
				ptx.a.addressMode == ir::PTXOperand::Address;
			
			if(ptx.opcode != ir::PTXInstruction::Call && !isParamStore
				&& !isParamLoad)
			{
				split = true;
				break;
			}
		}

		if(split)
		{
			block = cfg.split_block(block, splitPoint, ir::Edge::FallThrough);
		}
		else
		{
			break;
		}
	}
}

void SplitBasicBlockPass::initialize(const ir::Module& m)
{

}

void SplitBasicBlockPass::runOnKernel(ir::IRKernel& k)
{
	for(ir::ControlFlowGraph::iterator block = k.cfg()->begin();
		block != k.cfg()->end(); ++block)
	{
		splitBlock(*k.cfg(), block, _maxSize);
	}
}

void SplitBasicBlockPass::finalize()
{

}

void SplitBasicBlockPass::setMaximumBlockSize(unsigned int s)
{
	_maxSize = s;
}

}

