/*! \brief  LoopUnrollingPass.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Wednesday May 2, 2012
	\brief  The source file for the LoopUnrollingPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/LoopUnrollingPass.h>

#include <ocelot/analysis/interface/LoopAnalysis.h>

// Standard Library Includes
#include <cassert>

namespace transforms
{

LoopUnrollingPass::LoopUnrollingPass()
: KernelPass({"LoopAnalysis"}, "LoopUnrollingPass")
{
	
}

void LoopUnrollingPass::initialize(const ir::Module& m)
{

}

void LoopUnrollingPass::runOnKernel(ir::IRKernel& k)
{
	Analysis* loopAnalysis = getAnalysis("LoopAnalysis");
	assert(loopAnalysis != 0);
	
	// TODO actually unroll something.
}

void LoopUnrollingPass::finalize()
{

}

}


