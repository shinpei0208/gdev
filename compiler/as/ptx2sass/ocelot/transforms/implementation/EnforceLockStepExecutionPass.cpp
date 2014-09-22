/*! \file   EnforceLockStepExecutionPass.cpp
	\date   Wednesday April 18, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the  EnforceLockStepExecutionPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/EnforceLockStepExecutionPass.h>

#include <ocelot/analysis/interface/ThreadFrontierAnalysis.h>
#include <ocelot/analysis/interface/ConvergentRegionAnalysis.h>
#include <ocelot/analysis/interface/DivergenceAnalysis.h>
#include <ocelot/analysis/interface/DominatorTree.h>

#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0 

namespace transforms
{

EnforceLockStepExecutionPass::EnforceLockStepExecutionPass()
: KernelPass({}, "EnforceLockStepExecutionPass")
{

}

void EnforceLockStepExecutionPass::initialize(const ir::Module& m)
{
	
}

void EnforceLockStepExecutionPass::runOnKernel(ir::IRKernel& k)
{
	report("Running Enforce-Lock-Step-Execution-Pass on kernel '"
		<< k.name << "'");
	
}

void EnforceLockStepExecutionPass::finalize()
{
	
}

}


