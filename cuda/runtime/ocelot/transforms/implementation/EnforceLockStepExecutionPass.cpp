/*! \file   EnforceLockStepExecutionPass.cpp
	\date   Wednesday April 18, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the  EnforceLockStepExecutionPass class.
*/

// Ocelot Includes
#include <ocelot/transforms/interface/EnforceLockStepExecutionPass.h>

#include <ocelot/analysis/interface/ThreadFrontierAnalysis.h>
#include <ocelot/analysis/interface/ConvergentRegionAnalysis.h>
#include <ocelot/analysis/interface/DominatorTree.h>

#include <ocelot/ir/interface/IRKernel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace transforms
{

EnforceLockStepExecutionPass::EnforceLockStepExecutionPass()
: KernelPass(Analysis::ThreadFrontierAnalysis |
	Analysis::DataflowGraphAnalysis | Analysis::ConvergentRegionAnalysis |
	Analysis::DominatorTreeAnalysis,
	"EnforceLockStepExecutionPass")
{

}

void EnforceLockStepExecutionPass::initialize(const ir::Module& m)
{

}

void EnforceLockStepExecutionPass::runOnKernel(ir::IRKernel& k)
{
	
}

void EnforceLockStepExecutionPass::finalize()
{
	
}

EnforceLockStepExecutionPass::StringVector
	EnforceLockStepExecutionPass::getDependentPasses() const
{
	return StringVector(1, "SimplifyControlFlowGraphPass");	
}

}


