/*! \file   PassFactory.cpp
	\date   Wednesday May 2, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the PassFactory class.
	
*/

// Ocelot Includes
#include <ocelot/transforms/interface/PassFactory.h>

#include <ocelot/transforms/interface/LinearScanRegisterAllocationPass.h>
#include <ocelot/transforms/interface/RemoveBarrierPass.h>
#include <ocelot/transforms/interface/StructuralTransform.h>
#include <ocelot/transforms/interface/ConvertPredicationToSelectPass.h>
#include <ocelot/transforms/interface/SubkernelFormationPass.h>
#include <ocelot/transforms/interface/MIMDThreadSchedulingPass.h>
#include <ocelot/transforms/interface/DeadCodeEliminationPass.h>
#include <ocelot/transforms/interface/SplitBasicBlockPass.h>
#include <ocelot/transforms/interface/SyncEliminationPass.h>
#include <ocelot/transforms/interface/HoistSpecialValueDefinitionsPass.h>
#include <ocelot/transforms/interface/LoopUnrollingPass.h>
#include <ocelot/transforms/interface/SimplifyControlFlowGraphPass.h>
#include <ocelot/transforms/interface/EnforceLockStepExecutionPass.h>
#include <ocelot/transforms/interface/FunctionInliningPass.h>
#include <ocelot/transforms/interface/SimplifyExternalCallsPass.h>
#include <ocelot/transforms/interface/GlobalValueNumberingPass.h>
#include <ocelot/transforms/interface/ConstantPropagationPass.h>
#include <ocelot/transforms/interface/SharedPtrAttribute.h>
#include <ocelot/transforms/interface/HoistParameterLoadsPass.h>
#include <ocelot/transforms/interface/MoveEliminationPass.h>

// Standard Library Includes
#include <stdexcept>

namespace transforms
{

Pass* PassFactory::createPass(const std::string& name)
{
	if( name == "linearscan" )
	{
		return new transforms::LinearScanRegisterAllocationPass;
	}
	else if( name == "remove-barriers" || name == "RemoveBarrierPass" )
	{
		return new transforms::RemoveBarrierPass;
	}
	else if( name == "reverse-if-conversion" ||
		name == "ConvertPredicationToSelectPass" )
	{
		return new transforms::ConvertPredicationToSelectPass;
	}
	else if( name == "structural-transform" || name == "StructuralTransform" )
	{
		return new transforms::StructuralTransform;
	}
	else if( name == "subkernel-formation" || name == "SubkernelFormationPass" )
	{
		return new transforms::SubkernelFormationPass;
	}
	else if( name == "mimd-threading" || name == "MIMDThreadSchedulingPass" )
	{
		return new transforms::MIMDThreadSchedulingPass;
	}
	else if( name == "dead-code-elimination" ||
		name == "DeadCodeEliminationPass" )
	{
		return new transforms::DeadCodeEliminationPass;
	}
	else if( name == "split-blocks" || name == "SplitBasicBlockPass" )
	{
		return new transforms::SplitBasicBlockPass;
	}
	else if( name == "sync-elimination" || name == "SyncEliminationPass" )
	{
		return new transforms::SyncEliminationPass;
	}
	else if( name == "hoist-special-definitions" ||
		name == "HoistSpecialValueDefinitionsPass" )
	{
		return new transforms::HoistSpecialValueDefinitionsPass;
	}
	else if( name == "simplify-cfg" || name == "SimplifyControlFlowGraphPass" )
	{
		return new transforms::SimplifyControlFlowGraphPass;
	}
	else if( name == "loop-unrolling" || name == "LoopUnrollingPass" )
	{
		return new transforms::LoopUnrollingPass;
	}
	else if( name == "lock-step" || name == "EnforceLockStepExecutionPass" )
	{
		return new transforms::EnforceLockStepExecutionPass;
	}
	else if( name == "function-inlining" || name == "FunctionInliningPass" )
	{
		return new transforms::FunctionInliningPass;
	}
	else if( name == "simplify-calls" || name == "SimplifyExternalCallsPass" )
	{
		return new transforms::SimplifyExternalCallsPass;
	}
	else if( name == "global-value-numbering"
		|| name == "GlobalValueNumberingPass" )
	{
		return new transforms::GlobalValueNumberingPass;
	}
	else if (name == "shared-ptr-attribute")
	{
		return new transforms::SharedPtrAttribute;
	}
	else if (name == "constant-propagation" ||
		name == "ConstantPropagationPass")
	{
		return new transforms::ConstantPropagationPass;
	}
	else if (name == "hoist-parameters" || name == "HoistParameterLoadsPass")
	{
		return new transforms::HoistParameterLoadsPass;
	}
	else if (name == "move-elimination" || name == "MoveEliminationPass")
	{
		return new transforms::MoveEliminationPass;
	}
	else
	{
		throw std::runtime_error("Invalid pass name " + name);
	}
	
	return 0;
}

}


