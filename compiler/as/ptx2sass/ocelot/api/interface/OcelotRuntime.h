/*! \file OcelotRuntime.h
	\author Gregory Diamos
	\date Tuesday August 11, 2009
	\brief The header file for the OcelotRuntime class.
*/

#ifndef OCELOT_RUNTIME_H_INCLUDED
#define OCELOT_RUNTIME_H_INCLUDED

#include <ocelot/api/interface/OcelotConfiguration.h>

#include <ocelot/trace/interface/MemoryChecker.h>
#include <ocelot/trace/interface/MemoryRaceDetector.h>
#include <ocelot/trace/interface/InteractiveDebugger.h>
#include <ocelot/trace/interface/KernelTimer.h>

#include <ocelot/transforms/interface/StructuralTransform.h>
#include <ocelot/transforms/interface/ConvertPredicationToSelectPass.h>
#include <ocelot/transforms/interface/LinearScanRegisterAllocationPass.h>
#include <ocelot/transforms/interface/MIMDThreadSchedulingPass.h>
#include <ocelot/transforms/interface/SyncEliminationPass.h>
#include <ocelot/transforms/interface/HoistSpecialValueDefinitionsPass.h>
#include <ocelot/transforms/interface/SimplifyControlFlowGraphPass.h>
#include <ocelot/transforms/interface/EnforceLockStepExecutionPass.h>
#include <ocelot/transforms/interface/FunctionInliningPass.h>

namespace ocelot
{
	/*! \brief This is an interface for managing state associated with Ocelot */
	class OcelotRuntime	{
	private:
		trace::MemoryChecker _memoryChecker;
		trace::MemoryRaceDetector _raceDetector;
		trace::InteractiveDebugger _debugger;
		trace::KernelTimer _kernelTimer;

		transforms::StructuralTransform _structuralTransform;
		transforms::ConvertPredicationToSelectPass _predicationToSelect;
		transforms::LinearScanRegisterAllocationPass _linearScanAllocation;
		transforms::MIMDThreadSchedulingPass _mimdThreadScheduling;
		transforms::SyncEliminationPass _syncElimination;
		transforms::HoistSpecialValueDefinitionsPass _hoistSpecialValues;
		transforms::SimplifyControlFlowGraphPass _simplifyCFG;
		transforms::EnforceLockStepExecutionPass _enforceLockStepExecution;
		transforms::FunctionInliningPass _inliner;
		
		bool _initialized;
		
	public:
		//! \brief initializes Ocelot runtime state
		OcelotRuntime();
	
		//! \brief initializes the Ocelot runtime object with the 
		//         Ocelot configuration object
		void configure( const api::OcelotConfiguration &c );
					
	};
}

#endif

