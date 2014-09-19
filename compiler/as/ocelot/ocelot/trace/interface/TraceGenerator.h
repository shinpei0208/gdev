/*! \file TraceGenerator.h

	\author Andrew Kerr <arkerr@gatech.edu>

	\brief declares a TraceGenerator class to be the base class for application-dependent
		trace generators
*/

#ifndef TRACE_TRACEGENERATOR_H_INCLUDED
#define TRACE_TRACEGENERATOR_H_INCLUDED

// C++ includes
#include <string>
#include <vector>

// forward declare TraceEvent
namespace trace {
	class TraceEvent;
}

// forward declare EmulatedKernel
namespace executive {
	class ExecutableKernel;
}

namespace trace {

	/*!
		Base class for generating traces
	*/
	class TraceGenerator {
	public:
		/*!
			\brief The different possible types of trace formats
		*/
		enum TraceFormat {
			ParallelismTraceFormat,
			BranchTraceFormat,
			MemoryTraceFormat,
			SharedComputationTraceFormat,
			InstructionTraceFormat,
			KernelDimensionsFormat,
			MachineAttributesFormat,
			WatchTraceFormat,
			WarpSynchronousTraceFormat,
			PerformanceBoundTraceFormat,
			ConvergenceTraceFormat,
			BasicBlockCountFormat,
			InvalidTraceFormat
		};
		
	public:
		/*! \brief The name of the database file to store references 
				to kernel traces.
		*/
		std::string database;
	
	public:
		TraceGenerator();
		virtual ~TraceGenerator();

		/*! \brief called when a traced kernel is launched to retrieve some 
				parameters from the kernel
		*/
		virtual void initialize(const executive::ExecutableKernel& kernel);

		/*! \brief Called whenever an event takes place.

			Note, the const reference 'event' is only valid until event() 
			returns
		*/
		virtual void event(const TraceEvent & event);
		
		/*! \brief called when an event is committed
		*/
		virtual void postEvent(const TraceEvent & event);
		
		/*! \brief Called when a kernel is finished. There will be no more 
				events for this kernel.
		*/
		virtual void finish();
	};

	typedef std::vector< TraceGenerator *> TraceGeneratorVector;
}

#endif

