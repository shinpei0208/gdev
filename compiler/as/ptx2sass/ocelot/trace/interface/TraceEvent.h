/*! \file TraceEvent.h
	\author Andrew Kerr <arkerr@gatech.edu>
	\brief declares TraceEvent class for recording runtime information
*/

#ifndef TRACE_TRACEEVENT_H_INCLUDED
#define TRACE_TRACEEVENT_H_INCLUDED

#include <boost/dynamic_bitset.hpp>
#include <ocelot/ir/interface/Dim3.h>
#include <ocelot/ir/interface/PTXInstruction.h>
#include <deque>

/*! \brief A namespace for trace generation related classes */
namespace trace {

	/*!
		\brief trace events concerned specifically with thread divergence and reconvergence
	*/
	class ReconvergenceTraceEvent {
	public:
		ReconvergenceTraceEvent();
		
		/*!
			\brief resets instruction-specific events to their 'off' state
		*/
		void reset();
		
	public:
		ir::PTXU32 stackVisitNodes;

		ir::PTXU32 stackVisitEnd;
		
		ir::PTXU32 stackVisitMiddle;
		
		ir::PTXU32 stackInsert;
		
		ir::PTXU32 stackMerge;
		
		bool conservativeBranch;
	};

	class TraceEvent {
	public:
		typedef std::vector< ir::PTXU64 > U64Vector;
		typedef boost::dynamic_bitset<> BitMask;
		
	public:
		/*! Default constructor */
		TraceEvent();

		/*!
			Constructs a trace event object

			\param blockId ID of block that generated the event
			\param PC index into EmulatedKernel's packed instruction sequence
			\param instruction const reference to instruction pointed to by PC
			\param active bit mask of active threads that executed this 
				instruction
			\param memory_addresses vector of memory addresses possibly 
				generated for this instruction
			\param memory_sizes vector of sizes of memory operations possibly 
				issued by this instruction
		*/
		TraceEvent(
			ir::Dim3 blockId,
			ir::PTXU64 PC, 
			const ir::PTXInstruction* instruction, 
			const boost::dynamic_bitset<> & active,
			const U64Vector & memory_addresses,
			ir::PTXU32 memory_size,
			ir::PTXU32 ctxStackSize = 1);

		/*!
			\brief resets instruction-specific events to their 'off' state
		*/
		void reset();

	public:

		/*!
			ID of the block that generated the event
		*/
		ir::Dim3 blockId;

		/*!
			PC index into EmulatedKernel's packed instruction sequence
		*/
		ir::PTXU64 PC;

		/*!
			Depth of call stack [i.e. number of contexts on the runtime stack]
		*/
		ir::PTXU32 contextStackSize;

		/*!
			instruction const pointer to instruction pointed to by PC
		*/
		const ir::PTXInstruction* instruction;

		/*!
			bit mask of active threads that executed this instruction
		*/
		BitMask active;
		
		/*!
			\brief Taken thread mask in case of a branch
		*/
		BitMask taken;
		
		/*!
			\brief Fall through thread mask in case of a branch
		*/
		BitMask fallthrough;

		/*!
			vector of memory addresses possibly generated for this instruction - only contains
			actually accessed memory addresses, so the nth element in memory_addresses corresponds to
			the nth 1 in the active thread mask
		*/
		U64Vector memory_addresses;

		/*!
			vector of sizes of memory operations possibly issued by this 
				instruction
		*/
		ir::PTXU32 memory_size;

		/*!
			dimensions of the kernel grid that generated the event
		*/
		ir::Dim3 gridDim;


		/*!
				dimensions of the kernel block that generated the event
		*/
		ir::Dim3 blockDim;
		
		/*!
			\brief event capturing just events related to thread divergence and reconvergence
		*/
		ReconvergenceTraceEvent reconvergence;

	public:
	
		/*!
			Convert to string
		*/
		std::string toString() const;

	};

}

#endif

