/*!	\file InteractiveDebugger.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday June 1, 2010
	\brief The header for the InteractiveDebugger class.
*/

#ifndef INTERACTIVE_DEBUGGER_H_INCLUDED
#define INTERACTIVE_DEBUGGER_H_INCLUDED

// Ocelot Includes
#include <ocelot/trace/interface/TraceGenerator.h>
#include <ocelot/trace/interface/TraceEvent.h>
#include <ocelot/ir/interface/PTXOperand.h>

// Standard Library Includes
#include <unordered_set>
#include <list>

namespace executive
{
	class Device;
}

namespace trace
{
	class InteractiveDebugger;
	
	/*! \brief A heavy-weight tool for debugging the emulator.
	
		This should provide an interface similar to the gdb command line for 
		the emulator, or anything else that exports the trace generator 
		interface.
	*/
	class InteractiveDebugger : public TraceGenerator
	{	
	public:
	
		/*!
			\brief watchpoint datastructure
		*/
		class Watchpoint {
		public:
		
			enum Type {
				Register_location,
				Global_location,
				Shared_location,
				Param_location,
				Texture_location,
				Local_location,
				Unknown_location,			
			};
			
			enum Reference {
				Address,
				Symbol,
				Unknown_reference
			};
			
			enum HitType {
				Any_access,
				Read_access,
				Write_access,
				Unknown_access
			};
			
		public:
			Watchpoint();
			
			//! \brief tests an event against a watch point and returns true if it hit
			bool test(const trace::TraceEvent &event);
			
		private:
		
			bool _testGlobal(const trace::TraceEvent &event);
			
		public:
		
			//! \brief state space of watchpoint
			Type type;
			
			//! \brief reference type of watchpoint
			Reference reference;
		
			//! \brief symbol name (assumes reference == Symbol)
			std::string symbol;
			
			//! \brief indicates type of data
			ir::PTXOperand::DataType dataType;
			
			//! \brief location in memory
			void *ptr;
			
			//! \brief number of dataType elements in watched region
			size_t elements;
			
			//! \brief size of watched region in bytes
			size_t size;
			
			//! \brief filters on a particular thread
			int threadId;
			
			//! \brief indicates whether to break on read, write, or any access to watchpoint
			HitType hitType;
			
			//! \brief if true, watchpoint is a breakpoint
			bool breakOnAccess;
			
			//! \brief number of times a thread reads the watchpoint
			size_t reads;
			
			//! \brief number of times a thread writes to the watchpoint
			size_t writes;
			
			//! \brief debugger
			InteractiveDebugger *debugger;
		
		public:
		
			//! \brief message recorded on most recent hit of watchpoint
			std::string hitMessage;
		};
		
		typedef std::list< Watchpoint > WatchpointList;
	
	public:
		std::string filter;
		bool alwaysAttach;
	
	public:
		InteractiveDebugger();
		void initialize(const executive::ExecutableKernel& kernel);
		void event(const TraceEvent& event);
		void finish();

	private:
		/* \brief A set of program counters */
		typedef std::unordered_set<unsigned int> ProgramCounterSet;
	
	private:	
		/*! \brief A pointer to the kernel being debugged */
		const executive::ExecutableKernel* _kernel;
		/*! \brief A set of program counters with active breakpoints */
		ProgramCounterSet _breakpoints;
		/*! \brief Should the debugger break on the next instruction */
		bool _breakNow;
		/*! \brief Should the compiler break after the current instruction */
		bool _breakNext;
		/*! \brief Should the debugger continue processing commands */
		bool _processCommands;
		/*! \brief Is the debugger attached? */
		bool _attached;
		/*! \brief The current trace event */
		TraceEvent _event;
		/*! \brife list of watchpoints */
		WatchpointList _watchpoints;

	private:
		/*! \brief Parse a command */
		void _command(const std::string& command);
		/*! \brief Print out a help message */
		void _help() const;

	private:
		/*! \brief Step to the next instruction */
		void _step();
		/*! \brief Break out of execution of the program into the debugger */
		void _break();
		/*! \brief Continue execution */
		void _continue();
	
	private:
		/*! \brief Jump to a specific PC */
		void _jump(unsigned int PC);
		/*! \brief Remove a breakpoint at a specific PC */
		void _removeBreakpoint(unsigned int PC);
		/*! \brief Set a breakpoint at a specific PC */
		void _setBreakpoint(unsigned int PC);
public:
		/*! \brief gets the value of a register */
		ir::PTXU64 _getRegister(unsigned int thread,
			ir::PTXOperand::RegisterType reg) const;
public:
		/*! \brief View the value of a register */
		void _printRegister(unsigned int thread,
			ir::PTXOperand::RegisterType reg) const;
		/*! \brief View the value of a register */
		void _printRegisterAsMask(ir::PTXOperand::RegisterType reg) const;
		/*! \brief prints watchpoints according to formatting information
			from command */
		void _listWatchpoints(const std::string &command) const;
		/*! \brief print the value of the region specified in a watchpoint */
		void _printWatchpoint(const Watchpoint &watch) const;
		/*! \brief */
		void _clearWatchpoint(const std::string &command);
		/*! \brief sets a watchpoint */
		void _setWatchpoint(const std::string &command);
		/*! \brief finds the watch points triggered by the event and
			prints their values */
		void _testWatchpoints(const trace::TraceEvent &event);
		/*! \brief View values in memory near the specified device address */
		void _printMemory(ir::PTXU64 address) const;
		/*! \brief View the kernel assembly code near the specified address */
		void _printAssembly(unsigned int pc, unsigned int count) const;
		/*! \brief Print the internal state of the currently executing warp */
		void _printWarpState() const;
		/*! \brief Print the PC of the currently executing warp */
		void _printPC() const;
		/*! \brief Print the location of the nearest source code line */
		void _printLocation(unsigned int pc) const;
		/*! \brief Print the set of functions on the call stack  */
		void _backtrace() const;
		
	};
}


#endif

