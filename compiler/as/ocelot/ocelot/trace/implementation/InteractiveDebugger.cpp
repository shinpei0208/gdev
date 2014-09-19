/*!	\file InteractiveDebugger.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday June 1, 2010
	\brief The source for the InteractiveDebugger class.
*/

#ifndef INTERACTIVE_DEBUGGER_CPP_INCLUDED
#define INTERACTIVE_DEBUGGER_CPP_INCLUDED

// C++ includes
#include <algorithm>

// Boost includes
#include <boost/tokenizer.hpp>

// Ocelot Includes
#include <ocelot/trace/interface/InteractiveDebugger.h>
#include <ocelot/executive/interface/EmulatedKernel.h>
#include <ocelot/executive/interface/Device.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>
#include <hydrazine/interface/Casts.h>
#include <hydrazine/interface/SystemCompatibility.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

// Standard Library Includes
#include <iostream>

namespace trace
{

InteractiveDebugger::InteractiveDebugger() : alwaysAttach(false)
{

}

void InteractiveDebugger::initialize(const executive::ExecutableKernel& kernel)
{
	_breakNow = alwaysAttach || kernel.name == filter;
	_attached = _breakNow;
	_breakNext = false;
	if(_attached)
	{
		std::cout << "(ocelot-dbg) Attaching debugger to kernel '" 
		<< kernel.name << "'\n";
	}
	_kernel = &kernel;
}

void InteractiveDebugger::event(const TraceEvent& event)
{
	bool memoryError = false;
	for(TraceEvent::U64Vector::const_iterator 
		address = event.memory_addresses.begin(); 
		address != event.memory_addresses.end() && !memoryError; ++address)
	{
		switch(event.instruction->addressSpace)
		{
		case ir::PTXInstruction::Const:
		{
			if(*address + event.memory_size > _kernel->constMemorySize())
			{
				std::cout << "(ocelot-dbg) Constant memory access violation at " 
					<< (void*)*address << "\n";
				memoryError = true;
			}
		}
		break;
		case ir::PTXInstruction::Global:
		{
			if(!_kernel->device->checkMemoryAccess(
				(void*)*address, event.memory_size))
			{
				std::cout << "(ocelot-dbg) Global memory access violation at " 
					<< (void*)*address << "\n";
				memoryError = true;
			}
		}
		break;
		case ir::PTXInstruction::Local:
		{
			if(*address + event.memory_size > _kernel->localMemorySize())
			{
				std::cout << "(ocelot-dbg) Local memory access violation at " 
					<< (void*)*address << "\n";
				memoryError = true;
			}
		}
		break;
		case ir::PTXInstruction::Param:
		{
			bool isArgument = event.instruction->opcode 
				== ir::PTXInstruction::St ? event.instruction->d.isArgument 
				: event.instruction->a.isArgument;
			if(isArgument)
			{
				if(*address + event.memory_size > _kernel->argumentMemorySize())
				{
					std::cout << "(ocelot-dbg) Argument memory access " 
						<< "violation at " << (void*)*address << " (" 
						<< _event.memory_size << " bytes)\n";

					memoryError = true;
				}
			}
			else
			{
				if(*address + event.memory_size
					> _kernel->parameterMemorySize())
				{
					std::cout << "(ocelot-dbg) Parameter memory access " 
						<< "violation at " << (void*)*address << " (" 
						<< _event.memory_size << " bytes)\n";

					memoryError = true;
				}
			}
		}
		break;
		case ir::PTXInstruction::Shared:
		{
			if(*address + event.memory_size > _kernel->totalSharedMemorySize())
			{
				std::cout << "(ocelot-dbg) Shared memory access violation at " 
					<< (void*)*address << "\n";
				memoryError = true;
			}
		}
		break;
		default:
		{
		
		}
		break;	
		}
	}
	
	_testWatchpoints(event);

	if(memoryError && !_attached)
	{
		std::cout << "(ocelot-dbg) Attaching ocelot debugger.\n";
		_attached = true;
	}

	if(!_attached) return;
	
	if(memoryError && !_breakNow)
	{
		std::cout << "(ocelot-dbg) Breaking into program now!\n";
		_breakNow = true;
	}
	
	if(!_breakNow)
	{
		_breakNow = _breakpoints.count(event.PC) != 0;
	}
	
	if(_breakNow)
	{
		_event = event;
		_break();
	}
	
	if (_breakNext) {
		_breakNext = false;
		_breakNow = true;
	}
}

void InteractiveDebugger::finish()
{
	if(_attached)
	{
		std::cout << "(ocelot-dbg) Kernel '" << _kernel->name 
			<< "' finished, detaching.\n";
	}
	_breakpoints.clear();
}

void InteractiveDebugger::_command(const std::string& command)
{
	std::stringstream stream(command);
	
	std::string base;
	stream >> base;
	
	_processCommands = false;
	
	if (base == "")
	{
		_processCommands = true;
	}
	else if(base == "help" || base == "h")
	{
		_help();
		_processCommands = true;
	}
	else if(base == "jump" || base == "j")
	{
		unsigned int PC = 0;
		stream >> PC;
		_jump(PC);
		_processCommands = true;
	}
	else if(base == "remove" || base == "r")
	{
		unsigned int PC = 0;
		stream >> PC;
		_removeBreakpoint(PC);
	}
	else if(base == "break" || base == "b")
	{
		unsigned int PC = 0;
		stream >> PC;
		_setBreakpoint(PC);
	}
	else if (base == "list")
	{
		_listWatchpoints(command);
	}
	else if (base == "backtrace" || base == "bt")
	{
		_backtrace();
	}
	else if (base == "clear")
	{
		_clearWatchpoint(command);
	}
	else if (base == "watch")
	{
		_setWatchpoint(command);
	}	
	else if(base == "print" || base == "p")
	{
		std::string modifier;
		stream >> modifier;
		_processCommands = true;
		if (modifier == "watch") {
			int wp;
			stream >> wp;
			if (wp >= 1 && wp <= (int)_watchpoints.size()) {
				WatchpointList::iterator wpit = _watchpoints.begin();
				for (int i = 1; i < wp && wpit != _watchpoints.end(); i++, ++wpit) {
				}
				if (wpit != _watchpoints.end()) {
					_printWatchpoint(*wpit);
				}
			}
			else {
				std::cout << "watchpoint #" << wp << " out of range\n";
				_breakNow = true;
			}
		}
		else if(modifier == "asm" || modifier == "a")
		{
			unsigned int PC = _event.PC;
			stream >> PC;
			unsigned int count = 10;
			stream >> count;
			_printAssembly(PC, count);
		}
		else if(modifier == "reg" || modifier == "r")
		{
			unsigned int thread = 0;
			stream >> thread;
			unsigned int reg = 0;
			stream >> reg;
			_printRegister(thread, reg);
		}
		else if(modifier == "mask")
		{
			unsigned int reg = 0;
			stream >> reg;
			_printRegisterAsMask(reg);
		}
		else if(modifier == "mem" || modifier == "m")
		{
			ir::PTXU64 address = 0;
			stream >> std::hex;
			stream >> address;
			_printMemory(address);
		}
		else if(modifier == "pc")
		{
			_printPC();
		}
		else if(modifier == "warp" || modifier == "w")
		{
			_printWarpState();
		}
		else if(modifier == "loc" || modifier == "l")
		{
			unsigned int pc = _event.PC;
			stream >> pc;
			_printLocation(pc);
		}
		else
		{
			std::cout << "(ocelot-dbg) Unknown print command '" 
				<< modifier << "'\n";
		}
	}
	else if(base == "step" || base == "s")
	{
		_step();
	}
	else if(base == "continue" || base == "c")
	{
		_continue();
	}
	else if(base == "quit" || base == "q")
	{
		_breakNow = false;
		_attached = false;
	}
	else
	{
		_processCommands = true;
		std::cout << "(ocelot-dbg) Unknown command '" << base << "'\n";
	}
}

void InteractiveDebugger::_help() const
{
	std::cout << "\n";
	std::cout << "  _                ___       _.--.\n";
    std::cout << "  \\`.|\\..----...-'`   `-._.-'_.-'`\n";
    std::cout << "  /  ' `         ,       __.--'   \n";
    std::cout << "  )/' _/     \\   `-_,   /         \n";
    std::cout << "  `-'\" `\"\\_  ,_.-;_.-\\_ ',      \n";
    std::cout << "      _.-'_./   {_.'   ; /        \n";
    std::cout << "     {_.-``-'         {_/         \n";
	std::cout << "\n";
	std::cout << "This is the Ocelot Interactive PTX Debugger\n";
	std::cout << "\n";
	std::cout << " Commands:\n";
	std::cout << "  help     (h) - Print this message.\n";
	std::cout << "  jump     (j) - Jump current warp to the specified PC.\n";
	std::cout << "  remove   (r) - Remove a breakpoint from a specific PC.\n";
	std::cout << "  break    (b) - Set a breakpoint at the specified PC.\n";
	
	std::cout << "  list         - List watchpoints.\n";
	std::cout << "  clear        - Clear location or register from watch list.\n";
	std::cout << "  watch        - Break on accesses to location or register.\n";
	std::cout << "    global address <address> <ptx-type>\n";
	std::cout << "    global address <address> <ptx-type>[elements]\n";
	
	std::cout << "  print (p)  - Print the value of a resource.\n";
	std::cout << "   asm  (a)  - Print instructions near the specified PC.\n";
	std::cout << "   reg  (r)  - Print the value of a register.\n";
	std::cout << "   mask      - Print the of a register as a 1-bit-per-thread mask.\n";
	std::cout << "   mem  (m)  - Print the values near an address.\n";
	std::cout << "   warp (w)  - Print the current warp status.\n";
	std::cout << "   pc        - Print the PC of the current warp.\n";
	std::cout << "   loc  (l)  - Print the nearest CUDA source line.\n";
	std::cout << "   watch <#> - Print the value of a watch point identified by #.\n";
	std::cout << "  step     (s) - Execute the next instruction.\n";
	std::cout << "  continue (c) - Run until the next breakpoint.\n";
	std::cout << "  quit     (q) - Detach the debugger, resume execution.\n";
}

void InteractiveDebugger::_step()
{
	std::cout << "(" << _event.PC << ") - " 
		<< _event.instruction->toString();
	if(_event.instruction->opcode == ir::PTXInstruction::Bra) 
	{
		std::cout << " [target " 
			<< _event.instruction->branchTargetInstruction << "]";
	}
	std::cout << "\n";
}

void InteractiveDebugger::_break()
{
	std::string command;

	_processCommands = true;

	while(_processCommands)
	{
		std::cout << "(ocelot-dbg) ";
		std::getline(std::cin, command);
		_command(command);
	}
}

void InteractiveDebugger::_continue()
{
	_breakNow = false;
}

void InteractiveDebugger::_jump(unsigned int PC)
{
	switch(_kernel->ISA)
	{
	case ir::Instruction::Emulated:
	{
		const executive::EmulatedKernel& constKernel =
			static_cast<const executive::EmulatedKernel&>(*_kernel);
		executive::EmulatedKernel& kernel = 
			const_cast<executive::EmulatedKernel&>(constKernel);
		
		if(PC < kernel.instructions.size())
		{
			kernel.jumpToPC(PC);
			_event.PC = PC;
			_event.instruction = &kernel.instructions[PC];
		}
		else
		{
			std::cout << "(ocelot-dbg) Cannot jump to PC " << PC 
				<< ", kernel only has " 
				<< kernel.instructions.size() << " instructions.\n";
		}
	}
	break;
	default:
	{
		assertM(false, "Jump to PC not implemented for '" 
			<< ir::Instruction::toString(_kernel->ISA) << "' kernels.");
	}
	break;
	}
}

void InteractiveDebugger::_removeBreakpoint(unsigned int PC)
{
	_breakpoints.erase(PC);
}

void InteractiveDebugger::_setBreakpoint(unsigned int PC)
{
	_breakpoints.insert(PC);
}

/*! \brief gets the value of a register */
ir::PTXU64 InteractiveDebugger::_getRegister(unsigned int tid, 
	ir::PTXOperand::RegisterType reg) const {
	
	const executive::EmulatedKernel& kernel = static_cast<
		const executive::EmulatedKernel&>(*_kernel);
	executive::EmulatedKernel::RegisterFile file =
		kernel.getCurrentRegisterFile();
	
	unsigned int threads = _event.blockDim.x *
		_event.blockDim.y * _event.blockDim.z;
	unsigned int registers = file.size() / threads;
	
	return file[tid * registers + reg];
}

void InteractiveDebugger::_printRegister(unsigned int thread, 
	ir::PTXOperand::RegisterType reg) const
{
	switch(_kernel->ISA)
	{
	case ir::Instruction::Emulated:
	{
		const executive::EmulatedKernel& kernel =
			static_cast<const executive::EmulatedKernel&>(*_kernel);
		executive::EmulatedKernel::RegisterFile 
			file = kernel.getCurrentRegisterFile();
		
		unsigned int threads = _event.blockDim.x * _event.blockDim.y 
			* _event.blockDim.z;
		unsigned int registers = file.size() / threads;
		
		unsigned int maxReg = std::min(registers, reg + 10);
		unsigned int maxThread = std::min(threads, thread + 5);

		std::cout << "     ";
		std::cout << std::right;
		for(unsigned int tid = thread; tid < maxThread; ++tid)
		{
			std::stringstream stream;
			stream << "THREAD " << tid;
			std::cout.fill( ' ' );
			std::cout.width( 16 );
			std::cout << stream.str();
			std::cout.width( 1 );
			std::cout.fill( ' ' );
			std::cout << " ";
		}
		std::cout << "\n";
		for(; reg < maxReg; ++reg)
		{
			std::cout << std::left;
			std::cout.width( 1 );
			std::cout.fill( ' ' );
			std::cout << "R";
			std::cout.width( 3 );
			std::cout.fill( ' ' );
			std::cout << reg;
			std::cout.width( 1 );
			std::cout.fill( ' ' );
			std::cout << " ";
			std::cout << std::hex;
			std::cout << std::right;
			for(unsigned int tid = thread; tid < maxThread; ++tid)
			{
				std::cout.width( 16 );
				std::cout.fill( ' ' );
				std::cout << file[tid * registers + reg];
				std::cout.width( 1 );
				std::cout << " ";
			}
			std::cout << std::dec << "\n";
		}
	}
	break;
	default:
	{
		assertM(false, "Print registers not implemented for '" 
			<< ir::Instruction::toString(_kernel->ISA) << "' kernels.");
	}
	break;
	}

}

void InteractiveDebugger::_printRegisterAsMask(
	ir::PTXOperand::RegisterType reg) const
{
	switch(_kernel->ISA)
	{
	case ir::Instruction::Emulated:
	{
		const executive::EmulatedKernel& kernel =
			static_cast<const executive::EmulatedKernel&>(*_kernel);
		executive::EmulatedKernel::RegisterFile 
			file = kernel.getCurrentRegisterFile();
		
		unsigned int threads = _event.blockDim.x * _event.blockDim.y 
			* _event.blockDim.z;
		unsigned int registers = file.size() / threads;

		std::cout << "r" << reg << " as mask: ";
		
		unsigned int count = 0;
		for(unsigned int tid = 0; tid < threads; ++tid)
		{
			std::cout
				<< ((file[tid * registers + reg] & 0x1) == 0x1 ? "1" : "0");
			count += file[tid * registers + reg] & 0x1;
		}

		std::cout << " [" << count << " bits set]\n";
	}
	break;
	default:
	{
		assertM(false, "Print registers as masks not implemented for '" 
			<< ir::Instruction::toString(_kernel->ISA) << "' kernels.");
	}
	break;
	}

}

void InteractiveDebugger::_printMemory(ir::PTXU64 address) const
{
	const executive::Device* device = _kernel->device;

	std::cout.width( 16 );
	std::cout.fill( '0' );
	std::cout << std::right;
	for(unsigned int row = 0; row != 10; ++row)
	{
		std::cout << std::hex;
		std::cout.width( 16 );
		std::cout.fill( '0' );
		std::cout << address;
		std::cout.width( 1 );
		std::cout << " | ";
		for(unsigned int col = 0; col != 5; ++col)
		{
			for(unsigned int byte = 0; byte != 8; ++byte)
			{
				void* byteAddress = (void*) (7 - byte + address);
				if(device->checkMemoryAccess(byteAddress, 1))
				{
					std::cout.width( 2 );
					std::cout.fill( '0' );
					std::cout << ((unsigned int)(*(char*)byteAddress) & 0xff);
				}
				else
				{
					std::cout.width( 2 );
					std::cout << "XX";
				}
			}
			std::cout.width( 1 );
			std::cout << " ";
			address += 8;
		}
		std::cout << std::dec << "\n";
	}

}

void InteractiveDebugger::_printAssembly(unsigned int PC,
	unsigned int count) const
{
	switch(_kernel->ISA)
	{
	case ir::Instruction::Emulated:
	{
		const executive::EmulatedKernel& kernel =
			static_cast<const executive::EmulatedKernel&>(*_kernel);
		
		for(unsigned int pc = PC; 
			pc < std::min(kernel.instructions.size(), (size_t)(PC + count));
			++pc)
		{
			std::cout << "(" << pc << ") - " 
				<< kernel.instructions[pc].toString();
				
			if(kernel.instructions[pc].opcode == ir::PTXInstruction::Bra)
			{
				std::cout << " (target "
					<< kernel.instructions[pc].branchTargetInstruction << ")";
			}
			
			std::cout << "\n";
		}
	}
	break;
	default:
	{
		assertM(false, "Print assembly not implemented for '" 
			<< ir::Instruction::toString(_kernel->ISA) << "' kernels.");
	}
	break;
	}
}

void InteractiveDebugger::_printWarpState() const
{
	std::cout << "CTA ID:              (" << _event.blockId.x << ", " 
		<< _event.blockId.y << ", " << _event.blockId.z << ")\n";
	std::cout << "Warp ID:             " << 0 << "\n";
	std::cout << "PC:                  " << _event.PC << "\n";
	std::cout << "Context Stack Depth: " << _event.contextStackSize << "\n";
	std::cout << "Active Mask:         [" << _event.active << "]\n";
}

void InteractiveDebugger::_printPC() const
{
	std::cout << "(ocelot-dbg) Current PC is " << _event.PC << "\n";
}

void InteractiveDebugger::_printLocation(unsigned int pc) const
{
	switch(_kernel->ISA)
	{
	case ir::Instruction::Emulated:
	{
		const executive::EmulatedKernel& kernel =
			static_cast<const executive::EmulatedKernel&>(*_kernel);
		
		pc = std::min(kernel.instructions.size() - 1, (size_t)(pc));
		
		std::cout << "(ocelot-dbg) Near: " << kernel.location(pc) << "\n";
	}
	break;
	default:
	{
		assertM(false, "Print location not implemented for '" 
			<< ir::Instruction::toString(_kernel->ISA) << "' kernels.");
	}
	break;
	}
}

void InteractiveDebugger::_backtrace() const
{
	switch(_kernel->ISA)
	{
	case ir::Instruction::Emulated:
	{
		const executive::EmulatedKernel& kernel =
			static_cast<const executive::EmulatedKernel&>(*_kernel);
		
		unsigned int frames = kernel.getStackFrameCount();
		
		std::cout << (frames - 1) << " stack frames\n";
		
		for(unsigned int frame = frames; frame > 1; --frame)
		{
			unsigned int pc = _event.PC;
			
			if(frame != frames)
			{
				auto info = kernel.getStackFrameInfo(frame);
			
				pc = info.pc;
			}
			
			auto frameKernel = kernel.getKernelContainingThisPC(pc);
			assert(frameKernel != 0);
		
			std::cout << "frame " << (frames - frame)
				<< ": (PC " << pc << ") ";
			
			if(hydrazine::isMangledCXXString(frameKernel->name))
			{
				std::cout << hydrazine::demangleCXXString(frameKernel->name)
					<< "\n";
			}
			else
			{
				std::cout << frameKernel->name << "\n";
			}
		}
	}
	break;
	default:
	{
		assertM(false, "Backtrace for '" 
			<< ir::Instruction::toString(_kernel->ISA) << "' kernels.");
	}
	break;
	}
}

////////////////////////////////////////////////////////////////////////////////


/*!
	\brief command token
*/
class Token {
public:
	enum Type {
		String,
		Number,
		PtxType,
		VectorType,
		BracketOpen,
		BracketClose,
		Unknown
	};
	
private:
	/*!
		\brief accept either hexadecimal or decimal
	*/
	bool _lexNumber(const std::string &str) {
		if (str.size() >= 3 && str[0] == '0' && str[1] == 'x') {
			valNumber = 0;
			for (size_t n = 2; n < str.size(); n++) {
				char c = str[n];
				size_t d = 0;
				if ( (c >= '0' && c <= '9') ) {
					d = (size_t)(c - '0');
				}
				else if (c >= 'a' && c <= 'f') {
					d = (size_t)(c - 'a') + 10;
				}
				else if (c >= 'A' && c <= 'F') {
					d = (size_t)(c - 'A') + 10;
				}
				else {
					return false;
				}
				valNumber = (valNumber << 4) + d;
			}
			type = Number;
			return true;
		}
		else {
			valNumber = 0;
			for (size_t n = 0; n < str.size(); n++) {
				char c = str[n];
				if (c >= '0' && c <= '9') {
					valNumber = (valNumber * 10) + (size_t)(c - '0');
				}
				else {
					return false;
				}
			}
			type = Number;
			return true;
		}
		return false;
	}
	
	bool _lexPtxType(const std::string &str) {
		ir::PTXOperand::DataType types[] = {
			ir::PTXOperand::s8,
			ir::PTXOperand::u8,
			ir::PTXOperand::b8,
			ir::PTXOperand::s16,
			ir::PTXOperand::u16,
			ir::PTXOperand::b16,
			ir::PTXOperand::s32,

			ir::PTXOperand::u32,
			ir::PTXOperand::b32,
			ir::PTXOperand::f32,
			ir::PTXOperand::s64,
			ir::PTXOperand::u64,
			ir::PTXOperand::b64,
			ir::PTXOperand::f64
		};
		const char *names[] = {
			"s8", "u8", "b8", "s16", "u16", "b16", "s32", "u32",
			"b32", "f32", "s64", "u64", "b64", "f64",0
		};
		for (int n = 0; names[n]; n++) {
			if (str == std::string(names[n])) {
				valPtxType = types[n];
				type = PtxType;
				return true;
			}
		}
		return false;
	}
	
	bool _lexVecType(const std::string &str) {
		ir::PTXOperand::Vec types[] = {
			ir::PTXOperand::v1,
			ir::PTXOperand::v2,
			ir::PTXOperand::v4,
		};
		const char *names[] = {
			"v1", "v2", "v4", 0
		};
		for (int n = 0; names[n]; n++) {
			if (str == std::string(names[n])) {
				valVecType = types[n];
				type = VectorType;
				return true;
			}
		}
		return false;
	}

public:

	Token(const std::string &str) {
		valString = str;
		if (str == "[") {
			type = BracketOpen;
		}
		else if (str == "]") {
			type = BracketClose;
		}
		else if (_lexNumber(str)) {
			
		}
		else if (_lexPtxType(str)) {
		
		}
		else if (_lexVecType(str)) {
		
		}
		else {
			type = String;
		}
	}

	/*!
		\brief given a command, break into a sequence of tokens
	*/
	static std::vector< Token > tokenize(const std::string & str) {
		std::vector< Token > tokenVector;
		boost::char_separator<char> sep(" ", "[]");
		typedef boost::tokenizer<boost::char_separator<char> > Tokenizer;
		Tokenizer tokens(str, sep);
		for (Tokenizer::iterator tok = tokens.begin(); tok != tokens.end(); ++tok) {
			Token token(*tok);
			tokenVector.push_back(token);
		}
		return tokenVector;
	}

public:
	Type type;
	
	std::string valString;
	
	size_t valNumber;
	
	ir::PTXOperand::DataType valPtxType;
	
	ir::PTXOperand::Vec valVecType;
};
typedef std::vector< Token > TokenVector;

////////////////////////////////////////////////////////////////////////////////

std::ostream & operator<< (std::ostream &out,
	const InteractiveDebugger::Watchpoint &watch)
{
	std::stringstream ss;

	switch (watch.type) {	
	case trace::InteractiveDebugger::Watchpoint::Register_location:
	{
		unsigned int reg = 0;
		hydrazine::bit_cast(reg, watch.ptr);
		switch (watch.reference) {
		case trace::InteractiveDebugger::Watchpoint::Symbol:
			ss << "watchpoint register " << watch.symbol;
			break;
		case trace::InteractiveDebugger::Watchpoint::Address:
			ss << "watchpoint register " << reg;
			break;
		default:
			return out;
		}
	}
		break;
	case trace::InteractiveDebugger::Watchpoint::Global_location:
	{
		switch (watch.reference) {
		case trace::InteractiveDebugger::Watchpoint::Symbol:
			ss << "watchpoint global symbol " << watch.symbol;
			break;
		case trace::InteractiveDebugger::Watchpoint::Address:
			ss << "watch global address " << watch.ptr << " " 
				<< ir::PTXOperand::toString(watch.dataType) << "[" << watch.elements << "] - " 
				<< watch.size << " bytes";
			break;
		default:
			return out;
		}
	}
		break;
	case trace::InteractiveDebugger::Watchpoint::Shared_location:
	{
		switch (watch.reference) {
		case trace::InteractiveDebugger::Watchpoint::Symbol:
			ss << "watchpoint shared symbol " << watch.symbol;
			break;
		case trace::InteractiveDebugger::Watchpoint::Address:
			ss << "watchpoint shared address " << watch.ptr;
			break;
		default:
			return out;
		}
	}
		break;
	case trace::InteractiveDebugger::Watchpoint::Param_location:
	{
		switch (watch.reference) {
		case trace::InteractiveDebugger::Watchpoint::Symbol:
			ss << "watchpoint parameter " << watch.symbol;
			break;
		default:
			return out;
		}
	}
		break;
	case trace::InteractiveDebugger::Watchpoint::Local_location:
	{
		switch (watch.reference) {
		case trace::InteractiveDebugger::Watchpoint::Symbol:
			ss << "watchpoint local symbol " << watch.symbol;
			break;
		case trace::InteractiveDebugger::Watchpoint::Address:
			ss << "watchpoint local address " << watch.ptr;
			break;
		default:
			return out;
		}
	}
		break;
	case trace::InteractiveDebugger::Watchpoint::Texture_location:
	{
		switch (watch.reference) {
		case trace::InteractiveDebugger::Watchpoint::Symbol:
			ss << "watchpoint texture symbol " << watch.symbol;
			break;
		case trace::InteractiveDebugger::Watchpoint::Address:
			ss << "watchpoint texture address " << watch.ptr;
			break;
		default:
			return out;
		}
	}
		break;
	default:
		return out;
	}
	
	out << ss.str();
	return out;
}

InteractiveDebugger::Watchpoint::Watchpoint():
	type(Unknown_location),
	reference(Address),
	ptr(0),
	size(0),
	hitType(Any_access),
	breakOnAccess(true),
	reads(0),
	writes(0)
{
}

static void printPtxValue(std::ostream &out, volatile const void *ptr, ir::PTXOperand::DataType dataType) {
	switch (dataType) {
	case ir::PTXOperand::s8:
		out << (int)(*(volatile const ir::PTXS8 *)ptr);
		break;
	case ir::PTXOperand::s16:
		out << *(volatile const ir::PTXS16 *)ptr;
		break;
	case ir::PTXOperand::s32:
		out << *(volatile const ir::PTXS32 *)ptr;
		break;
	case ir::PTXOperand::s64:
		out << *(volatile const ir::PTXS64 *)ptr;
		break;
		
	case ir::PTXOperand::u8:
		out << (int)(*(volatile const ir::PTXU8 *)ptr);
		break;
	case ir::PTXOperand::u16:
		out << *(volatile const ir::PTXU16 *)ptr;
		break;
	case ir::PTXOperand::u32:
		out << *(volatile const ir::PTXU32 *)ptr;
		break;
	case ir::PTXOperand::u64:
		out << *(volatile const ir::PTXU64 *)ptr;
		break;
		
	case ir::PTXOperand::f32:
		out << *(volatile const ir::PTXF32 *)ptr;
		break;
		
	case ir::PTXOperand::f64:
		out << *(volatile const ir::PTXF64 *)ptr;
		break;
		
	default:
		break;
	}
}

//! \brief tests an event against a watch point and returns true if it hit
bool InteractiveDebugger::Watchpoint::test(const trace::TraceEvent &event) {
		
	bool hitLd = (hitType == Any_access || hitType == Read_access);
	bool hitSt = (hitType == Any_access || hitType == Write_access);
	
	if (!(event.instruction->addressSpace == ir::PTXInstruction::Global &&
			((event.instruction->opcode == ir::PTXInstruction::Ld && hitLd) || 
				(event.instruction->opcode == ir::PTXInstruction::St && hitSt))
		)) {
		return false;
	}
	bool result = false;
	switch (type) {
	case Global_location:
		{
			result = _testGlobal(event);
		}
	break;
	default:
		break;
	}
	return result;
}

/*!
	\brief tests for accesses to global memory
*/
bool InteractiveDebugger::Watchpoint::_testGlobal(const trace::TraceEvent &event) {
	
	int hitTid = -1;	// tid of first thread to hit a watchpoint or -1 if not hit
	std::stringstream ss;
	
	trace::TraceEvent::U64Vector::const_iterator addr_itr = event.memory_addresses.begin();
	for (size_t tid = 0; tid < event.active.size(); tid++) {
		if (event.active[tid]) {
			void *addr = 0;
			hydrazine::bit_cast(addr, *addr_itr);

			bool store = (event.instruction->opcode == ir::PTXInstruction::St);
			size_t elementCount = (store ? event.instruction->a.vec : event.instruction->d.vec);
			
			size_t memorySize = event.memory_size * elementCount;
			
			char *regStart = (char *)ptr, *regEnd = (char *)ptr + size;
			char *hitStart = (char *)addr, *hitEnd = (char *)hitStart + memorySize;
			
			if (hitEnd > regStart && hitStart < regEnd) {

				if (hitTid < 0) {
					hitTid = (int)tid;
					hitMessage = "";
					ss << "CTA (" << event.blockId.x << ", " << event.blockId.y << ")" << "\n";
				}
				
				size_t tidz = tid / (event.blockDim.x * event.blockDim.y);
				size_t tidy = ((tid - tidz * (event.blockDim.x * event.blockDim.y)) / event.blockDim.x);
				size_t tidx = tid - (tidy + tidz * event.blockDim.y) * event.blockDim.x;
				
				ss << "  thread (" << tidx << ", " << tidy << ", " << tidz << ") - " 
					<< (store ? "store to " : "load from ")
					<< addr << " " << memorySize << " bytes\n";
				
				for (int i = 0; i * ir::PTXOperand::bytes(dataType) < memorySize; i++) {
					if (!i) {
						ss << "  " << (store ? "old": "   ") << " value = ";
					}
					else {
						ss << "            = ";
					}
					printPtxValue(ss, addr, dataType);
					ss << "\n";
				}	
				if (store) {
					// print the value from the register file
					ir::PTXU64 srcValues[4];
					
					if (event.instruction->a.vec > 1) {
						for (int i = 0; i < event.instruction->a.vec; i++) {
							ir::PTXU64 srcValue = debugger->_getRegister(tid, event.instruction->a.array[i].reg);
							srcValues[i] = srcValue;
						}
					}
					else {
						ir::PTXU64 srcValue = debugger->_getRegister(tid, event.instruction->a.reg);
						srcValues[0] = srcValue;
					}
					for (int i = 0; i * ir::PTXOperand::bytes(dataType) < memorySize; i++) {
						if (!i) {
							ss << "  new value = ";
						}
						else {
							ss << "            = ";
						}
						printPtxValue(ss, srcValues + i, dataType);
						ss << "\n";
					}	
				}			
			}
			++addr_itr;
		}
	}
	if (hitTid >= 0) {
		hitMessage = ss.str();
	}
	return (hitTid >= 0);
}

/*! \brief prints watchpoints according to formatting information from command */
void InteractiveDebugger::_listWatchpoints(const std::string &command) const {
	int n = 1;
	for (WatchpointList::const_iterator w_it = _watchpoints.begin(); 
		w_it != _watchpoints.end(); ++w_it, n++) {
		std::cout << "#" << n << " - " << *w_it << "\n";
	}
}

/*! \brief print the value of the region specified in a watchpoint */
void InteractiveDebugger::_printWatchpoint(const Watchpoint &watch) const {
	if (watch.type == Watchpoint::Global_location && watch.reference == Watchpoint::Address) {
		std::cout << watch << "\n";
		for (size_t i = 0; i < watch.elements; i++) {
			volatile const char *ptr = (const char *)watch.ptr + i * ir::PTXOperand::bytes(watch.dataType);
			std::cout << "  [" << i << "]  " << (void *)ptr << "   ";
			printPtxValue(std::cout, ptr, watch.dataType);
			std::cout << "\n";
		}
	}
}

/*! \brief clears indicated watchpoints

*/
void InteractiveDebugger::_clearWatchpoint(const std::string &command) {
	// clear number
	TokenVector tokens = Token::tokenize(command);
	if (tokens.size() == 1 && tokens[0].valString == "clear") {
		_watchpoints.clear();
	}
	else if (tokens.size() > 1 && tokens[1].type == Token::Number) {
		WatchpointList::iterator wit = _watchpoints.begin();
		for (size_t n = 1; n < tokens[1].valNumber; n++) {
			++wit;
		}
		if (wit != _watchpoints.end()) {
			_watchpoints.erase(wit);
		}
		else {
			std::cout << "undefined watchpoint\n";
			_listWatchpoints(command);
		}
	}
	else {
		std::cout 
			<< "expected:\n  clear\n"
			<< "  clear <watchpoint number>\n";
	}
}

/*! \brief sets a watchpoint */
void InteractiveDebugger::_setWatchpoint(const std::string &command) {
	//
	// future features:
	//
	// watch register {symbol|number} {|read|write} {|break|nobreak} {thread number}
	// watch global {symbol|address {ptx-type|number}} {|read|write} {|break|nobreak} {thread number}
	// watch shared {symbol|address {ptx-type|number}} {|read|write} {|break|nobreak} {thread number}
	// watch local {symbol|address {ptx-type|number}} {|read|write} {|break|nobreak} {thread number}
	// watch texture {symbol|address vector-type ptx-type} {|read|write} {|break|nobreak} {thread number}
	// watch parameter {symbol|address ptx-type} {|read|write} {|break|nobreak} {thread number}
	
	TokenVector tokens = Token::tokenize(command);
	
	if (tokens.size() < 3) {
		std::cout << "expected: watch global address <.global address> <ptx-type> - no watchpoint set\n";
		return;
	}
	
	Watchpoint watch;
	
	if (tokens[1].valString == "global") {
		
		if (tokens.size() >= 4 &&
			tokens[2].valString == "address") {

			watch.ptr = (void *)tokens[3].valNumber;
			watch.reference = Watchpoint::Address;
		}
		else if (tokens.size() >= 4 && tokens[2].valString == "symbol") {
			std::cout << "watch global symbol not yet implemented\n";
			return;
		}
		else {
			std::cout << "unrecognized reference type '" << tokens[2].valString 
				<< "' - only 'address' and 'symbol' supported\n";
			return;
		}
			
		if (!(tokens.size() >= 5 &&
			tokens[3].type == Token::Number &&
			tokens[4].type == Token::PtxType) ) {
			
			std::cout << "expected: watch global address <.global address> <ptx-type> - no watchpoint set\n";
			return;
		}
			
		watch.type = Watchpoint::Global_location;
		watch.dataType = tokens[4].valPtxType;
		watch.threadId = -1;
		watch.hitType = Watchpoint::Write_access;
		watch.breakOnAccess = true;
		watch.debugger = this;
		
		if (tokens.size() >= 8 &&
			tokens[5].type == Token::BracketOpen &&
			tokens[6].type == Token::Number &&
			tokens[7].type == Token::BracketClose) {
			//
			// watch global allocation <address> <ptx-type>[<number>] - 8 tokens
			//
			watch.elements = tokens[6].valNumber;
		}
		else {
			//
			// watch global allocation <address> <ptx-type>	
			//
			watch.elements = 1;
		}
		watch.size = watch.elements * ir::PTXOperand::bytes(watch.dataType);
		
		_watchpoints.push_back(watch);
		
		std::cout << "set #" << _watchpoints.size() << ": " << watch << "\n";
	}
	else {
		std::cout << "only global addresses may be watched at this time - no watchpoint set\n";
		std::cout << "expected: watch global address <.global address> <ptx-type> - no watchpoint set\n";
	}
}

/*! \brief finds the watch points triggered by the event and prints their values */
void InteractiveDebugger::_testWatchpoints(const trace::TraceEvent &event) {
	
	const executive::EmulatedKernel& kernel =
		static_cast<const executive::EmulatedKernel&>(*_kernel);
		
	int n = 1;
	int hittingWPs = 0;
	
	for (WatchpointList::iterator w_it = _watchpoints.begin(); 
		w_it != _watchpoints.end(); ++w_it, n++) {
		
		if (w_it->test(event)) {
			if (!hittingWPs) {
				std::cout << kernel.instructions[event.PC].toString() << "\n";
			}
			++hittingWPs;
			std::cout << "watchpoint #" << n << " - " << " " << w_it->hitMessage;
			
			if (w_it->breakOnAccess) {
				_breakNow = true;
			}
		}
	}
	if (hittingWPs && _breakNow) {
		std::cout << "break on watchpoint\n";
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////

}

#endif

