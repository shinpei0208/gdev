/*! \file IRKernel.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date Tuesday January 22, 2011
	\brief The source file for the IRKernel class.
*/

#ifndef IR_KERNEL_CPP_INCLUDED
#define IR_KERNEL_CPP_INCLUDED

#include <ocelot/ir/interface/IRKernel.h>
#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/ControlFlowGraph.h>

#include <hydrazine/interface/Version.h>
#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

ir::IRKernel::IRKernel(Instruction::Architecture isa, const std::string& n, 
	bool isFunction, const ir::Module* m, Id id)
: Kernel(isa, n, isFunction, m, id), _cfg(0) {

}

ir::IRKernel::~IRKernel() {
	delete _cfg;
}

ir::IRKernel::IRKernel(const IRKernel &kernel) : Kernel(kernel) {
	// deep copy the elements from a kernel to this one

	_cfg = new ControlFlowGraph(this);
	*_cfg = *kernel._cfg;
}

const ir::IRKernel& ir::IRKernel::operator=(const IRKernel &kernel) {
	// deep copy the elements from a kernel to this one
	if( &kernel == this ) return *this;
	
	Kernel::operator=(kernel);
	
	delete _cfg;
	
	_cfg = new ControlFlowGraph(this);
	*_cfg = *kernel._cfg;

	return *this;	
}

ir::ControlFlowGraph* ir::IRKernel::cfg() {
	return _cfg;
}

const ir::ControlFlowGraph* ir::IRKernel::cfg() const {
	return _cfg;
}

bool ir::IRKernel::executable() const {
	return false;
}

std::string ir::IRKernel::getLocationString(
	const Instruction& instruction) const {

	if(instruction.ISA != ir::Instruction::PTX) return "";

	assert(module != 0);

	auto ptx = static_cast<const ir::PTXInstruction&>(instruction);

	unsigned int statement = ptx.statementIndex;
	
	auto s_it = module->statements().begin();
	std::advance(s_it, statement);
	auto s_rit = ir::Module::StatementVector::const_reverse_iterator(s_it);
	unsigned int program = 0;
	unsigned int line = 0;
	unsigned int col = 0;
	for ( ; s_rit != module->statements().rend(); ++s_rit) {
		if (s_rit->directive == ir::PTXStatement::Loc) {
			line = s_rit->sourceLine;
			col = s_rit->sourceColumn;
			program = s_rit->sourceFile;
			break;
		}
	}
	
	std::string fileName;
	for ( s_it = module->statements().begin(); 
		s_it != module->statements().end(); ++s_it ) {
		if (s_it->directive == ir::PTXStatement::File) {
			if (s_it->sourceFile == program) {
				fileName = s_it->name;
				break;
			}
		}
	}
	
	std::stringstream stream;
	stream << fileName << ":" << line << ":" << col;
	return stream.str();

}

#endif

