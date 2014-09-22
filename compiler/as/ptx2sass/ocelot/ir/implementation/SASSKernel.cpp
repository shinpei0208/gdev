/*!
 * \file SASSKernel.cpp
 */

#include <iostream>
#include <sstream>
#include <iomanip>
#include <ocelot/ir/interface/Module.h>
#include <ocelot/ir/interface/PTXKernel.h>
#include <ocelot/ir/interface/SASSKernel.h>
#include <ocelot/ir/interface/SASSStatement.h>
#include <ocelot/ir/interface/SASSInstruction.h>

#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif
#define REPORT_BASE 0

namespace ir {
	SASSKernel::Param::Param(unsigned int o, unsigned int s) :
		offset(o), size(s) {
	}

	SASSKernel::SASSKernel() {
		ISA = Instruction::SASS;
		_labels.clear();
	}

	SASSKernel::SASSKernel(const IRKernel &k) : IRKernel(k) {
		ISA = Instruction::SASS;
		const Module* module = k.module;
		std::vector<std::string> targets = module->target().targets;
		if (targets.size() >= 1) {
			_arch = targets[0];
		}
		_mach = module->addressSize();
		_labels.clear();
		_shared = 0;
		_shared_gmap.clear();
		_shared_lmap.clear();
	}

	void SASSKernel::assemble() {
		_code.clear();
		_code += "!Arch " + _arch + "\n";
		_code += "!Machine " + std::to_string(_mach) + "\n";
		_code += "!Kernel " + name + "\n";
		if (_shared > 0) {
			_code += "!Shared " + std::to_string(_shared) + "\n";
		}
		_code += "!Param E ";
		for (std::map<std::string,SASSKernel::Param>::const_iterator p =
			_params.begin(); p != _params.end(); p++) {
			_code += std::to_string(p->second.size) + " ";
		}
		_code += "\n";
		int addr = 0;
		for (std::vector<SASSStatement>::const_iterator s =
			_statements.begin(); s != _statements.end();
			s++, addr += instrBytes()) {
			SASSStatement state = static_cast<SASSStatement>(*s);
			if (state.type != SASSStatement::Instr) {
				continue;
			}
			std::stringstream ss;
			ss << "/*" << std::hex << std::setw(4) <<
				std::setfill('0') << addr << "*/ " <<
				state.instr.toString(&_labels) << std::endl;
			_code += ss.str();
		}
		_code += "!EndKernel\n";
	}

	const std::string& SASSKernel::code() const {
		return _code;
	}

	void SASSKernel::addStatement(SASSStatement s) {
		_statements.push_back(s);
	}

	void SASSKernel::setParameters(const PTXKernel &k) {
		unsigned int total = 0;
		report("SASSKernel setParameters:");
		_params.clear();
		for (ir::PTXKernel::Prototype::ArgumentVector::const_iterator it = k.arguments.begin();
			it != k.arguments.end(); it++) {
			SASSKernel::Param param;
			int size = it->getSize();
			total = (total + size - 1) & ~(size - 1);
			param.offset = total;
			param.size = size;
			report("  add: name=" << it->name <<
				", offset=" << total << ", size=" << size);
			_params[it->name] = param;
			total += size;
		}
	}

	unsigned int SASSKernel::getParameterOffset(std::string name) {
		return _params[name].offset;
	}

	unsigned int SASSKernel::getParameterSize(std::string name) {
		return _params[name].size;
	}

	bool SASSKernel::isParameter(std::string name) {
		return _params.find(name) != _params.end();
	}

	void SASSKernel::setLabel(std::string name) {
		_labels[name] = address();
	}

	void SASSKernel::setSharedMemory(const PTXKernel *k) {
		_shared_gmap.clear();
		_shared_lmap.clear();
		_shared = k->getSharedMemoryLayout(_shared_gmap, _shared_lmap);
	}

	unsigned int SASSKernel::getLocalSharedMemory(std::string name) {
		std::map<std::string, unsigned int>::const_iterator it =
			_shared_lmap.find(name);
		if (it != _shared_lmap.end()) {
			return it->second;
		} else {
			return (unsigned int)-1;
		}
	}

	std::string SASSKernel::toString() const {
		std::string ret = name + ":";
		for (std::vector<SASSStatement>::const_iterator s =
			_statements.begin(); s != _statements.end(); s++) {
			ret += "[" +
				(static_cast<SASSStatement>(*s)).toString() +
				"]";
		}
		return ret;
	}

	unsigned int SASSKernel::size() {
		return _statements.size();
	}

	unsigned int SASSKernel::instrBytes() {
		return _mach/8;
	}

	unsigned int SASSKernel::address() {
		return size()*instrBytes();
	}
}
