/*!
 * \file SASSStatement.cpp
 */

#include <ocelot/ir/interface/SASSStatement.h>
#include <ocelot/ir/interface/SASSInstruction.h>

namespace ir {
	SASSStatement::SASSStatement(Type type,
		SASSInstruction::Opcode op) : type(type), instr(op) {
	}

	SASSStatement::~SASSStatement() {
	}
	
	std::string SASSStatement::toString(Type t) {
		switch(t) {
		case Entry:
			return "Entry";
		case Param:
			return "Param";
		case Instr:
			return "Instr";
		default:
			return "*UNKNOWN_STATEMENT*";
		}
	}

	std::string SASSStatement::toString() const {
		if (type == Instr) {
			return instr.toString();
		} else {
			return toString(type);
		}
	}

	const SASSStatement& SASSStatement::operator=(
		const SASSStatement& s) {
		type = s.type;
		instr = *(s.instr.clone());
		return *this;
	}
}
