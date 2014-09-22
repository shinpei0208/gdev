/*!
 * \file SASSInstruction.cpp
 */

#include <iostream>
#include <ocelot/ir/interface/SASSInstruction.h>
#include <ocelot/translator/interface/PTXToSASSTranslator.h>

namespace ir {

	SASSInstruction::SASSInstruction(Opcode op) : opcode(op) {
	}

	std::string SASSInstruction::toString(Opcode opcode) {
		switch(opcode) {
		case Atom:
			return "ATOM";
		case B2r:
			return "B2R";
		case Bar:
			return "BAR";
		case Bfe:
			return "BFE";
		case Bfi:
			return "BFI";
		case Bpt:
			return "BPT";
		case Bra:
			return "BRA";
		case Brk:
			return "BRK";
		case Brx:
			return "BRX";
		case Cal:
			return "CAL";
		case Cctl:
			return "CCTL";
		case Cctll:
			return "CCTLL";
		case Cont:
			return "CONT";
		case Cset:
			return "CSET";
		case Csetp:
			return "CSETP";
		case Dadd:
			return "DADD";
		case Dfma:
			return "DFMA";
		case Dmnmx:
			return "DMNMX";
		case Dmul:
			return "DMUL";
		case Dset:
			return "DSET";
		case Dsetp:
			return "DSETP";
		case Exit:
			return "EXIT";
		case F2f:
			return "F2F";
		case F2i:
			return "F2I";
		case Fadd:
			return "FADD";
		case Fchk:
			return "FCHK";
		case Fcmp:
			return "FCMP";
		case Ffma:
			return "FFMA";
		case Flo:
			return "FLO";
		case Fmnmx:
			return "FMNMX";
		case Fmul:
			return "FMUL";
		case Fset:
			return "FSET";
		case Fsetp:
			return "FSETP";
		case Fswz:
			return "FSWZ";
		case I2f:
			return "I2F";
		case I2i:
			return "I2I";
		case Iadd:
			return "IADD";
		case Icmp:
			return "ICMP";
		case Imad:
			return "IMAD";
		case Imadsp:
			return "IMADSP";
		case Imnmx:
			return "IMNMX";
		case Imul:
			return "IMUL";
		case Isad:
			return "ISAD";
		case Iscadd:
			return "ISCADD";
		case Iset:
			return "ISET";
		case Isetp:
			return "ISETP";
		case Jcal:
			return "JCAL";
		case Jmp:
			return "JMP";
		case Jmx:
			return "JMX";
		case Ld:
			return "LD";
		case Ldc:
			return "LDC";
		case Ldg:
			return "LDG";
		case Ldl:
			return "LDL";
		case Lds:
			return "LDS";
		case Ldslk:
			return "LDSLK";
		case Lop:
			return "LOP";
		case Membar:
			return "MEMBAR";
		case Mov:
			return "MOV";
		case Mov32i:
			return "MOV32I";
		case Mufu:
			return "MUFU";
		case Nop:
			return "NOP";
		case P2r:
			return "P2R";
		case Pbk:
			return "PBK";
		case Pcnt:
			return "PCNT";
		case Popc:
			return "POPC";
		case Pret:
			return "PRET";
		case Prmt:
			return "PRMT";
		case Pset:
			return "PSET";
		case Psetp:
			return "PSETP";
		case R2p:
			return "R2P";
		case Red:
			return "RED";
		case Ret:
			return "RET";
		case Rro:
			return "RRO";
		case S2r:
			return "S2R";
		case Sel:
			return "SEL";
		case Shf:
			return "SHF";
		case Shfl:
			return "SHFL";
		case Shl:
			return "SHL";
		case Shr:
			return "SHR";
		case Ssy:
			return "SSY";
		case St:
			return "ST";
		case Stl:
			return "STL";
		case Sts:
			return "STS";
		case Stscul:
			return "STSCUL";
		case Stul:
			return "STUL";
		case Stsul:
			return "STSUL";
		case Subfm:
			return "SUBFM";
		case Suclamp:
			return "SUCLAMP";
		case Sueau:
			return "SUEAU";
		case Suld:
			return "SULD";
		case Suldga:
			return "SULDGA";
		case Sulea:
			return "SULEA";
		case Sust:
			return "SUST";
		case Sured:
			return "SURED";
		case Sustga:
			return "SUSTGA";
		case Suq:
			return "SUQ";
		case Tex:
			return "TEX";
		case Texdepbar:
			return "TEXDEPBAR";
		case Tld:
			return "TLD";
		case Tld4:
			return "TLD4";
		case Txq:
			return "TXQ";
		case Vabsdiff4:
			return "VABSDIFF4";
		case Vote:
			return "VOTE";
		default:
			return "*UNKNOWN*";
		}
	}

	std::string SASSInstruction::toString() const {
		return toString((std::map<std::string, unsigned int> *)NULL);
	}

	std::string SASSInstruction::toString(
		std::map<std::string, unsigned int> *m) const {
		std::string ret = toString(opcode);
		if (_predicate.length() > 0) {
			ret = "@" + _predicate + " " + ret;
		}
		for (std::vector<std::string>::const_iterator mit =
			_modifiers.begin();
			mit != _modifiers.end(); mit++) {
			ret += "." + *mit;
		}
		ret += " ";
		for (std::vector<std::string>::const_iterator oit =
			_operands.begin();
			oit != _operands.end(); oit++) {
			std::string opstr = *oit;
			if (m && m->find(opstr) != m->end()) {
				unsigned int addr = (*m)[opstr];
				opstr = translator::PTXToSASSTranslator::makeImmediate(addr);
			}
			ret += opstr +
				(((oit+1) == _operands.end()) ? ";":", ");
		}
		return ret;
	}

	const SASSInstruction& SASSInstruction::operator=(
		const SASSInstruction& instr) {
		opcode = instr.opcode;
		_predicate = instr.getPredicate();
		_modifiers = instr.getModifiers();
		_operands = instr.getOperands();
		return *this;
	}

	SASSInstruction* SASSInstruction::clone() const {
		SASSInstruction* ret = new SASSInstruction(opcode);
		ret->setPredicate(_predicate);
		for (std::vector<std::string>::const_iterator mit =
			_modifiers.begin();
			mit != _modifiers.end(); mit++) {
			ret->addModifier(*mit);
		}
		for (std::vector<std::string>::const_iterator oit =
			_operands.begin();
			oit != _operands.end(); oit++) {
			ret->addOperand(*oit);
		}
		return ret;
	}

	void SASSInstruction::setPredicate(std::string predicate) {
		_predicate = predicate;
	}

	void SASSInstruction::addModifier(std::string modifier) {
		_modifiers.push_back(modifier);
	}

	void SASSInstruction::addOperand(std::string operand) {
		_operands.push_back(operand);
	}

	std::string SASSInstruction::getPredicate() const {
		return _predicate; 
	}

	std::vector<std::string> SASSInstruction::getModifiers() const {
		std::vector<std::string> ret;
		std::copy(_modifiers.begin(), _modifiers.end(),
			std::back_inserter(ret));
		return ret;
	}

	std::vector<std::string> SASSInstruction::getOperands() const {
		std::vector<std::string> ret;
		std::copy(_operands.begin(), _operands.end(),
			std::back_inserter(ret));
		return ret;
	}
}
