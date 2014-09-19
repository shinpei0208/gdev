/*!
 * \file SASSInstruction.h
*/

#ifndef __SASS_INSTRUCTION_H__
#define __SASS_INSTRUCTION_H__

#include <string>
#include <vector>
#include <map>

namespace ir {
	class SASSInstruction {
	public:
		enum Opcode {
			Opcode_Invalid = -1,
			Atom,
			B2r,
			Bar,
			Bfe,
			Bfi,
			Bpt,
			Bra,
			Brk,
			Brx,
			Cal,
			Cctl,
			Cctll,
			Cont,
			Cset,
			Csetp,
			Dadd,
			Dfma,
			Dmnmx,
			Dmul,
			Dset,
			Dsetp,
			Exit,
			F2f,
			F2i,
			Fadd,
			Fchk,
			Fcmp,
			Ffma,
			Flo,
			Fmnmx,
			Fmul,
			Fset,
			Fsetp,
			Fswz,
			I2f,
			I2i,
			Iadd,
			Icmp,
			Imad,
			Imadsp,
			Imnmx,
			Imul,
			Isad,
			Iscadd,
			Iset,
			Isetp,
			Jcal,
			Jmp,
			Jmx,
			Ld,
			Ldc,
			Ldg,
			Ldl,
			Lds,
			Ldslk,
			Long_jmp,
			Lop,
			Membar,
			Mov,
			Mov32i,
			Mufu,
			Nop,
			P2r,
			Pbk,
			Pcnt,
			Plong_jmp,
			Popc,
			Pret,
			Prmt,
			Pset,
			Psetp,
			R2p,
			Red,
			Ret,
			Rro,
			S2r,
			Sel,
			Shf,
			Shfl,
			Shl,
			Shr,
			Ssy,
			St,
			Stl,
			Sts,
			Stscul,
			Stul,
			Stsul,
			Subfm,
			Suclamp,
			Sueau,
			Suld,
			Suldga,
			Sulea,
			Sust,
			Sured,
			Sustga,
			Suq,
			Tex,
			Texdepbar,
			Tld,
			Tld4,
			Txq,
			Vabsdiff4,
			Vote,
		};

		Opcode opcode;

		SASSInstruction(Opcode op = Opcode_Invalid);
		std::string toString() const;
		std::string toString(std::map<std::string, unsigned int> *m) const;
		static std::string toString(Opcode op);
		const SASSInstruction& operator=(const SASSInstruction& instr);
		SASSInstruction* clone() const;

		void setPredicate(std::string predicate);
		void addModifier(std::string modifier);
		void addOperand(std::string operand);

		std::string getPredicate() const;
		std::vector<std::string> getModifiers() const;
		std::vector<std::string> getOperands() const;

	private:
		std::string _predicate;
		std::vector<std::string> _modifiers;
		std::vector<std::string> _operands;
	};

}

#endif /* __SASS_INSTRUCTION_H__ */
